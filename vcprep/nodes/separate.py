"""Stage 3 - vocal isolation (UVR models, headless).

Runs UVR's models through ``audio-separator``, the maintained library
extraction of UVR's inference code. Defaults to Mel-Band RoFormer (12.60 vocal
SDR) over Kim Vocal 2 (10.18); either works, and so does any other model in the
registry or a local checkpoint.

**Why this stage batches.** These separators run a fixed-size inference window
- Kim Vocal 2's is 256 frames x 1024 hop = 5.94 s at 44.1 kHz - and pad
anything shorter up to it. Filimo utterances average about two seconds, so
separating them one at a time throws away most of the GPU: measured 1.54x
realtime per clip against 9.65x on a long file, i.e. ~76% of the time was
per-call overhead rather than model compute. Packing many clips into one long
input and slicing the result apart afterwards recovers that.

Two details make the packing faithful, so that a clip's output does not depend
on which batch it happened to land in:

* each clip is peak-normalised before packing and has its gain restored after,
  because the separator normalises its input globally - otherwise one loud clip
  would quietly attenuate its neighbours;
* a short silence is inserted between clips, so neighbouring audio cannot bleed
  across a boundary that falls inside the model's receptive field.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

from ..audio import load_audio, save_audio
from ..manifest import REJECTED, Record
from .base import Node

log = logging.getLogger(__name__)


class SeparateNode(Node):
    name = "separate"

    def __init__(self, cfg, manifest=None):
        super().__init__(cfg, manifest)
        self.out_dir = Path(cfg.paths.work_dir) / "vocals"
        self.staging = self.out_dir / "_staging"
        self._separator = None

    # ------------------------------------------------------------------
    def _load(self):
        if self._separator is not None:
            return self._separator

        from audio_separator.separator import Separator

        cfg = self.cfg.separate.resolve()
        model_dir = Path(os.path.expanduser(cfg.model_dir))
        model_dir.mkdir(parents=True, exist_ok=True)
        local = model_dir / cfg.model_name
        if not local.exists():
            log.info("%s not in %s - audio-separator will fetch it",
                     cfg.model_name, model_dir)

        # audio-separator resolves models by *basename* against its own
        # registry (that is where the architecture parameters come from) and
        # looks for the file in model_file_dir. It also captures output_dir at
        # load time and rewrites custom output names, so we let it write into a
        # staging directory and place the results ourselves.
        self.staging.mkdir(parents=True, exist_ok=True)
        sep = Separator(
            model_file_dir=str(model_dir),
            output_dir=str(self.staging),
            output_format=cfg.output_format,
            output_single_stem=None if cfg.keep_instrumental else "Vocals",
            sample_rate=cfg.sample_rate,
            normalization_threshold=cfg.normalization_threshold,
            use_autocast=cfg.use_autocast,
            mdx_params=dict(cfg.mdx_params),
            mdxc_params=dict(cfg.mdxc_params),
        )
        log.info("loading separation model %s from %s", cfg.model_name, model_dir)
        try:
            sep.load_model(model_filename=cfg.model_name)
        except Exception as exc:
            # An interrupted download leaves a truncated checkpoint behind, and
            # audio-separator will happily reuse it forever. Say so plainly.
            if local.exists():
                size_mb = local.stat().st_size / (1024 ** 2)
                raise RuntimeError(
                    f"could not load {local} ({size_mb:.1f} MB): {exc}\n"
                    f"A partial download is the usual cause. Delete it and "
                    f"re-run:\n    rm {local}"
                ) from exc
            raise
        self._separator = sep
        return sep

    # ------------------------------------------------------------------
    def run(self, shard: Optional[int] = None,
            source: Optional[str] = None) -> dict:
        records = self.pending(shard, source)
        stats = {"seen": 0, "separated": 0, "failed": 0, "batches": 0}
        if not records:
            return stats
        if not self.cfg.separate.enabled:
            for rec in records:
                self.advance(rec)
            self.manifest.flush()
            return {**stats, "seen": len(records), "skipped": True}

        sep = self._load()
        out_dir = self.unit_dir(self.out_dir, source, shard)

        if self.cfg.separate.batch_clips:
            stats.update(self._run_batched(sep, records, out_dir))
        else:
            stats.update(self._run_per_clip(sep, records, out_dir))

        self.manifest.flush()
        return stats

    # ------------------------------------------------------------------
    def _run_per_clip(self, sep, records: List[Record], out_dir: Path) -> dict:
        stats = {"seen": 0, "separated": 0, "failed": 0, "batches": 0}
        for i, rec in enumerate(tqdm(records, desc="separate", unit="clip",
                                     leave=False)):
            stats["seen"] += 1
            try:
                outputs = sep.separate(rec.path)
                produced = _pick_vocal(outputs, self.staging)
                if produced is None:
                    raise RuntimeError("separator produced no vocal stem")
                final = out_dir / f"{rec.uid}{produced.suffix}"
                shutil.move(str(produced), str(final))
                _clear_dir(self.staging)
                self._commit(rec, final)
                stats["separated"] += 1
            except Exception as exc:
                log.warning("separation failed for %s: %s", rec.uid, exc)
                rec.reject("separation_failed", REJECTED)
                stats["failed"] += 1
            self.advance(rec)
            if (i + 1) % 200 == 0:
                self.manifest.flush()
        return stats

    # ------------------------------------------------------------------
    def _run_batched(self, sep, records: List[Record], out_dir: Path) -> dict:
        cfg = self.cfg.separate
        stats = {"seen": 0, "separated": 0, "failed": 0, "batches": 0}
        batches = _plan_batches(records, cfg.batch_seconds)
        log.info("separating %d clips in %d packed batch(es) of <=%.0fs",
                 len(records), len(batches), cfg.batch_seconds)

        for batch in tqdm(batches, desc="separate (packed)", unit="batch",
                          leave=False):
            stats["seen"] += len(batch)
            stats["batches"] += 1
            try:
                done = self._separate_batch(sep, batch, out_dir)
                stats["separated"] += done
                stats["failed"] += len(batch) - done
            except Exception as exc:
                log.warning("packed batch failed (%d clips): %s", len(batch), exc)
                # Never lose a whole batch to one bad clip - retry it singly.
                # Skip any clip the batch had already committed, or we would
                # separate an output file a second time.
                retry = [r for r in batch
                         if "separated_from" not in r.metrics and r.alive]
                stats["separated"] += len(batch) - len(retry)
                if retry:
                    fallback = self._run_per_clip(sep, retry, out_dir)
                    stats["separated"] += fallback["separated"]
                    stats["failed"] += fallback["failed"]
            for rec in batch:
                self.advance(rec)
            self.manifest.flush()
        return stats

    # ------------------------------------------------------------------
    def _separate_batch(self, sep, batch: List[Record], out_dir: Path) -> int:
        cfg = self.cfg.separate
        sr = cfg.sample_rate
        guard = np.zeros((int(cfg.batch_guard_seconds * sr), 2), dtype=np.float32)

        pieces: List[np.ndarray] = []
        spans: List[Tuple[Record, int, int, float]] = []
        cursor = 0
        for rec in batch:
            try:
                audio = _load_stereo(rec.path, sr)
            except Exception as exc:
                log.debug("unreadable in batch: %s (%s)", rec.uid, exc)
                rec.reject("unreadable", REJECTED)
                continue
            peak = float(np.max(np.abs(audio))) or 1.0
            gain = cfg.batch_peak / peak
            pieces.append(audio * gain)
            spans.append((rec, cursor, cursor + len(audio), gain))
            cursor += len(audio)
            pieces.append(guard)
            cursor += len(guard)

        if not spans:
            return 0

        packed = np.concatenate(pieces, axis=0)
        _clear_dir(self.staging)
        packed_path = self.staging / "_packed.wav"
        save_audio(str(packed_path), packed, sr, subtype="FLOAT")

        outputs = sep.separate(str(packed_path))
        produced = _pick_vocal(outputs, self.staging, exclude={"_packed"})
        if produced is None:
            raise RuntimeError("separator produced no vocal stem for the batch")

        vocals, out_sr = load_audio(str(produced), mono=False)
        if vocals.ndim == 1:
            vocals = vocals[:, None]
        if out_sr != sr:
            raise RuntimeError(f"separator returned {out_sr} Hz, expected {sr}")

        done = 0
        for rec, start, end, gain in spans:
            end = min(end, len(vocals))
            if end <= start:
                rec.reject("separation_failed", REJECTED)
                continue
            # Undo the packing gain so the clip comes back at its own level.
            clip = vocals[start:end] / gain
            final = out_dir / f"{rec.uid}.wav"
            save_audio(str(final), clip, sr)
            self._commit(rec, final)
            done += 1

        _clear_dir(self.staging)
        return done

    # ------------------------------------------------------------------
    def _commit(self, rec: Record, final: Path) -> None:
        rec.metrics["separated_from"] = rec.path
        if self.cfg.fetch.delete_after_separate:
            _unlink(rec.path)
        rec.path = str(final)


# ---------------------------------------------------------------------------
def _plan_batches(records: List[Record], target_seconds: float
                  ) -> List[List[Record]]:
    """Pack records into batches of roughly ``target_seconds`` of audio.

    Durations come from the prefilter stage; a record without one is assumed
    short rather than skipped, so a missing measurement cannot silently create
    an enormous batch.
    """
    batches: List[List[Record]] = []
    current: List[Record] = []
    total = 0.0
    for rec in records:
        duration = rec.duration if rec.duration and rec.duration > 0 else 3.0
        if current and total + duration > target_seconds:
            batches.append(current)
            current, total = [], 0.0
        current.append(rec)
        total += duration
    if current:
        batches.append(current)
    return batches


def _load_stereo(path: str, sr: int) -> np.ndarray:
    """Load at ``sr`` as (n, 2) - packing needs a uniform channel count."""
    audio, _ = load_audio(path, sr=sr, mono=False)
    if audio.ndim == 1:
        audio = np.stack([audio, audio], axis=1)
    elif audio.shape[1] == 1:
        audio = np.repeat(audio, 2, axis=1)
    elif audio.shape[1] > 2:
        audio = audio[:, :2]
    return np.ascontiguousarray(audio, dtype=np.float32)


def _pick_vocal(outputs, staging: Path, exclude=frozenset()) -> Optional[Path]:
    """Resolve the vocal stem; returned names are relative to output_dir."""
    candidates = []
    for name in outputs or []:
        path = Path(name)
        if not path.is_absolute():
            path = staging / path.name
        if path.exists() and path.stem not in exclude:
            candidates.append(path)
    if not candidates:
        # Defensive: if the return value stops matching what was written, fall
        # back to whatever landed in the single-use staging directory.
        candidates = [p for p in staging.iterdir()
                      if p.is_file() and p.stem not in exclude]
    if not candidates:
        return None
    for path in candidates:
        if "instrument" not in path.stem.lower():
            return path
    return candidates[0]


def _clear_dir(directory: Path) -> None:
    if not directory.exists():
        return
    for path in directory.iterdir():
        if path.is_file():
            _unlink(str(path))


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
