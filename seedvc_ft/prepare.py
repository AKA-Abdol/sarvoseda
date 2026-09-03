"""Assemble a seed-vc training directory from pipeline output.

seed-vc's fine-tuning loader takes a flat directory of audio files, so this
stage's job is selection and linking, not conversion - ``vcprep materialize``
has already written 22.05 kHz mono FLAC.

Files are **hardlinked** by default: the training set costs no extra disk and
still looks like an ordinary directory to seed-vc.
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import DataConfig

log = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".mp3", ".m4a", ".opus", ".ogg"}


class Candidate:
    __slots__ = ("path", "uid", "source", "duration", "text", "scores")

    def __init__(self, path: Path, uid: str, source: str, duration: float,
                 text: str, scores: Dict[str, float]):
        self.path = path
        self.uid = uid
        self.source = source
        self.duration = duration
        self.text = text
        self.scores = scores


def collect(cfg: DataConfig) -> List[Candidate]:
    """Read every source's ``metadata.csv``, falling back to a directory scan."""
    roots: List[Path] = [Path(cfg.clean_dir)]
    if cfg.include_low_quality:
        roots.append(Path(cfg.low_quality_dir))

    found: List[Candidate] = []
    for root in roots:
        if not root.exists():
            log.warning("missing input directory: %s", root)
            continue
        metadata_files = sorted(root.glob("*/metadata.csv"))
        if metadata_files:
            for meta in metadata_files:
                found.extend(_from_metadata(meta))
        else:
            log.info("no metadata.csv under %s - scanning for audio", root)
            found.extend(_from_scan(root))
    return found


def _from_metadata(meta: Path) -> List[Candidate]:
    out: List[Candidate] = []
    with open(meta, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rel = row.get("file") or ""
            if not rel:
                continue
            path = (meta.parent / rel).resolve()
            if not path.exists():
                continue
            scores = {}
            for key in ("dnsmos_ovrl", "dnsmos_bak", "dnsmos_sig", "nisqa_dis"):
                try:
                    scores[key] = float(row[key])
                except (KeyError, TypeError, ValueError):
                    pass
            try:
                duration = float(row.get("duration") or 0.0)
            except ValueError:
                duration = 0.0
            out.append(Candidate(path, row.get("uid") or path.stem,
                                 row.get("source") or meta.parent.name,
                                 duration, row.get("text") or "", scores))
    return out


def _from_scan(root: Path) -> List[Candidate]:
    import soundfile as sf

    out: List[Candidate] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in AUDIO_EXTS:
            continue
        try:
            info = sf.info(str(path))
            duration = float(info.duration)
        except Exception:
            continue
        source = path.parent.name if path.parent != root else "unknown"
        out.append(Candidate(path.resolve(), path.stem, source, duration, "", {}))
    return out


def filter_candidates(items: List[Candidate], cfg: DataConfig
                      ) -> Tuple[List[Candidate], Dict[str, int]]:
    wanted_sources = set(cfg.sources)
    dropped: Dict[str, int] = {}

    def drop(reason: str) -> None:
        dropped[reason] = dropped.get(reason, 0) + 1

    kept: List[Candidate] = []
    for item in items:
        if wanted_sources and item.source not in wanted_sources:
            drop("other_source")
            continue
        if item.duration and not (cfg.min_duration <= item.duration <= cfg.max_duration):
            drop("duration")
            continue
        ovrl = item.scores.get("dnsmos_ovrl")
        if cfg.min_dnsmos_ovrl and ovrl is not None and ovrl < cfg.min_dnsmos_ovrl:
            drop("below_ovrl_floor")
            continue
        bak = item.scores.get("dnsmos_bak")
        if cfg.min_dnsmos_bak and bak is not None and bak < cfg.min_dnsmos_bak:
            drop("below_bak_floor")
            continue
        kept.append(item)
    return kept, dropped


def build(cfg: DataConfig) -> Dict[str, object]:
    """Select, link and split. Returns a summary dict (also written to disk)."""
    items = collect(cfg)
    log.info("found %d candidate clips", len(items))
    kept, dropped = filter_candidates(items, cfg)
    log.info("%d pass the training filters", len(kept))
    if not kept:
        raise RuntimeError(
            "no clips survived filtering - check --clean-dir and the duration "
            "and score floors"
        )

    rng = random.Random(cfg.seed)
    rng.shuffle(kept)
    if cfg.max_files:
        kept = kept[: cfg.max_files]

    n_val = int(len(kept) * cfg.val_fraction)
    # A val split is only meaningful if it leaves a real training set behind.
    if len(kept) < 50:
        n_val = 0
    val, train = kept[:n_val], kept[n_val:]

    dataset_dir = Path(cfg.dataset_dir)
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    for directory in (train_dir, val_dir):
        directory.mkdir(parents=True, exist_ok=True)

    _place(train, train_dir, cfg.materialize)
    _place(val, val_dir, cfg.materialize)

    _write_index(dataset_dir / "train.csv", train)
    if val:
        _write_index(dataset_dir / "val.csv", val)

    hours = sum(c.duration for c in train) / 3600.0
    by_source: Dict[str, int] = {}
    for candidate in train:
        by_source[candidate.source] = by_source.get(candidate.source, 0) + 1

    summary = {
        "dataset_dir": str(dataset_dir.resolve()),
        "train_dir": str(train_dir.resolve()),
        "train_files": len(train),
        "val_files": len(val),
        "train_hours": round(hours, 2),
        "by_source": by_source,
        "dropped": dropped,
        "sample_rate": cfg.sample_rate,
        "materialize": cfg.materialize,
    }
    with open(dataset_dir / "dataset_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)
    log.info("prepared %d train / %d val clips (%.2f h) in %s",
             len(train), len(val), hours, dataset_dir)
    return summary


def _place(items: List[Candidate], dest_dir: Path, mode: str) -> None:
    for item in items:
        target = dest_dir / f"{item.uid}{item.path.suffix}"
        if target.exists():
            continue
        try:
            if mode == "link":
                os.link(item.path, target)
            elif mode == "symlink":
                os.symlink(item.path, target)
            else:
                shutil.copy2(item.path, target)
        except OSError:
            # Hardlinks fail across filesystems; copying always works.
            shutil.copy2(item.path, target)


def _write_index(path: Path, items: List[Candidate]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["uid", "source", "duration", "text", "dnsmos_ovrl"])
        for item in items:
            writer.writerow([item.uid, item.source, f"{item.duration:.3f}",
                             item.text, item.scores.get("dnsmos_ovrl", "")])
