"""Stage 6 - write the verdicts out as folders on disk.

Deliberately the *only* stage that decides where a file lives. Every score
already sits in the manifest, so changing a threshold means re-running this
stage (seconds) rather than re-scoring the corpus (days).

Layout, named by the Hugging Face repo each clip came from::

    out/
      clean/<repo-slug>/<repo-slug>__<original>.flac
      low_quality/<repo-slug>/...
      rejected/<repo-slug>/...              (only with keep_rejected_audio)
      clean/<repo-slug>/metadata.csv        uid,file,duration,text,scores
      clean/metadata.csv                    the same, across all sources
"""
from __future__ import annotations

import csv
import logging
import math
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from ..audio import load_audio, save_audio
from ..manifest import KEEP, LOW_QUALITY, PENDING, REJECTED
from .base import Node

log = logging.getLogger(__name__)

_SUBTYPES = {"FLAC": "PCM_16", "WAV": "PCM_16"}

_SCORE_COLUMNS = ["dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak", "nisqa_mos",
                  "nisqa_dis", "squim_pesq", "vad_snr_db", "bandwidth_hz",
                  "speech_ratio"]


class MaterializeNode(Node):
    name = "materialize"

    def __init__(self, cfg, manifest=None):
        super().__init__(cfg, manifest)
        self.out_root = Path(cfg.paths.out_dir)
        #: set by the runner; metadata.csv spans every partition, while audio
        #: placement only ever touches the current work unit.
        self.store = None

    # ------------------------------------------------------------------
    def run(self, shard: Optional[int] = None,
            source: Optional[str] = None) -> dict:
        cfg = self.cfg.materialize
        stats = {"clean": 0, "low_quality": 0, "rejected": 0, "failed": 0}
        if not cfg.enabled:
            return {**stats, "skipped": True}

        # PENDING counts as clean: it means the record survived every stage
        # that ran and was never rejected or demoted. Without this, disabling
        # the quality stage (--no-quality) would leave everything pending and
        # produce an empty output directory.
        targets = {
            KEEP: cfg.clean_dir,
            PENDING: cfg.clean_dir,
            LOW_QUALITY: cfg.low_quality_dir,
        }
        if cfg.keep_rejected_audio:
            targets[REJECTED] = cfg.rejected_dir

        records = [r for r in self.manifest.records()
                   if r.status in targets
                   and (source is None or r.source == source)
                   and (shard is None or r.shard == shard)]

        ext = cfg.output_format.lower()
        for rec in tqdm(records, desc="materialize", unit="clip", leave=False):
            bucket = targets[rec.status]
            dest_dir = self.out_root / bucket / (rec.source or "unknown")
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / f"{rec.uid}.{ext}"

            if dest.exists() and rec.stage == self.name:
                stats[_key(rec.status)] += 1
                continue
            if not os.path.exists(rec.path):
                # Already materialised on a previous run and the source removed.
                if dest.exists():
                    stats[_key(rec.status)] += 1
                    rec.path = str(dest)
                    self.advance(rec)
                    continue
                log.warning("missing audio for %s (%s)", rec.uid, rec.path)
                stats["failed"] += 1
                continue

            try:
                self._write(rec, dest, cfg)
                stats[_key(rec.status)] += 1
            except Exception as exc:
                log.warning("materialize failed for %s: %s", rec.uid, exc)
                stats["failed"] += 1
                continue
            self.advance(rec)

        self.manifest.flush()
        self._write_metadata(cfg)
        return stats

    # ------------------------------------------------------------------
    def _write(self, rec, dest: Path, cfg) -> None:
        audio, sr = load_audio(rec.path, sr=cfg.sample_rate, mono=cfg.mono)
        save_audio(str(dest), audio, sr,
                   subtype=_SUBTYPES.get(cfg.output_format.upper(), "PCM_16"))
        rec.duration = len(audio) / float(sr)
        rec.sample_rate = sr
        if cfg.mode == "move":
            try:
                os.unlink(rec.path)
            except OSError:
                pass
        rec.path = str(dest)

    # ------------------------------------------------------------------
    def _write_metadata(self, cfg) -> None:
        """One CSV per source plus a combined one, for each bucket."""
        for status, bucket in ((KEEP, cfg.clean_dir),
                               (LOW_QUALITY, cfg.low_quality_dir)):
            source = self.store if self.store is not None else self.manifest
            records = [r for r in source.records()
                       if r.stage == self.name
                       and (r.status == status
                            or (status == KEEP and r.status == PENDING))]
            if not records:
                continue
            root = self.out_root / bucket
            self._dump_csv(root / "metadata.csv", records)

            by_source: Dict[str, List] = {}
            for rec in records:
                by_source.setdefault(rec.source or "unknown", []).append(rec)
            for slug, group in by_source.items():
                self._dump_csv(root / slug / "metadata.csv", group)

    def _dump_csv(self, path: Path, records: List) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = ["uid", "source", "file", "duration", "text"] + _SCORE_COLUMNS
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for rec in sorted(records, key=lambda r: r.uid):
                rel = os.path.relpath(rec.path, path.parent)
                row = [rec.uid, rec.source, rel, f"{rec.duration:.3f}", rec.text]
                row += [_fmt(rec.metrics.get(col)) for col in _SCORE_COLUMNS]
                writer.writerow(row)
        log.info("wrote %s (%d rows)", path, len(records))


def _fmt(value) -> str:
    """Unmeasurable metrics render as an empty cell, never the string "nan"."""
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return "" if not math.isfinite(number) else f"{number:.4f}"


def _key(status: str) -> str:
    return {KEEP: "clean", PENDING: "clean", LOW_QUALITY: "low_quality",
            REJECTED: "rejected"}.get(status, "failed")
