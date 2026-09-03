"""Stage 4 - silence / no-speech detection on the separated vocals.

Two jobs:

1. **Reject** clips with no usable speech. This is where music-only clips
   finally reveal themselves: with the instrumental stripped, what remains is
   separation residue whose speech ratio collapses toward zero. An energy
   threshold cannot make that distinction; a neural VAD can.
2. **Trim** to the speech span. Movie clips carry dead air at both ends, and
   seed-vc trains better on utterances that are mostly voice.

Fully parallel - see :mod:`vcprep.backends`.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from .base import Node

log = logging.getLogger(__name__)


class VadNode(Node):
    name = "vad"

    def __init__(self, cfg, manifest=None):
        super().__init__(cfg, manifest)
        self.out_dir = Path(cfg.paths.work_dir) / "trimmed"

    def run(self, shard: Optional[int] = None,
            source: Optional[str] = None) -> dict:
        records = self.pending(shard, source)
        stats = {"seen": len(records), "kept": 0, "rejected": 0}
        if not records:
            return stats
        if not self.cfg.vad.enabled:
            for rec in records:
                self.advance(rec)
            self.manifest.flush()
            return {**stats, "kept": len(records), "skipped": True}

        unit = self.unit_dir(self.out_dir, source, shard)
        results = self.backend.map("vad", records,
                                   extra={"out_dir": str(unit)},
                                   desc="silence/VAD")
        for rec in results:
            stats["kept" if rec.alive else "rejected"] += 1
            self.advance(rec)

        self.manifest.flush()
        return stats
