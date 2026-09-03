"""Stage 2 - cheap gate on *raw* audio, ahead of the GPU.

Separation is the most expensive node in the pipeline, and a movie-sourced
dataset contains many clips that are pure music, effects or room tone. There is
no reason to spend GPU time isolating vocals from a clip that has none.

Deliberately **lenient**: it drops only what is unambiguously unusable (too
short, too long, near-silent, no detectable speech at all). Borderline calls go
to the post-separation VAD stage, which sees a much cleaner signal.

Fully parallel - see :mod:`vcprep.backends`.
"""
from __future__ import annotations

import logging
from typing import Optional

from .base import Node

log = logging.getLogger(__name__)


class PrefilterNode(Node):
    name = "prefilter"

    def run(self, shard: Optional[int] = None,
            source: Optional[str] = None) -> dict:
        records = self.pending(shard, source)
        stats = {"seen": len(records), "kept": 0, "rejected": 0}
        if not records:
            return stats
        if not self.cfg.prefilter.enabled:
            for rec in records:
                self.advance(rec)
            self.manifest.flush()
            return {**stats, "kept": len(records), "skipped": True}

        for rec in self.backend.map("prefilter", records, desc="prefilter"):
            stats["kept" if rec.alive else "rejected"] += 1
            self.advance(rec)

        self.manifest.flush()
        return stats
