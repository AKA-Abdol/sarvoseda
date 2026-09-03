"""Stage 5 - perceptual quality scoring and the clean / low-quality split.

Scorers, and why each is here:

  **DNSMOS P.835** (primary). All three axes are used, not just OVRL. BAK
  grades residual background, which is a direct measurement of how well the
  separator did on this clip. SIG catches voices the separator damaged while
  cleaning.

  **NISQA v2** (artifacts). Contributes the *discontinuity* axis, which DNSMOS
  does not have. These separators fail by warbling and chopping the vocal, and
  that is what ``dis_pred`` measures.

  **SQUIM** (second opinion). Reference-free PESQ/STOI from torchaudio.

  **Heuristics** (structural). Bandwidth, clipping and inter-word SNR -
  defects MOS models score right through but that hurt a 22.05 kHz vocoder.

Scores are written to the manifest unconditionally; thresholds only set the
``status`` field, so retuning costs a ``materialize`` re-run rather than a
re-score. Run ``vcprep calibrate`` before trusting the shipped defaults.

Per-clip scoring is parallel. NISQA is not: it runs once per directory, and its
results are attached to each record before dispatch.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

from ..manifest import KEEP, LOW_QUALITY
from .base import Node

log = logging.getLogger(__name__)


class QualityNode(Node):
    name = "quality"

    def __init__(self, cfg, manifest=None):
        super().__init__(cfg, manifest)
        self._nisqa = None

    def _get_nisqa(self):
        if self._nisqa is None:
            from ..metrics.nisqa import Nisqa
            self._nisqa = Nisqa(self.cfg.quality.nisqa_repo,
                                self.cfg.quality.nisqa_weights,
                                device=self.cfg.quality.device)
        return self._nisqa

    # ------------------------------------------------------------------
    def run(self, shard: Optional[int] = None,
            source: Optional[str] = None) -> dict:
        cfg = self.cfg.quality
        records = self.pending(shard, source)
        stats = {"seen": len(records), "keep": 0, "low_quality": 0, "failed": 0}
        if not records:
            return stats
        if not cfg.enabled:
            for rec in records:
                rec.status = KEEP
                self.advance(rec)
            self.manifest.flush()
            return {**stats, "keep": len(records), "skipped": True}

        # NISQA has real startup cost and a directory-batch interface, so it
        # runs once for the whole unit and its scores ride along on each record.
        if cfg.use_nisqa:
            audio_dir = str(Path(records[0].path).parent)
            log.info("running NISQA over %s", audio_dir)
            try:
                scores = self._get_nisqa().score_dir(audio_dir)
                for rec in records:
                    found = scores.get(Path(rec.path).stem)
                    if found:
                        rec.metrics.update(found)
            except Exception as exc:
                log.error("NISQA unavailable (%s) - continuing without it", exc)

        for rec in self.backend.map("quality", records, desc="quality"):
            if rec.status == KEEP:
                stats["keep"] += 1
            elif rec.status == LOW_QUALITY:
                stats["low_quality"] += 1
            else:
                stats["failed"] += 1
            self.advance(rec)

        self.manifest.flush()
        return stats
