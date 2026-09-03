"""Cheap signal measurements that MOS predictors do not report.

These cost microseconds and catch failure modes DNSMOS scores right through:

  * **bandwidth**  - streaming rips lowpassed at 8-11 kHz. A 22.05 kHz seed-vc
    model trained on these learns to synthesise a band that is not in the data.
  * **clipping**   - encoder or mastering clipping survives separation and
    becomes a hard artifact the vocoder will faithfully reproduce.
  * **VAD SNR**    - direct measurement of residual noise between words, which
    is where separation leftovers are most audible.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..audio import (clipping_ratio, dc_offset, loudness_lufs, peak_dbfs,
                     rms_dbfs, snr_estimate_db, spectral_bandwidth_hz)


def speech_mask_from_segments(n_samples: int, sr: int,
                              segments: List[Tuple[float, float]]) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    for start, end in segments:
        a = max(0, int(start * sr))
        b = min(n_samples, int(end * sr))
        if b > a:
            mask[a:b] = True
    return mask


def compute(audio: np.ndarray, sr: int,
            speech_segments: Optional[List[Tuple[float, float]]] = None
            ) -> Dict[str, float]:
    out: Dict[str, float] = {
        "rms_dbfs": rms_dbfs(audio),
        "peak_dbfs": peak_dbfs(audio),
        "clipping_ratio": clipping_ratio(audio),
        "dc_offset": dc_offset(audio),
        "bandwidth_hz": spectral_bandwidth_hz(audio, sr),
        "lufs": loudness_lufs(audio, sr),
    }
    if speech_segments:
        mask = speech_mask_from_segments(len(audio), sr, speech_segments)
        out["vad_snr_db"] = snr_estimate_db(audio, mask)
    return {k: (float(v) if np.isfinite(v) else float("nan")) for k, v in out.items()}
