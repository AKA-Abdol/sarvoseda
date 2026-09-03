"""Threshold calibration.

Published DNSMOS thresholds come from English noisy-speech corpora. This is
Persian film audio that has been through MDX-Net separation, so the absolute
values sit somewhere else entirely, and a threshold copied from a paper will
either keep everything or throw the corpus away.

Run the pipeline over a small sample with quality scoring on but thresholds
wide open, then use this to read the real distribution and pick a percentile.
"""
from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence

from .manifest import Manifest

log = logging.getLogger(__name__)

#: metric -> (direction, config key). "min" = higher is better.
CALIBRATABLE = {
    "dnsmos_ovrl": ("min", "quality.min_ovrl"),
    "dnsmos_sig": ("min", "quality.min_sig"),
    "dnsmos_bak": ("min", "quality.min_bak"),
    "nisqa_dis": ("min", "quality.min_nisqa_dis"),
    "nisqa_mos": ("min", None),
    "squim_pesq": ("min", None),
    "vad_snr_db": ("min", None),
    "bandwidth_hz": ("min", "quality.min_bandwidth_hz"),
    "speech_ratio": ("min", "vad.min_speech_ratio"),
    "clipping_ratio": ("max", "quality.max_clipping_ratio"),
}

PERCENTILES = (1, 5, 10, 25, 50, 75, 90, 99)


def percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * (pct / 100.0)
    low = math.floor(pos)
    high = math.ceil(pos)
    if low == high:
        return ordered[int(pos)]
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def collect(manifest: Manifest, source: Optional[str] = None
            ) -> Dict[str, List[float]]:
    """Gather every finite value of every calibratable metric."""
    pools: Dict[str, List[float]] = {k: [] for k in CALIBRATABLE}
    for rec in manifest.records():
        if source and rec.source != source:
            continue
        for key in CALIBRATABLE:
            value = rec.metrics.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                pools[key].append(number)
    return {k: v for k, v in pools.items() if v}


def report(pools: Dict[str, List[float]], target_keep: float = 0.80) -> str:
    """Human-readable distribution plus a suggested threshold per metric.

    ``target_keep`` is the fraction to retain *per axis*. Axes are correlated,
    so the joint keep rate lands above the naive product but below any single
    axis - check the printed joint estimate rather than assuming.
    """
    lines: List[str] = []
    lines.append(f"{'metric':<18} {'n':>7} " +
                 " ".join(f"p{p:<4}" for p in PERCENTILES))
    lines.append("-" * (18 + 8 + 7 * len(PERCENTILES)))
    for key, values in pools.items():
        cells = " ".join(f"{percentile(values, p):<5.2f}" for p in PERCENTILES)
        lines.append(f"{key:<18} {len(values):>7} {cells}")

    lines.append("")
    lines.append(f"suggested thresholds (keep ~{target_keep:.0%} per axis):")
    for key, values in pools.items():
        direction, cfg_key = CALIBRATABLE[key]
        if direction == "min":
            cut = percentile(values, (1.0 - target_keep) * 100.0)
        else:
            cut = percentile(values, target_keep * 100.0)
        label = cfg_key or f"({key}: informational only)"
        lines.append(f"  {label:<28} {cut:.4f}")
    return "\n".join(lines)


def joint_keep_rate(manifest: Manifest, thresholds: Dict[str, float],
                    source: Optional[str] = None) -> float:
    """Fraction of scored clips that would pass all given thresholds at once."""
    total = passed = 0
    for rec in manifest.records():
        if source and rec.source != source:
            continue
        if not any(k in rec.metrics for k in thresholds):
            continue
        total += 1
        ok = True
        for key, limit in thresholds.items():
            value = rec.metrics.get(key)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(number):
                continue
            direction = CALIBRATABLE.get(key, ("min", None))[0]
            if (direction == "min" and number < limit) or \
               (direction == "max" and number > limit):
                ok = False
                break
        passed += int(ok)
    return (passed / total) if total else 0.0
