"""Per-record stage logic, as free functions.

Extracted from the node classes so that the serial path and the parallel
workers run *exactly* the same code. A worker process calls these directly;
the nodes call them in a loop. There is no second implementation to drift.

Anything expensive and reusable (VAD, DNSMOS, SQUIM) is cached per process in
:data:`_MODELS`, so a pool worker pays the load cost once, not once per clip.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .audio import (active_speech_ratio, load_audio, probe, rms_dbfs,
                    save_audio, snr_estimate_db, trim_to)
from .config import PipelineConfig
from .manifest import KEEP, LOW_QUALITY, PENDING, REJECTED, Record
from .metrics import heuristics
from .metrics.heuristics import speech_mask_from_segments

log = logging.getLogger(__name__)

#: process-local model cache
_MODELS: Dict[str, Any] = {}


def get_dnsmos(cfg: PipelineConfig):
    if "dnsmos" not in _MODELS:
        from .metrics.dnsmos import DNSMOS
        _MODELS["dnsmos"] = DNSMOS(cfg.quality.dnsmos_dir,
                                   device=cfg.quality.device)
    return _MODELS["dnsmos"]


def get_squim(cfg: PipelineConfig):
    if "squim" not in _MODELS:
        from .metrics.squim import Squim
        _MODELS["squim"] = Squim(device=cfg.quality.device)
    return _MODELS["squim"]


# ---------------------------------------------------------------------------
# prefilter
# ---------------------------------------------------------------------------
def prefilter_record(rec: Record, cfg: PipelineConfig) -> Record:
    """Cheap gate on raw audio, ahead of the GPU. Lenient by design."""
    pf = cfg.prefilter
    try:
        duration, sr, _ = probe(rec.path)
        rec.duration = duration
        rec.sample_rate = sr

        if duration < pf.min_duration:
            rec.reject("too_short", REJECTED)
            return rec
        if duration > pf.max_duration:
            rec.reject("too_long", REJECTED)
            return rec

        audio, sr = load_audio(rec.path, mono=True)
        level = rms_dbfs(audio)
        rec.metrics["raw_rms_dbfs"] = round(level, 2)
        if level < pf.min_dbfs:
            rec.reject("silent_raw", REJECTED)
            return rec

        rec.metrics["raw_active_ratio"] = round(active_speech_ratio(audio, sr), 4)

        if pf.use_vad:
            from . import vad_engine
            info = vad_engine.analyse(
                audio, sr,
                threshold=pf.vad_threshold,
                min_speech_duration_ms=150,
                min_silence_duration_ms=200,
                speech_pad_ms=100,
            )
            rec.metrics["raw_speech_seconds"] = round(info["speech_seconds"], 3)
            rec.metrics["raw_speech_ratio"] = round(info["speech_ratio"], 4)
            if info["speech_seconds"] < pf.min_speech_seconds:
                rec.reject("no_speech_raw", REJECTED)
    except Exception as exc:
        log.debug("prefilter failed on %s: %s", rec.uid, exc)
        rec.reject("unreadable", REJECTED)
    return rec


# ---------------------------------------------------------------------------
# silence / VAD
# ---------------------------------------------------------------------------
def vad_record(rec: Record, cfg: PipelineConfig, out_dir: str) -> Record:
    """Reject no-speech clips and trim the survivors to their speech span."""
    vc = cfg.vad
    try:
        from . import vad_engine

        audio, sr = load_audio(rec.path, mono=True)
        level = rms_dbfs(audio)
        rec.metrics["vocal_rms_dbfs"] = round(level, 2)
        if level < vc.min_dbfs:
            rec.reject("silent_after_separation", REJECTED)
            return rec

        info = vad_engine.analyse(
            audio, sr,
            threshold=vc.threshold,
            min_speech_duration_ms=vc.min_speech_duration_ms,
            min_silence_duration_ms=vc.min_silence_duration_ms,
            speech_pad_ms=vc.speech_pad_ms,
        )
        rec.metrics.update({
            "speech_seconds": round(float(info["speech_seconds"]), 3),
            "speech_ratio": round(float(info["speech_ratio"]), 4),
            "num_speech_segments": int(info["num_speech_segments"]),
        })
        rec.metrics["speech_segments"] = [
            [round(a, 3), round(b, 3)] for a, b in info["speech_segments"]
        ]

        # Inter-word SNR must be measured on the *untrimmed* signal: trimming
        # removes the silence the estimate needs, and a well-trimmed clip has
        # no non-speech frames left to measure a noise floor against.
        mask = speech_mask_from_segments(len(audio), sr, info["speech_segments"])
        snr = snr_estimate_db(audio, mask)
        if snr == snr:                                   # not NaN
            rec.metrics["vad_snr_db"] = round(snr, 2)

        if info["speech_seconds"] < vc.min_speech_seconds:
            rec.reject("no_speech", REJECTED)
            return rec
        if info["speech_ratio"] < vc.min_speech_ratio:
            rec.reject("low_speech_ratio", REJECTED)
            return rec

        if not vc.trim:
            rec.duration = float(info["duration"])
            return rec

        pad = vc.trim_pad_ms / 1000.0
        start = max(0.0, float(info["speech_start"]) - pad)
        end = min(float(info["duration"]), float(info["speech_end"]) + pad)
        cut = trim_to(audio, sr, start, end)
        new_duration = len(cut) / float(sr)

        if new_duration < vc.min_trimmed_seconds:
            rec.reject("too_short_after_trim", REJECTED)
            return rec
        if new_duration > vc.max_trimmed_seconds:
            rec.reject("too_long_after_trim", REJECTED)
            return rec

        out_path = Path(out_dir) / f"{rec.uid}.wav"
        save_audio(str(out_path), cut, sr)
        _unlink(rec.path)

        rec.path = str(out_path)
        rec.duration = new_duration
        rec.sample_rate = sr
        rec.metrics["trim_start"] = round(start, 3)
        rec.metrics["trim_end"] = round(end, 3)
        rec.metrics["speech_segments"] = [
            [round(max(0.0, a - start), 3), round(max(0.0, b - start), 3)]
            for a, b in info["speech_segments"]
        ]
    except Exception as exc:
        log.debug("VAD failed on %s: %s", rec.uid, exc)
        rec.reject("vad_failed", REJECTED)
    return rec


# ---------------------------------------------------------------------------
# quality
# ---------------------------------------------------------------------------
def quality_record(rec: Record, cfg: PipelineConfig) -> Record:
    """Score, then apply thresholds.

    NISQA scores, when enabled, are attached to ``rec.metrics`` by the caller
    before dispatch - NISQA runs per directory, not per clip.
    """
    qc = cfg.quality
    try:
        audio, sr = load_audio(rec.path, mono=True)

        if qc.use_heuristics:
            segments = [tuple(s) for s in rec.metrics.get("speech_segments", [])]
            scores = heuristics.compute(audio, sr, segments or None)
            # Do not overwrite the real pre-trim SNR with the NaN a trimmed
            # clip yields.
            if "vad_snr_db" in rec.metrics:
                scores.pop("vad_snr_db", None)
            rec.metrics.update(scores)

        if qc.use_dnsmos:
            rec.metrics.update(get_dnsmos(cfg).score(audio, sr))
        if qc.use_squim:
            rec.metrics.update(get_squim(cfg).score(audio, sr))

        decide_quality(rec, cfg)
    except Exception as exc:
        log.debug("quality scoring failed for %s: %s", rec.uid, exc)
        rec.reject("quality_failed", REJECTED)
    return rec


def decide_quality(rec: Record, cfg: PipelineConfig) -> Record:
    """Apply thresholds. Failing is a *demotion*, not a deletion."""
    import math

    qc = cfg.quality
    m = rec.metrics
    failures = []

    def below(key: str, minimum: float, label: str) -> None:
        """An unmeasurable value is not evidence of a bad clip - skip it."""
        value = m.get(key)
        if value is None:
            return
        try:
            number = float(value)
        except (TypeError, ValueError):
            return
        if math.isfinite(number) and number < minimum:
            failures.append(label)

    if qc.use_dnsmos:
        below("dnsmos_ovrl", qc.min_ovrl, "low_ovrl")
        below("dnsmos_sig", qc.min_sig, "low_sig")
        below("dnsmos_bak", qc.min_bak, "residual_background")
    if qc.use_nisqa and "nisqa_dis" in m:
        below("nisqa_dis", qc.min_nisqa_dis, "separation_artifacts")
    if qc.use_heuristics:
        below("bandwidth_hz", qc.min_bandwidth_hz, "band_limited")
        clip = m.get("clipping_ratio")
        try:
            if clip is not None and float(clip) > qc.max_clipping_ratio:
                failures.append("clipped")
        except (TypeError, ValueError):
            pass

    rec.metrics["quality_failures"] = failures
    if failures:
        rec.status = LOW_QUALITY
        for reason in failures:
            if reason not in rec.reasons:
                rec.reasons.append(reason)
    else:
        rec.status = KEEP
    return rec


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
