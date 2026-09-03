"""Silero VAD wrapper - the core of silence / no-speech detection.

Energy thresholding alone cannot do this job. The dominant failure mode in a
movie-sourced dataset is not digital silence but clips that were pure music or
sound effects: after Kim Vocal 2 strips the instrumental, what remains is
low-level separation residue that sits comfortably *above* any sane dBFS gate
while containing no speech at all. A neural VAD separates those two cases;
an RMS threshold cannot.

The model is ~1 MB and MIT licensed, so it is cheap enough to run twice - once
as a lenient prefilter on raw audio (to avoid paying GPU separation cost on
dead clips) and once strictly on the separated vocals.
"""
from __future__ import annotations

import logging
import threading
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

VAD_SR = 16000

_model = None
_lock = threading.Lock()


def get_model():
    """Load (once, process-wide) the Silero VAD model."""
    global _model
    with _lock:
        if _model is None:
            from silero_vad import load_silero_vad
            _model = load_silero_vad()
            log.info("Silero VAD loaded")
    return _model


def speech_timestamps(audio: np.ndarray, sr: int,
                      threshold: float = 0.5,
                      min_speech_duration_ms: int = 250,
                      min_silence_duration_ms: int = 300,
                      speech_pad_ms: int = 100,
                      max_speech_duration_s: float = float("inf"),
                      ) -> List[Tuple[float, float]]:
    """Return speech spans as ``[(start_s, end_s), ...]``."""
    import torch
    from silero_vad import get_speech_timestamps

    if sr != VAD_SR:
        import librosa
        audio = librosa.resample(np.asarray(audio, dtype=np.float32),
                                 orig_sr=sr, target_sr=VAD_SR)
    audio = np.asarray(audio, dtype=np.float32).flatten()
    if audio.size < VAD_SR // 20:            # < 50 ms is not worth asking
        return []

    tensor = torch.from_numpy(audio)
    with torch.no_grad():
        spans = get_speech_timestamps(
            tensor,
            get_model(),
            threshold=threshold,
            sampling_rate=VAD_SR,
            min_speech_duration_ms=min_speech_duration_ms,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
            max_speech_duration_s=max_speech_duration_s,
            return_seconds=True,
        )
    return [(float(s["start"]), float(s["end"])) for s in spans]


def analyse(audio: np.ndarray, sr: int, **kwargs) -> Dict[str, object]:
    """Speech statistics for one utterance.

    ``speech_ratio`` is the discriminative one: real speech typically lands
    well above 0.5, while music residue and room tone collapse toward 0.
    """
    duration = len(audio) / float(sr) if sr else 0.0
    spans = speech_timestamps(audio, sr, **kwargs)
    speech_seconds = float(sum(end - start for start, end in spans))
    return {
        "duration": duration,
        "speech_segments": spans,
        "num_speech_segments": len(spans),
        "speech_seconds": speech_seconds,
        "speech_ratio": (speech_seconds / duration) if duration > 0 else 0.0,
        "speech_start": spans[0][0] if spans else None,
        "speech_end": spans[-1][1] if spans else None,
    }
