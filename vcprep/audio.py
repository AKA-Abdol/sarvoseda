"""Audio I/O and cheap signal measurements shared by several nodes."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import soundfile as sf

log = logging.getLogger(__name__)

EPS = 1e-10


def load_audio(path: str, sr: Optional[int] = None, mono: bool = True
               ) -> Tuple[np.ndarray, int]:
    """Load audio as float32. Resamples only when ``sr`` is given and differs."""
    try:
        data, native_sr = sf.read(path, dtype="float32", always_2d=False)
    except Exception:
        # mp3 via libsndfile needs >=1.1; fall back to librosa/audioread
        import librosa
        data, native_sr = librosa.load(path, sr=None, mono=False)
        data = np.asarray(data, dtype=np.float32)
        if data.ndim == 2:
            data = data.T

    if data.ndim == 2 and mono:
        data = data.mean(axis=1)
    if sr is not None and native_sr != sr:
        import librosa
        axis = 0 if data.ndim == 1 else 1
        data = librosa.resample(
            data if data.ndim == 1 else data.T, orig_sr=native_sr, target_sr=sr, axis=axis
        )
        if data.ndim == 2:
            data = data.T
        native_sr = sr
    return np.ascontiguousarray(data, dtype=np.float32), native_sr


def save_audio(path: str, data: np.ndarray, sr: int, subtype: str = "PCM_16") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, data, sr, subtype=subtype)


def probe(path: str) -> Tuple[float, int, int]:
    """(duration_seconds, sample_rate, channels) without decoding the payload."""
    try:
        info = sf.info(path)
        return float(info.duration), int(info.samplerate), int(info.channels)
    except Exception:
        data, sr = load_audio(path, mono=False)
        n = data.shape[0]
        ch = 1 if data.ndim == 1 else data.shape[1]
        return n / float(sr), sr, ch


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------
def rms_dbfs(x: np.ndarray) -> float:
    """Full-file RMS in dBFS."""
    if x.size == 0:
        return -np.inf
    return float(20.0 * np.log10(np.sqrt(np.mean(np.square(x))) + EPS))


def peak_dbfs(x: np.ndarray) -> float:
    if x.size == 0:
        return -np.inf
    return float(20.0 * np.log10(np.max(np.abs(x)) + EPS))


def clipping_ratio(x: np.ndarray, threshold: float = 0.99) -> float:
    """Fraction of samples at or beyond full scale."""
    if x.size == 0:
        return 0.0
    return float(np.mean(np.abs(x) >= threshold))


def dc_offset(x: np.ndarray) -> float:
    return float(np.mean(x)) if x.size else 0.0


def spectral_bandwidth_hz(x: np.ndarray, sr: int, energy_fraction: float = 0.99,
                          n_fft: int = 2048, max_seconds: float = 10.0) -> float:
    """Highest frequency still carrying signal, from an averaged power spectrum.

    Streaming rips are frequently lowpassed at 8-11 kHz by the encoder. That is
    invisible to MOS predictors but genuinely harmful to a 22.05 kHz seed-vc
    model, which would be asked to synthesise a band the data never contains.
    """
    seg = x[: int(sr * max_seconds)]
    if seg.size < n_fft:
        return 0.0
    hop = n_fft // 2
    window = np.hanning(n_fft).astype(np.float32)
    n_frames = 1 + (len(seg) - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        seg,
        shape=(n_frames, n_fft),
        strides=(seg.strides[0] * hop, seg.strides[0]),
    )
    power = np.mean(np.abs(np.fft.rfft(frames * window, axis=1)) ** 2, axis=0)
    total = power.sum()
    if total <= EPS:
        return 0.0
    cumulative = np.cumsum(power) / total
    idx = int(np.searchsorted(cumulative, energy_fraction))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    return float(freqs[min(idx, len(freqs) - 1)])


def loudness_lufs(x: np.ndarray, sr: int) -> float:
    """ITU-R BS.1770 integrated loudness; -inf when the file is too short."""
    try:
        import pyloudnorm as pyln
        if len(x) < sr * 0.4:      # meter needs ~400 ms
            return float("-inf")
        meter = pyln.Meter(sr)
        return float(meter.integrated_loudness(x))
    except Exception:
        return float("nan")


def frame_energy_db(x: np.ndarray, sr: int, frame_ms: float = 20.0) -> np.ndarray:
    """Per-frame RMS in dBFS - the basis of the cheap silence gate."""
    n = max(1, int(sr * frame_ms / 1000.0))
    if x.size < n:
        return np.array([rms_dbfs(x)], dtype=np.float32)
    trimmed = x[: (len(x) // n) * n].reshape(-1, n)
    rms = np.sqrt(np.mean(np.square(trimmed), axis=1))
    return (20.0 * np.log10(rms + EPS)).astype(np.float32)


def active_speech_ratio(x: np.ndarray, sr: int, rel_threshold_db: float = 25.0
                        ) -> float:
    """Fraction of frames within ``rel_threshold_db`` of the loudest frame.

    Relative rather than absolute, so it survives the level changes that vocal
    separation introduces.
    """
    energies = frame_energy_db(x, sr)
    if energies.size == 0:
        return 0.0
    return float(np.mean(energies > (energies.max() - rel_threshold_db)))


def snr_estimate_db(x: np.ndarray, speech_mask: np.ndarray) -> float:
    """SNR from VAD segmentation: speech-frame power vs non-speech-frame power.

    ``speech_mask`` is a per-sample boolean of equal length to ``x``.
    """
    if x.size == 0 or speech_mask.size != x.size:
        return float("nan")
    speech = x[speech_mask]
    noise = x[~speech_mask]
    if speech.size < 100 or noise.size < 100:
        return float("nan")
    p_s = float(np.mean(np.square(speech)))
    p_n = float(np.mean(np.square(noise)))
    if p_n <= EPS:
        return 60.0                      # noise floor below measurable
    return float(10.0 * np.log10((p_s + EPS) / (p_n + EPS)))


def trim_to(x: np.ndarray, sr: int, start_s: float, end_s: float) -> np.ndarray:
    a = max(0, int(start_s * sr))
    b = min(len(x), int(end_s * sr))
    return x[a:b] if b > a else x[:0]
