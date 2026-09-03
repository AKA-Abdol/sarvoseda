"""TorchAudio SQUIM - reference-free PESQ / STOI / SI-SDR prediction.

Ships inside torchaudio, so it costs no extra dependency and batches on GPU.
Used as a second opinion alongside DNSMOS: SQUIM's predicted PESQ correlates
with perceptual damage from separation in a way DNSMOS's SIG axis does not
fully capture.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np

log = logging.getLogger(__name__)

SQUIM_SR = 16000


class Squim:
    def __init__(self, device: str = "auto"):
        import torch
        import torchaudio

        self.torch = torch
        self.device = _resolve_device(device, torch)
        bundle = torchaudio.pipelines.SQUIM_OBJECTIVE
        self.model = bundle.get_model().to(self.device).eval()
        log.info("SQUIM objective loaded on %s", self.device)

    def score(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        torch = self.torch
        if sr != SQUIM_SR:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SQUIM_SR)
        # SQUIM's receptive field needs a bit of context to be meaningful.
        if len(audio) < SQUIM_SR // 2:
            return {"squim_stoi": float("nan"),
                    "squim_pesq": float("nan"),
                    "squim_si_sdr": float("nan")}
        wav = torch.from_numpy(np.asarray(audio, dtype=np.float32))[None, :].to(self.device)
        with torch.no_grad():
            stoi, pesq, si_sdr = self.model(wav)
        return {
            "squim_stoi": float(stoi[0].item()),
            "squim_pesq": float(pesq[0].item()),
            "squim_si_sdr": float(si_sdr[0].item()),
        }


def _resolve_device(device: str, torch) -> str:
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device
