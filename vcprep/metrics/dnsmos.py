"""DNSMOS P.835 - non-intrusive speech quality on three axes.

Faithful port of Microsoft's ``dnsmos_local.py`` (DNS-Challenge, MIT).
The three outputs matter individually for this pipeline:

  SIG   speech signal quality      -> did separation damage the voice?
  BAK   background intrusiveness   -> how much BGM did Kim Vocal 2 leave behind?
  OVRL  overall                    -> the headline number

BAK is the reason DNSMOS is the right primary gate here: it grades the
separator's own output, which is precisely the question this stage asks.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np

log = logging.getLogger(__name__)

SAMPLING_RATE = 16000
INPUT_LENGTH = 9.01          # seconds per analysis window

P835_MODEL = "sig_bak_ovr.onnx"
P835_URL = (
    "https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/"
    "DNSMOS/DNSMOS/sig_bak_ovr.onnx"
)

# Third-order (personalised) / second-order (standard) polynomial mappings from
# raw network output to the MOS scale. Verbatim from the reference code.
_POLY = {
    False: {  # non-personalised
        "ovr": np.poly1d([-0.06766283, 1.11546468, 0.04602535]),
        "sig": np.poly1d([-0.08397278, 1.22083953, 0.0052439]),
        "bak": np.poly1d([-0.13166888, 1.60915514, -0.39604546]),
    },
    True: {   # personalised
        "ovr": np.poly1d([-0.00533021, 0.005101, 1.18058466, -0.11236046]),
        "sig": np.poly1d([-0.01019296, 0.02751166, 1.19576786, -0.24348726]),
        "bak": np.poly1d([-0.04976499, 0.44276479, -0.1644611, 0.96883132]),
    },
}


class DNSMOS:
    """Stateful scorer; construct once and reuse across the whole shard."""

    def __init__(self, model_dir: str, personalized: bool = False,
                 device: str = "auto", num_threads: int = 0):
        ort = _import_onnxruntime()

        model_path = Path(model_dir) / P835_MODEL
        if not model_path.exists():
            raise FileNotFoundError(
                f"DNSMOS model not found at {model_path}.\n"
                f"Fetch it with:  python -m vcprep.cli fetch-models "
                f"--dnsmos-dir {model_dir}"
            )

        providers = _providers(device, ort)
        opts = ort.SessionOptions()
        if num_threads:
            opts.intra_op_num_threads = num_threads
        self.session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=providers
        )
        self.personalized = personalized
        self.poly = _POLY[personalized]
        log.info("DNSMOS loaded (%s) providers=%s", model_path.name,
                 self.session.get_providers())

    # ------------------------------------------------------------------
    def score(self, audio: np.ndarray, sr: int) -> Dict[str, float]:
        """Score one utterance. Returns SIG / BAK / OVRL averaged over windows."""
        if sr != SAMPLING_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLING_RATE)
        audio = np.asarray(audio, dtype=np.float32).flatten()

        len_samples = int(INPUT_LENGTH * SAMPLING_RATE)
        # The reference implementation tiles short clips up to one full window.
        while len(audio) < len_samples:
            audio = np.append(audio, audio)

        num_hops = int(np.floor(len(audio) / SAMPLING_RATE) - INPUT_LENGTH) + 1
        hop_samples = SAMPLING_RATE

        sigs, baks, ovrs = [], [], []
        for idx in range(max(num_hops, 1)):
            start = idx * hop_samples
            seg = audio[start: start + len_samples]
            if len(seg) < len_samples:
                continue
            feed = {"input_1": seg.astype(np.float32)[np.newaxis, :]}
            raw_sig, raw_bak, raw_ovr = self.session.run(None, feed)[0][0]
            sigs.append(raw_sig)
            baks.append(raw_bak)
            ovrs.append(raw_ovr)

        if not sigs:
            return {"dnsmos_sig": float("nan"),
                    "dnsmos_bak": float("nan"),
                    "dnsmos_ovrl": float("nan")}

        return {
            "dnsmos_sig": float(self.poly["sig"](float(np.mean(sigs)))),
            "dnsmos_bak": float(self.poly["bak"](float(np.mean(baks)))),
            "dnsmos_ovrl": float(self.poly["ovr"](float(np.mean(ovrs)))),
        }

    def score_file(self, path: str) -> Dict[str, float]:
        from ..audio import load_audio
        audio, sr = load_audio(path, sr=SAMPLING_RATE, mono=True)
        return self.score(audio, sr)


def _import_onnxruntime():
    """Import onnxruntime, turning a half-installed package into a clear error.

    onnxruntime and onnxruntime-gpu share the same ``onnxruntime/`` directory,
    so uninstalling one deletes files the other needs while leaving its
    dist-info behind. pip then calls the survivor "already satisfied" and will
    not repair it. The result imports but has no attributes, which otherwise
    surfaces as a baffling AttributeError deep in a run.
    """
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError(
            "onnxruntime is not installed (needed for DNSMOS).\n"
            "  CPU:  pip install onnxruntime\n"
            "  GPU:  pip install onnxruntime-gpu"
        ) from exc
    if not hasattr(ort, "get_available_providers"):
        raise RuntimeError(
            f"onnxruntime is installed but broken (loaded {ort.__file__}, "
            f"no get_available_providers).\n"
            "This happens when onnxruntime and onnxruntime-gpu were both "
            "installed and one was then uninstalled - they share a directory.\n"
            "Repair with:\n"
            "  pip uninstall -y onnxruntime onnxruntime-gpu\n"
            "  pip install --force-reinstall --no-cache-dir onnxruntime-gpu"
        )
    return ort


def _providers(device: str, ort) -> list:
    available = ort.get_available_providers()
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device in ("cuda", "auto") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if device == "cuda":
        log.warning("CUDAExecutionProvider unavailable; falling back to CPU. "
                    "Install onnxruntime-gpu for GPU DNSMOS.")
    return ["CPUExecutionProvider"]


def download_model(model_dir: str) -> str:
    """Fetch sig_bak_ovr.onnx (~1 MB) into ``model_dir``."""
    import urllib.request

    dest_dir = Path(model_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / P835_MODEL
    if dest.exists():
        log.info("DNSMOS model already present at %s", dest)
        return str(dest)
    log.info("downloading DNSMOS P.835 model -> %s", dest)
    tmp = dest.with_suffix(".onnx.part")
    urllib.request.urlretrieve(P835_URL, tmp)
    os.replace(tmp, dest)
    return str(dest)
