"""Report what each stage will actually run on, and why.

The pipeline uses two different accelerators, and confusing them wastes hours:

* **PyTorch** drives RoFormer/MDXC separation, Silero VAD and SQUIM.
* **onnxruntime** drives DNSMOS, and MDX-Net (``.onnx``) separation only.

So audio-separator's ``CUDAExecutionProvider not available`` warning is
irrelevant when the model is a RoFormer ``.ckpt`` - that runs on torch, and the
line is printed at import time no matter what you load. It does mean DNSMOS
falls back to CPU, which is survivable (it is a tiny model) but worth fixing.
"""
from __future__ import annotations

import importlib.metadata as md
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

OK = "OK"
WARN = "WARN"
BAD = "FAIL"


def _torch_info() -> Dict[str, object]:
    out: Dict[str, object] = {"installed": False}
    try:
        import torch
    except ImportError:
        return out
    out.update(
        installed=True,
        version=torch.__version__,
        cuda_available=torch.cuda.is_available(),
        cuda_version=getattr(torch.version, "cuda", None),
        cudnn=torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        device_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
    )
    if out["cuda_available"]:
        out["devices"] = [torch.cuda.get_device_name(i)
                          for i in range(int(out["device_count"]))]
    mps = getattr(torch.backends, "mps", None)
    out["mps"] = bool(mps and mps.is_available())
    return out


def _ort_info() -> Dict[str, object]:
    out: Dict[str, object] = {"installed": False, "distributions": []}
    for dist in md.distributions():
        name = (dist.metadata["Name"] or "").lower()
        if name.startswith("onnxruntime"):
            out["distributions"].append(f"{dist.metadata['Name']}=={dist.version}")
    try:
        import onnxruntime as ort
    except ImportError:
        return out
    out["installed"] = True
    out["healthy"] = hasattr(ort, "get_available_providers")
    if out["healthy"]:
        out["version"] = ort.__version__
        out["providers"] = list(ort.get_available_providers())
    return out


def _probe_cuda_session(model_path: Optional[str]) -> Tuple[bool, str]:
    """Actually create a session on CUDA, to separate 'listed' from 'works'.

    ``get_available_providers()`` reports what the wheel was *compiled* with.
    onnxruntime then loads the CUDA shared libraries lazily and falls back to
    CPU **silently** if a version does not match, so the provider being listed
    proves nothing on its own. Only building a session settles it.
    """
    if not model_path or not Path(model_path).exists():
        return False, "no model available to probe with"
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(model_path),
                                    providers=["CUDAExecutionProvider"])
        used = sess.get_providers()
        if "CUDAExecutionProvider" in used:
            return True, "session created on CUDA"
        return False, f"silently fell back to {used}"
    except Exception as exc:                              # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"


def _diagnose_ort(ort: Dict[str, object], torch_info: Dict[str, object],
                  model_path: Optional[str] = None) -> List[Tuple[str, str]]:
    """Explain a missing CUDAExecutionProvider.

    Keyed off the **provider list**, not the installed distribution names: what
    a wheel claims in its dist-info can disagree with the files actually on
    disk, whereas the provider list reflects the binary that really loaded.
    """
    notes: List[Tuple[str, str]] = []
    providers = ort.get("providers") or []

    if not ort.get("installed"):
        return [(BAD, "onnxruntime is not installed - DNSMOS cannot run.\n"
                      "      pip install onnxruntime-gpu   (or onnxruntime on CPU)")]
    if not ort.get("healthy", True):
        return [(BAD,
                 "onnxruntime is installed but broken (no get_available_providers).\n"
                 "      onnxruntime and onnxruntime-gpu share one directory, so\n"
                 "      uninstalling either guts the other. Remove both, reinstall one:\n"
                 "        pip uninstall -y onnxruntime onnxruntime-gpu\n"
                 "        pip install --force-reinstall --no-cache-dir onnxruntime-gpu")]

    if not torch_info.get("cuda_available"):
        return [(OK, "No CUDA on this machine at all, so a CPU-only "
                     "onnxruntime is correct here.")]

    if "CUDAExecutionProvider" not in providers:
        # Compiled-in support is absent: this is the CPU wheel, whatever the
        # dist-info metadata happens to say.
        return [(BAD,
                 "The installed onnxruntime has **no CUDA support compiled in**.\n"
                 f"      It offers only: {', '.join(providers)}\n"
                 "      The GPU wheel always lists CUDAExecutionProvider here, even\n"
                 "      when its CUDA libraries fail to load - so this is the\n"
                 "      CPU-only 'onnxruntime' package. Nothing to do with any model.\n"
                 "      Fix (remove BOTH first - they share one directory):\n"
                 "        pip uninstall -y onnxruntime onnxruntime-gpu\n"
                 "        pip install --force-reinstall --no-cache-dir onnxruntime-gpu")]

    # CUDA is compiled in. Does it actually build a session?
    works, detail = _probe_cuda_session(model_path)
    if works:
        return [(OK, "CUDAExecutionProvider is available and a session really "
                     "builds on it - DNSMOS runs on GPU.")]

    version = str(ort.get("version", ""))
    cudnn = torch_info.get("cudnn")
    torch_cuda = str(torch_info.get("cuda_version") or "")
    torch_major = torch_cuda.split(".")[0] if torch_cuda else ""
    ort_major = _ort_cuda_major(version)

    notes.append((WARN,
                  f"CUDAExecutionProvider is listed but unusable ({detail}).\n"
                  "      onnxruntime loads its CUDA libraries lazily and falls back\n"
                  "      to CPU silently, so nothing fails loudly at import time."))

    if torch_major and ort_major and torch_major != ort_major:
        want = "<1.27" if torch_major == "12" else ">=1.27"
        notes.append((BAD,
                      f"CUDA major mismatch: onnxruntime {version} is built for "
                      f"CUDA {ort_major},\n"
                      f"      but torch {torch_info.get('version')} ships CUDA "
                      f"{torch_cuda}.\n"
                      f"      That is the 'libcublasLt.so.{ort_major}: cannot open "
                      f"shared object file' error.\n"
                      f"      onnxruntime-gpu <= 1.26.0 targets CUDA 12; "
                      f">= 1.27.0 targets CUDA 13.\n"
                      f"      Fix:\n"
                      f"        pip uninstall -y onnxruntime onnxruntime-gpu\n"
                      f"        pip install --no-cache-dir 'onnxruntime-gpu{want}'"))
    else:
        notes.append((WARN,
                      f"onnxruntime {version} targets CUDA {ort_major or '?'} "
                      f"and cuDNN 9;\n"
                      f"      torch reports CUDA {torch_cuda or '?'}, cuDNN {cudnn}."))

    notes.append((WARN,
                  "Full loader error (onnxruntime hides it by default):\n"
                  "        python -c \"import onnxruntime as o; "
                  "o.set_default_logger_severity(0); "
                  "o.InferenceSession('m.onnx', providers=['CUDAExecutionProvider'])\""))
    notes.append((WARN,
                  "DNSMOS is a ~1 MB model - running it on CPU costs about an\n"
                  "      hour across the whole corpus. If this turns into a rabbit\n"
                  "      hole, it is entirely reasonable to proceed on CPU: RoFormer\n"
                  "      separation runs on torch and is unaffected."))
    return notes


def _ort_cuda_major(version: str) -> str:
    """Which CUDA major an onnxruntime-gpu release is built against.

    Verified from the wheels' own ``requires_dist``: 1.26.0 and earlier depend
    on ``nvidia-*-cu12``, 1.27.0 and later on ``nvidia-*-cu13``.
    """
    try:
        parts = [int(x) for x in version.split(".")[:2]]
    except ValueError:
        return ""
    if len(parts) < 2:
        return ""
    return "13" if (parts[0], parts[1]) >= (1, 27) else "12"


def run(cfg=None) -> int:
    torch_info = _torch_info()
    ort = _ort_info()

    print("=" * 68)
    print("environment")
    print("=" * 68)

    if not torch_info["installed"]:
        print("  torch            NOT INSTALLED")
    else:
        dev = ("CUDA" if torch_info["cuda_available"]
               else ("MPS" if torch_info.get("mps") else "CPU"))
        print(f"  torch            {torch_info['version']}  ->  {dev}")
        if torch_info["cuda_available"]:
            print(f"                   CUDA {torch_info['cuda_version']}, "
                  f"cuDNN {torch_info['cudnn']}, "
                  f"{torch_info['device_count']} device(s)")
            for name in torch_info.get("devices", []):
                print(f"                   - {name}")

    dists = ort.get("distributions") or ["(none)"]
    print(f"  onnxruntime pkgs {', '.join(dists)}")
    if ort.get("healthy"):
        print(f"  onnxruntime      {ort.get('version')}")
        print(f"  providers        {', '.join(ort.get('providers', []))}")

    print(f"  ffmpeg           {'found' if shutil.which('ffmpeg') else 'NOT FOUND'}")

    print()
    print("=" * 68)
    print("what each stage will use")
    print("=" * 68)

    torch_dev = ("cuda" if torch_info.get("cuda_available")
                 else ("mps" if torch_info.get("mps") else "cpu"))
    ort_dev = ("cuda" if "CUDAExecutionProvider" in (ort.get("providers") or [])
               else "cpu")

    model = getattr(getattr(cfg, "separate", None), "model_name", "") if cfg else ""
    is_onnx = model.lower().endswith(".onnx")
    if model:
        engine = "onnxruntime" if is_onnx else "torch"
        dev = ort_dev if is_onnx else torch_dev
        print(f"  separation       {model}")
        print(f"                   engine={engine}  device={dev}")
        if not is_onnx:
            print("                   (a RoFormer .ckpt runs on torch - "
                  "onnxruntime\n"
                  "                    warnings from audio-separator do not "
                  "apply to it)")
    print(f"  silence / VAD    torch      device={torch_dev}")
    print(f"  DNSMOS           onnxruntime device={ort_dev}")
    print(f"  SQUIM            torch      device={torch_dev}")

    print()
    print("=" * 68)
    print("findings")
    print("=" * 68)
    dnsmos_model = None
    if cfg is not None:
        candidate = Path(getattr(cfg.quality, "dnsmos_dir", "")) / "sig_bak_ovr.onnx"
        if candidate.exists():
            dnsmos_model = str(candidate)
    notes = _diagnose_ort(ort, torch_info, dnsmos_model)
    if torch_info["installed"] and not torch_info["cuda_available"] \
            and not torch_info.get("mps"):
        notes.insert(0, (WARN, "torch sees no GPU - separation will run on CPU, "
                               "which is very slow for a full corpus."))
    if not shutil.which("ffmpeg"):
        notes.append((BAD, "ffmpeg is missing; mp3 decoding will fail.\n"
                           "      apt-get install -y ffmpeg"))
    if not notes:
        notes = [(OK, "no problems found")]
    for level, text in notes:
        print(f"  [{level:4}] {text}")
    print()
    return 0 if all(lvl != BAD for lvl, _ in notes) else 1
