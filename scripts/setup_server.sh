#!/usr/bin/env bash
# Preprocessing environment (vcprep).
#
# IMPORTANT: this is venv #1 of two. seed-vc pins torch==2.4.0 / numpy==1.26.4 /
# transformers==4.46.3, which cannot coexist with the modern stack the
# separation and VAD models need. Install training separately with
# scripts/setup_seedvc.sh in its own virtualenv.
set -euo pipefail

echo "==> checking system prerequisites"
missing=""
command -v ffmpeg >/dev/null 2>&1 || missing="$missing ffmpeg"
python -c "import ctypes.util,sys; sys.exit(0 if ctypes.util.find_library('sndfile') else 1)" \
  2>/dev/null || missing="$missing libsndfile1"
if [ -n "$missing" ]; then
  echo "ERROR: missing system packages:$missing" >&2
  echo "  Debian/Ubuntu:  sudo apt-get install -y ffmpeg libsndfile1" >&2
  echo "  RHEL/Fedora:    sudo dnf install -y ffmpeg libsndfile" >&2
  exit 1
fi
echo "ffmpeg and libsndfile present"

# Torch must go in FIRST, from the index that matches the driver. Installing it
# later lets a transitive dependency pick a build for the wrong CUDA version.
if ! python -c "import torch" 2>/dev/null; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    CUDA_TAG="${CUDA_TAG:-cu124}"
    echo "==> installing torch for CUDA ($CUDA_TAG)"
    echo "    override with: CUDA_TAG=cu118 bash scripts/setup_server.sh"
    pip install torch torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
  else
    echo "==> no nvidia-smi; installing CPU torch"
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
else
  python - <<'PY'
import torch
print(f"==> torch {torch.__version__} already installed "
      f"(CUDA available: {torch.cuda.is_available()})")
PY
fi

echo "==> python dependencies"
pip install -r requirements.txt
pip install -e .

echo "==> separation + ONNX backend"
if command -v nvidia-smi >/dev/null 2>&1; then
  pip install "audio-separator[gpu]"

  # onnxruntime and onnxruntime-gpu install into the SAME onnxruntime/
  # directory. Uninstalling one deletes the shared files the other still needs,
  # and pip then reports the survivor as "already satisfied" and refuses to
  # repair it - leaving a package that imports but has no attributes
  # (AttributeError: no attribute 'get_available_providers').
  # So: remove BOTH, then force a clean single install.
  echo "==> installing a clean onnxruntime-gpu"
  pip uninstall -y onnxruntime onnxruntime-gpu >/dev/null 2>&1 || true
  SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
  rm -rf "$SITE/onnxruntime" "$SITE"/onnxruntime-*.dist-info \
         "$SITE"/onnxruntime_gpu-*.dist-info 2>/dev/null || true

  # onnxruntime-gpu must match torch's CUDA *major* version, or it fails to
  # load its provider library ("libcublasLt.so.13: cannot open shared object
  # file") and silently drops to CPU. The cutover:
  #   onnxruntime-gpu <= 1.26.0  -> CUDA 12
  #   onnxruntime-gpu >= 1.27.0  -> CUDA 13
  ORT_SPEC="$(python - <<'PY'
try:
    import torch
    cuda = torch.version.cuda or ""
except Exception:
    cuda = ""
major = cuda.split(".")[0] if cuda else ""
if major == "12":
    print("onnxruntime-gpu<1.27")     # newest CUDA 12 line
elif major == "13":
    print("onnxruntime-gpu>=1.27")
else:
    print("onnxruntime-gpu")          # unknown; take the default
PY
)"
  echo "    torch CUDA -> installing '$ORT_SPEC'"
  pip install --force-reinstall --no-cache-dir "$ORT_SPEC"
else
  pip install "audio-separator[cpu]"
fi

# A gutted install imports fine and only fails later, mid-run, so call into it.
python - <<'PY'
import sys
try:
    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__}: {ort.get_available_providers()}")
except Exception as exc:
    print(f"ERROR: onnxruntime is broken: {exc}", file=sys.stderr)
    print("Repair with:", file=sys.stderr)
    print("  pip uninstall -y onnxruntime onnxruntime-gpu", file=sys.stderr)
    print("  pip install --force-reinstall --no-cache-dir onnxruntime-gpu",
          file=sys.stderr)
    sys.exit(1)
PY

# DNSMOS weights come before the health check: `vcprep doctor` uses the model
# to build a real CUDA session, which is the only way to tell a working
# provider from one that is merely listed and silently falls back.
echo "==> DNSMOS weights"
vcprep fetch-models --dnsmos-dir models/dnsmos

echo "==> optional: NISQA (adds the discontinuity axis; academic licence)"
bash scripts/setup_nisqa.sh || echo "NISQA setup skipped"

echo
echo "==> environment check"
vcprep doctor || true

echo
echo "Smoke test:"
echo "  vcprep run --shards 1 --limit-per-shard 25"
