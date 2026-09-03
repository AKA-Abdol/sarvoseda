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
  # Both packages register the same module name; the CPU one wins if present,
  # silently costing you the GPU.
  pip uninstall -y onnxruntime >/dev/null 2>&1 || true
  pip install onnxruntime-gpu
else
  pip install "audio-separator[cpu]"
fi

echo "==> DNSMOS weights"
vcprep fetch-models --dnsmos-dir models/dnsmos

echo "==> optional: NISQA (adds the discontinuity axis; academic licence)"
bash scripts/setup_nisqa.sh || echo "NISQA setup skipped"

echo
python - <<'PY'
import torch
import onnxruntime as ort
print("=" * 60)
print(f"torch      {torch.__version__}  CUDA={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"           {torch.cuda.device_count()} GPU(s): "
          f"{torch.cuda.get_device_name(0)}")
print(f"onnxruntime providers: {ort.get_available_providers()}")
print("=" * 60)
PY
echo
echo "Smoke test:"
echo "  vcprep run --shards 1 --limit-per-shard 25"
