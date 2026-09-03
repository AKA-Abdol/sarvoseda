#!/usr/bin/env bash
# One-shot server bootstrap: dependencies, models, sanity check.
set -euo pipefail

echo "==> python dependencies"
pip install -r requirements.txt

echo "==> onnxruntime / audio-separator backend"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "CUDA detected"
  pip install "audio-separator[gpu]"
  pip uninstall -y onnxruntime >/dev/null 2>&1 || true
  pip install onnxruntime-gpu
else
  echo "no CUDA - installing CPU backends"
  pip install "audio-separator[cpu]"
fi

echo "==> DNSMOS weights"
python -m vcprep.cli fetch-models --dnsmos-dir models/dnsmos

echo "==> optional: NISQA (discontinuity axis)"
bash scripts/setup_nisqa.sh || echo "NISQA setup skipped"

echo
echo "Smoke test (the separation model downloads on first use):"
echo "  python -m vcprep.cli run --shards 1 --limit-per-shard 25"
echo
echo "To use a local UVR model instead, add:"
echo "  --uvr-model /path/to/Kim_Vocal_2.onnx"
echo
echo "For the distributed backend (only useful across machines):"
echo "  pip install 'ray[default]' && ... --backend ray"
