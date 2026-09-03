#!/usr/bin/env bash
# Training environment (seed-vc + vcft).
#
# Run this inside a SEPARATE virtualenv from the preprocessing one. seed-vc
# pins torch==2.4.0, numpy==1.26.4 and transformers==4.46.3; the preprocessing
# stack needs much newer versions of all three. Installing both in one
# environment breaks whichever was installed first.
#
#   python -m venv .venv-train && source .venv-train/bin/activate
#   bash scripts/setup_seedvc.sh
set -euo pipefail

DEST="${1:-third_party/seed-vc}"
mkdir -p "$(dirname "$DEST")"

python - <<'PY'
import sys
try:
    import audio_separator  # noqa: F401
except ImportError:
    sys.exit(0)
print("WARNING: audio-separator is installed in this environment.", file=sys.stderr)
print("You are probably in the preprocessing venv. seed-vc's pins will "
      "downgrade torch and numpy and break it.", file=sys.stderr)
print("Create a separate venv first:  python -m venv .venv-train", file=sys.stderr)
sys.exit(1)
PY

if [ -d "$DEST/.git" ]; then
  echo "seed-vc already present at $DEST"
else
  git clone https://github.com/Plachtaa/seed-vc.git "$DEST"
fi

# seed-vc's requirements.txt carries both nightly --index-url lines and hard
# pins (torch==2.4.0) for the same packages. Letting pip see the index lines
# pulls nightly builds that then conflict with the pins, so install torch
# explicitly first and let the pinned lines find it already satisfied.
CUDA_TAG="${CUDA_TAG:-cu124}"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "==> installing torch 2.4.0 for $CUDA_TAG"
  pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
else
  echo "==> no GPU detected; installing CPU torch 2.4.0 (training will be slow)"
  pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu
fi

echo "==> seed-vc requirements"
grep -v -- '--index-url' "$DEST/requirements.txt" > /tmp/seedvc_reqs.txt
pip install -r /tmp/seedvc_reqs.txt
pip install accelerate

echo
echo "seed-vc ready at $DEST"
echo "Pretrained V2 weights download automatically on the first run."
echo
echo "vcft runs as a module here (do NOT 'pip install -e .' in this venv -"
echo "that would pull audio-separator and undo the pins):"
echo "  python -m seedvc_ft.cli prepare --clean-dir /data/out/clean"
echo "  python -m seedvc_ft.cli train --repo-dir $DEST --dry-run"
