#!/usr/bin/env bash
# Clone seed-vc and install its training requirements.
set -euo pipefail

DEST="${1:-third_party/seed-vc}"
mkdir -p "$(dirname "$DEST")"

if [ -d "$DEST/.git" ]; then
  echo "seed-vc already present at $DEST"
else
  git clone https://github.com/Plachtaa/seed-vc.git "$DEST"
fi

echo "installing seed-vc requirements..."
pip install -r "$DEST/requirements.txt"
pip install accelerate

echo
echo "seed-vc ready at $DEST"
echo "Pretrained V2 weights download automatically on first run."
echo "Next:  vcft prepare && vcft train --repo-dir $DEST"
