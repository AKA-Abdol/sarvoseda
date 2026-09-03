#!/usr/bin/env bash
# NISQA v2 - adds the discontinuity axis that DNSMOS does not have.
# Separate repo under an academic / non-commercial licence, so it is cloned
# rather than vendored. Review the licence before any commercial use.
set -euo pipefail

DEST="${1:-third_party/NISQA}"
mkdir -p "$(dirname "$DEST")"

if [ -d "$DEST/.git" ]; then
  echo "NISQA already present at $DEST"
else
  git clone --depth 1 https://github.com/gabrielmittag/NISQA.git "$DEST"
fi

if [ ! -f "$DEST/weights/nisqa.tar" ]; then
  echo "ERROR: $DEST/weights/nisqa.tar is missing." >&2
  echo "The pretrained weights ship in the repo; check the clone." >&2
  exit 1
fi

echo "NISQA ready at $DEST"
echo "Enable it with:  vcprep run --with-nisqa --nisqa-repo $DEST"
