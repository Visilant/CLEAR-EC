#!/usr/bin/env bash
# Download the SAM ViT-B checkpoint into ./weights/ (public URL from the
# segment-anything README). cpsam downloads its own weights on first use.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p weights
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
OUT="weights/sam_vit_b_01ec64.pth"
if [ -f "$OUT" ]; then
  echo "weights already present: $OUT"
  exit 0
fi
curl -L -o "$OUT" "$URL"
echo "downloaded $OUT"
