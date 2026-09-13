#!/usr/bin/env bash
# Overnight spike driver: cpsam overlay grid, single sam_vit_b overlay run,
# pick best cpsam params from the 19 train overlays, then eval_val on 300 val images.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs results
PY=.venv/bin/python

echo "=== weights ==="
./download_weights.sh

DIAMETERS=(None 22 28)
FLOWS=(0.4 0.8)
CELLPROBS=(0.0 -2.0)
CLAHES=(0 1)

echo "=== cpsam overlay grid (12 runs x 19 train + 6 held-out) ==="
for d in "${DIAMETERS[@]}"; do
  for f in "${FLOWS[@]}"; do
    for c in "${CELLPROBS[@]}"; do
     for e in "${CLAHES[@]}"; do
      DARG=()
      if [ "$d" != "None" ]; then
        DARG=(--diameter "$d")
      fi
      EARG=()
      if [ "$e" = "1" ]; then
        EARG=(--clahe)
      fi
      echo "--- cpsam diameter=$d flow=$f cellprob=$c clahe=$e ---"
      $PY eval_overlays.py --model cpsam "${DARG[@]}" --flow "$f" --cellprob "$c" "${EARG[@]}" \
        2>&1 | tee "logs/overlay_cpsam_d${d}_f${f}_c${c}_e${e}.log"
     done
    done
  done
done

echo "=== sam_vit_b overlay run ==="
$PY eval_overlays.py --model sam_vit_b 2>&1 | tee logs/overlay_sam_vit_b.log

echo "=== picking best cpsam params (lowest mean CD/HEX MAPE on 19 train overlays) ==="
BEST=$($PY - <<'EOF'
import glob
import re
import pandas as pd

best = None
for path in sorted(glob.glob("results/cpsam_d*.csv")):
    df = pd.read_csv(path)
    df = df[df["split"] == "train"]
    if df.empty:
        continue
    score = df[["CD_ape_det", "HEX_ape_det"]].mean().mean()
    if best is None or score < best[0]:
        best = (score, path)
if best is None:
    raise SystemExit("no cpsam overlay results found in results/cpsam_d*.csv")
m = re.search(r"cpsam_d(None|[\d.]+)_f([\d.]+)_c(-?[\d.]+)_e([01])\.csv", best[1])
print(m.group(1), m.group(2), m.group(3), m.group(4))
print(f"# best score={best[0]:.3f} from {best[1]}")
EOF
)
BEST_LINE=$(echo "$BEST" | head -1)
read -r BEST_D BEST_F BEST_C BEST_E <<< "$BEST_LINE"
echo "$BEST"
echo "best cpsam params: diameter=$BEST_D flow=$BEST_F cellprob=$BEST_C clahe=$BEST_E"
BEST_EARG=()
if [ "$BEST_E" = "1" ]; then
  BEST_EARG=(--clahe)
fi

BEST_DARG=()
if [ "$BEST_D" != "None" ]; then
  BEST_DARG=(--diameter "$BEST_D")
fi

echo "=== eval_val cpsam n=300 with best params ==="
$PY eval_val.py --model cpsam --n 300 "${BEST_DARG[@]}" --flow "$BEST_F" --cellprob "$BEST_C" "${BEST_EARG[@]}" \
  2>&1 | tee logs/eval_val_cpsam.log

echo "run_all.sh done"
