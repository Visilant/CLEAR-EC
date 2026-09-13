#!/usr/bin/env bash
# Score last.pt of every finished single-split arm on val (no TTA and flips) if not already scored.
PY=/home/visilant/CLEAR-EC/code/.venv/bin/python; CODE=/home/visilant/CLEAR-EC/.worktrees/regression-night/code
R=/home/visilant/CLEAR-EC/results/night_20260912; GPU=${1:-1}
for mj in "$R"/*/regression_cnn/seed_*/metrics.json; do
  d=$(dirname "$mj"); name=$(basename "$(dirname "$(dirname "$d")")")
  [[ -f "$d/last.pt" ]] || continue; grep -q val_mape "$mj" || continue
  [[ -f "$R/tta/${name}_val_last.json" ]] || CUDA_VISIBLE_DEVICES=$GPU "$PY" "$CODE/scripts/night_predict.py" --ckpt_dir "$d" --which last --indices val --tta none --out "$R/tta/${name}_val_last.csv" --gpu 0 2>/dev/null | tail -1
  [[ -f "$R/tta/${name}_val_last_flips.json" ]] || CUDA_VISIBLE_DEVICES=$GPU "$PY" "$CODE/scripts/night_predict.py" --ckpt_dir "$d" --which last --indices val --tta flips --out "$R/tta/${name}_val_last_flips.csv" --gpu 0 2>/dev/null | tail -1
done
