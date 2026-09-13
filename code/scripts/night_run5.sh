#!/usr/bin/env bash
set -uo pipefail
PY=/home/visilant/CLEAR-EC/code/.venv/bin/python; CODE=/home/visilant/CLEAR-EC/.worktrees/regression-night/code
R=/home/visilant/CLEAR-EC/results/night_20260912; CACHE=/home/visilant/CLEAR-EC/data/cache; LABELS=/home/visilant/CLEAR-EC/data/final_train_ids.csv
log() { echo "$(date '+%F %T') $*" >> "$R/logs/driver.log"; }
until grep -q "queue1c done" "$R/logs/driver.log"; do sleep 60; done
log "START v2refit_seed7 gpu=1 (phase5) --all_data"
CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE/scripts/run_training.py" --method regression --gpu 0 --cache_dir "$CACHE" --labels_csv "$LABELS" \
  --results_dir "$R/v2refit_seed7" --seeds 7 --model timm:convnextv2_tiny.fcmae_ft_in22k_in1k --input_mode whole --loss relative \
  --batch_size 8 --lr 1e-4 --weight_decay 1e-4 --amp --channels_last --augment_flips --ema 0.999 --sched cosine --warmup_epochs 1 --patience 0 --epochs 8 --all_data \
  > "$R/logs/v2refit_seed7.log" 2>&1; log "END v2refit_seed7 exit=$?"
