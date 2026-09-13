#!/usr/bin/env bash
set -uo pipefail
PY=/home/visilant/CLEAR-EC/code/.venv/bin/python; CODE=/home/visilant/CLEAR-EC/.worktrees/regression-night/code
R=/home/visilant/CLEAR-EC/results/night_20260912; CACHE=/home/visilant/CLEAR-EC/data/cache; LABELS=/home/visilant/CLEAR-EC/data/final_train_ids.csv
BASE=(--input_mode whole --loss relative --batch_size 8 --lr 1e-4 --weight_decay 1e-4 --amp --channels_last --augment_flips --ema 0.999 --sched cosine --warmup_epochs 1 --patience 0 --epochs 8)
log() { echo "$(date '+%F %T') $*" >> "$R/logs/driver.log"; }
train() { local name=$1; shift; local dir="$R/$name"
  if compgen -G "$dir/regression_cnn/seed_*/metrics.json" > /dev/null; then log "SKIP $name"; return 0; fi
  log "START $name gpu=1 (phase4) $*"
  CUDA_VISIBLE_DEVICES=1 "$PY" "$CODE/scripts/run_training.py" --method regression --gpu 0 --cache_dir "$CACHE" --labels_csv "$LABELS" \
    --results_dir "$dir" "${BASE[@]}" "$@" > "$R/logs/$name.log" 2>&1; log "END $name exit=$?"; }
train x_v2fcmae   --seeds 123 --model timm:convnextv2_tiny.fcmae
train x_v2nano    --seeds 123 --model timm:convnextv2_nano.fcmae_ft_in22k_in1k
train v2fold0_s42 --seeds 42  --model timm:convnextv2_tiny.fcmae_ft_in22k_in1k --fold 0 --n_folds 5
log "queue1c done"
