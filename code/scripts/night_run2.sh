#!/usr/bin/env bash
# Phase 2b: extra single-split arms once the phase-1 queues free a GPU. Same recipe/gate as night_run.sh.
set -uo pipefail
PY=/home/visilant/CLEAR-EC/code/.venv/bin/python; CODE=/home/visilant/CLEAR-EC/.worktrees/regression-night/code
R=/home/visilant/CLEAR-EC/results/night_20260912; CACHE=/home/visilant/CLEAR-EC/data/cache; LABELS=/home/visilant/CLEAR-EC/data/final_train_ids.csv
BASE=(--model convnext_tiny --input_mode whole --loss relative --batch_size 8 --lr 1e-4 --weight_decay 1e-4 --amp --channels_last --augment_flips)
EMA=(--ema 0.999 --sched cosine --warmup_epochs 1 --epochs 12 --patience 0)
log() { echo "$(date '+%F %T') $*" >> "$R/logs/driver.log"; }
train() { local name=$1 gpu=$2; shift 2; local dir="$R/$name"
  if compgen -G "$dir/regression_cnn/seed_*/metrics.json" > /dev/null; then log "SKIP $name"; return 0; fi
  log "START $name gpu=$gpu (phase2b) $*"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$CODE/scripts/run_training.py" --method regression --gpu 0 --cache_dir "$CACHE" --labels_csv "$LABELS" \
    --results_dir "$dir" "${BASE[@]}" "${EMA[@]}" "$@" > "$R/logs/$name.log" 2>&1; log "END $name exit=$?"; }
q0() { while pgrep -f "results_dir.*[x]_wd5e-2" > /dev/null; do sleep 30; done
  train x_dp03      0 --seeds 123 --drop_path 0.3
  train x_reg_e20   0 --seeds 123 --weight_decay 5e-2 --drop_path 0.3 --epochs 20
  log "queue0b done"; }
q1() { V2=(--model timm:convnextv2_tiny.fcmae_ft_in22k_in1k --epochs 8)
  while pgrep -f "results_dir.*[v]2fold" > /dev/null; do sleep 30; done
  for k in 0 1 2 3 4; do train "v2fold$k" 1 --seeds 123 --fold $k --n_folds 5 "${V2[@]}"; done
  train v2refit_seed123 1 --seeds 123 --all_data "${V2[@]}"
  train v2refit_seed42  1 --seeds 42  --all_data "${V2[@]}"
  log "queue1b done"; }
q0 & q1 & wait; log "PHASE2B COMPLETE"
