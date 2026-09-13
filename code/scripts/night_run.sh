#!/usr/bin/env bash
# Overnight driver, 2026-09-12. Two GPU queues + gate + OOF analysis.
#   GPU0: [r3_seed123 already running] -> gate -> folds 0,1,2 -> all-data refit seed 123
#   GPU1: wait for SAM spike -> r3_seed42 -> gate -> folds 3,4 -> all-data refit seed 42
set -uo pipefail
PY=/home/visilant/CLEAR-EC/code/.venv/bin/python
CODE=/home/visilant/CLEAR-EC/.worktrees/regression-night/code
R=/home/visilant/CLEAR-EC/results/night_20260912
CACHE=/home/visilant/CLEAR-EC/data/cache
LABELS=/home/visilant/CLEAR-EC/data/final_train_ids.csv
mkdir -p "$R/logs" "$R/oof" "$R/tta"
BASE=(--model convnext_tiny --input_mode whole --loss relative --batch_size 8 --lr 1e-4 --weight_decay 1e-4 --amp --channels_last --augment_flips)
EMA=(--ema 0.999 --sched cosine --warmup_epochs 1 --epochs 12 --patience 0)
PLAIN=(--epochs 20 --patience 5)
GATE_MAX=9.70   # last-epoch EMA val mean must be <= this to adopt the EMA recipe for folds/refit
log() { echo "$(date '+%F %T') $*" >> "$R/logs/driver.log"; }

train() {  # train <name> <gpu> <recipe-array-name> [extra...]
  local name=$1 gpu=$2 recipe=$3; shift 3
  local dir="$R/$name"
  if compgen -G "$dir/regression_cnn/seed_*/metrics.json" > /dev/null; then log "SKIP $name"; return 0; fi
  local -n RC=$recipe
  log "START $name gpu=$gpu recipe=$recipe $*"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" "$CODE/scripts/run_training.py" --method regression --gpu 0 \
    --cache_dir "$CACHE" --labels_csv "$LABELS" --results_dir "$dir" "${BASE[@]}" "${RC[@]}" "$@" \
    > "$R/logs/$name.log" 2>&1
  log "END $name exit=$?"
}
wait_file() { while [[ ! -f "$1" ]]; do sleep 60; done; }

gate() {  # decide recipe once r3_seed123 is done; writes $R/recipe.txt
  wait_file "$R/r3_seed123/regression_cnn/seed_123/metrics.json"
  if [[ -f "$R/recipe.txt" ]]; then return; fi
  CUDA_VISIBLE_DEVICES=$1 "$PY" "$CODE/scripts/night_predict.py" --ckpt_dir "$R/r3_seed123/regression_cnn/seed_123" \
    --which last --indices val --tta none --out "$R/tta/r3_seed123_val_last.csv" --gpu 0 > "$R/logs/gate.log" 2>&1
  local last; last=$("$PY" -c "import json;print(json.load(open('$R/tta/r3_seed123_val_last.json'))['mape']['mean'])")
  local best; best=$("$PY" -c "import json;print(json.load(open('$R/r3_seed123/regression_cnn/seed_123/metrics.json'))['val_best_mape_mean'])")
  if "$PY" -c "import sys; sys.exit(0 if $last <= $GATE_MAX else 1)"; then echo EMA > "$R/recipe.txt"; else echo PLAIN > "$R/recipe.txt"; fi
  log "GATE r3_seed123 last-epoch val=$last best-epoch val=$best -> recipe $(cat "$R/recipe.txt")"
}

queue0() {
  gate 0
  local rc; rc=$(cat "$R/recipe.txt")
  for k in 0 1 2 3 4; do train "fold$k" 0 "$rc" --seeds 123 --fold $k --n_folds 5; done
  train refit_seed123 0 "$rc" --seeds 123 --all_data
  log "queue0 done"
}
queue1() {
  while pgrep -f "spike/sam/run_all.sh|eval_val.py|eval_overlays.py" > /dev/null; do sleep 60; done
  log "GPU1 free"
  while pgrep -f "results_dir.*r3_seed42" > /dev/null; do sleep 60; done   # 20-epoch seed-42 curve, launched by the first driver
  log "r3_seed42 finished"
  # exploration arms, all on the EMA recipe at seed 123 so they compare to r3_seed123
  train x_crop06   1 EMA --seeds 123 --crop_scale 0.6
  train x_logspace 1 EMA --seeds 123 --target_space log
  train x_small    1 EMA --seeds 123 --model convnext_small
  train x_in12k    1 EMA --seeds 123 --model timm:convnext_tiny.in12k_ft_in1k
  train x_v2tiny   1 EMA --seeds 123 --model timm:convnextv2_tiny.fcmae_ft_in22k_in1k
  wait_file "$R/recipe.txt"; local rc; rc=$(cat "$R/recipe.txt")
  train refit_seed42 1 "$rc" --seeds 42 --all_data
  log "queue1 done"
}
queue0 & Q0=$!
queue1 & Q1=$!
wait $Q0 $Q1
log "both queues done; scoring OOF"
for k in 0 1 2 3 4; do
  d="$R/fold$k/regression_cnn/seed_123"
  for w in last best; do for t in none flips; do
    if [[ $w == best ]]; then [[ -f "$d/best_model.pt" ]] || continue; else [[ -f "$d/last.pt" ]] || continue; fi
    CUDA_VISIBLE_DEVICES=$(( k % 2 )) "$PY" "$CODE/scripts/night_predict.py" --ckpt_dir "$d" --which $w --indices "fold:$k/5" --tta $t \
      --out "$R/oof/fold${k}_${w}_${t}.csv" --gpu 0 >> "$R/logs/oof.log" 2>&1
  done; done
done
log "PHASE1 COMPLETE"
