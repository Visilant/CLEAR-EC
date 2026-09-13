#!/usr/bin/env bash
# Sequential overnight regression-lever queue.
#
# Usage: night_queue.sh <gpu_index> [start_index]
#   gpu_index   - CUDA device to bind this queue to (never launch on GPU 1
#                 before the watcher releases it around 05:00 UTC).
#   start_index - 0-based index into the flattened arm list below; lets a
#                 second copy start later in the queue on another GPU.
#
# Every arm skips automatically if its results dir already has metrics.json
# or last.pt/best_model.pt (safe to re-run after a crash or a manual resume).
set -uo pipefail

GPU="${1:?usage: night_queue.sh <gpu_index> [start_index]}"
START_INDEX="${2:-0}"

PY=/home/visilant/CLEAR-EC/.worktrees/spike-sam/spike/sam/.venv/bin/python
CODE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR=/home/visilant/CLEAR-EC/data/cache
LABELS_CSV=/home/visilant/CLEAR-EC/data/final_train_ids.csv
RESULTS_ROOT=/home/visilant/CLEAR-EC/results/night_20260912
LOG_DIR="$RESULTS_ROOT/logs"
mkdir -p "$LOG_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"

# Shared recipe: convnext_tiny whole-image relative loss + EMA + cosine warmup.
COMMON_ARGS=(--input_mode whole --loss relative --lr 1e-4 --weight_decay 1e-4
             --amp --channels_last --augment_flips
             --ema 0.999 --sched cosine --warmup_epochs 1 --epochs 20 --patience 0)

is_done() {
    local dir="$1"
    [[ -f "$dir/metrics.json" || -f "$dir/last.pt" || -f "$dir/best_model.pt" ]]
}

# run_job <name> <arm_dir> <seed> <batch_size> <extra run_training.py args...>
run_job() {
    local name="$1" arm_dir="$2" seed="$3" batch_size="$4"
    shift 4
    local nested="$arm_dir/regression_cnn/seed_${seed}"
    local log="$LOG_DIR/${name}.log"
    if is_done "$nested"; then
        echo "[night_queue] SKIP $name (already has results in $nested)"
        return 0
    fi
    echo "[night_queue] RUN $name -> $arm_dir (seed=$seed batch=$batch_size)"
    "$PY" "$CODE_DIR/scripts/run_training.py" \
        --method regression --gpu 0 \
        --cache_dir "$CACHE_DIR" --labels_csv "$LABELS_CSV" \
        --results_dir "$arm_dir" --seeds "$seed" \
        --batch_size "$batch_size" \
        "${COMMON_ARGS[@]}" "$@" \
        > "$log" 2>&1
    local status=$?
    if [[ $status -ne 0 ]]; then
        echo "[night_queue] FAILED $name (exit $status); see $log"
    else
        echo "[night_queue] DONE $name"
    fi
    return $status
}

# run_convnext_tiny <name> <arm_dir> <seed> <extra args...>
run_convnext_tiny() {
    local name="$1" arm_dir="$2" seed="$3"
    shift 3
    run_job "$name" "$arm_dir" "$seed" 8 --model convnext_tiny "$@"
}

# --- Arm (a): convnext_tiny baseline + EMA, seeds 123 and 42 ---
arm_a() {
    run_convnext_tiny a_seed123 "$RESULTS_ROOT/a_seed123" 123
    run_convnext_tiny a_seed42  "$RESULTS_ROOT/a_seed42"  42
}

# --- Arm (f): convnext_tiny --target_space log ---
arm_f() {
    run_convnext_tiny f "$RESULTS_ROOT/f" 123 --target_space log
}

# --- Arm (g): convnext_tiny --photometric ---
arm_g() {
    run_convnext_tiny g "$RESULTS_ROOT/g" 123 --photometric
}

# --- Arm (b): timm convnext_tiny.in12k_ft_in1k ---
arm_b() {
    run_job b "$RESULTS_ROOT/b" 123 8 --model "timm:convnext_tiny.in12k_ft_in1k"
}

# --- Arm (c): timm convnextv2_tiny.fcmae_ft_in22k_in1k ---
arm_c() {
    run_job c "$RESULTS_ROOT/c" 123 8 --model "timm:convnextv2_tiny.fcmae_ft_in22k_in1k"
}

# --- Arm (d): timm tf_efficientnetv2_s.in21k_ft_in1k ---
arm_d() {
    run_job d "$RESULTS_ROOT/d" 123 8 --model "timm:tf_efficientnetv2_s.in21k_ft_in1k"
}

# --- Arm (e): timm convnext_small.fb_in22k_ft_in1k, batch 4 if 8 does not fit ---
arm_e() {
    local arm_dir="$RESULTS_ROOT/e"
    local nested="$arm_dir/regression_cnn/seed_123"
    if is_done "$nested"; then
        echo "[night_queue] SKIP e (already has results in $nested)"
        return 0
    fi
    if run_job e "$arm_dir" 123 8 --model "timm:convnext_small.fb_in22k_ft_in1k"; then
        return 0
    fi
    if grep -qi "out of memory" "$LOG_DIR/e.log"; then
        echo "[night_queue] e OOM at batch 8, retrying at batch 4"
        local arm_dir4="$RESULTS_ROOT/e_batch4"
        run_job e_batch4 "$arm_dir4" 123 4 --model "timm:convnext_small.fb_in22k_ft_in1k"
    fi
}

# --- Arm (h): 3 slide-grouped folds of convnext_tiny ---
arm_h() {
    run_convnext_tiny h_fold0 "$RESULTS_ROOT/h_fold0" 123 --fold 0 --n_folds 3
    run_convnext_tiny h_fold1 "$RESULTS_ROOT/h_fold1" 123 --fold 1 --n_folds 3
    run_convnext_tiny h_fold2 "$RESULTS_ROOT/h_fold2" 123 --fold 2 --n_folds 3
}

# --- Arm (i): final refit on train+val+test, no validation ---
arm_i() {
    run_convnext_tiny i_seed123 "$RESULTS_ROOT/i_seed123" 123 --all_data
    run_convnext_tiny i_seed42  "$RESULTS_ROOT/i_seed42"  42  --all_data
}

# Flattened order for --start_index bookkeeping. Each entry name maps to the
# arm function that owns it; a function may run more than one job, so once a
# function starts, all of its remaining jobs run (a function's own is_done
# checks make re-entry safe).
ARM_NAMES=(a_seed123 a_seed42 f g b c d e h_fold0 h_fold1 h_fold2 i_seed123 i_seed42)
ARM_FUNCS=(arm_a     arm_a    arm_f arm_g arm_b arm_c arm_d arm_e arm_h arm_h arm_h arm_i arm_i)

if [[ "$START_INDEX" -lt 0 || "$START_INDEX" -ge "${#ARM_NAMES[@]}" ]]; then
    echo "[night_queue] start_index must be in [0, ${#ARM_NAMES[@]})" >&2
    exit 1
fi

echo "[night_queue] GPU=$GPU start_index=$START_INDEX (${ARM_NAMES[$START_INDEX]})"

ran_funcs=""
for ((i = START_INDEX; i < ${#ARM_NAMES[@]}; i++)); do
    fn="${ARM_FUNCS[$i]}"
    if [[ "$ran_funcs" == *"|$fn|"* ]]; then
        continue  # this arm's jobs already ran as part of an earlier index
    fi
    ran_funcs="$ran_funcs|$fn|"
    "$fn"
done

echo "[night_queue] queue complete"
