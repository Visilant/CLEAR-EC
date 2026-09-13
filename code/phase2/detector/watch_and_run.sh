#!/usr/bin/env bash
# Wait until scripts/run_training.py (the other user's training job on GPU 1)
# has not been running for 3 consecutive 60s checks, then run run_all.sh on GPU 1.
# nohup-safe: run as `nohup ./watch_and_run.sh &`.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs

clear_checks=0
while [ "$clear_checks" -lt 3 ]; do
  if pgrep -f "scripts/run_training.py" > /dev/null; then
    clear_checks=0
    echo "$(date): run_training.py still running, waiting" >> logs/watch.log
  else
    clear_checks=$((clear_checks + 1))
    echo "$(date): run_training.py not seen ($clear_checks/3)" >> logs/watch.log
  fi
  sleep 60
done

echo "$(date): run_training.py gone for 3 checks, starting run_all.sh on GPU 1" >> logs/watch.log
CUDA_VISIBLE_DEVICES=1 ./run_all.sh >> logs/watch.log 2>&1
echo "$(date): run_all.sh exited with code $?" >> logs/watch.log
