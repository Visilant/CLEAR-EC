#!/usr/bin/env bash
# Artifact cleanup approved on 2026-09-13 (aggressive level: dead/empty/diverged results, superseded
# docker images, the unused seed123 submission worktree). Checkpoints referenced by any REPORT or
# manifest are kept. Usage: bash cleanup_20260913.sh [--dry-run]
set -u
REPO=/home/visilant/CLEAR-EC
DRY=${1:-}
PATHS=(
  "$REPO/results/failed_run_20260827_125247"   # aborted first overnight attempt, superseded by results/overnight
  "$REPO/results/baseline"                     # empty stub
  "$REPO/results/training"                     # empty stub (logs/ only)
  "$REPO/results/night_20260912/v2fold0_s42"   # diverged seed-42 V2 run (val 9.95); REPORT says deleted; clipped rerun kept
)
IMAGES=(clear_ec_audit:local clear_ec_phase1:seed123)   # superseded small-CNN containers; clear_ec_phase1:v2ens stays
for p in "${PATHS[@]}"; do
  [ -e "$p" ] || { echo "absent  $p"; continue; }
  echo "$(du -sh "$p" | cut -f1)  $p"
  [ "$DRY" = "--dry-run" ] || rm -rf "$p"
done
for img in "${IMAGES[@]}"; do
  docker image inspect "$img" > /dev/null 2>&1 || { echo "absent  $img"; continue; }
  echo "$(docker images --format '{{.Size}}' "$img")  docker image $img"
  [ "$DRY" = "--dry-run" ] || docker rmi "$img" > /dev/null
done
find "$REPO/code" -name __pycache__ -prune -print0 | xargs -0 rm -rf
echo "done (${DRY:-real run})"
