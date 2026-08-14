#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_NAME="formal-fastwam-zero2offload-bs4-acc1-4ep-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/so101_ahawam_square_v1/full_zero2_offload/$RUN_NAME"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > runs/so101_ahawam_square_v1/latest_run.txt

RUN_NAME="$RUN_NAME" nohup setsid bash \
  scripts/so101_pipeline/full_finetune_zero2_offload/run_train.sh \
  > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/launcher.pid"
printf '%s\n' "$PID" > runs/so101_ahawam_square_v1/latest_pid.txt
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Training exited during startup: $RUN_DIR/launcher.log" >&2
  exit 1
fi
echo "run_dir=$RUN_DIR"
echo "pid=$PID"
echo "log=$RUN_DIR/train.log"

