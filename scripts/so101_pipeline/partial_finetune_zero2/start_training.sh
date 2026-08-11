#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_NAME="formal-3gpu-zero2-4ep-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/so101_g05_square_v1/$RUN_NAME"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > runs/so101_g05_square_v1/latest_run.txt

RUN_NAME="$RUN_NAME" nohup setsid bash \
  scripts/so101_pipeline/partial_finetune_zero2/train_4epochs.sh \
  > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/launcher.pid"
printf '%s\n' "$PID" > runs/so101_g05_square_v1/latest_pid.txt
sleep 2
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Training process exited during startup. See $RUN_DIR/launcher.log"
  exit 1
fi
echo "run_dir=$RUN_DIR"
echo "pid=$PID"
echo "log=$RUN_DIR/train.log"
