#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_NAME="formal-padapterv2-flash-ode2-from-step005328-4xa100-$(date +%Y%m%d-%H%M%S)"
RUN_ROOT="runs/so101_ahawam_padapter_v2/flash_distillation"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"
printf '%s\n' "$RUN_DIR" > runs/so101_ahawam_padapter_v2/latest_flash_run.txt

RUN_NAME="$RUN_NAME" RUN_ROOT="$RUN_ROOT" \
GPU_IDS="0,1,2,3" NUM_GPUS=4 MIN_FREE_MIB=36000 MAIN_PORT=29624 \
SETTINGS="scripts/so101_pipeline/flash_distillation/settings_4gpu.yaml" \
  nohup setsid bash scripts/so101_pipeline/flash_distillation/run_train.sh \
  > "$RUN_DIR/launcher.log" 2>&1 < /dev/null &
PID=$!
printf '%s\n' "$PID" > "$RUN_DIR/launcher.pid"
printf '%s\n' "$PID" > runs/so101_ahawam_padapter_v2/latest_flash_pid.txt

sleep 3
if ! kill -0 "$PID" 2>/dev/null; then
  echo "Four-GPU Flash distillation exited during startup: $RUN_DIR/launcher.log" >&2
  tail -n 80 "$RUN_DIR/launcher.log" >&2 || true
  exit 1
fi
echo "run_dir=$RUN_DIR"
echo "pid=$PID"
echo "log=$RUN_DIR/train.log"
