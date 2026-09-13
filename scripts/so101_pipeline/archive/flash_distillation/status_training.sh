#!/usr/bin/env bash
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUN_DIR="$(cat runs/so101_ahawam_padapter_v2/latest_flash_run.txt)"
PID="$(cat runs/so101_ahawam_padapter_v2/latest_flash_pid.txt 2>/dev/null || true)"
echo "run_dir=$RUN_DIR"
echo "pid=${PID:-unknown}"
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  echo "status=running"
else
  echo "status=not-running"
fi
tail -n 60 "$RUN_DIR/train.log" 2>/dev/null || tail -n 60 "$RUN_DIR/launcher.log"

