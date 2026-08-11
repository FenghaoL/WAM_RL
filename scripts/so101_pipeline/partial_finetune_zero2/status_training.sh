#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_DIR="$(cat runs/so101_g05_square_v1/latest_run.txt)"
PID="$(cat runs/so101_g05_square_v1/latest_pid.txt 2>/dev/null || true)"
echo "run_dir=$RUN_DIR"
echo "pid=${PID:-unknown}"
if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
  echo "status=running"
else
  echo "status=not-running"
fi
tail -n 30 "$RUN_DIR/train.log" 2>/dev/null || tail -n 30 "$RUN_DIR/launcher.log"
