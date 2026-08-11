#!/usr/bin/env bash
set -eo pipefail

ROOT="runs/so101_g05_square_v1/full_zero3"
LAUNCH_LOG="$ROOT/formal_launcher.log"
EXIT_FILE="$ROOT/formal_launcher.exit_code"
PID_FILE="$ROOT/formal_launcher.pid"
SESSION_NAME="ahawam_so101_full_zero3"
mkdir -p "$ROOT"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Formal training is already running in tmux session: $SESSION_NAME"
  exit 2
fi
if [[ -f "$PID_FILE" ]]; then
  OLD_PID="$(cat "$PID_FILE")"
  if [[ "$OLD_PID" =~ ^[0-9]+$ ]] && kill -0 "$OLD_PID" 2>/dev/null; then
    echo "Formal training is already running with process-group leader PID $OLD_PID."
    exit 2
  fi
fi

REPO_ROOT="$(pwd)"
tmux new-session -d -s "$SESSION_NAME" -c "$REPO_ROOT" \
  "bash scripts/so101_pipeline/full_finetune_zero3/train_4epochs.sh > '$REPO_ROOT/$LAUNCH_LOG' 2>&1; echo \$? > '$REPO_ROOT/$EXIT_FILE'"
echo "Started formal ZeRO-3 training in tmux session: $SESSION_NAME"
echo "Launcher log: $LAUNCH_LOG"
