#!/usr/bin/env bash
set -eo pipefail

SESSION_NAME="ahawam_so101_full_zero3"
PID_FILE="runs/so101_g05_square_v1/full_zero3/formal_launcher.pid"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  tmux kill-session -t "$SESSION_NAME"
  echo "Stopped training tmux session: $SESSION_NAME"
  exit 0
fi

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if [[ "$PID" =~ ^[0-9]+$ ]] && kill -0 "$PID" 2>/dev/null; then
    kill -- "-$PID"
    echo "Stopped training process group: $PID"
    exit 0
  fi
fi

echo "Training is already stopped."
