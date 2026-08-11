#!/usr/bin/env bash
set -eo pipefail

ROOT="runs/so101_g05_square_v1/full_zero3"
LATEST_FILE="$ROOT/latest_formal.txt"
PID_FILE="$ROOT/formal_launcher.pid"
SESSION_NAME="ahawam_so101_full_zero3"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  echo "Training tmux session is running: $SESSION_NAME"
elif [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Training process group is running: PID $(cat "$PID_FILE")"
else
  echo "Training is not running."
fi

if [[ -f "$LATEST_FILE" ]]; then
  RUN_DIR="$(cat "$LATEST_FILE")"
  echo "Latest run: $RUN_DIR"
  if [[ -f "$RUN_DIR/train.log" ]]; then
    tail -n 20 "$RUN_DIR/train.log"
  fi
  if [[ -f "$RUN_DIR/summary.md" ]]; then
    echo "Final summary: $RUN_DIR/summary.md"
  elif ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "No final summary exists yet."
  fi
fi
