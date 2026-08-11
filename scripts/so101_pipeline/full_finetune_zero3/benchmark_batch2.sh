#!/usr/bin/env bash
set -eo pipefail

# Test micro batch 2 at the same effective batch (2 x 4 GPUs x accumulation 2 = 16).
# Run this after the current formal job finishes; it deliberately refuses to compete
# with the formal job for GPU memory.
SESSION_NAME="ahawam_so101_full_zero3"
ROOT="runs/so101_g05_square_v1/full_zero3"
LATEST_FILE="$ROOT/latest_formal.txt"
PID_FILE="$ROOT/formal_launcher.pid"
FORMAL_ACTIVE=0

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
  FORMAL_ACTIVE=1
fi
if [[ -f "$PID_FILE" ]]; then
  FORMAL_PID="$(cat "$PID_FILE")"
  if [[ "$FORMAL_PID" =~ ^[0-9]+$ ]] && kill -0 "$FORMAL_PID" 2>/dev/null; then
    FORMAL_ACTIVE=1
  fi
fi
# Some containers cannot access the host tmux socket or PID namespace. A formal
# log updated in the last ten minutes is another conservative activity signal.
if [[ -f "$LATEST_FILE" ]]; then
  FORMAL_DIR="$(cat "$LATEST_FILE")"
  FORMAL_LOG="$FORMAL_DIR/train.log"
  if [[ -f "$FORMAL_LOG" ]]; then
    LOG_AGE_SECONDS=$(( $(date +%s) - $(stat -c %Y "$FORMAL_LOG") ))
    if (( LOG_AGE_SECONDS < 600 )); then
      FORMAL_ACTIVE=1
    fi
  fi
fi

if (( FORMAL_ACTIVE == 1 )); then
  echo "Formal training is still active; batch-2 benchmark was not started."
  echo "Wait for it to finish, then run this script again."
  exit 3
fi

SMOKE_STEPS=4 \
SMOKE_BATCH_SIZE=2 \
SMOKE_GRAD_ACCUM=2 \
SMOKE_FIRST_HISTORY_LEN=5 \
bash scripts/so101_pipeline/full_finetune_zero3/smoke_test.sh
