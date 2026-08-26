#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/runs/so101_ahawam_nw_2/freeze_video_continuations/gpu_reservations"
GPU_IDS="${GPU_IDS:-1,2,3}"

IFS=',' read -ra GPUS <<< "$GPU_IDS"
for gpu in "${GPUS[@]}"; do
    pid_file="$RUNTIME_DIR/gpu_${gpu}.pid"
    [[ -f "$pid_file" ]] || continue
    pid="$(<"$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
        kill "$pid"
        for _ in $(seq 1 40); do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "GPU $gpu reservation PID $pid did not stop after 40 seconds" >&2
            exit 1
        fi
        echo "released GPU $gpu reservation PID $pid"
    fi
done
