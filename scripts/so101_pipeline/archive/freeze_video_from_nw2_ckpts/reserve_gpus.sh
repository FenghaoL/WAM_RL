#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RUNTIME_DIR="$REPO_ROOT/runs/so101_ahawam_nw_2/freeze_video_continuations/gpu_reservations"
GPU_IDS="${GPU_IDS:-1,2,3}"
PYTHON_BIN="/home/fenghao/miniconda3/envs/ahawam/bin/python"
mkdir -p "$RUNTIME_DIR"
source /home/fenghao/use-nvidia-580159.sh

IFS=',' read -ra GPUS <<< "$GPU_IDS"
for gpu in "${GPUS[@]}"; do
    log="$RUNTIME_DIR/gpu_${gpu}.log"
    pid_file="$RUNTIME_DIR/gpu_${gpu}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
        echo "GPU $gpu already reserved by PID $(<"$pid_file")"
        continue
    fi
    CUDA_VISIBLE_DEVICES="$gpu" setsid nohup "$PYTHON_BIN" \
        "$REPO_ROOT/scripts/so101_pipeline/freeze_video_from_nw2_ckpts/reserve_gpus.py" \
        > "$log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$pid_file"
    echo "GPU $gpu reservation PID $pid"
done

sleep 4
for gpu in "${GPUS[@]}"; do
    log="$RUNTIME_DIR/gpu_${gpu}.log"
    if [[ -f "$log" ]]; then
        printf 'GPU %s: ' "$gpu"
        tail -n 1 "$log"
    fi
done
