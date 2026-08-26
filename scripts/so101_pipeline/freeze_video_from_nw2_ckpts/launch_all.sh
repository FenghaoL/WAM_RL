#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-6}"
STAMP="${STAMP:-$(date +%Y%m%d-%H%M%S)}"
SOURCE_RUN="runs/so101_ahawam_nw_2/full_zero2_offload_ahawam_base/formal-nw2-ahawambase-zero2offload-bs4-acc1-5ep-3gpu-20260818-232749"
PIPELINE="scripts/so101_pipeline/freeze_video_from_nw2_ckpts"

# Reservations remain in place until this script is ready to replace them with
# the actual jobs. release_gpus.sh waits for memory to be returned.
GPU_IDS=1,2,3 bash "$PIPELINE/release_gpus.sh"

steps=(001988 002982 003976)
gpus=(1 2 3)
ports=(29641 29642 29643)
run_dirs=()

for i in 0 1 2; do
    step="${steps[$i]}"
    gpu="${gpus[$i]}"
    port="${ports[$i]}"
    run_name="formal-action-only-freeze-video-from-${step}-${NUM_EPOCHS}ep-1gpu-bs${BATCH_SIZE}-${STAMP}"
    run_dir="$SOURCE_RUN/freeze_video_continuations/from_step_${step}/$run_name"
    mkdir -p "$run_dir"
    setsid nohup env \
        GPU_IDS="$gpu" \
        NUM_GPUS=1 \
        SOURCE_STEP="$step" \
        NUM_EPOCHS="$NUM_EPOCHS" \
        BATCH_SIZE="$BATCH_SIZE" \
        OMP_NUM_THREADS=8 \
        MAIN_PORT="$port" \
        MIN_FREE_MIB=36000 \
        RUN_NAME="$run_name" \
        bash "$PIPELINE/run_one.sh" \
        > "$run_dir/launcher.log" 2>&1 < /dev/null &
    pid=$!
    printf '%s\n' "$pid" > "$run_dir/launcher.pid"
    run_dirs+=("$run_dir")
    echo "launched source_step=$step gpu=$gpu pid=$pid run_dir=$run_dir"
done

batch_dir="$SOURCE_RUN/freeze_video_continuations/formal_batch_${STAMP}"
mkdir -p "$batch_dir"
printf '%s\n' "${run_dirs[@]}" > "$batch_dir/run_dirs.txt"
printf '%s\n' "$batch_dir" > "$SOURCE_RUN/freeze_video_continuations/latest_batch.txt"

sleep 3
for run_dir in "${run_dirs[@]}"; do
    pid="$(<"$run_dir/launcher.pid")"
    if ! kill -0 "$pid" 2>/dev/null; then
        echo "Launcher PID $pid exited early: $run_dir" >&2
        tail -n 30 "$run_dir/launcher.log" >&2 || true
        exit 1
    fi
done
echo "batch_dir=$batch_dir"
