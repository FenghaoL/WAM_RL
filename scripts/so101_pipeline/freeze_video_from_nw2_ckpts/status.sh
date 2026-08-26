#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
SOURCE_RUN="runs/so101_ahawam_nw_2/full_zero2_offload_ahawam_base/formal-nw2-ahawambase-zero2offload-bs4-acc1-5ep-3gpu-20260818-232749"
pointer="$SOURCE_RUN/freeze_video_continuations/latest_batch.txt"
[[ -f "$pointer" ]] || { echo "No formal batch pointer found"; exit 1; }
batch_dir="$(<"$pointer")"
echo "batch_dir=$batch_dir"
while IFS= read -r run_dir; do
    [[ -n "$run_dir" ]] || continue
    pid="$(<"$run_dir/launcher.pid")"
    if kill -0 "$pid" 2>/dev/null; then state=running; else state=stopped; fi
    echo "state=$state pid=$pid run_dir=$run_dir"
    grep -aE '^\[schedule\]|^\[parameters\]|step=[0-9]+/|Traceback|OutOfMemory|^exit=' \
        "$run_dir/train.log" 2>/dev/null | tail -n 3 || true
done < "$batch_dir/run_dirs.txt"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader
