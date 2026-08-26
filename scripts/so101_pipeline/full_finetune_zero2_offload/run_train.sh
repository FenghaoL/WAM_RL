#!/usr/bin/env bash
set -eo pipefail

GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
MIN_FREE_MIB="${MIN_FREE_MIB:-36000}"
MAIN_PORT="${MAIN_PORT:-29561}"
SETTINGS="${SETTINGS:-scripts/so101_pipeline/full_finetune_zero2_offload/settings.yaml}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/runs/cache/torch_extensions"

IFS=',' read -ra SELECTED_GPUS <<< "$GPU_IDS"
if (( ${#SELECTED_GPUS[@]} != NUM_GPUS )); then
  echo "GPU_IDS contains ${#SELECTED_GPUS[@]} GPUs, but NUM_GPUS=$NUM_GPUS" >&2
  exit 2
fi
for gpu_id in "${SELECTED_GPUS[@]}"; do
  free_mib="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits)"
  if (( free_mib < MIN_FREE_MIB )); then
    echo "GPU $gpu_id has only ${free_mib} MiB free; need ${MIN_FREE_MIB} MiB." >&2
    exit 3
  fi
done

python scripts/so101_pipeline/full_finetune_zero2_offload/preflight.py \
  --settings "$SETTINGS"

RUN_NAME="${RUN_NAME:-formal-fastwam-zero2offload-bs4-acc1-4ep-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/so101_ahawam_square_v1/full_zero2_offload}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
mkdir -p "$RUN_ROOT"
mkdir -p "$RUN_DIR"
export TRITON_CACHE_DIR="$REPO_ROOT/$RUN_DIR/triton_cache"
export WANDB_DIR="$REPO_ROOT/$RUN_DIR/wandb"
mkdir -p "$TRITON_CACHE_DIR" "$WANDB_DIR" "$TORCH_EXTENSIONS_DIR"
# Derive the pointer location from RUN_ROOT. Hardcoding square_v1 here made a
# run on any other dataset overwrite that dataset's pointer.
printf '%s\n' "$RUN_DIR" > "$(dirname "$RUN_ROOT")/latest_run.txt"

python scripts/so101_pipeline/benchmark/gpu_monitor.py \
  --output "$RUN_DIR/gpu.csv" --interval 2.0 &
GPU_MONITOR_PID=$!

echo "epoch_s,mem_total_mb,mem_used_mb,mem_available_mb" > "$RUN_DIR/ram.csv"
( while true; do
    free -m | awk -v t="$(date +%s)" '/^Mem:/ {print t","$2","$3","$7}' >> "$RUN_DIR/ram.csv"
    sleep 2
  done ) &
RAM_MONITOR_PID=$!

cleanup() {
  kill "$GPU_MONITOR_PID" "$RAM_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" "$RAM_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

# Do not restate batch/epoch here -- they live in $SETTINGS and the authoritative
# values are printed by train_entry.py on the [schedule] line below.
echo "run=$RUN_NAME settings=$SETTINGS GPUs=$GPU_IDS OMP=$OMP_NUM_THREADS"
set +e
accelerate launch \
  --config_file scripts/so101_pipeline/full_finetune_zero2_offload/accelerate.yaml \
  --num_processes "$NUM_GPUS" \
  --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "$MAIN_PORT" \
  scripts/so101_pipeline/full_finetune_zero2_offload/train_entry.py \
  --settings "$SETTINGS" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$RUN_DIR/model" \
  --wandb-name "$RUN_NAME" \
  2>&1 | tee "$RUN_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

cleanup
trap - EXIT
echo "exit=$TRAIN_EXIT run_dir=$RUN_DIR"
exit "$TRAIN_EXIT"
