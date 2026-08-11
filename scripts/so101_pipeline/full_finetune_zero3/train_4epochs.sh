#!/usr/bin/env bash
set -eo pipefail

# The tested 4 x A100 40GB profile: ZeRO-3 with 35% optimizer CPU offload.
GPU_IDS="0,1,2,3"
NUM_GPUS=4
MIN_FREE_MIB=30000
SETTINGS="scripts/so101_pipeline/full_finetune_zero3/settings.yaml"
CONFIG_FILE="scripts/so101_pipeline/full_finetune_zero3/accelerate_zero3_hybrid.yaml"
RUN_NAME="formal-4gpu-zero3-hybrid-full-4ep-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/so101_g05_square_v1/full_zero3/$RUN_NAME"

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
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="$REPO_ROOT/$RUN_DIR/triton_cache"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/runs/cache/torch_extensions"
export WANDB_DIR="$REPO_ROOT/$RUN_DIR/wandb"
mkdir -p "$RUN_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$WANDB_DIR"
printf '%s\n' "$RUN_DIR" > runs/so101_g05_square_v1/full_zero3/latest_formal.txt

for free_mib in $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); do
  if (( free_mib < MIN_FREE_MIB )); then
    echo "Refusing to start: need ${MIN_FREE_MIB} MiB free per GPU, found ${free_mib} MiB."
    exit 3
  fi
done

python scripts/so101_pipeline/benchmark/gpu_monitor.py \
  --output "$RUN_DIR/gpu.csv" --interval 1.0 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

set +e
accelerate launch \
  --config_file "$CONFIG_FILE" \
  --num_processes "$NUM_GPUS" \
  scripts/so101_pipeline/full_finetune_zero3/train_entry.py \
  --settings "$SETTINGS" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$RUN_DIR/model" \
  --wandb-name "$RUN_NAME" \
  2>&1 | tee "$RUN_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT
python scripts/so101_pipeline/benchmark/summarize.py \
  --name "$RUN_NAME" \
  --log "$RUN_DIR/train.log" \
  --gpu-csv "$RUN_DIR/gpu.csv" \
  --exit-code "$TRAIN_EXIT" \
  --output "$RUN_DIR/summary.md"
exit "$TRAIN_EXIT"
