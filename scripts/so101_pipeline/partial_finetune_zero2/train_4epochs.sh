#!/usr/bin/env bash
set -eo pipefail

NUM_GPUS=3
GPU_IDS="0,1,2"
SETTINGS="scripts/so101_pipeline/partial_finetune_zero2/settings.yaml"
RUN_NAME="${RUN_NAME:-formal-3gpu-zero2-4ep-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="runs/so101_g05_square_v1/$RUN_NAME"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
set -u
source /home/fenghao/use-nvidia-580159.sh
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="$REPO_ROOT/$RUN_DIR/triton_cache"
export WANDB_DIR="$REPO_ROOT/$RUN_DIR/wandb"
mkdir -p "$TRITON_CACHE_DIR" "$WANDB_DIR"

printf '%s\n' "$RUN_DIR" > runs/so101_g05_square_v1/latest_run.txt
python scripts/so101_pipeline/benchmark/gpu_monitor.py \
  --output "$RUN_DIR/gpu.csv" --interval 1.0 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

set +e
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes "$NUM_GPUS" \
  scripts/so101_pipeline/partial_finetune_zero2/train_entry.py \
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
