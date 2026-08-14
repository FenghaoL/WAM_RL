#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_NAME="smoke-fastwam-zero2offload-bs4-acc1-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/so101_ahawam_square_v1/full_zero2_offload/$RUN_NAME"
mkdir -p "$RUN_DIR"

source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2,3}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export TRITON_CACHE_DIR="$REPO_ROOT/$RUN_DIR/triton_cache"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/runs/cache/torch_extensions"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

python scripts/so101_pipeline/benchmark/gpu_monitor.py \
  --output "$RUN_DIR/gpu.csv" --interval 1.0 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

accelerate launch \
  --config_file scripts/so101_pipeline/full_finetune_zero2_offload/accelerate.yaml \
  --num_processes 4 --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${MAIN_PORT:-29560}" \
  scripts/so101_pipeline/full_finetune_zero2_offload/train_entry.py \
  --settings scripts/so101_pipeline/full_finetune_zero2_offload/settings.yaml \
  --num-gpus 4 --output-dir "$RUN_DIR/model" --wandb-name "$RUN_NAME" \
  --smoke-steps "${SMOKE_STEPS:-3}" \
  2>&1 | tee "$RUN_DIR/train.log"
