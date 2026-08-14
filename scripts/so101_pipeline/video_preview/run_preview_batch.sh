#!/usr/bin/env bash
set -eo pipefail

# Edit these values directly before running.
GPU=0
NUM_SAMPLES=8
STEPS=20
SEED=42
FPS=3.75

CONFIG="runs/so101_g05_square_v1/full_zero3/formal-4gpu-zero3-hybrid-full-4ep-20260809-104429/model/config.yaml"
CHECKPOINT="runs/so101_g05_square_v1/full_zero3/formal-4gpu-zero3-hybrid-full-4ep-20260809-104429/model/checkpoints/weights/step_004896.pt"
OUTPUT="runs/so101_g05_square_v1/full_zero3/video_preview_online_history"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export DIFFSYNTH_MODEL_BASE_PATH="$REPO_ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python scripts/so101_pipeline/archive/video_path_experiments/batch/batch_exact_video_path.py \
  --mode so101 \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --device cuda:0 \
  --num-samples "$NUM_SAMPLES" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --fps "$FPS" \
  --history-mode online
