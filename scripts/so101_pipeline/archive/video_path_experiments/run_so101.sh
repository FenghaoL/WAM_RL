#!/usr/bin/env bash
set -eo pipefail

# Change only this block for another checkpoint or sample.
GPU="${GPU:-0}"
STEPS="${STEPS:-20}"
SEED="${SEED:-42}"
SAMPLE_INDEX="${SAMPLE_INDEX:-8696}"
RUN_DIR="runs/so101_g05_square_v1/full_zero3/video_path_experiment"
CONFIG="runs/so101_g05_square_v1/full_zero3/formal-4gpu-zero3-hybrid-full-4ep-20260809-104429/model/config.yaml"
#CHECKPOINT="runs/so101_g05_square_v1/formal-3gpu-zero2-4ep-20260807-143159/model/checkpoints/weights/step_006524.pt"
CHECKPOINT="runs/so101_g05_square_v1/full_zero3/formal-4gpu-zero3-hybrid-full-4ep-20260809-104429/model/checkpoints/weights/step_004896.pt"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python scripts/so101_pipeline/archive/video_path_experiments/exact_video_path.py \
  --mode so101 \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --sample-index "$SAMPLE_INDEX" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --fps 3.75 \
  --device cuda:0 \
  --output "$RUN_DIR"
