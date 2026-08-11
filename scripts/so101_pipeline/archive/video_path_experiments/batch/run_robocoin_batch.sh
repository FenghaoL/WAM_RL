#!/usr/bin/env bash
set -eo pipefail

GPU="${GPU:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
STEPS="${STEPS:-20}"
SEED="${SEED:-42}"
RUN_DIR="runs/robocoin/video_path_batch_history"
DATASET_DIR="data/robocoin/Cobot_Magic_move_the_bread"
TEXT_CACHE="data/robocoin/Cobot_Magic_move_the_bread_text_embeds"
CONFIG="runs/so101_g05_square_v1/formal-3gpu-zero2-4ep-20260807-143159/model/config.yaml"
CHECKPOINT="checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python scripts/so101_pipeline/video_path_experiments/batch/batch_exact_video_path.py \
  --mode robocoin \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --dataset-dir "$DATASET_DIR" \
  --text-cache-dir "$TEXT_CACHE" \
  --num-samples "$NUM_SAMPLES" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --fps 6.25 \
  --device cuda:0 \
  --output "$RUN_DIR"
