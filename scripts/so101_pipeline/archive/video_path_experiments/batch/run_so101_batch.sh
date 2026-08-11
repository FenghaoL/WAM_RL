#!/usr/bin/env bash
set -eo pipefail

# Edit these defaults directly, or override a value for one run: NUM_SAMPLES=12 bash ...
GPU="${GPU:-0}"
NUM_SAMPLES="${NUM_SAMPLES:-8}"
STEPS="${STEPS:-20}"
SEED="${SEED:-42}"
RUN_DIR="runs/so101_g05_square_v1/video_path_batch_history"
CONFIG="runs/so101_g05_square_v1/formal-3gpu-zero2-4ep-20260807-143159/model/config.yaml"
CHECKPOINT="runs/so101_g05_square_v1/formal-3gpu-zero2-4ep-20260807-143159/model/checkpoints/weights/step_006524.pt"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

python scripts/so101_pipeline/video_path_experiments/batch/batch_exact_video_path.py \
  --mode so101 \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --num-samples "$NUM_SAMPLES" \
  --steps "$STEPS" \
  --seed "$SEED" \
  --fps 3.75 \
  --device cuda:0 \
  --output "$RUN_DIR"
