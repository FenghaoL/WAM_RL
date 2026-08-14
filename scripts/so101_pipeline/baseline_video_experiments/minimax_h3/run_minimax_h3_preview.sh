#!/usr/bin/env bash
# MiniMax-H3 image-to-video baseline on the exported SO101 windows.
#
# Prerequisites (both are one-off):
#   bash scripts/so101_pipeline/baseline_video_experiments/minimax_h3/setup_env.sh
#   huggingface-cli download MiniMaxAI/MiniMax-H3 --local-dir checkpoints/MiniMaxAI/MiniMax-H3 \
#     --include "modular_model_index.json" "model_index.json" "transformer/*" \
#               "text_encoder/*" "vae/*" "audio_vae/*" "processor/*" "tokenizer/*" \
#               "scheduler/*" "audio_scheduler/*"
#
# Then:
#   bash scripts/so101_pipeline/baseline_video_experiments/minimax_h3/run_minimax_h3_preview.sh
set -eo pipefail

GPU="${GPU:-1}"
LIMIT="${LIMIT:-6}"
HEIGHT="${HEIGHT:-576}"
WIDTH="${WIDTH:-480}"
STEPS="${STEPS:-32}"      # MiniMax-H3 defaults to 50 sigma points; 32 keeps quality at ~2/3 the cost
SEED="${SEED:-42}"
# int8 or int4. Both quantize on the fly from the bfloat16 checkpoint already on
# disk -- neither needs an extra download. int4 uses torchao tinygemm, the only
# int4 packing format A100 (sm_80) supports.
QUANT="${QUANT:-int8}"
ENV_NAME="${ENV_NAME:-minimax_h3}"
EXTRA_ARGS=("$@")

SAMPLES_DIR="runs/so101_g05_square_v1/baseline_experiments/samples"
OUTPUT="${OUTPUT:-runs/so101_g05_square_v1/baseline_experiments/minimax_h3_$QUANT}"
MODEL_PATH="checkpoints/MiniMaxAI/MiniMax-H3"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_NAME"
source /home/fenghao/use-nvidia-580159.sh

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

python scripts/so101_pipeline/baseline_video_experiments/minimax_h3/run_minimax_h3_preview.py \
  --samples-dir "$SAMPLES_DIR" \
  --output "$OUTPUT" \
  --model-path "$MODEL_PATH" \
  --run-name "minimax_h3_$QUANT" \
  --device cuda:0 \
  --quantization "$QUANT" \
  --height "$HEIGHT" \
  --width "$WIDTH" \
  --num-inference-steps "$STEPS" \
  --seed "$SEED" \
  --limit "$LIMIT" \
  "${EXTRA_ARGS[@]}"
