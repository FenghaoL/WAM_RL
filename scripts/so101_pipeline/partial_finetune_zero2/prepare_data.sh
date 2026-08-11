#!/usr/bin/env bash
set -eo pipefail

# This is the only data-preparation entry point. Edit these defaults as needed.
SOURCE_DATA="data/so101_g05_square_v1"
OUTPUT_DATA="data/so101_g05_square_v1_ahawam_v21"
TEXT_CACHE="data/so101_g05_square_v1_text_embeds_cache"
STATS_DIR="runs/so101_g05_square_v1/assets"
SETTINGS="scripts/so101_pipeline/partial_finetune_zero2/settings.yaml"
GPU_ID=0

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam

if [[ -f "$OUTPUT_DATA/meta/conversion_manifest.json" ]]; then
  echo "Converted dataset already exists; validating it: $OUTPUT_DATA"
else
  python scripts/so101_pipeline/partial_finetune_zero2/convert_v3_to_v21.py \
    --source "$SOURCE_DATA" \
    --output "$OUTPUT_DATA"
fi

python scripts/so101_pipeline/partial_finetune_zero2/validate_dataset.py \
  --dataset "$OUTPUT_DATA"

python scripts/so101_pipeline/partial_finetune_zero2/make_stats.py \
  --settings "$SETTINGS" \
  --output-dir "$STATS_DIR"

source /home/fenghao/use-nvidia-580159.sh
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
torchrun --standalone --nproc_per_node=1 scripts/precompute_text_embeds.py \
  task=robotwin_ahawam_offset \
  "data.train.dataset_dirs=[$OUTPUT_DATA]" \
  data.val=null \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  +overwrite=false
