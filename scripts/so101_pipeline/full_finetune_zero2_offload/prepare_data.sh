#!/usr/bin/env bash
set -eo pipefail

SETTINGS="scripts/so101_pipeline/full_finetune_zero2_offload/settings.yaml"
STATS_DIR="runs/so101_ahawam_square_v1/assets"
TEXT_CACHE="data/so101_ahawam_square_v1_text_embeds_cache"
GPU_ID="${GPU_ID:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

python scripts/so101_pipeline/full_finetune_zero2_offload/preflight.py \
  --settings "$SETTINGS" --metadata-only

python scripts/so101_pipeline/partial_finetune_zero2/make_stats.py \
  --settings "$SETTINGS" --output-dir "$STATS_DIR"

torchrun --standalone --nproc_per_node=1 scripts/precompute_text_embeds.py \
  task=robotwin_ahawam_offset \
  "data.train.dataset_dirs=[data/so101_ahawam_square_v1/01_pick_up_white_block/20260813_144830,data/so101_ahawam_square_v1/01_pick_up_white_block/20260813_151605,data/so101_ahawam_square_v1/03_put_white_block_on_paper_roll/20260813_154522,data/so101_ahawam_square_v1/03_put_white_block_on_paper_roll/20260813_155331]" \
  data.val=null \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  +overwrite=false

python scripts/so101_pipeline/full_finetune_zero2_offload/preflight.py \
  --settings "$SETTINGS"
