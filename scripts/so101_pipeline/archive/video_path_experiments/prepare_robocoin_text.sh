#!/usr/bin/env bash
set -eo pipefail

GPU="${GPU:-0}"
DATASET_DIR="data/robocoin/Cobot_Magic_move_the_bread"
TEXT_CACHE="data/robocoin/Cobot_Magic_move_the_bread_text_embeds"
TASK="grasp the yellow bread and place it into the yellow plate, then retrieve it."
ACTION_DIT="checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u

export CUDA_VISIBLE_DEVICES="$GPU"
export HF_HOME="$REPO_ROOT/.cache/huggingface"

python scripts/precompute_text_embeds.py \
  task=robotwin_ahawam \
  model.action_dit_pretrained_path="$ACTION_DIT" \
  "data.train.dataset_dirs=[$DATASET_DIR]" \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  data.val=null \
  "+override_instruction='$TASK'" \
  +overwrite=false
