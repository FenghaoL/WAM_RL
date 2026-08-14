#!/usr/bin/env bash
set -eo pipefail

# Control experiment: identical official launcher, but VideoDiT is frozen.
GPU_IDS="0,1,2,3"
NUM_GPUS=4

ROBOTWIN_DATA="data/robotwin2.0-fastwam/robotwin2.0"
ROBOTWIN_STATS="checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json"
TEXT_CACHE="data/robotwin_text_embeds_cache"
ACTION_DIT="checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

RUN_DIR="runs/zero2_repro/official-freeze-video"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="$REPO_ROOT/$RUN_DIR/triton_cache"
mkdir -p "$TRITON_CACHE_DIR"

echo "Expected result: one complete optimizer step without OOM."
echo "Log: $RUN_DIR/train.log"

bash scripts/train_zero2.sh "$NUM_GPUS" \
  task=robotwin_ahawam \
  model=ahawam \
  model.action_dit_pretrained_path="$ACTION_DIT" \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.train.pretrained_norm_stats="$ROBOTWIN_STATS" \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  data.val=null \
  output_dir="$RUN_DIR/model" \
  wandb.enabled=false \
  batch_size=1 \
  num_workers=0 \
  gradient_accumulation_steps=1 \
  max_steps=1 \
  log_every=1 \
  save_every=1000000 \
  eval_every=0 \
  freeze_video_dit=true \
  model.mot_checkpoint_mixed_attn=true \
  2>&1 | tee "$RUN_DIR/train.log"
