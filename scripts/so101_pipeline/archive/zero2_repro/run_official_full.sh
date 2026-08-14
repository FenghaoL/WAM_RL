#!/usr/bin/env bash
set -eo pipefail

# This is the minimal four-GPU, full-finetuning reproduction.
GPU_IDS="0,1,2,3"
NUM_GPUS=4

ROBOTWIN_DATA="data/robotwin2.0-fastwam/robotwin2.0"
ROBOTWIN_STATS="checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json"
TEXT_CACHE="data/robotwin_text_embeds_cache"
ACTION_DIT="checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

WORK_DIR="runs/zero2_repro/official-full"
RECORD_DIR="scripts/so101_pipeline/zero2_repro/records/official-full"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="$REPO_ROOT/$WORK_DIR/triton_cache"
mkdir -p "$TRITON_CACHE_DIR" "$RECORD_DIR"

GPU_LOG="$RECORD_DIR/gpu_monitor.csv"
TRAIN_LOG="$RECORD_DIR/train_output.log.md"
echo "timestamp, gpu_index, gpu_name, gpu_uuid, memory_total_mib, memory_used_mib, memory_free_mib, gpu_utilization_percent, memory_utilization_percent, temperature_c, power_draw_w" > "$GPU_LOG"
nvidia-smi \
  --id="$GPU_IDS" \
  --query-gpu=timestamp,index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
  --format=csv,noheader,nounits \
  --loop=1 >> "$GPU_LOG" 2>&1 &
GPU_MONITOR_PID=$!

stop_gpu_monitor() {
  kill "$GPU_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" 2>/dev/null || true
  if [[ -f "$WORK_DIR/model/config.yaml" ]]; then
    cp "$WORK_DIR/model/config.yaml" "$RECORD_DIR/config.yaml" || true
  fi
}
trap stop_gpu_monitor EXIT

echo "Expected result: OOM at the first optimizer.step() while Adam allocates exp_avg_sq."
echo "Log: $TRAIN_LOG"
echo "GPU monitor: $GPU_LOG"

bash scripts/train_zero2.sh "$NUM_GPUS" \
  task=robotwin_ahawam \
  model=ahawam \
  model.action_dit_pretrained_path="$ACTION_DIT" \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.train.pretrained_norm_stats="$ROBOTWIN_STATS" \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  data.val=null \
  output_dir="$WORK_DIR/model" \
  wandb.enabled=false \
  batch_size=1 \
  num_workers=0 \
  gradient_accumulation_steps=1 \
  max_steps=1 \
  log_every=1 \
  save_every=1000000 \
  eval_every=0 \
  freeze_video_dit=false \
  model.mot_checkpoint_mixed_attn=true \
  2>&1 | tee "$TRAIN_LOG"
