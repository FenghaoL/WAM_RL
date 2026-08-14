#!/usr/bin/env bash
set -eo pipefail

# Same full-finetune workload as run_official_full.sh, but with ZeRO-2 +
# CPU optimizer offload -- i.e. the configuration the A30 machine actually uses.
# Expected result: no OOM, one complete optimizer step.
#
# GPU_IDS/NUM_GPUS are overridable: NUM_GPUS=3 GPU_IDS=0,1,2 bash <this script>
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"

ROBOTWIN_DATA="data/robotwin2.0-fastwam/robotwin2.0"
ROBOTWIN_STATS="checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json"
TEXT_CACHE="data/robotwin_text_embeds_cache"
ACTION_DIT="checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt"

WORK_DIR="runs/zero2_repro/offload-full"
RECORD_DIR="scripts/so101_pipeline/zero2_repro/records/offload-full"

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
export HYDRA_FULL_ERROR=1
# With optimizer offload, Adam runs on the CPU over ~1.8B params per rank.
# torchrun defaults OMP_NUM_THREADS to 1, which makes that step crawl.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
mkdir -p "$TRITON_CACHE_DIR" "$RECORD_DIR"

GPU_LOG="$RECORD_DIR/gpu_monitor.csv"
RAM_LOG="$RECORD_DIR/ram_monitor.csv"
TRAIN_LOG="$RECORD_DIR/train_output.log"
echo "timestamp, gpu_index, gpu_name, memory_total_mib, memory_used_mib, memory_free_mib, gpu_utilization_percent" > "$GPU_LOG"
nvidia-smi \
  --id="$GPU_IDS" \
  --query-gpu=timestamp,index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader,nounits \
  --loop=2 >> "$GPU_LOG" 2>&1 &
GPU_MONITOR_PID=$!

echo "epoch_s,mem_total_mb,mem_used_mb,mem_available_mb" > "$RAM_LOG"
( while true; do
    free -m | awk -v t="$(date +%s)" '/^Mem:/ {print t","$2","$3","$7}' >> "$RAM_LOG"
    sleep 2
  done ) &
RAM_MONITOR_PID=$!

cleanup() {
  kill "$GPU_MONITOR_PID" "$RAM_MONITOR_PID" 2>/dev/null || true
  wait "$GPU_MONITOR_PID" "$RAM_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "GPUs: $GPU_IDS (n=$NUM_GPUS)  |  log: $TRAIN_LOG"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_offload.yaml \
  --num_processes "$NUM_GPUS" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port 29517 \
  scripts/train.py \
  task=robotwin_ahawam \
  model=ahawam \
  model.action_dit_pretrained_path="$ACTION_DIT" \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.train.pretrained_norm_stats="$ROBOTWIN_STATS" \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  data.val=null \
  output_dir="$WORK_DIR/model" \
  wandb.enabled=false \
  batch_size=8 \
  num_workers=0 \
  gradient_accumulation_steps=1 \
  max_steps=20 \
  log_every=1 \
  save_every=1000000 \
  save_at_end=false \
  eval_every=0 \
  freeze_video_dit=false \
  model.mot_checkpoint_mixed_attn=true \
  2>&1 | tee "$TRAIN_LOG"

exit "${PIPESTATUS[0]}"
