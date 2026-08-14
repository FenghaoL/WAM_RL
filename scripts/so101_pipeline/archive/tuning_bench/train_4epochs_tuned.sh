#!/usr/bin/env bash
set -eo pipefail
# ============================================================================
# SO101 full fine-tune, 4 epochs, tuned config for this box (4 x A100-40GB).
# Derived from the 16-run sweep in scripts/so101_pipeline/tuning_bench/.
# See docs/AHAWAM_ZERO2_TUNING_ZH.md for why each value is what it is.
#
#   ACCUM=1  effective batch 16   ~8.0 h   lr 1e-5    (matches the ZeRO-3 baseline)
#   ACCUM=2  effective batch 32   ~6.1 h   lr 1.4e-5  (default)
#   ACCUM=4  effective batch 64   ~4.3 h   lr 2e-5    (fastest sensible)
# ============================================================================

MICRO_BS="${MICRO_BS:-4}"          # 31,958 MiB steady, 8.3 GB headroom
ACCUM="${ACCUM:-2}"
LR="${LR:-1.4e-5}"
GPU_IDS="${GPU_IDS:-0,1,2,3}"
NUM_GPUS="${NUM_GPUS:-4}"
MIN_FREE_MIB="${MIN_FREE_MIB:-36000}"   # DeepSpeed init peaks near 31 GB

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u

RUN_NAME="tuned-4gpu-zero2-offload-bs${MICRO_BS}-acc${ACCUM}-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/so101_g05_square_v1/tuned/$RUN_NAME"
mkdir -p "$RUN_DIR"

# --- environment -----------------------------------------------------------
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# The single biggest win in the whole sweep: torchrun defaults this to 1, which
# leaves the CPU-side Adam (1.84B params per rank) single-threaded -> 42% slower.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TRITON_CACHE_DIR="$REPO_ROOT/$RUN_DIR/triton_cache"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/runs/cache/torch_extensions"
export WANDB_DIR="$REPO_ROOT/$RUN_DIR/wandb"
mkdir -p "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$WANDB_DIR"

for free_mib in $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); do
  if (( free_mib < MIN_FREE_MIB )); then
    echo "Refusing to start: need ${MIN_FREE_MIB} MiB free per GPU, found ${free_mib} MiB." >&2
    exit 3
  fi
done

# --- DeepSpeed config ------------------------------------------------------
DS_JSON="$RUN_DIR/ds_zero2_offload.json"
cat > "$DS_JSON" <<JSON
{
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": "auto",
  "gradient_accumulation_steps": "auto",
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": { "device": "cpu", "pin_memory": true },
    "offload_param": { "device": "none" },
    "round_robin_gradients": true,
    "overlap_comm": false,
    "contiguous_gradients": true,
    "reduce_bucket_size": 2e8,
    "allgather_bucket_size": 2e8
  },
  "steps_per_print": 2000000
}
JSON

ACCEL_YAML="$RUN_DIR/accelerate.yaml"
cat > "$ACCEL_YAML" <<YAML
compute_environment: LOCAL_MACHINE
debug: false
distributed_type: DEEPSPEED
deepspeed_config:
  deepspeed_config_file: $REPO_ROOT/$DS_JSON
  deepspeed_multinode_launcher: nossh
  zero3_init_flag: false
mixed_precision: null
machine_rank: 0
main_training_function: main
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
use_cpu: false
YAML

# --- training settings -----------------------------------------------------
SETTINGS="$RUN_DIR/settings.yaml"
sed -e "s/^batch_size: .*/batch_size: ${MICRO_BS}/" \
    -e "s/^gradient_accumulation_steps: .*/gradient_accumulation_steps: ${ACCUM}/" \
    -e "s/^learning_rate: .*/learning_rate: ${LR}/" \
    -e "s/^num_workers: .*/num_workers: 4/" \
    scripts/so101_pipeline/full_finetune_zero3/settings.yaml > "$SETTINGS"

echo "=========================================================="
echo " run        : $RUN_NAME"
echo " micro_bs   : $MICRO_BS   accum: $ACCUM   effective: $((MICRO_BS*ACCUM*NUM_GPUS))"
echo " lr         : $LR"
echo " OMP_THREADS: $OMP_NUM_THREADS"
echo " log        : $RUN_DIR/train.log"
echo "=========================================================="

python scripts/so101_pipeline/benchmark/gpu_monitor.py \
  --output "$RUN_DIR/gpu.csv" --interval 2.0 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

set +e
accelerate launch \
  --config_file "$ACCEL_YAML" \
  --num_processes "$NUM_GPUS" \
  --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port 29555 \
  scripts/so101_pipeline/partial_finetune_zero2/train_entry.py \
  --settings "$SETTINGS" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$RUN_DIR/model" \
  --wandb-name "$RUN_NAME" \
  2>&1 | tee "$RUN_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

kill "$MONITOR_PID" 2>/dev/null || true
trap - EXIT
echo "exit=$TRAIN_EXIT  log=$RUN_DIR/train.log"
exit "$TRAIN_EXIT"
