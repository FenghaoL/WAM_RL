#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${HIL_SAMPLING_STRATEGY:-}" || -z "${HIL_RUN_VARIANT:-}" ]]; then
  echo "This helper is called by run_natural_5ep.sh or run_balanced_2plus2_5ep.sh." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

GPU_IDS="${GPU_IDS:-0,1,2,3,4,5,6,7}"
NUM_GPUS="${NUM_GPUS:-8}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MAIN_PORT="${MAIN_PORT:-29571}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SETTINGS="scripts/so101_pipeline/hil_human_only_training/settings.yaml"
GPU_MEMORY_MODE="${GPU_MEMORY_MODE:-auto}"

IFS=',' read -r -a SELECTED_GPUS <<< "$GPU_IDS"
if (( ${#SELECTED_GPUS[@]} != NUM_GPUS )); then
  echo "GPU_IDS contains ${#SELECTED_GPUS[@]} GPUs, but NUM_GPUS=$NUM_GPUS" >&2
  exit 3
fi
command -v "$PYTHON_BIN" >/dev/null
command -v accelerate >/dev/null
command -v nvidia-smi >/dev/null
MIN_TOTAL_MIB=999999
declare -A GPU_FREE_MIB
for gpu_id in "${SELECTED_GPUS[@]}"; do
  total_mib="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.total --format=csv,noheader,nounits)"
  free_mib="$(nvidia-smi --id="$gpu_id" --query-gpu=memory.free --format=csv,noheader,nounits)"
  GPU_FREE_MIB["$gpu_id"]="$free_mib"
  if (( total_mib < MIN_TOTAL_MIB )); then
    MIN_TOTAL_MIB="$total_mib"
  fi
done

if [[ "$GPU_MEMORY_MODE" == "auto" ]]; then
  if (( MIN_TOTAL_MIB >= 70000 )); then
    GPU_MEMORY_MODE="no_offload"
  else
    GPU_MEMORY_MODE="offload"
  fi
fi
case "$GPU_MEMORY_MODE" in
  no_offload)
    ACCELERATE_CONFIG="scripts/so101_pipeline/hil_human_only_training/accelerate.yaml"
    REQUIRED_FREE_MIB="${MIN_FREE_MIB:-60000}"
    ;;
  offload)
    ACCELERATE_CONFIG="scripts/so101_pipeline/full_finetune_zero2_offload/accelerate.yaml"
    REQUIRED_FREE_MIB="${MIN_FREE_MIB:-36000}"
    ;;
  *)
    echo "GPU_MEMORY_MODE must be auto, no_offload, or offload; got $GPU_MEMORY_MODE" >&2
    exit 4
    ;;
esac
for gpu_id in "${SELECTED_GPUS[@]}"; do
  free_mib="${GPU_FREE_MIB[$gpu_id]}"
  if (( free_mib < REQUIRED_FREE_MIB )); then
    echo "GPU $gpu_id has ${free_mib} MiB free; need $REQUIRED_FREE_MIB MiB for $GPU_MEMORY_MODE." >&2
    exit 5
  fi
done

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$REPO_ROOT/runs/cache/huggingface/datasets_v2}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$REPO_ROOT/runs/cache/torch_extensions}"

"$PYTHON_BIN" scripts/so101_pipeline/full_finetune_zero2_offload/preflight.py \
  --settings "$SETTINGS"

RUN_NAME="${RUN_NAME:-formal-human-only-${HIL_RUN_VARIANT}-bs4-5ep-8xa800-$(date +%Y%m%d-%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-runs/so101_ahawam_plug_hil/human_only_zero2}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR" "$TORCH_EXTENSIONS_DIR"
export TRITON_CACHE_DIR="$REPO_ROOT/$RUN_DIR/triton_cache"
export WANDB_DIR="$REPO_ROOT/$RUN_DIR/wandb"
mkdir -p "$TRITON_CACHE_DIR" "$WANDB_DIR"

echo "run=$RUN_NAME sampling=$HIL_SAMPLING_STRATEGY GPUs=$GPU_IDS memory_mode=$GPU_MEMORY_MODE"
set +e
accelerate launch \
  --config_file "$ACCELERATE_CONFIG" \
  --num_processes "$NUM_GPUS" \
  --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "$MAIN_PORT" \
  scripts/so101_pipeline/full_finetune_zero2_offload/train_entry.py \
  --settings "$SETTINGS" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$RUN_DIR/model" \
  --wandb-name "$RUN_NAME" \
  2>&1 | tee "$RUN_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
echo "exit=$TRAIN_EXIT run_dir=$RUN_DIR"
exit "$TRAIN_EXIT"
