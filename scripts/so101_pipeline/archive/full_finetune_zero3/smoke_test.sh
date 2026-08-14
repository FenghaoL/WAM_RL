#!/usr/bin/env bash
set -eo pipefail

GPU_IDS="0,1,2,3"
NUM_GPUS=4
MIN_FREE_MIB=30000
SMOKE_STEPS="${SMOKE_STEPS:-1}"
SMOKE_GRAD_ACCUM="${SMOKE_GRAD_ACCUM:-1}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-1}"
SMOKE_FIRST_HISTORY_LEN="${SMOKE_FIRST_HISTORY_LEN:-}"
SAVE_WEIGHTS="${SAVE_WEIGHTS:-0}"
SETTINGS="scripts/so101_pipeline/full_finetune_zero3/settings.yaml"
CONFIG_FILE="scripts/so101_pipeline/full_finetune_zero3/accelerate_zero3_hybrid.yaml"
RUN_NAME="smoke-4gpu-zero3-hybrid-b${SMOKE_BATCH_SIZE}-full-$(date +%Y%m%d-%H%M%S)"
RUN_DIR="runs/so101_g05_square_v1/full_zero3/$RUN_NAME"
EXTRA_ARGS=()
if [[ "$SMOKE_STEPS" != "1" ]]; then
  RUN_NAME="steps${SMOKE_STEPS}-${RUN_NAME}"
  RUN_DIR="runs/so101_g05_square_v1/full_zero3/$RUN_NAME"
fi
if [[ "$SMOKE_GRAD_ACCUM" != "1" ]]; then
  RUN_NAME="accum${SMOKE_GRAD_ACCUM}-${RUN_NAME}"
  RUN_DIR="runs/so101_g05_square_v1/full_zero3/$RUN_NAME"
fi
if [[ "$SAVE_WEIGHTS" == "1" ]]; then
  RUN_NAME="checkpoint-${RUN_NAME}"
  RUN_DIR="runs/so101_g05_square_v1/full_zero3/$RUN_NAME"
  EXTRA_ARGS+=(--smoke-save)
fi
if [[ -n "$SMOKE_FIRST_HISTORY_LEN" ]]; then
  RUN_NAME="history${SMOKE_FIRST_HISTORY_LEN}-${RUN_NAME}"
  RUN_DIR="runs/so101_g05_square_v1/full_zero3/$RUN_NAME"
  EXTRA_ARGS+=(--smoke-first-history-len "$SMOKE_FIRST_HISTORY_LEN")
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRITON_CACHE_DIR="$REPO_ROOT/$RUN_DIR/triton_cache"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/runs/cache/torch_extensions"
mkdir -p "$RUN_DIR" "$TRITON_CACHE_DIR" "$TORCH_EXTENSIONS_DIR"

for free_mib in $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits); do
  if (( free_mib < MIN_FREE_MIB )); then
    echo "Refusing to start: one selected GPU has only ${free_mib} MiB free; need ${MIN_FREE_MIB} MiB."
    exit 3
  fi
done

python scripts/so101_pipeline/benchmark/gpu_monitor.py \
  --output "$RUN_DIR/gpu.csv" --interval 1.0 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true' EXIT

set +e
accelerate launch \
  --config_file "$CONFIG_FILE" \
  --num_processes "$NUM_GPUS" \
  scripts/so101_pipeline/full_finetune_zero3/train_entry.py \
  --settings "$SETTINGS" \
  --num-gpus "$NUM_GPUS" \
  --output-dir "$RUN_DIR/model" \
  --wandb-name "$RUN_NAME" \
  --smoke-steps "$SMOKE_STEPS" \
  --smoke-gradient-accumulation "$SMOKE_GRAD_ACCUM" \
  --micro-batch-size "$SMOKE_BATCH_SIZE" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$RUN_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true
trap - EXIT
python scripts/so101_pipeline/benchmark/summarize.py \
  --name "$RUN_NAME" \
  --log "$RUN_DIR/train.log" \
  --gpu-csv "$RUN_DIR/gpu.csv" \
  --exit-code "$TRAIN_EXIT" \
  --output "$RUN_DIR/summary.md"
printf '%s\n' "$RUN_DIR" > runs/so101_g05_square_v1/full_zero3/latest_smoke.txt
exit "$TRAIN_EXIT"
