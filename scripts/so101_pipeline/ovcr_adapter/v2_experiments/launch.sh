#!/usr/bin/env bash
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u
export CUDA_HOME="$CONDA_PREFIX"
export CUDA_VISIBLE_DEVICES="${GPU_IDS:-0,1,2}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/runs/cache/torch_extensions"
export TRITON_CACHE_DIR="$REPO_ROOT/runs/cache/triton_ovcr_v2"
export WANDB_MODE="${WANDB_MODE:-disabled}"  # callers opt in with WANDB_MODE=online
mkdir -p "$HF_DATASETS_CACHE" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"
IFS=',' read -ra selected <<< "$CUDA_VISIBLE_DEVICES"
[[ ${#selected[@]} == 3 ]] || { echo 'Exactly three GPUs required'; exit 2; }
for id in "${selected[@]}"; do
  # NVML can transiently print "Failed to ..." instead of a number; retry briefly.
  for attempt in 1 2 3 4 5 6 7 8 9 10; do
    free_mib="$(nvidia-smi --id="$id" --query-gpu=memory.free --format=csv,noheader,nounits 2>&1 || true)"
    [[ "$free_mib" =~ ^[0-9]+$ ]] && break
    echo "nvidia-smi GPU $id attempt $attempt returned: $free_mib"; sleep 30
  done
  [[ "$free_mib" =~ ^[0-9]+$ ]] || { echo "GPU $id: no memory reading from nvidia-smi"; exit 3; }
  (( free_mib >= 36000 )) || { echo "GPU $id: only $free_mib MiB free"; exit 3; }
done
exec accelerate launch \
  --config_file scripts/so101_pipeline/ovcr_adapter/v2_experiments/accelerate_zero2_offload.yaml \
  --num_processes 3 --num_machines 1 --machine_rank 0 \
  --main_process_ip 127.0.0.1 --main_process_port "${MAIN_PORT:-29573}" \
  "${ENTRYPOINT:-scripts/so101_pipeline/ovcr_adapter/v2_experiments/run.py}" "$@"
