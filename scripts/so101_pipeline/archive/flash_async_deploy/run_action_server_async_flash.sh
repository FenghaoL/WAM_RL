#!/usr/bin/env bash
# Some conda CUDA activation hooks probe optional shell variables. Enable
# nounset only after the environment is active, then keep strict mode for the
# deployment itself.
set -eo pipefail

INSTRUCTION="${INSTRUCTION:-Remove the power adapter from the holder and plug it into the power strip.}"
GPU_ID="${GPU_ID:-3}"
SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
PORT="${PORT:-12000}"
MAX_RETURN_ACTIONS="${MAX_RETURN_ACTIONS:-8}"
MIN_FREE_MIB="${MIN_FREE_MIB:-24000}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
set -u
source /home/fenghao/use-nvidia-580159.sh

free_mib="$(nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits)"
if (( free_mib < MIN_FREE_MIB )); then
  echo "GPU $GPU_ID has only ${free_mib} MiB free; need ${MIN_FREE_MIB} MiB." >&2
  exit 3
fi

DEPLOY_CONFIG="$(python scripts/so101_pipeline/flash_async_deploy/resolve_flash_checkpoint.py)"
python scripts/so101_pipeline/online_deploy/deployment_preflight.py "$DEPLOY_CONFIG"

echo "Starting 2-step Flash + single-GPU async AHA-WAM service"
echo "config=$DEPLOY_CONFIG host=$SERVER_HOST port=$PORT physical_gpu=$GPU_ID"
CUDA_VISIBLE_DEVICES="$GPU_ID" python deploy/server/wam_policy_server.py \
  --policy-path "$DEPLOY_CONFIG" \
  --instruction "$INSTRUCTION" \
  --action-dim 6 \
  --host "$SERVER_HOST" \
  --port "$PORT" \
  --max-return-actions "$MAX_RETURN_ACTIONS" \
  --threaded-connections \
  --no-policy-worker-process \
  --async-mode \
  --async-timing-config scripts/so101_pipeline/flash_async_deploy/async_timing.yaml
