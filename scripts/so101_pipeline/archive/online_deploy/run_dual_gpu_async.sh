#!/usr/bin/env bash
set -eo pipefail

# Edit these defaults before starting the asynchronous server.
DEPLOY_CONFIG="scripts/so101_pipeline/settings/deploy_so101.yaml"
INSTRUCTION="pick up the white block"
VIDEO_GPU=0
ACTION_GPU=1
SERVER_HOST="127.0.0.1"
PORT=10000
MAX_RETURN_ACTIONS=16
WORKER_WARMUP_ROUNDS=2

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
set -u
source /home/fenghao/use-nvidia-580159.sh

python scripts/so101_pipeline/deployment_preflight.py "$DEPLOY_CONFIG"

python deploy/server/wam_policy_server.py \
  --dual-gpu-async \
  --video-gpu "$VIDEO_GPU" \
  --action-gpu "$ACTION_GPU" \
  --worker-warmup-rounds "$WORKER_WARMUP_ROUNDS" \
  --policy-path "$DEPLOY_CONFIG" \
  --instruction "$INSTRUCTION" \
  --action-dim 6 \
  --host "$SERVER_HOST" \
  --port "$PORT" \
  --max-return-actions "$MAX_RETURN_ACTIONS"
