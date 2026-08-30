#!/usr/bin/env bash
set -eo pipefail

# Action-only online service. Edit these simple defaults when needed.
DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/settings.yaml"
INSTRUCTION="Pick up the white block."
GPU_ID=0
# Keep pickle TCP private. The robot computer should connect through an SSH
# tunnel; see online_deploy/README.md.
SERVER_HOST="127.0.0.1"
PORT=12001
MAX_RETURN_ACTIONS=8

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
set -u
source /home/fenghao/use-nvidia-580159.sh

python scripts/so101_pipeline/online_deploy/deployment_preflight.py "$DEPLOY_CONFIG"

echo "Starting action-only AHA-WAM service on ${SERVER_HOST}:${PORT}, physical GPU ${GPU_ID}"
CUDA_VISIBLE_DEVICES="$GPU_ID" python deploy/server/wam_policy_server.py \
  --policy-path "$DEPLOY_CONFIG" \
  --instruction "$INSTRUCTION" \
  --action-dim 6 \
  --host "$SERVER_HOST" \
  --port "$PORT" \
  --max-return-actions "$MAX_RETURN_ACTIONS" \
  --no-threaded-connections \
  --policy-worker-process
