#!/usr/bin/env bash
set -eo pipefail

# Native FastWAM checkpoint test. This returns 14-D RoboTwin actions and must
# not be connected to the 6-D SO101 client.
DEPLOY_CONFIG="scripts/so101_pipeline/fastwam_deploy/settings_native_robotwin.yaml"
CHECKPOINT="checkpoints/FastWAM-SharePoint/step_137500.pt"
INSTRUCTION="Pick up the white block."
GPU_ID=0
SERVER_HOST="127.0.0.1"
PORT=12010
MAX_RETURN_ACTIONS=8

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
set -u
source /home/fenghao/use-nvidia-580159.sh

python scripts/so101_pipeline/fastwam_deploy/check_checkpoint.py \
  "$CHECKPOINT" \
  --expected-action-dim 14 \
  --dataset-stats checkpoints/FastWAM-SharePoint/robotwin_uncond_3cam_384_dataset_stats.json

echo "Starting native 14-D FastWAM service on ${SERVER_HOST}:${PORT}, physical GPU ${GPU_ID}"
CUDA_VISIBLE_DEVICES="$GPU_ID" python deploy/server/wam_policy_server.py \
  --policy-path "$DEPLOY_CONFIG" \
  --policy-module deploy.server.ahawam_policy \
  --policy-class FastWAMPolicy \
  --instruction "$INSTRUCTION" \
  --action-dim 14 \
  --host "$SERVER_HOST" \
  --port "$PORT" \
  --max-return-actions "$MAX_RETURN_ACTIONS" \
  --no-threaded-connections \
  --policy-worker-process
