#!/usr/bin/env bash
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
set -u
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python deploy/client/wam_policy_async_dummy_client.py \
  --host "${SERVER_HOST:-127.0.0.1}" \
  --port "${PORT:-12000}" \
  --instruction "${INSTRUCTION:-Remove the power adapter from the holder and plug it into the power strip.}" \
  --action-dim 6 \
  --camera-payload so101-raw \
  --image-height 384 \
  --image-width 320 \
  --image-push-fps 2 \
  --action-request-rate 1 \
  --num-action-requests 3 \
  --prefill-wait 10 \
  --socket-timeout 300
