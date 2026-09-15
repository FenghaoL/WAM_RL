#!/usr/bin/env bash
set -eo pipefail
# ============================================================================
# Action-only AHA-WAM service.  Run with no arguments:
#
#   bash scripts/so101_pipeline/online_deploy/run_action_server.sh
#
# Pick the model by editing DEPLOY_CONFIG below. GPU, port, host and the task
# instruction live inside that YAML's `serving:` block, so switching models is
# a one-line change here and never leaves a stale instruction behind.
# ============================================================================

# ---- which model to serve --------------------------------------------------
# Available (scripts/so101_pipeline/online_deploy/deployments/):
#
#   cylinder_v1_sft              cylinder insertion, 4ep SFT       GPU0 :12000
#                                task: "Pick up the cylinder ..."
#   padapter_v2_sft              power adapter v2, 4ep SFT         GPU1 :12001
#   ovcr_p1_full_balanced_video  OVCR p1, full finetune, no adapter GPU2 :12002
#   ovcr_p1_v32_balanced         OVCR p1, adapter mode=value        GPU3 :12003
#   ovcr_p1_t4_balanced          OVCR p1, adapter mode=token_kv     GPU0 :12004
#   ovcr_p1_q32_balanced         OVCR p1, adapter mode=query        GPU1 :12005
#
# All ovcr_* and padapter_v2_sft serve the power-adapter task string;
# only cylinder_v1_sft serves the cylinder one.
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v1_sft.yaml"
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v1_full_pretrained_natural_4ep.yaml"
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v1_p1_v32_balanced.yaml"

#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v1_sft_nohist.yaml"
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v1_p1_v32_balanced_recover.yaml"

#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_v32_balanced.yaml"


#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_full_balanced_video.yaml"
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_full_balanced_video_vidext.yaml"
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_full_balanced_video_actext.yaml"

#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/padapter_v2_sft.yaml"
DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/padapter_v3_p1_full_balanced_video_vidext.yaml"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
set -u
source /home/fenghao/use-nvidia-580159.sh

# Read the serving block out of the chosen YAML.
eval "$(python - "$DEPLOY_CONFIG" <<'PY'
import shlex, sys
from omegaconf import OmegaConf
s = (OmegaConf.load(sys.argv[1]).get("serving") or {})
for key, default in (("gpu_id", 0), ("port", 12000), ("host", "127.0.0.1"),
                     ("action_dim", 6), ("max_return_actions", 8),
                     ("instruction", "")):
    value = s.get(key, default)
    print(f"CFG_{key.upper()}={shlex.quote(str(default if value is None else value))}")
PY
)"

if [[ -z "$CFG_INSTRUCTION" ]]; then
  echo "serving.instruction is empty in $DEPLOY_CONFIG." >&2
  echo "It must be the task string this checkpoint trained on: a prompt the run" >&2
  echo "never saw is out-of-distribution conditioning and fails silently." >&2
  exit 3
fi

python scripts/so101_pipeline/online_deploy/deployment_preflight.py "$DEPLOY_CONFIG"

echo "deployment : $DEPLOY_CONFIG"
echo "instruction: $CFG_INSTRUCTION"
echo "serving    : ${CFG_HOST}:${CFG_PORT} on physical GPU ${CFG_GPU_ID}"

CUDA_VISIBLE_DEVICES="$CFG_GPU_ID" python deploy/server/wam_policy_server.py \
  --policy-path "$DEPLOY_CONFIG" \
  --instruction "$CFG_INSTRUCTION" \
  --action-dim "$CFG_ACTION_DIM" \
  --host "$CFG_HOST" \
  --port "$CFG_PORT" \
  --max-return-actions "$CFG_MAX_RETURN_ACTIONS" \
  --no-threaded-connections \
  --policy-worker-process
