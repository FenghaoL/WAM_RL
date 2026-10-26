#!/usr/bin/env bash
set -eo pipefail
# ============================================================================
# Action-only AHA-WAM service.  Run with no arguments:
#
#   bash scripts/so101_pipeline/online_deploy/run_action_server.sh
#   bash scripts/so101_pipeline/online_deploy/run_action_server.sh path/to/model.yaml
#
# Pick the model by editing DEPLOY_CONFIG below. GPU, port, host and the task
# instruction live inside that YAML's `serving:` block, so switching models is
# a one-line change here and never leaves a stale instruction behind.
# ============================================================================

# ---- which model to serve --------------------------------------------------
# Available (scripts/so101_pipeline/online_deploy/deployments/).
# Physical GPU is read from each YAML, not inferred from the model name.
# Every model has its own port; the last digit is the variant:
#   0 SFT   1 full (1:1)   2 vidext   3 segdemo (human spans as demos)   4 demo2to1
#   5 action-DiT-only (video DiT frozen)      121x = power adapter continued to 4073 steps
# Older configs live in deployments/archive/.

# cylinder: "Pick up the cylinder and insert it into the matching hole."
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v1_sft.yaml"                              # :12100
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_full_balanced_video.yaml"           # :12101
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_full_balanced_video_vidext.yaml"    # :12102
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_full_balanced_video_segdemo.yaml"   # :12103
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_full_balanced_video_demo2to1.yaml"  # :12104
DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_action_dit_only.yaml"        # :12105 action DiT only

# Cylinder offline RL comparisons (all use collector SFT 006572 as the base).
# These use OfflineRLPolicy: one 8-command chunk, chunk index 0, per-request planner.
# Existing robot client / model_to_arm coordinates are unchanged.
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_iql.yaml"             # GPU 0 :12130 IQL small actor, step 10000
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_td3_bc.yaml"          # GPU 1 :12131 TD3+BC small actor, step 10000
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_iql_action_dit.yaml"  # GPU 2 :12132 IQL weighted-FM Action DiT LoRA, step 2000

# power adapter: "Remove the power adapter from the holder and plug it into the power strip."
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/padapter_v2_sft.yaml"                              # :12110
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/padapter_v3_p1_full_balanced_video.yaml"           # :12111
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/padapter_v3_p1_full_balanced_video_vidext.yaml"    # :12112
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/padapter_v3_p1_full_balanced_video_segdemo.yaml"   # :12113
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/padapter_v3_p1_full_balanced_video_demo2to1.yaml"  # :12114
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/padapter_v3_p1_action_dit_only.yaml"        # :12115 action DiT only
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/padapter_v3_p1_full_balanced_video_cont4073.yaml"         # :12121
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/padapter_v3_p1_full_balanced_video_vidext_cont4073.yaml"  # :12122
#DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/deployments/padapter_v3_p1_full_balanced_video_segdemo_cont4073.yaml" # :12123

# Optional argument or environment override allows three independent terminals
# without repeatedly editing this file. No argument preserves the selection above.
DEPLOY_CONFIG="${1:-${AHAWAM_DEPLOY_CONFIG:-$DEPLOY_CONFIG}}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
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
                     ("instruction", ""), ("policy_module", "deploy.server.ahawam_policy"),
                     ("policy_class", "AHAWAMPolicy")):
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
  --policy-module "$CFG_POLICY_MODULE" \
  --policy-class "$CFG_POLICY_CLASS" \
  --instruction "$CFG_INSTRUCTION" \
  --action-dim "$CFG_ACTION_DIM" \
  --host "$CFG_HOST" \
  --port "$CFG_PORT" \
  --max-return-actions "$CFG_MAX_RETURN_ACTIONS" \
  --no-threaded-connections \
  --policy-worker-process
