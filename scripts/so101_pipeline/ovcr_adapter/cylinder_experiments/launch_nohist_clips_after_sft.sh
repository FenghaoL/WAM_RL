#!/usr/bin/env bash
# Queue the four no-history correction runs (HIL HUMAN + 2026-09-11 correction
# clips) behind the running no-history cylinder SFT, on the same three GPUs:
#   p1_full_balanced_video_nohist_clips -> p1_v32_balanced_nohist_clips
#   -> p1_q32_balanced_nohist_clips -> p1_t4_balanced_nohist_clips
# The tmux session first waits for the SFT queue to exit and checks that the SFT
# completed and wrote step_006572.pt; only then does the new queue start.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
SFT_ROOT="runs/ovcr_adapter_cylinder_v1"
INIT="$SFT_ROOT/sft_nohist_pretrained_4ep/model/checkpoints/weights/step_006572.pt"
export RUN_ROOT="runs/ovcr_adapter_cylinder_v1_nohist_clips"
export NAMES="p1_full_balanced_video_nohist_clips p1_v32_balanced_nohist_clips p1_q32_balanced_nohist_clips p1_t4_balanced_nohist_clips"
for name in $NAMES; do
  [[ -f "$RUN_ROOT/configs/$name.yaml" ]] || {
    echo "Missing $RUN_ROOT/configs/$name.yaml; run cylinder_experiments/prepare_nohist_clips.py first"; exit 2; }
done
export SESSION="${SESSION:-ovcr_cylinder_nohist_clips}" GPU_IDS="${GPU_IDS:-0,1,2}"
export PRE_COMMAND="bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/wait_for_queue.sh $SFT_ROOT sft_nohist_pretrained_4ep $INIT"
exec bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_suite.sh
