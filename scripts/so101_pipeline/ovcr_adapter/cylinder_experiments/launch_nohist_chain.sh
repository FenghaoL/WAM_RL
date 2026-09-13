#!/usr/bin/env bash
# Run the two no-history cylinder experiments back to back on three GPUs:
#   1. sft_nohist_pretrained_4ep       AHA pretrained -> cylinder task SFT, num_history_frames=0
#   2. p1_full_balanced_video_nohist   1:1 demo/HUMAN full fine-tune on top of (1), no history
# (2) starts only after (1) exits 0: suite.py stops the queue at the first failure,
# and (2) loads (1)'s final step_006572.pt.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
RUN_ROOT="runs/ovcr_adapter_cylinder_v1"
for name in sft_nohist_pretrained_4ep p1_full_balanced_video_nohist; do
  [[ -f "$RUN_ROOT/configs/$name.yaml" ]] || {
    echo "Missing $RUN_ROOT/configs/$name.yaml; run cylinder_experiments/prepare_nohist.py first"; exit 2; }
done
if [[ -f "$RUN_ROOT/suite.pid" ]] && kill -0 "$(cat "$RUN_ROOT/suite.pid")" 2>/dev/null; then
  echo "A queue is already running in $RUN_ROOT (pid $(cat "$RUN_ROOT/suite.pid"))"; exit 4
fi
# suite.py rewrites suite_status.json; keep the record of the finished history-6 queue.
if [[ -f "$RUN_ROOT/suite_status.json" ]]; then
  cp "$RUN_ROOT/suite_status.json" "$RUN_ROOT/suite_status_before_nohist_$(date +%Y%m%d-%H%M%S).json"
fi
export RUN_ROOT SESSION="${SESSION:-ovcr_cylinder_nohist}" GPU_IDS="${GPU_IDS:-0,1,2}"
export NAMES="sft_nohist_pretrained_4ep p1_full_balanced_video_nohist"
exec bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_suite.sh
