#!/usr/bin/env bash
# Queue the cylinder-v1 correction experiments on three GPUs inside one tmux session.
#
#   bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_suite.sh
#   NAMES="p1_full_balanced_video" bash .../launch_suite.sh      # a subset, same queue rules
#
# The queue runs one experiment at a time, stops at the first failure, refuses to
# overwrite an existing output directory, and keeps GPU 3 free for deployment.
# PRE_COMMAND (optional, no double quotes) runs first inside the session; the
# queue starts only if it exits 0 -- used to chain behind another queue.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"
RUN_ROOT="${RUN_ROOT:-runs/ovcr_adapter_cylinder_v1}"
SESSION="${SESSION:-ovcr_cylinder_v1}"
GPU_IDS="${GPU_IDS:-0,1,2}"
NAMES="${NAMES:-p1_v32_balanced p1_q32_balanced p1_t4_balanced p1_full_balanced_video b_full_pretrained_natural_4ep}"
[[ -d "$RUN_ROOT/configs" ]] || { echo "Run prepare.py first: no $RUN_ROOT/configs"; exit 2; }
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux session '$SESSION' already exists; attach with: tmux attach -t $SESSION"; exit 3
fi
if [[ -f "$RUN_ROOT/suite.pid" ]] && kill -0 "$(cat "$RUN_ROOT/suite.pid")" 2>/dev/null; then
  echo "A queue is already running (pid $(cat "$RUN_ROOT/suite.pid"))"; exit 4
fi
mkdir -p "$RUN_ROOT/logs"
COMMAND="cd '$REPO_ROOT' \
&& source /home/fenghao/miniconda3/etc/profile.d/conda.sh && conda activate ahawam \
&& export WANDB_MODE=online GPU_IDS='$GPU_IDS' PYTHONPATH='$REPO_ROOT/src:$REPO_ROOT' \
${PRE_COMMAND:+&& $PRE_COMMAND} \
&& python scripts/so101_pipeline/ovcr_adapter/v2_experiments/suite.py \
     --run-root '$RUN_ROOT' --names $NAMES"
tmux new-session -d -s "$SESSION" "bash -lc \"$COMMAND\" 2>&1 | tee -a '$RUN_ROOT/logs/suite_queue.log'"
echo "Started tmux session '$SESSION' on GPUs $GPU_IDS"
echo "Queue:   $NAMES"
echo "Status:  python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py $RUN_ROOT"
echo "Logs:    $RUN_ROOT/logs/<experiment>.log"
