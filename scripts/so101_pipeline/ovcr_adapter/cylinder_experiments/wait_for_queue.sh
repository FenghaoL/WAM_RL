#!/usr/bin/env bash
# Block until the experiment queue of RUN_ROOT has exited, then require that
# EXPERIMENT completed with exit code 0, that REQUIRED_FILE exists, and that every
# GPU in GPU_IDS has MIN_FREE_MIB free.  Any failure exits non-zero, so a queue
# chained after this script with && never starts on a bad or missing checkpoint.
#   wait_for_queue.sh RUN_ROOT EXPERIMENT REQUIRED_FILE
set -euo pipefail
run_root=$1; experiment=$2; required=$3
gpu_ids=${GPU_IDS:-0,1,2}; min_free=${MIN_FREE_MIB:-36000}
say() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
# Same NVIDIA userspace workaround launch.sh applies before calling nvidia-smi.
source /home/fenghao/use-nvidia-580159.sh >/dev/null || say "NVIDIA compat environment not applied"
# Free MiB of one GPU, or nothing: NVML can transiently print "Failed to ..."
# instead of a number, which must mean "not ready yet", never a crash.
gpu_free_mib() {
  local out
  out=$(nvidia-smi --id="$1" --query-gpu=memory.free --format=csv,noheader,nounits 2>&1 || true)
  if [[ "$out" =~ ^[0-9]+$ ]]; then echo "$out"; else say "nvidia-smi GPU $1 returned: ${out//$'\n'/ }" >&2; fi
}
if [[ -f "$run_root/suite.pid" ]]; then
  pid=$(cat "$run_root/suite.pid")
  say "waiting for queue pid $pid in $run_root to exit"
  while kill -0 "$pid" 2>/dev/null; do sleep 60; done
fi
python3 - "$run_root/suite_status.json" "$experiment" <<'PY'
import json, sys
records = [r for r in json.load(open(sys.argv[1]))["experiments"] if r["name"] == sys.argv[2]]
if not records or records[-1].get("state") != "completed" or records[-1].get("exit_code") != 0:
    sys.exit(f"{sys.argv[2]} did not complete successfully: {records[-1] if records else 'no record'}")
print(f"{sys.argv[2]} completed with exit code 0")
PY
[[ -s "$required" ]] || { say "required file missing: $required"; exit 5; }
IFS=',' read -ra ids <<< "$gpu_ids"
deadline=$((SECONDS + ${GPU_WAIT_SECONDS:-1800}))  # GPU_WAIT_SECONDS: how long to wait for free GPUs
for id in "${ids[@]}"; do
  until free=$(gpu_free_mib "$id"); [[ -n "$free" ]] && (( free >= min_free )); do
    (( SECONDS < deadline )) || { say "GPU $id not ready after ${GPU_WAIT_SECONDS:-1800}s (last reading: ${free:-none})"; exit 6; }
    sleep 20
  done
done
say "ready: $required"
