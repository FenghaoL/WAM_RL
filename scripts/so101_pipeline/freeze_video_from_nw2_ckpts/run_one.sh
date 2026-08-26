#!/usr/bin/env bash
set -eo pipefail

GPU_IDS="${GPU_IDS:?Set GPU_IDS}"
NUM_GPUS="${NUM_GPUS:-1}"
SOURCE_STEP="${SOURCE_STEP:?Set SOURCE_STEP, for example 001988}"
NUM_EPOCHS="${NUM_EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
MAIN_PORT="${MAIN_PORT:-29611}"
MIN_FREE_MIB="${MIN_FREE_MIB:-36000}"
SMOKE_STEPS="${SMOKE_STEPS:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u

SETTINGS="scripts/so101_pipeline/full_finetune_zero2_offload/settings_nw2.yaml"
SOURCE_RUN="runs/so101_ahawam_nw_2/full_zero2_offload_ahawam_base/formal-nw2-ahawambase-zero2offload-bs4-acc1-5ep-3gpu-20260818-232749"
SOURCE_STATE="$SOURCE_RUN/model/checkpoints/step_${SOURCE_STEP}_trainer_state.json"
SOURCE_WEIGHTS="$SOURCE_RUN/model/checkpoints/weights/step_${SOURCE_STEP}.pt"
RUN_ROOT="$SOURCE_RUN/freeze_video_continuations/from_step_${SOURCE_STEP}"
RUN_NAME="${RUN_NAME:-freeze-video-${NUM_EPOCHS}ep-${NUM_GPUS}gpu-bs${BATCH_SIZE}-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="$RUN_ROOT/$RUN_NAME"

[[ -f "$SOURCE_STATE" ]] || { echo "Missing $SOURCE_STATE" >&2; exit 2; }
[[ -f "$SOURCE_WEIGHTS" ]] || { echo "Missing $SOURCE_WEIGHTS" >&2; exit 2; }
IFS=',' read -ra SELECTED_GPUS <<< "$GPU_IDS"
(( ${#SELECTED_GPUS[@]} == NUM_GPUS )) || {
    echo "GPU_IDS has ${#SELECTED_GPUS[@]} entries but NUM_GPUS=$NUM_GPUS" >&2
    exit 2
}
for gpu in "${SELECTED_GPUS[@]}"; do
    free_mib="$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)"
    (( free_mib >= MIN_FREE_MIB )) || {
        echo "GPU $gpu has ${free_mib} MiB free; need ${MIN_FREE_MIB} MiB" >&2
        exit 3
    }
done

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/runs/cache/torch_extensions"
export TRITON_CACHE_DIR="$REPO_ROOT/$RUN_DIR/triton_cache"
export WANDB_DIR="$REPO_ROOT/$RUN_DIR/wandb"
mkdir -p "$RUN_DIR" "$TRITON_CACHE_DIR" "$WANDB_DIR" "$TORCH_EXTENSIONS_DIR"

SOURCE_STATE="$SOURCE_STATE" SOURCE_WEIGHTS="$SOURCE_WEIGHTS" \
RUN_DIR="$RUN_DIR" GPU_IDS="$GPU_IDS" NUM_GPUS="$NUM_GPUS" \
NUM_EPOCHS="$NUM_EPOCHS" BATCH_SIZE="$BATCH_SIZE" \
python - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

state_path = Path(os.environ["SOURCE_STATE"])
manifest = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "source_trainer_state": str(state_path),
    "source_trainer_metadata": json.loads(state_path.read_text()),
    "source_weights": os.environ["SOURCE_WEIGHTS"],
    "optimizer_state_reused": False,
    "freeze_video_dit": True,
    "parameter_policy": {
        "train": [
            "action_expert",
            "mot.action_branch_embedding",
            "proprio_encoder",
            "action_obs_visual_proj",
            "chunk_obs_query_encoder"
        ],
        "freeze": ["video_expert", "mot.chunk_kv_cache_editor"]
    },
    "gpu_ids": os.environ["GPU_IDS"],
    "num_gpus": int(os.environ["NUM_GPUS"]),
    "num_epochs": int(os.environ["NUM_EPOCHS"]),
    "batch_size_per_gpu": int(os.environ["BATCH_SIZE"]),
}
Path(os.environ["RUN_DIR"], "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
PY

python scripts/so101_pipeline/freeze_video_from_nw2_ckpts/gpu_monitor.py \
    --output "$RUN_DIR/gpu.csv" --interval 2.0 &
GPU_MONITOR_PID=$!
cleanup() {
    kill "$GPU_MONITOR_PID" 2>/dev/null || true
    wait "$GPU_MONITOR_PID" 2>/dev/null || true
}
trap cleanup EXIT

EXTRA_ARGS=()
if [[ -n "$SMOKE_STEPS" ]]; then
    EXTRA_ARGS+=(--smoke-steps "$SMOKE_STEPS")
fi

echo "run=$RUN_NAME source_step=$SOURCE_STEP GPUs=$GPU_IDS epochs=$NUM_EPOCHS batch=$BATCH_SIZE"
set +e
accelerate launch \
    --config_file scripts/so101_pipeline/full_finetune_zero2_offload/accelerate.yaml \
    --num_processes "$NUM_GPUS" \
    --num_machines 1 --machine_rank 0 \
    --main_process_ip 127.0.0.1 --main_process_port "$MAIN_PORT" \
    scripts/so101_pipeline/freeze_video_from_nw2_ckpts/train_entry.py \
    --settings "$SETTINGS" \
    --num-gpus "$NUM_GPUS" \
    --output-dir "$RUN_DIR/model" \
    --wandb-name "$RUN_NAME" \
    --init-checkpoint "$SOURCE_WEIGHTS" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee "$RUN_DIR/train.log"
TRAIN_EXIT=${PIPESTATUS[0]}
set -e
cleanup
trap - EXIT
echo "exit=$TRAIN_EXIT run_dir=$RUN_DIR"
exit "$TRAIN_EXIT"
