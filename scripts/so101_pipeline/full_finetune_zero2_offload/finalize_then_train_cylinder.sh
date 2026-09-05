#!/usr/bin/env bash
set -eo pipefail
# ============================================================================
# Cylinder-insertion end-to-end: wait for upload -> encode videos -> text
# embedding -> dataset stats -> preflight -> train.
#
# The capture ships per-frame PNGs, so nothing downstream can read the data
# until finalize_on_server.sh has encoded and verified all 194 AV1 videos.
# Every stage is gated: a failure anywhere stops before training starts.
# ============================================================================

REPO_ROOT="/home/fenghao/workspace/AHA-WAM"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u

DATA_DIR="data/so101_ahawam_cylinder_insertion_v1/01_pick_up_and_insert_cylinder"
TOOLS="$DATA_DIR/_server_tools"
SETTINGS="scripts/so101_pipeline/full_finetune_zero2_offload/settings_cylinder_v1.yaml"
CACHE="data/so101_ahawam_square_v1_text_embeds_cache"
STATS_DIR="runs/so101_ahawam_cylinder_v1/assets"
EXPECTED_PNG="${EXPECTED_PNG:-40738}"
EXPECTED_MP4=194
GPU_IDS="${GPU_IDS:-0,1,2}"
NUM_GPUS="${NUM_GPUS:-3}"

export PYTHONPATH="$REPO_ROOT"
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/runs/cache/torch_extensions"
export OMP_NUM_THREADS=8

say() { echo "[$(date +%H:%M:%S)] $*"; }

# ---- Stage 0: wait until the upload is actually complete -----------------
# Counting PNGs is not a reliable gate: this upload pre-creates the files and
# then fills them in, so the count sits still for a long time while bytes keep
# arriving. Use the vendor precheck as the gate instead -- it verifies every
# episode has a contiguous frame range for BOTH cameras -- and only treat the
# transfer as dead when neither the byte total nor the file count moves.
say "STAGE 0  waiting for a complete upload"
STABLE=0
LAST_BYTES=-1
LAST_FILES=-1
while true; do
  BYTES=$(du -sb "$DATA_DIR" 2>/dev/null | cut -f1)
  FILES=$(find "$DATA_DIR" -name '*.png' 2>/dev/null | wc -l)
  if bash "$TOOLS/check_on_server.sh" >/tmp/cyl_precheck.json 2>&1; then
    say "  precheck READY (png=${FILES} bytes=${BYTES})"
    break
  fi
  if [[ "$BYTES" == "$LAST_BYTES" && "$FILES" == "$LAST_FILES" ]]; then
    STABLE=$((STABLE + 1))
  else
    STABLE=0
  fi
  if (( STABLE >= 30 )); then
    say "  no new bytes or files for 30 min and precheck still failing; aborting."
    tail -20 /tmp/cyl_precheck.json || true
    exit 10
  fi
  say "  png=${FILES} bytes=$(( BYTES / 1048576 ))MiB quiet=${STABLE}m"
  LAST_BYTES=$BYTES
  LAST_FILES=$FILES
  sleep 60
done

# ---- Stage 1: read-only precheck -----------------------------------------
say "STAGE 1  precheck"
chmod +x "$TOOLS"/*.sh 2>/dev/null || true
bash "$TOOLS/check_on_server.sh"

# ---- Stage 2: encode + verify AV1 videos ---------------------------------
say "STAGE 2  encoding videos (194 jobs)"
bash "$TOOLS/finalize_on_server.sh"
MP4=$(find "$DATA_DIR" -name '*.mp4' 2>/dev/null | wc -l)
say "  mp4=${MP4}/${EXPECTED_MP4}"
if (( MP4 < EXPECTED_MP4 )); then
  say "  encoding incomplete; aborting before training."
  exit 11
fi

# ---- Stage 3: text embedding for the new task string ---------------------
say "STAGE 3  text embedding"
torchrun --standalone --nproc_per_node=1 scripts/precompute_text_embeds.py \
  task=robotwin_ahawam_offset \
  "data.train.dataset_dirs=[$DATA_DIR/20260904_014410,$DATA_DIR/20260904_024009_val]" \
  data.val=null \
  data.train.text_embedding_cache_dir="$CACHE" \
  +overwrite=false

# ---- Stage 4: normalisation statistics ------------------------------------
say "STAGE 4  dataset stats"
python scripts/so101_pipeline/full_finetune_zero2_offload/preflight.py \
  --settings "$SETTINGS" --metadata-only
python scripts/so101_pipeline/full_finetune_zero2_offload/make_stats.py \
  --settings "$SETTINGS" --output-dir "$STATS_DIR"

# ---- Stage 5: full preflight ---------------------------------------------
say "STAGE 5  full preflight"
python scripts/so101_pipeline/full_finetune_zero2_offload/preflight.py --settings "$SETTINGS"

# ---- Stage 6: train -------------------------------------------------------
say "STAGE 6  training on GPU ${GPU_IDS}"
export SETTINGS RUN_ROOT="runs/so101_ahawam_cylinder_v1/full_zero2_offload_ahawam_base"
export RUN_NAME="formal-cylinderv1-sft-ahawambase-bs4-acc1-4ep-${NUM_GPUS}gpu-$(date +%Y%m%d-%H%M%S)"
export GPU_IDS NUM_GPUS MAIN_PORT="${MAIN_PORT:-29681}"
exec bash scripts/so101_pipeline/full_finetune_zero2_offload/run_train.sh
