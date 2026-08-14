#!/usr/bin/env bash
# Score the AHA-WAM video branch on the shared SO101 windows for several weight
# sets. Edit the variables below, then run:
#
#   bash scripts/so101_pipeline/baseline_video_experiments/wan22_base/run_video_branch_preview.sh
#
# CONDITIONS entries are "<run-name>|<weights>", where <weights> is either the
# literal `wan22-base` (raw Wan2.2-TI2V-5B initialisation, no AHA-WAM training)
# or a path to an AHA-WAM checkpoint.
set -eo pipefail

GPU=0
NUM_SAMPLES=16
STEPS=20
SEED=42
HISTORY_MODES="online,training,none"

CONFIG="runs/so101_g05_square_v1/full_zero3/formal-4gpu-zero3-hybrid-full-4ep-20260809-104429/model/config.yaml"
SAMPLES_DIR="runs/so101_g05_square_v1/baseline_experiments/samples"
OUTPUT_ROOT="runs/so101_g05_square_v1/baseline_experiments"

# Also run raw Wan2.2 in its own inference configuration (bidirectional mask,
# contiguous RoPE, no history). Without it the `wan22_base` number mostly
# measures how far AHA-WAM's mask/RoPE setup is from Wan2.2's, not the weights.
RUN_WAN22_NATIVE=1

CONDITIONS=(
  "wan22_base|wan22-base"
  "ahawam_pretrained|checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt"
  "so101_finetuned|runs/so101_g05_square_v1/full_zero3/formal-4gpu-zero3-hybrid-full-4ep-20260809-104429/model/checkpoints/weights/step_004896.pt"
)

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
cd "$REPO_ROOT"

source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh

export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export DIFFSYNTH_MODEL_BASE_PATH="$REPO_ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

if [ ! -f "$SAMPLES_DIR/manifest.json" ]; then
  echo "[stage1] exporting shared evaluation windows to $SAMPLES_DIR"
  python scripts/so101_pipeline/baseline_video_experiments/export_reference_samples.py \
    --config "$CONFIG" \
    --output "$SAMPLES_DIR" \
    --num-samples "$NUM_SAMPLES"
fi

for entry in "${CONDITIONS[@]}"; do
  name="${entry%%|*}"
  weights="${entry#*|}"
  echo "=== [$name] weights=$weights ==="
  python scripts/so101_pipeline/baseline_video_experiments/wan22_base/run_video_branch_preview.py \
    --config "$CONFIG" \
    --samples-dir "$SAMPLES_DIR" \
    --output "$OUTPUT_ROOT/$name" \
    --weights "$weights" \
    --run-name "$name" \
    --device cuda:0 \
    --steps "$STEPS" \
    --seed "$SEED" \
    --history-modes "$HISTORY_MODES"
done

if [ "$RUN_WAN22_NATIVE" = "1" ]; then
  echo "=== [wan22_native] raw Wan2.2 in Wan2.2's own inference configuration ==="
  python scripts/so101_pipeline/baseline_video_experiments/wan22_base/run_video_branch_preview.py \
    --config "$CONFIG" \
    --samples-dir "$SAMPLES_DIR" \
    --output "$OUTPUT_ROOT/wan22_native" \
    --weights wan22-base \
    --run-name wan22_native \
    --device cuda:0 \
    --steps "$STEPS" \
    --seed "$SEED" \
    --history-modes wan_native
fi

echo "=== comparison table ==="
python scripts/so101_pipeline/baseline_video_experiments/compare_runs.py \
  --runs "$OUTPUT_ROOT"/*/summary.json \
  --output "$OUTPUT_ROOT/comparison"
