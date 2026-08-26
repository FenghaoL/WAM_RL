#!/usr/bin/env bash
set -eo pipefail
# Score every checkpoint in WEIGHTS_DIR on the held-out val set, one GPU each,
# all in parallel. Every checkpoint sees the identical fixed sample subset.

WEIGHTS_DIR="${WEIGHTS_DIR:?set WEIGHTS_DIR}"
SETTINGS="${SETTINGS:-scripts/so101_pipeline/full_finetune_zero2_offload/settings.yaml}"
OUT_DIR="${OUT_DIR:-runs/so101_ahawam_square_v1/val_eval}"
NUM_SAMPLES="${NUM_SAMPLES:-200}"
STEPS="${STEPS:-10}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
set -u
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$REPO_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
mkdir -p "$OUT_DIR"

mapfile -t CKPTS < <(ls -1 "$WEIGHTS_DIR"/*.pt | sort)
echo "Scoring ${#CKPTS[@]} checkpoints on $NUM_SAMPLES val samples each"

PIDS=()
for i in "${!CKPTS[@]}"; do
  ckpt="${CKPTS[$i]}"
  tag="$(basename "$ckpt" .pt)"
  gpu=$(( i % 4 ))
  echo "  GPU$gpu <- $tag"
  CUDA_VISIBLE_DEVICES=$gpu python scripts/so101_pipeline/val_eval/eval_checkpoint.py \
    --checkpoint "$ckpt" \
    --settings "$SETTINGS" \
    --output-json "$OUT_DIR/$tag.json" \
    --num-samples "$NUM_SAMPLES" \
    --num-inference-steps "$STEPS" \
    > "$OUT_DIR/$tag.log" 2>&1 &
  PIDS+=($!)
done

FAIL=0
for pid in "${PIDS[@]}"; do wait "$pid" || FAIL=1; done
echo "all done (fail=$FAIL)"
python scripts/so101_pipeline/val_eval/summarize.py --dir "$OUT_DIR"
exit "$FAIL"
