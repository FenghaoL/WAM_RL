#!/usr/bin/env bash
# Isolated entry reusing the established data loader, queue, and three-GPU trainer.
# bash .../run.sh data|smoke|train RUN_ROOT EXPERIMENT
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
[[ $# == 3 ]] || { echo "Usage: $0 data|smoke|train RUN_ROOT EXPERIMENT"; exit 2; }
mode="$1"
run_root="$2"
experiment="$3"
[[ "$experiment" != */* ]] || { echo "Experiment must be a basename"; exit 2; }
config="$run_root/configs/$experiment.yaml"
[[ -f "$config" ]] || { echo "Missing generated configuration: $config"; exit 2; }
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export GPU_IDS="${GPU_IDS:-0,1,2}"
export MAIN_PORT="${MAIN_PORT:-29593}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
paper_python="${PAPER_PYTHON:-/home/fenghao/miniconda3/envs/ahawam/bin/python}"
case "$mode" in
  data) exec "$paper_python" scripts/so101_pipeline/ovcr_adapter/v2_experiments/run.py --config "$config" --data-only ;;
  smoke) exec "$paper_python" scripts/so101_pipeline/ovcr_adapter/v2_experiments/suite.py --run-root "$run_root" --names "$experiment" --smoke ;;
  train) exec "$paper_python" scripts/so101_pipeline/ovcr_adapter/v2_experiments/suite.py --run-root "$run_root" --names "$experiment" ;;
  *) echo "Unknown mode: $mode"; exit 2 ;;
esac
