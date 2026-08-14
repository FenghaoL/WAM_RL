#!/usr/bin/env bash
# Create the conda environment for the MiniMax-H3 baseline.
#
# It is deliberately separate from `ahawam`: MiniMax-H3 needs diffusers `main`
# (the integration is newer than any release) and a transformers with
# Qwen3-VL support, while `ahawam` pins transformers==4.49.0. Upgrading in
# place would put the training environment at risk.
#
#   bash scripts/so101_pipeline/baseline_video_experiments/minimax_h3/setup_env.sh
set -euo pipefail

ENV_NAME="${ENV_NAME:-minimax_h3}"

source /home/fenghao/miniconda3/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.12 -y
fi
conda activate "$ENV_NAME"

pip install -U pip setuptools wheel

# A100 is sm_80: no FP8 and no NVFP4 tensor cores, so int8 weight-only is the
# quantisation that actually runs here.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "git+https://github.com/huggingface/diffusers.git"
pip install -U transformers accelerate torchao safetensors sentencepiece protobuf
pip install -U "huggingface_hub[hf_transfer]" imageio "imageio-ffmpeg" av pillow numpy ftfy

python - <<'PY'
import torch, diffusers, transformers, torchao
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("diffusers", diffusers.__version__)
print("transformers", transformers.__version__)
print("torchao", torchao.__version__)
from diffusers import ModularPipeline, MiniMaxH3Transformer3DModel  # noqa: F401
from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
print("MiniMax-H3 classes import OK")
PY
