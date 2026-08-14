#!/usr/bin/env bash
# =============================================================================
# 阶段 1 续训：自动查找最新 run 下的最新 checkpoint 恢复训练（仅权重恢复）。
# 恢复内容：模型权重 + 对齐 step/epoch/lr（从文件名 step_XXXXXX 解析）。
# 优化器动量不恢复（从头累积），前几十步 loss 可能小幅波动，很快恢复。
#
# 关键：resume 与 init_checkpoint 互斥，命令行必须同时把 init_checkpoint 置 null。
# =============================================================================

set -euo pipefail

PROJECT_DIR="/data/wzy/AHA-WAM"
cd "${PROJECT_DIR}"

export PATH="${PROJECT_DIR}/env/bin:${PATH}"
export DIFFSYNTH_MODEL_BASE_PATH="${PROJECT_DIR}/checkpoints/wan_models"
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1
unset PYTORCH_CUDA_ALLOC_CONF

# ---- 自动查找最新 checkpoint ----
RUNS_DIR="${PROJECT_DIR}/runs"

# 1. 找最新的 loader_ahawam-* 目录（按名字排序取最后一个，日期时间格式天然支持）
LATEST_RUN=$(find "${RUNS_DIR}" -maxdepth 1 -type d -name "loader_ahawam-*" \
  | sort | tail -1)

if [ -z "${LATEST_RUN}" ]; then
  echo "ERROR: 未找到任何 loader_ahawam-* run 目录 in ${RUNS_DIR}"
  exit 1
fi

# 2. 在该 run 下找 step 最大的 .pt checkpoint
LATEST_CKPT=$(find "${LATEST_RUN}/checkpoints/weights" -maxdepth 1 -type f -name "step_*.pt" \
  | sort -V | tail -1)

if [ -z "${LATEST_CKPT}" ]; then
  echo "ERROR: 未找到任何 step_*.pt in ${LATEST_RUN}/checkpoints/weights/"
  exit 1
fi

RESUME_DIR="${LATEST_CKPT}"

NPROC=4
PORT=29511

echo "============================================================"
echo " AHA-WAM 阶段1 续训"
echo " run      : ${LATEST_RUN}"
echo " resume   : ${RESUME_DIR}"
echo " 显卡数   : ${NPROC}"
echo " 日志     : ${PROJECT_DIR}/train.log"
echo " 配置     : ZeRO-2 + 优化器offload, bs=1, 梯度检查点, 256x320"
echo "============================================================"

accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_offload.yaml \
  --num_processes "${NPROC}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${PORT}" \
  scripts/train.py \
  model=ahawam task=loader_ahawam \
  resume="${RESUME_DIR}" \
  init_checkpoint=null \
  batch_size=1 \
  num_workers=0 \
  model.mot_checkpoint_mixed_attn=true \
  save_every=600 \
  eval_every=601 \
  log_every=10 \
  2>&1 | tee "${PROJECT_DIR}/train.log"

exit "${PIPESTATUS[0]}"
