#!/usr/bin/env bash
# =============================================================================
# 阶段 1：在 loader（装载机）数据上全量微调 AHA-WAM，产出「教师」权重。
# 目标硬件：4 × A30 24GB（本机即训练机）。
#
# 已验证可行的配置（smoke8 跑通，loss 正常下降）：
#   - ZeRO-2 + 优化器 offload 到 CPU（A30 24G 放不下 5B 全参，no-offload 会 OOM）
#   - batch_size=1 + 梯度检查点（mot_checkpoint_mixed_attn=true）
#   - num_workers=0（A30 宿主 cgroup 内存上限 200GB，worker 多了会被 SIGKILL）
#   - 不设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments（GPU 驱动 470 会 CUDA error）
#   - 视频分辨率 256×320（latent 16×20，比例 1.25 匹配原图 768×960）
#
# 初始化权重：官方 robotwin_ahawam.pt（14维/3相机 → 形状自适应到 5维/6维/单相机）
#   checkpoint_shape_adapt: true 由 task 配置开启，自动跳过/池化不匹配的层
# =============================================================================

set -euo pipefail

# ---------- 路径 ----------
PROJECT_DIR="/data/wzy/AHA-WAM"
cd "${PROJECT_DIR}"

# conda prefix env（不是默认 -n 位置），直接用绝对 PATH
export PATH="${PROJECT_DIR}/env/bin:${PATH}"

# ---------- 关键环境变量 ----------
# Wan 组件（VAE + T5）从这里加载；DiT 主干来自 init_checkpoint，不依赖此目录的 DiT
export DIFFSYNTH_MODEL_BASE_PATH="${PROJECT_DIR}/checkpoints/wan_models"
# 用 modelscope / hf-mirror 下载源（huggingface.co 直连被墙）
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
export HF_ENDPOINT=https://hf-mirror.com
export HYDRA_FULL_ERROR=1

# 关键：不能设 expandable_segments:True —— GPU 驱动 470 下会触发
#   "CUDA driver error: invalid argument"（DeepSpeed zero stage1/2 init 处）。
# 官方 train_zero2.sh 第 182 行强制设了它，所以不能用官方脚本，必须手动 launch。
unset PYTORCH_CUDA_ALLOC_CONF

# ---------- 训练超参 ----------
# batch_size=1：A30 显存上限；smoke 验证 no-offload OOM，offload 后 1 卡 ~18GB 可行
# num_workers=0：宿主 cgroup 内存限 200GB，smoke8 用 0 跑通（峰值 ~192GB）
# save_every=200：持续产 checkpoint，随时可取去做阶段 2 蒸馏，不必等满 epoch
# eval_every=201：关掉 val（val 前向会叠加显存/内存峰值，先把训练跑稳）
NPROC=4
PORT=29510

echo "============================================================"
echo " AHA-WAM 阶段1 全量微调 (loader 教师)"
echo " 项目目录 : ${PROJECT_DIR}"
echo " 显卡数   : ${NPROC}"
echo " 日志     : ${PROJECT_DIR}/train.log"
echo " 初始化   : robotwin_ahawam.pt (形状自适应 14->5 / 3->1)"
echo " 配置     : ZeRO-2 + 优化器offload, bs=1, 梯度检查点, 256x320"
echo "============================================================"

# ---------- 启动 ----------
# accelerate 重定向：stdout/stderr 都进 train.log
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_offload.yaml \
  --num_processes "${NPROC}" \
  --num_machines 1 \
  --machine_rank 0 \
  --main_process_ip 127.0.0.1 \
  --main_process_port "${PORT}" \
  scripts/train.py \
  model=ahawam task=loader_ahawam \
  batch_size=1 \
  num_workers=0 \
  model.mot_checkpoint_mixed_attn=true \
  save_every=200\
  eval_every=201 \
  log_every=10 \
  2>&1 | tee "${PROJECT_DIR}/train.log"

# 退出码透传
exit "${PIPESTATUS[0]}"
