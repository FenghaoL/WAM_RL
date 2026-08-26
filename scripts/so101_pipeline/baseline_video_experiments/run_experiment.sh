#!/usr/bin/env bash
# ============================================================================
# 视频生成对照实验 —— 一键入口
#
#   bash scripts/so101_pipeline/baseline_video_experiments/run_experiment.sh
#
# 不需要任何参数。要改什么，只改下面【配置区】里的变量。
#
# 它按顺序做三件事：
#   1. 挑评测窗口 + 导出真值      -> $OUTPUT_ROOT/samples/
#   2. 每个 checkpoint 生成视频   -> $OUTPUT_ROOT/<checkpoint 名>/
#   3. 合并成对比表 + 并排视频    -> $OUTPUT_ROOT/comparison/
#
# 第 1 步只在窗口还不存在时跑。所有 checkpoint 打的是同一批窗口，数字才可比。
# ============================================================================
set -eo pipefail

# ============================== 配置区 ======================================

# 训练 run 的目录，里面要有 config.yaml 和 checkpoints/weights/
RUN_DIR="runs/so101_ahawam_plug/full_zero2_offload_ahawam_base/formal-plug-ahawambase-zero2offload-bs4-acc1-4ep-3gpu-20260825-145746/model"

# 打哪一批数据。改这一个词就行。
#   train -> 用 $RUN_DIR/config.yaml 里的训练集目录
#   val   -> 用下面 VAL_DIRS 里的留出目录
# 每一档的窗口都是各自挑的，所以输出目录自动分开，不会混在一张表里比。
PRESET="train"

VAL_DIRS=(
  data/so101_ahawam_plug/04_full_remove_and_insert/20260825_142447_val
  data/so101_ahawam_plug/04_full_remove_and_insert/20260825_142844_val
)

# 结果放哪。同一个 OUTPUT_ROOT 下的所有 run 共用一批评测窗口，
# 最后的对比表会把这个目录里**所有**跑过的 run 都合进来。
# 想和别的 run 对比，就用同一个 OUTPUT_ROOT + 不同的 NAME_PREFIX。
# 每档各自的输出目录和窗口数。验证集 episode 少，能挑出的窗口自然也少；
# 挑多了脚本会直接报 "Only N windows are available"，按提示调小即可。
#
# WINDOW_STRIDE：窗口在一条 episode 内部每隔多少个控制帧放一个（默认 48 = 一整段）。
#   调小 -> 每条 episode 出更多窗口。历史不受影响：历史帧永远取
#   sample_idx - 48*j，窗口落在哪都一样。
# PER_EPISODE_START=1：额外保证**每条 episode 的第 0 帧**都有一个窗口，
#   也就是机器人的真实初始状态（这种窗口没有历史，和在线部署刚开始时一致）。
#   顺带也救回那些太短、原本会被整条跳过的 episode。
if [ "$PRESET" = "val" ]; then
  # 4 条留出 episode。stride=48 正好等于一个窗口的长度，所以窗口首尾相接、
  # 无缝铺满整条 episode；再加上首/尾保留，等于把每条 episode 从初始状态一直
  # 走到任务结束都过一遍。
  OUTPUT_ROOT="runs/so101_ahawam_plug/baseline_experiments_val"
  DATASET_DIRS=("${VAL_DIRS[@]}")
  NUM_SAMPLES=26
  WINDOW_STRIDE=48
  PER_EPISODE_START=1
  PER_EPISODE_END=1
else
  OUTPUT_ROOT="runs/so101_ahawam_plug/baseline_experiments"
  DATASET_DIRS=()
  NUM_SAMPLES=24
  WINDOW_STRIDE=48
  PER_EPISODE_START=0
  PER_EPISODE_END=0
fi

# 给这一批 checkpoint 的输出目录加个前缀，避免和同名 step 撞车。
# 例：前缀 "ahawambase_" + checkpoint "step_003440" -> 目录 ahawambase_step_003440
NAME_PREFIX="plug_"

# 要评测哪些 checkpoint。写几个就跑几个，一张卡跑一个。
#   - 只写 step 名  -> 自动到 $RUN_DIR/checkpoints/weights/<名>.pt 找
#   - 写完整 .pt 路径 -> 直接用（可以是别的 run 的 checkpoint）
#   - 写字面量 wan22-base -> 不加载任何 checkpoint，用 Wan2.2 官方原始权重当对照
CHECKPOINTS=(
  "step_002434"
  "step_003651"
  "step_004868"
)

# 用哪些卡（物理卡号）。写几张就并行几个 checkpoint，多出来的排队等。
GPUS=(1 2 3)

STEPS=20                             # 去噪步数
SEED=42                              # 同一 seed 下各 checkpoint 拿到同一份初始噪声
HISTORY_MODES="online,training,none" # 历史模式，逗号分隔；第一个用于并排视频
FORCE_REEXPORT=0                     # 1 = 强制重新挑窗口（会让旧结果不可比）


# --------------------------- 三个常用配方 ---------------------------------
# 只要改上面配置区的几行就能切换，不用动别的文件。
#
# 配方 1：看同一个 run 的几个 checkpoint 训得怎么样（就是当前默认）
#   RUN_DIR   = 那个 run 的 model 目录
#   CHECKPOINTS = ("step_001977" "step_002636" "step_003295")
#
# 配方 2：加上"没训练过"的原始 Wan2.2 当下限对照
#   CHECKPOINTS = ("wan22-base" "step_003440")
#   （想看原生 Wan2.2 自己的推理配置能到什么水平，再单跑一次并把
#     HISTORY_MODES 改成 "wan_native"，CHECKPOINTS 只留 "wan22-base"）
#
# 配方 3：打验证集
#   PRESET="val"（换别的留出目录就改 VAL_DIRS）
#   注意 NUM_SAMPLES 别超过验证集能提供的窗口数
# --------------------------------------------------------------------------

# ============================ 配置区结束 ====================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$REPO_ROOT/.cache/huggingface"
export HF_DATASETS_CACHE="$REPO_ROOT/runs/cache/huggingface/datasets_v2"
export DIFFSYNTH_MODEL_BASE_PATH="$REPO_ROOT/checkpoints"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

HERE="scripts/so101_pipeline/baseline_video_experiments"
SAMPLES_DIR="$OUTPUT_ROOT/samples"
LOG_DIR="$OUTPUT_ROOT/logs"
CONFIG="$RUN_DIR/config.yaml"
mkdir -p "$LOG_DIR"

[ -f "$CONFIG" ] || { echo "找不到配置文件: $CONFIG"; exit 1; }

# 数据目录覆盖：两个 python 脚本要传一模一样的参数，否则全局索引对不上
DATA_ARGS=()
if [ ${#DATASET_DIRS[@]} -gt 0 ]; then
  DATA_ARGS=(--dataset-dirs "${DATASET_DIRS[@]}" --val-set-proportion 0)
fi

echo "=========================================================="
echo " run     : $RUN_DIR"
echo " 输出    : $OUTPUT_ROOT"
echo " ckpt    : ${CHECKPOINTS[*]}"
echo " 卡      : ${GPUS[*]}"
echo " 数据    : ${DATASET_DIRS[*]:-（用训练配置里的目录）}"
echo " 窗口数  : $NUM_SAMPLES   步数: $STEPS   seed: $SEED"
echo "=========================================================="

# ---------- 第 1 步：挑窗口 ----------
if [ "$FORCE_REEXPORT" = "1" ] || [ ! -f "$SAMPLES_DIR/manifest.json" ]; then
  echo "[1/3] 挑评测窗口 -> $SAMPLES_DIR"
  EXPORT_ARGS=()
  [ -n "$WINDOW_STRIDE" ] && EXPORT_ARGS+=(--window-stride "$WINDOW_STRIDE")
  [ "$PER_EPISODE_START" = "1" ] && EXPORT_ARGS+=(--per-episode-start)
  [ "$PER_EPISODE_END" = "1" ] && EXPORT_ARGS+=(--per-episode-end)
  python "$HERE/export_reference_samples.py" \
    --config "$CONFIG" \
    --output "$SAMPLES_DIR" \
    --num-samples "$NUM_SAMPLES" \
    "${EXPORT_ARGS[@]}" \
    "${DATA_ARGS[@]}"
else
  echo "[1/3] 窗口已存在，跳过（改 FORCE_REEXPORT=1 可强制重挑）"
fi

# ---------- 第 2 步：每个 checkpoint 生成视频 ----------
echo "[2/3] 生成视频"
RUN_NAMES=()
pids=()
slot=0
for ckpt in "${CHECKPOINTS[@]}"; do
  if [ "$ckpt" = "wan22-base" ]; then
    weights="wan22-base"; name="wan22_base"
  elif [[ "$ckpt" == *.pt ]]; then
    weights="$ckpt"; name="$(basename "$ckpt" .pt)"
  else
    weights="$RUN_DIR/checkpoints/weights/$ckpt.pt"; name="$ckpt"
  fi
  name="$NAME_PREFIX$name"
  if [ "$weights" != "wan22-base" ] && [ ! -f "$weights" ]; then
    echo "  找不到权重: $weights"; exit 1
  fi
  RUN_NAMES+=("$name")

  gpu="${GPUS[$((slot % ${#GPUS[@]}))]}"
  echo "  $name -> GPU $gpu   (日志: $LOG_DIR/$name.log)"
  CUDA_VISIBLE_DEVICES="$gpu" python -u "$HERE/wan22_base/run_video_branch_preview.py" \
    --config "$CONFIG" \
    --samples-dir "$SAMPLES_DIR" \
    --output "$OUTPUT_ROOT/$name" \
    --weights "$weights" \
    --run-name "$name" \
    --device cuda:0 \
    --steps "$STEPS" \
    --seed "$SEED" \
    --history-modes "$HISTORY_MODES" \
    "${DATA_ARGS[@]}" \
    > "$LOG_DIR/$name.log" 2>&1 &
  pids+=($!)
  slot=$((slot + 1))

  # 卡用满了就等这一批跑完，再开下一批
  if [ ${#pids[@]} -ge ${#GPUS[@]} ]; then
    for p in "${pids[@]}"; do wait "$p"; done
    pids=()
  fi
done
for p in "${pids[@]}"; do wait "$p"; done

# ---------- 第 3 步：合并对比 ----------
# 把 $OUTPUT_ROOT 下所有跑过的 run 都合进来，不只是这一批
echo "[3/3] 合并对比"
python "$HERE/compare_runs.py" \
  --runs "$OUTPUT_ROOT"/*/summary.json \
  --output "$OUTPUT_ROOT/comparison"

echo
echo "=========================================================="
echo " 完成。先看这两个："
echo "   对比表   : $OUTPUT_ROOT/comparison/comparison.md"
echo "   并排视频 : $OUTPUT_ROOT/comparison/side_by_side/"
echo "=========================================================="
