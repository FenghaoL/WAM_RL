# cylinder-v1：只用 09-11 recover 短片段（裁掉每条前 0.5 秒）的三组 adapter 训练

2026-09-11 按用户调整：从原带历史的 4-epoch SFT `step_006572` 出发，依次训练 V32 → Q32 → T4，
**纠正数据只用** `data/so101_ahawam_cylinder_insertion_v1/01_pick_up_and_insert_cylinder/20260911_185932_recover`，
不再使用 DAgger 人类接管数据，也不用 09-11 早些时候那批片段。

## 1. 先回答："这种短片段训练时带不带历史有区别吗？"

**对这批数据没有区别，实测是 0 区别。** 一帧视频历史要求采样起点距片段开头至少 48 帧
（`horizon_stride = (num_frames-1) × global_sample_stride = 48`），再多一帧又要 48 帧：

| 数据 | 平均历史帧数 | 完全无历史的样本 | 最大 |
|---|---:|---:|---:|
| demo（19,708 帧） | 1.763 | 22.9% | 6 |
| DAgger 人类接管（2,393 帧，77 段） | 0.148 | 88.7% | 3 |
| 09-11 早批纠正片段（4,706 帧） | 0.070 | 93.5% | 2 |
| **本轮 recover 片段（923 帧，裁剪后最长 47 帧）** | **0.000** | **100%** | **0** |

所以对纠正样本而言，`num_history_frames` 设 6 还是 0 完全不影响它们的输入。有影响的只是同一个
batch 里另一半 demo 样本，以及在 demo 上计算的 P0 anchor。本轮起点是带历史训练的 SFT，因此保持
`num_history_frames: 6`，避免给 demo 侧和 anchor 引入与起点不一致的条件。

## 2. 裁掉每条 episode 前 0.5 秒的机制

**做法：加载时的逻辑视图，不改磁盘数据、不重新编码视频。** `SO101CorrectionClipDataset` 新增
`trim_leading_frames`：每条 episode 的逻辑起点后移 N 帧，之后所有动作、本体状态、图像查询都由同一个
后移后的存储下标解析，因此不可能出现"动作和视频帧错位"；历史与片段起点边界一并后移；不足 N 帧的
episode 会被丢弃（本轮为 0 条）。这与 DAgger 人类片段"只暴露接管区间"的既有机制是同一套做法。

- 0.5 秒 @ 15fps = 7.5 帧，向上取整为 **8 帧**；30 条 episode 共 1,163 帧 → 裁剪后 **923 帧**。
- 配置里由 `data.train.correction_clip_trim_leading_frames` 控制，秒数写在预设里（`trim_leading_seconds`），
  帧数由数据集 fps 换算并记进 manifest，方便以后换数值或换数据集复用。

**正确性验证**（`check_trim_alignment.py`，可复用）：构造一个同样 session、`trim=0` 的对照数据集，
逐样本比较"裁剪后第 i 帧"与"原始第 i+8 帧"。本轮比较了全部 30 条 episode、120 对样本，
`action`、`proprio`、`video`、`chunk_obs_images`、`chunk_obs_images_no_offset`、`action_offset`
以及三个 padding 掩码**全部逐张量相等**；反向对照确认裁剪确实移动了起点。

一个实现细节值得记下：`action_offset` 是每次取样用 `np.random` 抽的（0–7 增强），`chunk_obs_images`
按它挑帧，所以比较两个数据集前必须固定随机种子，否则会误报不一致——脚本已经这样做。

另外实测这批数据开头静止时间**普遍超过 0.5 秒**：首次运动帧中位数 10.5 帧（0.70 秒），p90 为 13.8 帧，
个别 episode 第 0 帧就在动。按要求先按 0.5 秒裁；若要更激进，把预设里的 `trim_leading_seconds` 调大即可。

## 3. 数据与训练设置

| 项 | 值 |
|---|---|
| 起点 | 原 cylinder SFT `formal-cylinderv1-sft-...-20260905-022135/step_006572.pt`（history 6） |
| 纠正数据 | 只有 recover：30 条、1,163 帧 → 裁剪后 **923 帧**，无 DAgger 数据 |
| demo | 与之前相同的 4 个 session，19,708 帧 |
| 采样 | demo:纠正 = 1:1，全局 batch 12，3285 步 |
| 其余 | lr 1e-4、P0 anchor λ=1、冻结原模型只训 adapter、无视频损失、每 219 步验证 |
| 验证 | 不变：demo val `20260904_024009_val` + 13 条 held-out DAgger rollout（仅评估，不训练） |
| checkpoint | **每 547 步保存一次**（其余轮次是 1095），共 7 个 |

**必须提醒：纠正数据只有 923 帧，1:1 采样下会被重复约 21 遍**（之前几轮分别是 8.2 和 2.8 遍）。
过拟合风险明显更高，所以这轮把保存频率加密，方便挑早期权重；请重点看 HUMAN/demo 验证曲线何时开始变差，
不要默认最后一个权重最好。验证集仍是 DAgger 接管分布，与这批 recover 数据不同源，可以作为"是否伤害了
原有纠正能力"的参考，但它不衡量对 recover 这类场景的拟合。

## 4. 命令

```bash
# 生成（已执行；拒绝覆盖）
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/prepare_clips.py --preset recover
# 裁剪对齐验证（可复用，--episodes 0 表示全查）
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/check_trim_alignment.py \
  --config runs/ovcr_adapter_cylinder_v1_recover/configs/p1_v32_balanced_recover.yaml --episodes 0
# 片段边界 / 来源标签 / 混合 batch
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/check_clip_data.py \
  --config runs/ovcr_adapter_cylinder_v1_recover/configs/p1_v32_balanced_recover.yaml
# 启动（已执行）
RUN_ROOT=runs/ovcr_adapter_cylinder_v1_recover SESSION=ovcr_cylinder_recover GPU_IDS=0,1,2 \
NAMES="p1_v32_balanced_recover p1_q32_balanced_recover p1_t4_balanced_recover" \
bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_suite.sh
# 查看 / 停止
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/ovcr_adapter_cylinder_v1_recover
kill -TERM "$(cat runs/ovcr_adapter_cylinder_v1_recover/suite.pid)"
```

## 5. 以后要对别的数据做同样的事

1. 在 `prepare_clips.py` 的 `PRESETS` 里加一条：`clip_glob` 指向新 session，`trim_leading_seconds` 写秒数，
   `use_hil_human` 决定是否混入 DAgger 人类接管数据，`save_every` 控制存档密度；
2. 跑 `prepare_clips.py --preset <名字>` 生成 manifest 和三个配置；
3. 跑 `check_trim_alignment.py`（必做，验证裁剪没有错位）和 `check_clip_data.py`；
4. 用 `launch_suite.sh` 排队。

裁剪本身只改采样起点，不改磁盘数据，所以同一批原始数据可以并存多个不同裁剪量的实验。
