# cylinder HIL v2：历史可跨越接管边界的纠正训练

2026-09-12。起点仍是原 cylinder 任务 SFT `step_006572`（history 6）。纠正数据只用
`data/ahawam_hil_raw/so101_ahawam_cylinder_hil_v2` 的人类接管部分。

## 1. 这轮的新机制：历史可跨出接管区间

**当前帧、被监督的动作窗口、预测的视频都仍严格在人类接管段内**；只有 6 帧视频历史被允许取到
接管发生之前，也就是该 rollout 里模型自主运行的部分——这正是部署时模型真正拥有的记忆。历史仍然
以该条 rollout 的开头为下界，不会跨 rollout。

效果（真实数据实测）：

| | 平均历史帧 | 完全没有历史的样本 |
|---|---:|---:|
| 旧做法（历史也被截断在接管段内） | 0.04 | **95.5%** |
| 本轮（历史可跨界） | 2.39 | 4.5% |

也就是说，在此之前这批短接管片段的训练**几乎等于没有历史**，这轮才真正让记忆参与进来。

## 2. 历史帧间隔：为什么按"执行帧"数而不是按原始帧数

录制是固定 15Hz 连续记帧的（`loop_dt` 中位 67.8ms，p99 68.5ms），等待推理时的停顿会被完整记录：
v2 数据里 `control_source` 的分布是 POLICY 17,482 / HUMAN 3,629 / **SYSTEM_HOLD 22,569（52%）**。
因此在自主运行段里往回数 48 个原始帧，覆盖的任务进度远小于 demo 里的 48 帧。

实测对比（一次历史步覆盖的关节运动量，以 demo SFT 的 48 帧为 100%）：

| 方案 | 阶段覆盖 | 平均可用历史帧 | 中位 | 无历史样本 |
|---|---:|---:|---:|---:|
| 原始 48 帧（不跳停顿） | 38% | 4.20 | 6 | 0.0% |
| 原始 96 帧（2× 倍率） | 77% | 2.47 | 3 | 3.3% |
| **跳停顿 48 执行帧（本轮主实验）** | **65%** | **2.39** | **3** | **4.5%** |
| 跳停顿 72 执行帧 | 97% | 1.31 | 2 | 34.7% |
| 跳停顿 96 执行帧 | 130% | 0.85 | 1 | 44.7% |

主实验选"跳停顿 48"的两个理由：

1. **位置编码几何与 SFT 完全一致**。按执行帧计数后，历史帧的 RoPE 位置正好每步相隔 12 个单位，
   与 demo SFT 相同；而"原始 96"会变成 24 个单位，是冻结骨干没见过的几何。
2. **完全对齐阶段（跳停顿 72）代价过大**：35% 的样本会一帧历史都没有。

一个必须记录的矛盾：**部署端每次 prefill 把时间位置固定 +10 个单位（=40 帧）递增，与真实经过多少帧无关**
（[ahawam.py:490](../../../src/ahawam/models/wan22/ahawam.py#L490)），而部署时历史帧的真实内容间隔是两次
prefill 之间的实际帧数（v2 实测中位 31.5 帧）。所以"贴近 SFT 语义"和"贴近部署实况"是两个互相冲突的
目标，本轮按用户偏好选择前者。若要让前者在部署时也成立，需要改部署端保留更早的 prefill 作为历史。

## 3. 数据与划分

| 项 | 值 |
|---|---|
| 训练 session | 5 个，82 条含接管的 rollout，**3,460 帧 / 121 段** |
| 验证 session | `20260912_010747_val`，4 条 rollout，169 帧 / 7 段 |
| demo | 与之前相同的 4 个 session，19,708 帧；demo 验证 661 帧 |
| 划分规则 | 目录名以 `_val` 结尾的 session 整体作验证，其余全部训练（不在 session 内部按 episode 切） |

注意：HUMAN 验证集只有 169 帧，比之前几轮（674 帧）小很多，`human_*` 指标会更抖，且与之前几轮
**不可直接比较**（验证集换了）；`demo_*` 仍然可比。

## 4. 六个实验

前四个是主实验（按用户要求 V32 → Q32 → T4 → full），后两个是只改"历史间隔"这一个变量的 V32 对照：

| 实验 | 可训练部分 | 历史间隔 | 跳停顿 |
|---|---|---|---|
| `p1_v32_balanced_hilv2` | V32 adapter | 48 执行帧 | 是 |
| `p1_q32_balanced_hilv2` | Q32 adapter | 48 执行帧 | 是 |
| `p1_t4_balanced_hilv2` | T4 adapter | 48 执行帧 | 是 |
| `p1_full_balanced_video_hilv2` | 全量（含视频损失） | 48 执行帧 | 是 |
| `p1_v32_balanced_hilv2_raw48` | V32 adapter | 48 原始帧 | 否 |
| `p1_v32_balanced_hilv2_skip72` | V32 adapter | 72 执行帧 | 是 |

其余沿用既有协议：1:1 demo/纠正、3285 步、全局 batch 12、adapter lr 1e-4 + P0 anchor、
full lr 1e-5 + 视频损失、每 219 步验证、每 1095 步存权重。

## 5. 启动前验证

`check_history_crossing.py`（可复用）在 61 个真实样本上逐项确认：

- 143 帧历史确实取自接管之前；
- 每个被用作历史的帧都是执行帧，相邻历史帧之间正好相隔 48 个执行帧，平均每次回溯跳过约 115 帧 SYSTEM_HOLD；
- RoPE 位置非负、间距一致，且与当前帧共用同一原点；
- 动作监督窗口仍止于接管段末尾（有效动作数 = 段内剩余帧数）；
- 关掉跨界后同样的样本历史帧严格更少（0.04 vs 2.39）。

## 6. 命令

```bash
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/prepare_hil_v2.py
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/check_history_crossing.py \
  --config runs/ovcr_adapter_cylinder_hil_v2/configs/p1_v32_balanced_hilv2.yaml --samples 60
RUN_ROOT=runs/ovcr_adapter_cylinder_hil_v2 SESSION=ovcr_cylinder_hilv2 GPU_IDS=0,1,2 \
NAMES="p1_v32_balanced_hilv2 p1_q32_balanced_hilv2 p1_t4_balanced_hilv2 p1_full_balanced_video_hilv2 p1_v32_balanced_hilv2_raw48 p1_v32_balanced_hilv2_skip72" \
bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_suite.sh
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/ovcr_adapter_cylinder_hil_v2
kill -TERM "$(cat runs/ovcr_adapter_cylinder_hil_v2/suite.pid)"   # 停止
```

部署这些权重时用 `num_history_frames: 6`、cylinder stats 与原任务指令。
