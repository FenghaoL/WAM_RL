# AHA-WAM HIL 连续时间轴与训练 Mask 交接说明

本文面向训练服务器上的 Codex 或维护者，说明 `so101-ahawam-hil-rich-v3` 数据中新增的 `SYSTEM_HOLD` 帧、逐步 action 监督 mask，以及 AHA-WAM 40-step 训练端必须采用的处理方式。

## 1. 背景与问题

SO-101 客户端以 15 Hz 执行 AHA-WAM 返回的 8-step action chunk。一块动作只覆盖：

```text
8 / 15 = 0.533 s
```

当前服务器的一次请求通常慢于这一时长。旧 collector 在 action FIFO 为空时让 follower 继续保持上一目标，但不写相机、state 或 action。下一块动作到达后，LeRobot 又按固定 15 Hz 把前后两帧排成相邻帧，导致约 0.7 秒真实运动被压缩为一个 66.7 ms 时间步。

实测旧数据在该边界上的最大关节 state 变化中位数约为 3.34°、P90 为 6.86°、最大约 9.67°。MP4 帧数和 parquet 行数相符，所以这不是视频编码丢帧，而是 collector 从未提交等待期间的帧。

## 2. rich-v3 采集语义

从 `so101-ahawam-hil-rich-v3` 开始，AUTONOMOUS 阶段 action FIFO 为空时，每个 15 Hz tick 仍写入一行：

| 字段 | SYSTEM_HOLD 的值 | 含义 |
|---|---|---|
| `control_source` | `2` | 枚举名为 `SYSTEM_HOLD` |
| `phase` | `2` | 仍处于 `AUTONOMOUS` |
| `action` | 上一个实际 follower target | 当前舵机仍在执行的绝对标定关节目标 |
| `executed_action_arm` | 同上 | arm 坐标下的实际目标 |
| `policy_action` | 零占位 | 必须先看 `policy_action_valid=false` |
| `human_action` | 零占位 | 必须先看 `human_action_valid=false` |
| `action_supervision_mask` | `false` | 不得把网络等待产生的 hold 计入直接 action loss |
| `intervention` | `false` | 不是人工接管 |
| `policy_request_id` / `policy_chunk_id` | `-1` | 该 tick 没有新的策略动作 |

正常 POLICY 和 HUMAN 帧的 `action_supervision_mask=true`。这个字段表示“该行具有可作为行为标签的主动命令”，并不等价于“这是 HG-DAgger 专家标签”。

`action` 不能在 SYSTEM_HOLD 行写零，也不应写当前 measured state。SO-101 使用绝对关节目标；真实生效的命令仍是上一目标。写零会制造完全错误的绝对姿态，写 measured state 则会把舵机尚未到达目标的误差伪装成新命令。

## 3. 必须区分的三个 mask

训练端至少要区分：

1. `action_is_pad`：LeRobot episode 尾部补齐位置；
2. `action_supervision_mask`：collector 判断该 tick 是否有主动、可监督的 action；
3. `intervention` / `human_action_valid`：是否为 HG-DAgger 人工专家标签。

对于从时刻 `t` 开始的 40-step action target，必须读取未来 40 个 action 对应的 40 个 mask。不能只读取时刻 `t` 的一个布尔值再广播到整个 horizon。

严格 HG-DAgger action loss 应使用：

```python
# Build one mask value for every future action step.
valid = ~action_is_pad.bool()
valid &= action_supervision_mask.bool()
valid &= intervention.bool()
valid &= human_action_valid.bool()
```

这会同时排除：

- SYSTEM_HOLD；
- 自主 POLICY action；
- episode padding；
- 没有合法 human proposal 的行。

如果进行 IWR 或行为正则化，自主策略动作应单独构造 mask 和 sampler，不能与 HG expert 标签混为一类：

```python
# Keep policy self-imitation separate from expert supervision.
policy_valid = ~action_is_pad.bool()
policy_valid &= action_supervision_mask.bool()
policy_valid &= control_source.eq(0)
policy_valid &= policy_action_valid.bool()

expert_valid = ~action_is_pad.bool()
expert_valid &= action_supervision_mask.bool()
expert_valid &= control_source.eq(1)
expert_valid &= human_action_valid.bool()
```

## 4. AHA-WAM loss 的推荐修改

官方 `src/ahawam/models/wan22/ahawam.py` 中 `_compute_weighted_action_loss()` 已生成 `[B, T]` 的逐时间步 MSE，并支持 `action_is_pad`。应增加一个同形状的 `action_supervision_mask`，然后与 padding mask 相交。

推荐保留“每个 sample 按有效步数归一化”的现有语义，同时排除整条没有有效 action 标签的 sample：

```python
def masked_action_loss(
    action_loss_token,
    action_weight,
    action_is_pad,
    action_supervision_mask,
):
    # All tensors below are [B, T].
    valid = ~action_is_pad.bool()
    valid &= action_supervision_mask.bool()
    valid_f = valid.to(action_loss_token.dtype)

    valid_count = valid_f.sum(dim=1)
    per_sample = (
        action_loss_token * action_weight * valid_f
    ).sum(dim=1) / valid_count.clamp_min(1.0)

    has_label = valid_count > 0
    if not bool(has_label.any()):
        return action_loss_token.sum() * 0.0
    return per_sample[has_label].mean()
```

不能只把 masked token 的 action 改成零。这里的 action 是绝对关节角，零本身是合法且通常相差很大的姿态；正确做法是保留真实生效的 hold target，但让对应直接 action loss 为零。

## 5. Dataset loader 的对齐要求

训练服务器需要在 AHA-WAM 的 LeRobot loader 中，把以下字段使用与 `action` 完全相同的未来 delta indices 载入：

```text
action
action_supervision_mask
intervention
human_action_valid
control_source
tick_index
dropped_tick
```

最终 batch 中至少应有：

```text
action:                   [B, 40, 6]
action_is_pad:            [B, 40]
action_supervision_mask:  [B, 40]
intervention:             [B, 40]
human_action_valid:       [B, 40]
control_source:           [B, 40]
tick_index:               [B, 40]
dropped_tick:             [B, 40]
```

LeRobot feature 的原始 shape 是 `[1]`，loader 取出时间窗口后可能得到 `[B, 40, 1]`；进入 loss 前应安全地 `squeeze(-1)`，同时断言最终形状与 `action_is_pad` 完全一致。

需要重点检查：

- `src/ahawam/datasets/lerobot/base_lerobot_dataset.py`：额外字段是否被 LeRobot temporal query 保留下来；
- `src/ahawam/datasets/lerobot/robot_video_dataset.py`：是否把 mask 放入返回 sample；
- `src/ahawam/models/wan22/ahawam.py`：训练入口和 `_compute_weighted_action_loss()` 是否接收逐步 mask。

## 6. 时间连续性不能只看 LeRobot timestamp

rich-v3 修复的是 AUTONOMOUS action 队列等待。用户主动按 Space 后的 PAUSED、leader 对齐和 POLICY_WARMUP 仍不写训练帧，这是有意行为。因此完整 raw episode 中仍可能存在 deliberate pause 前后的相邻存储行。

训练窗口必须满足未来范围内 tick 连续：

```python
# Reject windows that cross omitted pause, warmup, or reset ticks.
tick_delta = tick_index[:, 1:] - tick_index[:, :-1]
window_is_continuous = tick_delta.eq(1).all(dim=1)

# dropped_tick marks a controller deadline miss.
window_is_continuous &= ~dropped_tick[:, 1:].bool().any(dim=1)
```

对于不连续窗口，建议重采样或跳过整个 sequence；不要插值生成虚假的 camera、state 或 action。窗口第一帧之前是否存在 gap 通常不影响以该帧为当前 observation 的预测，因此关键是窗口内部所有未来 transition 连续。

## 7. Video loss 如何处理 SYSTEM_HOLD

SYSTEM_HOLD 图像和 state 是真实、按 15 Hz 连续采集的物理过程：机器人在保持上一绝对目标时仍可能继续收敛、受重力下沉或发生接触。因此默认可以把这些帧用于 world/video loss，且 action conditioning 应使用真实 hold target。

但 `action_supervision_mask=false` 只保证没有直接 action MSE/flow loss，并不能数学上保证这些 sample 对联合模型“完全零影响”：AHA-WAM 的 video/action 分支共享或交互模块，video loss 仍可能通过共享模块产生梯度。

推荐把训练拆成两个清晰的数据用途：

- **HG action batch**：原始 demonstrations + HIL 人工 correction；action loss 只看 `expert_valid`；
- **可选 raw world batch**：连续完整 raw 用于 video/world loss，action loss 关闭或单独降低权重。

如果实验目标要求严格排除所有自主轨迹对模型参数的影响，就不要把完整 raw 用于 video loss，只使用原始 demonstrations 与 correction-only 导出。mask 能排除直接 action 标签，不能替代数据集采样策略。

## 8. Normalization stats

不要让大量重复的 SYSTEM_HOLD action 改变 action 均值和方差。

- strict HG：action stats 只使用原始 demonstrations 与 `expert_valid=true` 的 human action；
- IWR：分别统计并记录 expert/policy 来源，或使用与实际训练 sampler 一致的加权统计；
- state/image stats 可以使用时间连续、质量合格的 raw 帧；
- 所有零占位的 `policy_action` / `human_action` 必须先用 valid mask 排除。

## 9. 只训练人工 correction 的推荐入口

如果本轮实验的目标是“完全不让 autonomous / SYSTEM_HOLD 帧进入训练”，最稳妥的做法不是在服务器 loader 中临时过滤 raw 行，而是先运行本地 `export_hg_corrections.ps1`。导出的每个 LeRobot episode 都是一段时间连续的人工接管：

- 只选 `intervention=true && human_action_valid=true`；
- 在 episode、frame、`tick_index` 不连续或当前帧 `dropped_tick=true` 时断开；
- `action` 是当时真正发送给 follower 的绝对关节目标，不是 leader 相对位移；
- fixed/wrist 图像、state、action 保持标准 LeRobot v2.1 结构；
- `source_segments.jsonl` 保留每个输出帧到 raw 数据的可追溯映射。

这种 correction-only 输出不需要 rich-v3 的 `action_supervision_mask`，因为其中所有行都是专家标签。但 AHA-WAM 40-step target 在片段尾部仍必须使用与 action 同一 delta indices 产生的 `action_is_pad`；不得把 episode 尾部重复值当成 40 个有效 expert step。

如果本轮仍训练 future-video/world loss，还必须为每个 future video target 构造同样的逐帧 `video_is_pad`，或者只采样具有完整 future-video horizon 的 anchor。当前服务 health 所示 13 帧、3.75 fps 预览大约跨 3.2 秒，在 15 Hz 原始数据上约需从 anchor 起覆盖 49 帧；不能把短 correction episode 的末帧重复到 13 帧后继续计 video loss。若服务器暂时没有 video padding mask，第一次 HG 实验更稳的是只对 correction batch 开 action loss，同时继续从原始 demonstrations 提供正常 video/world batch。

不要直接从 raw parquet 删除 autonomous 行后继续复用原 MP4 和原 timestamp。这样会造成视频索引、frame index 和时序语义错位。应使用导出器重新物化视频与 episode。

推荐训练配方是“原始人工 demonstrations + correction-only 数据”，并显式记录两者采样比例。只使用 correction-only 做少量 epoch 的对照实验在技术上可行，但它集中在失败附近和插入末端，容易缩窄状态分布或遗忘正常抓取/搬运能力，不能仅凭格式校验保证策略性能。

## 10. 2026-08-29 本地数据审计

以下四个 run 已用当前校验器检查，所有已保存 episode 的 outcome 都是 `success`，parquet、manifest 和双相机视频数量一致；`20260829_110214` 原先只有 PNG，现已完成 4 个 MP4 的离线封装。

| run | contract | episode | human 帧 | 可导出人工段 | 段长 min / median / max | 结论 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| `20260829_004417` | rich-v1 | 1 | 96 | 2 | 26 / 48 / 70 | 可作 correction-only；样本很少 |
| `20260829_110214` | rich-v2 | 2 | 476 | 3 | 74 / 190 / 212 | 已完成视频，可作 correction-only |
| `20260829_112454` | rich-v2 | 14 | 3965 | 41 | 22 / 94 / 261 | 可作 correction-only |
| `20260829_155403` | rich-v3 | 18 | 4621 | 51 | 22 / 74 / 210 | 可作 correction-only，也可按本文件 mask 规则使用 raw |

四批中 `minimum_segment_frames=2` 时人工帧保留率都是 100%，没有被丢弃的孤立人工单帧。`112454` 有 2 个、`155403` 有 9 个 `dropped_tick=true` 行；导出器会在这些位置断开片段，当前帧本身仍是真实 observation/action，可以作为新片段开头。

短于 40 个控制帧的 correction 段数量依次为 1、0、4、6；短于约 49 帧 future-video 覆盖长度的段数量依次为 1、0、5、9。这些段不是坏数据，不应直接删除；它们要求 action/video 分别正确处理 padding，或者在采样 anchor 时保证目标 horizon 完整。

`155403` 唯一的 AUTONOMOUS-to-AUTONOMOUS gap 位于 episode 17、frame 688→689、tick 901→983，经过约 5.55 秒。`events.jsonl` 显示期间依次发生 `pause_requested(fine_alignment_error)`、leader 对齐、policy reset/warmup 和恢复自主控制，因此它是有意省略暂停阶段，不是视频编码掉帧。任何完整 raw future window 都必须在这里断开；correction-only 导出不会跨越它。

rich-v1/v2 的大量 autonomous gap 来自旧 collector 在等待下一动作块时没有写帧。它们不能用于连续 future-video 训练，但不影响导出连续人工接管段。不要给 rich-v1/v2 人工补一列 `action_supervision_mask` 后把它们冒充 rich-v3。

服务器端第一次使用这些数据时应保留并运行本目录的 `validate_hil_dataset.py`，把生成的 `validation_summary.json` 连同训练配置归档。若服务器只拿到 correction-only 输出，则还应检查每个 episode 的视频帧数、parquet 行数和 `action_is_pad` 单元测试。

## 11. 旧数据兼容性

`so101-ahawam-hil-rich-v2` 没有 `action_supervision_mask`，且 AUTONOMOUS 等待期间的帧从未采集，无法在服务器上事后恢复。

旧数据应采用以下任一方式：

1. 使用 `export_hg_corrections.py` 导出的 correction-only v2.1 数据；
2. 根据 `tick_index` / `monotonic_ns` 把完整 raw 切成连续区间，禁止窗口跨 gap；
3. 仅用于诊断，不用于 AHA-WAM 连续 future-video 训练。

绝对不能仅给旧数据补一列 mask 后宣称时间轴已经修复。

## 12. 服务器端验收测试

训练改动完成后至少增加以下测试：

1. 40-step action 与 40-step mask 使用同一 delta indices；
2. 混合 `POLICY → SYSTEM_HOLD → HUMAN` 窗口中，strict HG 只监督 HUMAN 步；
3. SYSTEM_HOLD 的绝对 action 保留上一目标而不是零；
4. `action_is_pad=true` 的位置无论其他 mask 如何都不参与 loss；
5. 没有任何 expert step 的 batch 不产生 NaN，也不稀释有标签 sample 的平均 loss；
6. `tick_index` 不连续或 `dropped_tick=true` 的 future window 不进入时序训练；
7. action normalization stats 排除 SYSTEM_HOLD 和无效零占位；
8. video loss 是否包含 raw world batch 由显式配置控制，而不是意外继承 action mask。

## 13. 本地 collector 相关文件

- `ahawam_hil_core.py`：`ControlSource.SYSTEM_HOLD = 2`；
- `ahawam_hil_collector.py`：空 action FIFO 时按 15 Hz 写入上一实际 follower target；
- `ahawam_hil_dataset.py`：rich-v3 schema 与 `action_supervision_mask`；
- `validate_hil_dataset.py`：检查 hold 来源、mask、上一目标一致性及 autonomous tick gap；
- `export_hg_corrections.py`：保守的 correction-only 导出，不依赖完整 raw 的 action mask。
- `export_hg_corrections.ps1`：顶部指定 source/output 的无参数 correction-only 导出入口。

训练服务器应把 `recording_context/collector_config.json` 中的 `contract_version` 作为 schema 分流依据。新数据应为：

```text
so101-ahawam-hil-rich-v3
```
