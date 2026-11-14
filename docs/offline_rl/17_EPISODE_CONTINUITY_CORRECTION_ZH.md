# 2026-10-26：暂停不拆 episode，修正历史与转移时间轴

用户确认采集时接管前后的完全暂停没有保存为视频帧，整个 episode 应连续使用。此前把 `tick_index` 跳号当作记录中断的处理不符合这批数据的语义。本次已修正代码并生成新的 CPU 转移数据；**尚未启动新的 GPU cache 导出或重训**。现有三项模型及价值视频仍属于旧协议。

## 原因与原始证据

`frame_index` 是 episode 中保存的帧序号；`tick_index` 是采集循环计数，两者不是同一时间轴。采集配置 `recording_context/collector_config.json` 中有 `control.pause_frames_stored: false`，暂停期间 tick 会增加，但不会新增保存帧。

全量 103 episode 的 250 处 tick 跳号，其相邻保存帧的时间区间内，全部可以匹配 `pause_requested`、leader 对齐、`human_control_started/stopped` 或 `policy_resume_requested` 等暂停/接管事件，没有无法由这些事件解释的跳号。例如 `20260911_183057` episode 0 的保存帧 98→99，tick 为 112→161，间隔约 3.33 秒，事件明确记录暂停、leader 对齐、进入人工控制。这不应清空任务历史。

另有 12 个保存行的 `dropped_tick` 为真，它们相邻的 tick 仍均为逐一递增。仅凭这个标记不能判断视频缺行。新协议将该标记与原始 tick 保留为审计信息，不据此切断任务时间轴。完整跳号列表、事件名称及这些标记行见 [continuity_audit_20261026.json](continuity_audit_20261026.json)。

## 修正后的约定

- `timeline_mode=episode_frames` 为新准备数据的默认值。历史、offset 均按同一个 episode 的保存帧序号取样，不因接管、恢复或 tick 跳号重置。
- planner anchor 为 `floor(frame_index / 8) * 8`，history 是 anchor 之前每隔 48 个保存帧取一帧，最多 6 帧。48 帧约为 3.2 秒保存视频时长，跨暂停时实际墙钟间隔可以更长。
- 构建转移继续只去掉 SYSTEM_HOLD 的动作，保留 POLICY＋HUMAN 的实际动作。可以跨过暂停和控制权切换，取自然往后的 8 个保留动作；末端观察仍是第 8 个动作对应原始行之后的第一张保存画面。
- 暂停本身不生成额外动作或奖励、不导致 terminal，也不清空历史；折扣继续按保留动作数计算。原始 `duration_seconds` 和 `duration_ticks` 仅记录实际时间跨度。
- 成功前移至 HOLD 第一帧、自然动作块可跨成功、成功窗口只奖励不自举、9 条提前结束录像排除、不同 episode 不拼接等约定保持执行。

这是按用户确认的任务语义压缩暂停时间；恢复后的观察可能有机械臂跟踪收敛，不能据此宣称暂停两端像素或关节读数完全相同。

## 实际影响与完成状态

| 项目 | 旧 strict_ticks | 新 episode_frames |
|---|---:|---:|
| 全部可训练转移 | 16,554 | 18,345 |
| 训练集转移 | 16,016 | 17,732 |
| 验证集转移 | 538 | 613 |
| 成功奖励窗口 | 535 | 580 |
| 能贡献成功奖励窗口的 episode | 79 | 81 |
| 全量观察含至少一帧 history | 68.60% | 88.68% |
| 全量观察含完整 6 帧 history | 9.15% | 34.73% |
| 训练起点含至少一帧 history | 62.79% | 88.05% |
| 训练起点含完整 6 帧 history | 7.90% | 34.01% |

旧协议的转移 metadata 和全部张量已逐项精确复现。新协议增加 1,791 条转移，没有删除旧起点；恢复了之前被过滤的两个成功事件。43,680 个观察中，23,545 个的 planner anchor 或 history 列表改变。新列 history 数字是 **CPU 计算出的取帧覆盖率，不是已经重新导出的 cache 实测**。

旧缓存中的 KV 和 reference 已受旧 history 条件化；这些内容不能靠修改 JSON 元数据修复。完成实验修正需要新目录导出 cache，再从采集用 SFT 重新训练 IQL、TD3+BC 和 IQL 加权 Action DiT，并重新生成价值视频。旧结果可保留为旧协议记录，不能说成已经采用连续 episode。全量新 cache 的张量形状、观察数不变，存储量仍约 76 GiB。

## 代码与复现

- `src/ahawam/offline_rl/timeline.py`：共享时间轴规则。`episode_frames` 不因 tick 跳号或标记清空历史/过滤转移；`strict_ticks` 仅供复现旧结果。
- `data.py`：`TransitionConfig.timeline_mode`，三种动作时序使用一致的连续性检查。
- `cache.py`：新数据导出使用 `causal_independent_anchor_v3_episode_frames`；spec 记录时间轴及相关源码 hash；`conditioning_from_spec` 对缺少时间轴字段的旧 spec 使用旧规则。
- `flow_actor.py`：原生 Action DiT 重算条件时读取 cache spec 的时间轴，防止与缓存 history 不一致。
- `value_diagnostics.py`、`value_videos.py`：诊断依训练数据时间轴取动作块/计算参考回报；旧数据继续按旧规则解释。
- `scripts/so101_pipeline/offline_rl/audit_continuity.py`：核对采集事件、精确复现旧转移、生成连续 episode 转移及覆盖率报告。
- `system.py:require_complete`：禁止新时间轴转移静默混用旧时间轴缓存。旧 cache 的源码 hash 校验仍有效，不能绕过校验重新做原生 Action DiT 训练。

本轮已生成的新数据为 `runs/offline_rl/cylinder103_episode_contiguous_20261026/data/transitions.pt`，对应 `audit.json`。运行审计：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 /home/fenghao/miniconda3/envs/ahawam/bin/python \
  scripts/so101_pipeline/offline_rl/audit_continuity.py \
  --original-transitions runs/offline_rl/cylinder103_skip_hold/data/transitions.pt \
  --output runs/offline_rl/cylinder103_episode_contiguous_20261026/data \
  --report docs/offline_rl/continuity_audit_20261026.json
```

后续四卡 cache 导出命令如下，**本轮未执行**：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 /home/fenghao/miniconda3/envs/ahawam/bin/python \
  scripts/so101_pipeline/offline_rl/system.py launch-cache \
  --transitions runs/offline_rl/cylinder103_episode_contiguous_20261026/data/transitions.pt \
  --cache runs/offline_rl/cylinder103_episode_contiguous_20261026/cache \
  --gpus 0 1 2 3
```

40 项 CPU 测试通过，覆盖跨暂停的历史与 offset、跨暂停成功奖励、保存行标记、旧 spec 兼容、诊断窗口与记录回报、新旧时间轴禁止混用，以及已有离线 learner、LoRA、部署测试。GPU 导出与新模型训练尚未验证。
