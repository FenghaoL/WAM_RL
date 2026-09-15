# full balanced video 三个单变量实验：视频延伸 / 动作延伸 / demo:DAgger = 2:1

2026-09-13。用户决定：暂停其他实验，重点只跑 `p1_full_balanced_video`；取代 09 文档里的 A/B 方案。
三个实验都以 `p1_full_balanced_video_hilv2`（原 cylinder SFT 出发、hil_v2 人类接管、历史跨界跳停顿 48 帧、1:1、3285 步）为基线，
**每个只改一处**。

## 1. 起点过滤：按用户定义评估后不启用

用户的定义：只删除"完全静止或只有一两帧在动"的样本。主臂读数最小变化一格约 0.07–0.09°，"在动"取任一关节变化 > 0；
窗口内在动的帧间步数 ≤2 则排除。

| 实验 | 被排除 | 其中窗口本身只剩 1–3 帧（接管段末尾起点） | 窗口较长但确实静止 |
|---|---:|---:|---:|
| 视频延伸（动作仍在接管段内） | 15.3% | 363 | 167（**4.8%**） |
| 动作延伸 | 2.1% | 36 | 36（1.0%） |

真正"长窗口却完全不动"的样本只有约 5%，其余是接管段最后几帧、本就几乎全是 padding 的样本。按用户指示不加过滤，并据此判断
**卡住的主因不是静止帧**。之前按"所有关节变化 <1°"得到的 32% 把按格的缓慢微调也算成了静止，口径不同。

更可能的主因（09 文档）：抓取对准接管段里人从未合爪（0/62），而动作窗口截断在接管段末尾，模型在对准位姿上看不到"接着合爪"。

## 2. 三个实验

| 顺序 | 实验 | 相对基线的唯一改动 |
|---|---|---|
| 1 | `p1_full_balanced_video_hilv2_vidext` | 纠正样本的**未来视频**延伸到接管之后（跳过 SYSTEM_HOLD，只在成功 rollout 上）；动作标签仍只用人类段 |
| 2 | `p1_full_balanced_video_hilv2_actext` | 视频**和动作/状态**都沿同一时间线延伸（交还后模型的合爪动作进入监督） |
| 3 | `p1_full_balanced_video_hilv2_demo2to1` | 不延伸；每个全局 batch 从 6 demo + 6 纠正改为 **8 demo + 4 纠正** |

- 视频延伸后，纠正起点中能拿到完整 13 张图视频监督的比例从基线的约 4.5% 升到约 55%（成功 rollout、跳停顿）。
- 2:1 保持 3285 步（与基线同算力），因此 demo 约过 1.33 遍、纠正约 3.8 遍（基线分别是 1.0 和 5.7 遍）。
- 验证集（demo val + 人类接管 val）全部与基线相同，`demo_*`、`human_*` 曲线可直接对比。

## 3. 2:1 采样器怎么实现

每卡 batch 4、3 卡，单卡内无法凑出 2:1。`GlobalRatioTwoSourceEpochSampler`（`samplers.py`）在**全局 batch** 上保证精确比例：
每步 12 个样本里 8 demo + 4 纠正，其中一张卡分到 2 个纠正、另外两张各 1 个，**哪张卡多拿一个每步轮换**；每卡 batch 大小不变，
因此各卡梯度权重一致。两个来源都是打乱后不放回抽取，用完再重新打乱。配置项 `sampling_strategy: balanced_2_1`。
单元测试覆盖：每步 8:4、每卡 [1,1,2] 组成且轮换、一个周期内 demo 不重复；并在真实数据集上复核。

## 4. 队列

GPU **0/1/2**（部署服务已由用户移到 GPU 3）。第一次按 0/1/3 启动时 GPU 3 已被部署服务占用，`launch.sh` 显存检查未通过（exit 3，未产生任何训练）；该记录归档为 `suite_status_gpu3_busy_*.json`、`logs/vidext_attempt_gpu3_busy.log`，随后在 0/1/2 上重新启动。tmux 会话 `ovcr_cylinder_hilv2_full`，
按上表顺序执行，每个约 8.5 小时。

```bash
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/ovcr_adapter_cylinder_hil_v2_full
kill -TERM "$(cat runs/ovcr_adapter_cylinder_hil_v2_full/suite.pid)"      # 停止
```

生成与验证：

```bash
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/prepare_hil_v2_full.py
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/check_future_window.py \
  --config runs/ovcr_adapter_cylinder_hil_v2_full/configs/p1_full_balanced_video_hilv2_actext.yaml --samples 12
```

## 5. 本轮停掉/取消的内容

- `p1_v32_balanced_hilv2_raw48`、`_skip72`：上一轮为"历史间隔"问题额外加的 V32 对照。raw48 已跑完；skip72 因 GPU 2 被部署服务占用而未能启动。
- 09 文档的 A/B 变体：被本方案取代，其配置目录已删除（未产生训练结果）。

部署这些权重时：`num_history_frames: 6`、cylinder stats、原任务指令。

## 6. 追加：人类段当 demo（2026-09-14 排队）

用户要求：纠正样本只用人类段本身，历史和未来视频都只在段内取，不够就按 demo 规则 padding / 不算视频 loss，不取接管前历史、不做延伸。

核查结论：hil_v2 上**没有训练过**这个设置。已有的 `p1_full_balanced_video_hilv2`、`_vidext`、`_actext`、`_demo2to1`
训练时都 `correction_history_crosses_segment: true`。但 ICRA 2027 会话已为此生成过配置
`runs/icra2027_cylinder_controls/configs/full_context_truncated.yaml`（09-13，`plan.json` 标记未训练）：与基线唯一差别
是 `data.train.correction_history_crosses_segment: false`，未来视频本来就不延伸，正是这个设置。为避免重复配置，直接训练它，
输出也落在 ICRA 文档预期的位置 `runs/icra2027_cylinder_controls/full_context_truncated/model`（最终 `step_003285.pt`，
wandb project `ahawam-so101-cylinder-hil-v1`，group `icra2027-context-controls`）。

- 历史不跨界时 skip-hold / stride 不起作用（只在跨界时间线上用），因此不必另改。
- 验证集规则不变：验证纠正样本仍跨界取历史，与本文其他实验 val loss 可比。
- data-only 长度 `[19708, 3460, 661, 169]`，与 ICRA 文档记录一致。

队列：tmux `ovcr_segdemo_chain`（GPU 0/1/2），在 power adapter `p1_full_balanced_video_hilv3` 完成之后启动，结束后接着跑
power adapter 的 `_segdemo` 和 `_demo2to1`（见 [padapter_hil_v3](../padapter_hil_v3/README_ZH.md)）。

```bash
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/icra2027_cylinder_controls
```
