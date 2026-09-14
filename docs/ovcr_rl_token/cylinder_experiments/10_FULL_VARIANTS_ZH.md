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
