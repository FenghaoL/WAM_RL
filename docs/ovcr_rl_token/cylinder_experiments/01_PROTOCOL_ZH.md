# cylinder-v1：数据、切分、训练预算与实验设置

本轮把 power-adapter v2 已经跑完的那套纠正训练协议，原样搬到 cylinder 插入任务上，并补两处此前缺少的东西：**wandb 在线记录**和**更大的固定验证子集**。方法本身没有新增算法。

## 1. 数据实测（2026-09-08 冻结）

`runs/ovcr_adapter_cylinder_v1/data_manifest.json` 是本轮唯一的数据事实来源，记录每个 parquet 的 sha256、每条 rollout 的 HUMAN 片段长度和 episode 划分。

| 项目 | 数值 |
|---|---|
| demo 训练数据 | `data/so101_ahawam_cylinder_insertion_v1/01_pick_up_and_insert_cylinder` 的 4 个采集 session，94 条 episode，19,708 帧 |
| demo 验证数据 | `20260904_024009_val`，661 帧（与任务 SFT 的 val 完全一致） |
| DAgger 原始数据 | `data/ahawam_hil_raw/so101_ahawam_cylinder_hil_v1`，5 个 session，80 条 rollout，36,355 帧 |
| 其中人类接管 | 3,067 帧、94 个连续 HUMAN 片段，占原始数据 8.4% |
| HUMAN 训练集 | 53 条 rollout，77 段，2,393 帧 |
| HUMAN 验证集 | 13 条 rollout，17 段，674 帧 |
| 片段长度分位数 | 最短 10、25% 18、中位 24.5、75% 38.75、最长 150 帧 |

HUMAN 划分方式与 padapter v2 相同：按 session 内 raw episode 做 sha256(`20260908:<session>:<episode>`) 排序，取前 20% 作为验证，和模型/优化器 seed 无关。80 条 rollout 中有 14 条没有任何有效 HUMAN 帧，只能进入统计、不进入任一集合。

只使用 `control_source==HUMAN & intervention & human_action_valid` 且时间连续的行；自主运行段和 SYSTEM_HOLD 不作为动作标签，也不跨接管边界拼历史。

**注意：** HIL 的最后一个 session（`20260908_020614`）文件写入时间是 09-08 03:07，就在本轮冻结之前。如果之后还会继续采集，新数据不会自动进入本轮；那需要重新生成 manifest 并作为新一轮实验，而不是往正在跑的配置里加数据。

## 2. 与 padapter v2 的量级差别

| 指标 | padapter v2 | cylinder v1 |
|---|---|---|
| demo 训练帧 | 15,984 | 19,708 |
| HUMAN 训练帧 | 4,437 | 2,393 |
| 自然混合中 HUMAN 占比 | 21.7% | 10.8% |
| 1:1 采样下 HUMAN 重复次数 | 约 3.6 遍 | 约 8.2 遍 |
| 满足 49 帧完整视频窗口的 HUMAN 起点 | 538 | 271（训练集） |

**这条必须先说清楚**：cylinder 的纠正数据只有 padapter 的一半，而 demo 更多，所以同样的“1:1 + 一遍 demo”预算下，HUMAN 起点会被重复约 8 遍，过拟合风险高于上一轮。因此本轮每个 P1 实验保存 3 个 checkpoint（1095/2190/3285），并把验证频率提高到每 219 步一次，用于观察 HUMAN 验证损失何时开始变差，而不是只看最后一个权重。如果确实看到过拟合，下一轮再加 3:1 或自然混合的单因素对照，而不是现在同时改比例和其他条件。

## 3. 五个实验

全部固定：3 张 GPU（默认 0/1/2，GPU3 留给部署）、每卡 batch 4、accumulation 1、全局 batch 12、bf16、DeepSpeed ZeRO2 + CPU optimizer offload、AdamW（weight_decay 0.01、grad clip 1.0）、cosine 调度 5% warmup、seed 42、只保存权重不保存 optimizer state。

| 实验 | 初始化 | 可训练部分 | 采样 | anchor | lr | 步数 | 保存 |
|---|---|---|---|---|---|---|---|
| `p1_v32_balanced` | cylinder SFT `step_006572` | V32 adapter（2,058,560 参数） | 1:1 | 1.0 | 1e-4 | 3285 | 1095/2190/3285 |
| `p1_q32_balanced` | 同上 | Q32 adapter（2,744,640） | 1:1 | 1.0 | 1e-4 | 3285 | 同上 |
| `p1_t4_balanced` | 同上 | T4 adapter（5,511,360） | 1:1 | 1.0 | 1e-4 | 3285 | 同上 |
| `p1_full_balanced_video` | 同上 | 全量（video/action DiT、原 OVCR、观测编码器；VAE 冻结） | 1:1 | 0 | 1e-5 | 3285 | 同上 |
| `b_full_pretrained_natural_4ep` | `checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt` | 全量 | 自然混合 | 0 | 1e-5 | 7368（4×1842） | 每轮各一个 |

- 3285 步 = 1:1 采样下 demo 起点正好过一遍（19,708 / 6）。这与 padapter 轮用的规则相同，所以两个任务的 P1 预算定义一致。
- 1842 步 = 自然混合一个 epoch（(19,708+2,393)/12）；四轮即每个起点约看 4 次，不是 1:1 的 8 遍。**两组预算不同，不能直接当作等算力对比。**
- 三个 adapter 冻结整个原模型和原 OVCR，`lambda_video=0`，up 层零初始化，起点输出等于 P0；另加同噪声同时间步的 demo P0 保持损失（anchor，λ=1，不作用于 HUMAN 标签）。
- 两个 full 实验 `lambda_video=1`，且 HUMAN 只有“当前起点的 13 张采样帧全部有效”的窗口才计算视频损失（`video_supervision_require_full_window=true`），训练集中只有 271 个这样的起点，其余 HUMAN 窗口只监督动作。
- 只有 B 使用 `checkpoint_shape_adapt: true`，因为它要从预训练的 14 维动作空间适配到 SO101 的 6 维，和原任务 SFT 的初始化方式一致；其余四个从已适配好的 cylinder SFT 严格加载。

## 4. 验证与 wandb

每次验证都在两个固定、互不重叠的集合上分别计算，并且关闭 anchor 正则，只报监督损失：

- `demo_*`：`20260904_024009_val`，661 帧，与任务 SFT 用的验证集完全相同；
- `human_*`：13 条 held-out rollout 的 674 个 HUMAN 帧。

子集固定为 seed 42 的置换切片，因此曲线变化来自模型而不是采样。本轮把每次验证的样本量从上一轮的 12 提到 **48 个 loss 样本/来源**，动作推理样本从 3 提到 **12 个/来源**（10 步 flow 推理），P1 每 219 步、B 每 461 步验证一次，外加训练前 step 0 的 P0 基线和训练结束后的一次。

wandb：project `ahawam-so101-cylinder-hil-v1`，group `cylinder-v1-ovcr-correction`，run 名即实验名，`mode=online`。训练曲线在 `train/*`，验证在 `eval/demo_*` 和 `eval/human_*`，其中 `eval/demo_loss_action`、`eval/human_loss_action`、`eval/demo_action_l1`、`eval/human_action_l1` 是判断“纠正是否学到、原任务是否退化”的主要两组量。上一轮 padapter 的 launcher 强制 `WANDB_MODE=disabled`，本轮改为可由调用方覆盖，队列启动脚本显式设 `online`。

**跨实验比较只能用分项，不能用总损失。** `eval/val_loss` 和 `eval/demo_loss` 是总损失，adapter 实验 `lambda_video=0`，总损失恒等于动作损失；两个 full 实验 `lambda_video=1`，总损失里还含视频重建项。实测 step 0（同一个 `step_006572`）：三个 adapter 的 val_loss 都是 0.16940，`p1_full_balanced_video` 是 0.28006 = 动作 0.16973 + 视频 0.11034，动作 L1 分别为 2.2979 与 2.2967 —— 起点相同，差的就是视频项。剩余约 0.2% 的动作损失差异属于数值层面（可训练参数集合不同，ZeRO-2 混合精度处理路径不同），未逐张量定位。同理 `eval/human_video_supervised_fraction` 在 adapter 是 0.625、full 是 0.1458，是因为 full 额外要求完整 49 帧窗口，而 adapter 的这个数字不进入损失。

单次验证仍然是小样本：48 个样本足以看趋势，不足以宣称显著差异；**验证损失下降不等于真机成功率提高**，最终仍要在同样起始位姿下做配对真机测试。

## 5. 队列与命令

```bash
# 生成 manifest 和五个配置（已执行；重复执行会因拒绝覆盖而报错）
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/prepare.py

# 只查数据，不需要 GPU
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/v2_experiments/run.py \
  --config runs/ovcr_adapter_cylinder_v1/configs/p1_v32_balanced.yaml --data-only

# 三卡两步工程检查，输出到独立 _smoke 目录
WANDB_MODE=online GPU_IDS=0,1,2 scripts/so101_pipeline/ovcr_adapter/v2_experiments/launch.sh \
  --config runs/ovcr_adapter_cylinder_v1/configs/p1_v32_balanced.yaml --smoke-steps 2

# 正式排队（tmux 会话 ovcr_cylinder_v1，顺序执行，一项失败即停）
bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_suite.sh

# 只跑其中一部分
NAMES="p1_full_balanced_video b_full_pretrained_natural_4ep" \
  bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_suite.sh

# 查看队列和最新验证
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/ovcr_adapter_cylinder_v1
```

队列复用 padapter 轮已经跑通的 `suite.py`/`launch.sh`/`run.py`，只是加了 `--run-root` 和 wandb 开关；`prepare.py` 是本任务新增的配置生成器。队列有进程锁、PID、独立日志，拒绝覆盖已有 output 目录；停止方式是向 `runs/ovcr_adapter_cylinder_v1/suite.pid` 发 SIGTERM，或 `tmux kill-session -t ovcr_cylinder_v1`。权重不含 optimizer state，中断后不能声称精确恢复训练。

## 6. 产物

- `runs/ovcr_adapter_cylinder_v1/data_manifest.json`：冻结的数据事实。
- `runs/ovcr_adapter_cylinder_v1/configs/*.yaml`：五个完整配置；每个 run 的 `model/experiment_config.yaml` 是启动时快照。
- `runs/ovcr_adapter_cylinder_v1/logs/<name>.log`：逐步 loss、grad norm、耗时。
- `runs/ovcr_adapter_cylinder_v1/<name>/model/validation.jsonl`：两个来源分开的验证记录。
- `runs/ovcr_adapter_cylinder_v1/<name>/model/checkpoints/weights/step_*.pt`：完整 AHA 权重（约 14.7GB/个，adapter 实验额外带 adapter 元数据），可直接被现有 action server 加载。

部署时使用 cylinder 的 stats `runs/so101_ahawam_cylinder_v1/assets/dataset_stats.json` 和指令 `Pick up the cylinder and insert it into the matching hole.`，不要与 power-adapter 的 stats/指令混用。
