# SO101 ZeRO-2 Offload 全量微调

这是 `data/so101_ahawam_square_v1` 的正式训练入口。它在 4 张 A100 40GB 上同时
训练 VideoDiT、ActionDiT、MoT/OVCR、AHA history/chunk 模块和 SO101 6D 适配层，
仅把 Adam optimizer state 放到 CPU。文本编码器和 VAE 仍是冻结的特征模块。

## 当前训练定义

| 项目 | 值 |
| --- | --- |
| 初始化 | 官方 AHA 完整权重，再覆盖 FastWAM 共享主干 |
| 原始数据 | 4 个 LeRobot v2.1 session，79 episodes，14,886 frames，15 Hz |
| 训练/留出 | 固定 seed 42，73 episodes / 13,745 frames 用于训练，其余留出 |
| 相机 | fixed 放上方；wrist 放左下；缺失的右腕相机用黑色补齐 |
| 时序窗口 | 49 个 15 Hz action 时刻；每 4 个时刻取一张视频帧，共 13 帧 |
| 动作 | 6D，40-step horizon，8-step chunk，随机 offset 0..7 |
| 并行 | 4 GPU，ZeRO-2，optimizer CPU offload，bf16 |
| batch | 每卡 4，accum 1，effective batch = 4 x 4 x 1 = 16 |
| 训练长度 | 860 optimizer steps/epoch，4 epochs，共 3,440 steps |
| 学习率 | `1e-5`，cosine decay，5% linear warmup，AdamW |
| 保存 | step 860/1720/2580/3440 各保存一份可部署 `.pt` |

旧数据 tuning bench 在同一台机器、同一模型结构和上述 batch 下实测约
2.72 samples/s、31.96 GiB/卡稳态显存。新数据按样本数估算约 5.6 小时，再加模型
初始化与四次 checkpoint I/O；这是实测吞吐外推，不是本次完整四轮的完成计时。

## 为什么可以用 FastWAM 初始化

FastWAM checkpoint 提供经过机器人数据训练的 5B VideoDiT、ActionDiT 主体和 14D
proprio encoder，比只从 Wan2.2 视频模型初始化多了机器人动作先验。但它不是原生
AHA-WAM checkpoint。精确审计发现 FastWAM 比官方 AHA checkpoint 少 302 个 MoT
键、1,227,634,718 个参数，几乎全部属于 chunk KV editor；再算顶层 observation
模块，FastWAM-only 会令约 1.329B AHA 参数从随机初始化开始。

这不适合仅有 79 episodes 的四轮小数据训练。因此 Trainer 使用两阶段初始化：先加载
官方 AHA-WAM checkpoint 作为完整骨架，再让 FastWAM 覆盖结构和 shape 完全相同的
6,020,688,078 个 MoT 参数以及 proprio encoder；FastWAM 不包含的键会保留第一阶段
的官方 AHA 权重。两次加载都发生在 DeepSpeed 创建 fp32 master weights 之前。加载到
当前 6D AHA-WAM 时仍有以下预期 adaptation：

1. 14D action encoder/head 的 3 个参数因 shape 不同而保留为新的 6D 参数。
2. proprio encoder 的输入维度由 14D 自适应到 6D。
3. AHA 专属的 branch embedding、chunk KV editor、visual projection 和 query
   encoder 来自官方 AHA checkpoint，随后在 SO101 上继续全量更新。

这个两阶段初始化适合作为“FastWAM 共享主干 + 官方 AHA 异步模块 + SO101 全量微调”的
初始化，不适合不训练就直接部署。它仍是一次 transfer experiment：FastWAM 主干和
官方 AHA chunk editor 并非联合训练所得，四轮训练的作用之一就是重新对齐它们。为
降低小数据全量训练破坏预训练表示的风险，学习率使用保守的 `1e-5`，而不是冻结
VideoDiT 时使用的 `5e-5`。

## 文件作用

| 文件 | 作用 |
| --- | --- |
| `settings.yaml` | 唯一的模型、数据、batch、epoch、LR 和 checkpoint 配置入口 |
| `ds_zero2_offload.json` | ZeRO-2 optimizer CPU offload 的 DeepSpeed 配置 |
| `accelerate.yaml` | Accelerate 如何启动 DeepSpeed 四进程 |
| `prepare_data.sh` | 计算新数据 stats，并生成两个任务的 UMT5 embedding |
| `preflight.py` | 检查四个数据集、6D schema、stats、embedding 和 checkpoint |
| `train_entry.py` | 计算精确 epoch 边界并创建仓库原生 AHA-WAM 模型 |
| `zero2_trainer.py` | 每 epoch 只保存可部署权重，避免巨大的 optimizer state |
| `smoke_test.sh` | 用正式配置跑 3 个 optimizer steps，默认关闭 W&B/保存 |
| `run_train.sh` | 前台执行正式四轮训练，并记录 GPU/RAM/train log |
| `start_training.sh` | 后台启动正式训练 |
| `status_training.sh` | 查看最新任务 PID 和训练日志 |
| `stop_training.sh` | 停止最新任务的整个进程组 |

`zero2_trainer.py` 只改变保存策略，不改变 loss、forward、backward 或 optimizer 更新。
默认不保存 DeepSpeed Adam state，因为旧 ZeRO-2 实测每轮 state 约 47 GiB；四轮加可
部署权重会占约 231 GiB。当前方案每轮只写约 11 GiB 的 `.pt`，但中断后只能从权重
重新开始 optimizer/scheduler，不能精确续接 Adam 动量。

## 第一次准备

通常只需运行一次：

```bash
bash scripts/so101_pipeline/full_finetune_zero2_offload/prepare_data.sh
```

它不会复制或转码视频。当前数据已经是 LeRobot v2.1，训练器直接联合读取四个
session。输出是：

```text
runs/so101_ahawam_square_v1/assets/dataset_stats.json
data/so101_ahawam_square_v1_text_embeds_cache/*.pt
```

stats 必须来自本批 SO101 轨迹，不能复用 RoboTwin/FastWAM 的 14D stats。embedding
按完整 prompt 的 SHA256 命名，缓存内容是 `[128,4096]` UMT5 context 和 mask。

## 启动训练

先确认四张卡每张至少有 36,000 MiB 空闲。其他任务即使在本脚本开始后才占卡，也会
导致 ZeRO-2 初始化或稳态训练 OOM。

```bash
bash scripts/so101_pipeline/full_finetune_zero2_offload/smoke_test.sh
bash scripts/so101_pipeline/full_finetune_zero2_offload/start_training.sh
```

查看状态：

```bash
bash scripts/so101_pipeline/full_finetune_zero2_offload/status_training.sh
```

停止：

```bash
bash scripts/so101_pipeline/full_finetune_zero2_offload/stop_training.sh
```

正式输出位于：

```text
runs/so101_ahawam_square_v1/full_zero2_offload/formal-*/
├── train.log
├── gpu.csv
├── ram.csv
├── wandb/
└── model/checkpoints/weights/step_*.pt
```

W&B project 是 `ahawam-so101-square-v1`。`train/loss_video`、
`train/loss_action`、总 loss、grad norm、LR 和各阶段耗时都由仓库 Trainer 上报。

## 修改配置

通常只编辑 `settings.yaml`。最常用的是：

```yaml
batch_size: 4
gradient_accumulation_steps: 1
learning_rate: 1.0e-5
num_epochs: 4
base_init_checkpoint: checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt
init_checkpoint: checkpoints/FastWAM-SharePoint/step_137500.pt
freeze_video_dit: false
```

若改 batch/accum，`train_entry.py` 会自动重算每轮 step 和保存点。若改数据目录或任务
文字，要重新运行 `prepare_data.sh`。若希望精确恢复 optimizer，可将
`zero2_save_training_state` 改为 `true`，但要先为四轮额外预留约 188 GiB。

## 2026-08-13 启动记录

- 数据准备成功：79 episodes / 14,886 frames / 2 prompts，资产检查 `status=ready`。
- FastWAM-only 审计发现约 1.329B AHA 参数缺失，因此正式配置改用上述两阶段初始化。
- 首次四卡短测正确进入 FastWAM shape adaptation 和 ZeRO-2 初始化。
- 第一次短测尚未进入 optimizer step：17:10:06 有四个独立 root 任务在短测启动后
  抢占每卡 15.6-17.2 GiB，DeepSpeed 分配 13.69 GiB flatten buffer 时外部竞争导致
  OOM。这次结果不能用于判断 `batch_size=4` 是否可行；旧 tuning bench 的无竞争
  `p1-bs4` 已完整通过 40/40 steps。
- 四个占用任务的父进程都是 `run_dataset.py --runs 10`，会在每张卡连续启动十轮，
  因此本次没有冒险启动正式 W&B 任务。待这些父任务停止或全部结束后，应重新运行
  `smoke_test.sh`；看到三个有限 loss/optimizer step 后再运行 `start_training.sh`。

### 最终成功启动

- 外部任务结束后，正式配置的 3-step smoke test 完整通过；三个 step 的总 loss
  分别为 `2.7977`、`3.9588`、`3.0285`，无 OOM、NaN 或数据错误。
- 正式任务于 22:05 启动：
  `formal-fastwam-zero2offload-bs4-acc1-4ep-20260813-220551`。
- 日程为 73 个训练 episodes、13,745 个训练 frames、每卡 batch 4、全局 batch 16、
  每轮 860 optimizer steps、四轮共 3,440 steps；每 860 steps 保存一次权重。
- ZeRO-2 初始化报告的 CUDA 最大 allocated memory 为 28.69 GB；实际训练监控的
  稳态显存约 32.46 GiB/卡，主机内存约使用 271 GB。
- 第 5 到 43 step 的稳态墙钟吞吐为约 2.42 samples/s。旧 `p1-bs4` 的 2.72
  samples/s 是同样去掉前四个预热 step 的口径；当前任务启用了在线 W&B，且是新的
  数据运行，因此不要把 trainer 早期含冷启动的约 2.2 samples/s 直接与 2.72 比较。
- W&B run：
  `https://wandb.ai/fenghaoliu8-university-of-michigan/ahawam-so101-square-v1/runs/ezvv7qjh`。
