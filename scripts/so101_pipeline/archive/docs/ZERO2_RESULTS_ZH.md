# SO101 G05 Square V1 实验记录

记录日期：2026-08-07（America/Detroit）

## 1. 数据审计

原目录包含 11 个录制根，每个根有原始机械臂坐标版和 `_g05_model_frame` 版。训练采用后者。

| 任务 | Episodes | Frames |
|---|---:|---:|
| Pick up the white block. | 33 | 5,097 |
| Pick up the paper roll. | 31 | 3,808 |
| Put the paper roll on the pink bowl. | 30 | 5,470 |
| Put the white block on the paper roll. | 31 | 6,381 |
| 合计 | 125 | 20,756 |

数据规格：15 Hz、6D action、6D observation state、固定相机 480×480、腕部相机 480×640、AV1 视频、LeRobot v3.0。

固定随机种子 42 做 episode 级 95%/5% 划分：

- 训练：118 episodes、19,570 frames
- 验证：7 episodes、1,186 frames
- 验证集覆盖全部 4 个任务，不是从同一条轨迹随机抽帧，因此没有同轨迹泄漏

## 2. 格式适配结果

仓库 vendored LeRobot 读取器使用 v2.1 的“一条 episode 一个 parquet/视频路径”布局；原数据的 v3 布局将多条 episode 合并在文件中。已生成零重编码 v2.1 兼容视图：

```text
data/so101_g05_square_v1_ahawam_v21
```

完整性检查结果：11 sources、125 episodes、20,756 frames、4 tasks、250 个 episode-camera 符号链接。抽查 episode 0、1、62、124 均能从共享 AV1 MP4 的正确绝对时间解码，得到固定相机 `[3,480,480]` 和腕部相机 `[3,480,640]`。

转换依据与官方格式设计一致：LeRobot v3 使用关系元数据从多 episode parquet/MP4 中恢复单条轨迹，而 v2 是 episode-per-file。[官方 LeRobot v3 文档](https://huggingface.co/docs/lerobot/lerobot-dataset-v3)

## 3. 模型输入验证

训练和验证各抽查首、中、末样本，六个样本全部通过：

```text
video   [3, 13, 384, 320]
action  [48, 6]
proprio [48, 6]
context/context_mask 已缓存且数值有限
action_offset 随机位于 [0, 7]
```

两个相机先各自缩放到 240×320，然后固定相机占组合画面顶部 256×320，腕部相机占底部 128×320，最终送入模型的画面是 384×320。

## 4. 初始化与可训练部分

初始化 checkpoint：

```text
checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt
```

该权重来自 14D 双臂数据，而 SO101 是 6D。`checkpoint_shape_adapt=true` 的实测行为：

- Video expert、Action expert 中形状兼容的时序/注意力参数和 OVCR 相关参数正常加载。
- `action_encoder.weight`、`action head.weight`、`action head.bias` 因 14D/6D 不同而保持 6D 新初始化。
- proprio encoder 的输入权重通过 adaptive average pooling 从 14D 适配为 6D。

本实验设置 `freeze_video_dit=true`：冻结 5.00B Video expert，训练 1.02B Action expert、未被冻结的 MoT/OVCR 路径和额外的 proprio/视觉条件模块。这不是 LoRA，也不是全量微调。

选择这个 baseline 的原因：数据只有 125 条轨迹，先保护视频世界先验更稳妥；同时 Action/OVCR 路径仍有很大容量来学习 SO101 的 6D 动作语义。AHA-WAM 论文将模型描述为低频 Video DiT 世界规划器和高频 Action DiT 执行器，并用 offset training/OVCR 处理异步观测；当前时间参数和冻结边界都保留了这条核心路径。[AHA-WAM 论文](https://arxiv.org/abs/2606.09811)

## 5. 正式超参数

| 参数 | 值 |
|---|---:|
| GPUs | 3 × A100-SXM4-40GB |
| DeepSpeed | ZeRO-2 |
| precision | bf16 |
| micro batch/GPU | 1 |
| gradient accumulation | 4 |
| effective global batch | 12 |
| epochs | 4 |
| steps/epoch | 1,631 |
| total optimizer steps | 6,524 |
| optimizer | AdamW, betas=(0.9,0.95) |
| learning rate | 5e-5 |
| scheduler | 5% warmup + cosine |
| warmup steps | 326 |
| weight decay | 1e-2 |
| gradient clipping | 1.0 |
| Video DiT | frozen |
| gradient checkpointing | disabled |

ZeRO-2 会分片 optimizer state 和 gradient，但每卡仍保留模型参数；这与实测每卡约 37.4 GiB 相符。官方阶段定义见 [DeepSpeed ZeRO 文档](https://deepspeed.readthedocs.io/en/stable/zero3.html)。

W&B 只由 rank 0 初始化一个 run，符合其分布式训练建议，避免三进程各建一个 run。[W&B 分布式日志文档](https://docs.wandb.ai/models/track/log/distributed-training)

## 6. 三卡冒烟实测

运行：`smoke-3gpu-zero2-20260807-142747`

```text
exit_code: 0
OOM: false
optimizer steps: 8
peak GPU memory: 37414 / 37412 / 37412 MiB
GPU 3: 14 MiB，未参与
observed speed: 0.21 optimizer step/s
observed sample rate: 0.63 sample/s（日志按每次 optimizer step 乘 micro-batch×GPU 计）
timing per micro step: data 0.128s, forward 0.432s, backward 0.555s
```

这证明 3 卡足够，不需要按备选方案改成 4 卡。显存余量约 3.5 GiB/卡，不建议在当前方案上增加 batch、解冻 Video DiT 或提高分辨率。

## 7. 正式训练状态

运行目录：

```text
runs/so101_g05_square_v1/formal-3gpu-zero2-4ep-20260807-143159
```

W&B：<https://wandb.ai/fenghaoliu8-university-of-michigan/ahawam-so101-g05-square/runs/vu1lpihj>

已确认：

- DeepSpeed 报告 `zero_stage=2`、`world_size=3`
- 14D 到 6D checkpoint 适配符合预期
- W&B 已在线创建并同步
- 已进入 `max_steps=6524` 的正式优化器循环
- 第 5 步：loss 2.1541、action loss 1.8717、video loss 0.2824
- 启动时 ETA 约 8 小时 28 分，checkpoint/验证写盘会增加少量时间

checkpoint 会出现在：

```text
model/checkpoints/weights/step_001631.pt
model/checkpoints/weights/step_003262.pt
model/checkpoints/weights/step_004893.pt
model/checkpoints/weights/step_006524.pt
```

对应的可恢复 DeepSpeed optimizer/scheduler 状态位于 `model/checkpoints/state/step_*`。单独的 `.pt` 权重适合推理；训练意外中断后应优先用同一步的 `state/step_*` 恢复。

## 8. 结论与下一步判断

当前方案适合作为 SO101 的第一个可解释 baseline：冻结 Video DiT，不做 LoRA，直接训练原生 Action/OVCR 条件路径。它已经通过真实数据、真实反向传播和在线 W&B。

训练结束后不要只看总 loss。应比较 4 个 epoch checkpoint 的验证 action L1/L2、生成视频指标，并做真机离线回放：先打印反归一化和 G05 逆变换后的 6D 指令，再限速、限幅、短 chunk 上电。若模型动作学习不足，再依次尝试：增加 SO101 轨迹、训练更久、对 Video DiT 加小范围 LoRA；不建议第一步就做全量 Video DiT 微调。
