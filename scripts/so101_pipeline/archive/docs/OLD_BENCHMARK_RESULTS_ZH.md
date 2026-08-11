# AHA-WAM 四卡训练实测与 SO101 方案

## 1. 测试结论

这台服务器的 `4 x A100-SXM4-40GB` 足够做推荐的 SO101 微调，但不够用当前仓库的 ZeRO-2 实现做 6B 专家主干的全量微调。这里的 4 卡结论针对训练；单卡部署只保存前向所需权重和临时张量，不需要梯度与 Adam 状态，两者不矛盾。

推荐配置是：

```text
4 GPU
BF16
DeepSpeed ZeRO-2
每卡 batch_size = 1
gradient_accumulation_steps = 4
freeze_video_dit = true
mot_checkpoint_mixed_attn = false
```

它不是“小动作头微调”。模型日志中的 Video-DiT 约 5.00B、Action-DiT 约 1.02B；此外，AHA 的 `LayerwiseChunkKVCacheEditor` 按当前 30 层、4096/3072 维配置约有 1.228B 参数。冻结 Video-DiT 后，Action-DiT、KV editor、proprio encoder、视觉投影和 query encoder 仍会训练，总规模约为 2.3B 参数级别。

## 2. 测试环境

```text
GPU: 4 x NVIDIA A100-SXM4-40GB
Python: 3.10
PyTorch: 2.7.1+cu128
DeepSpeed: 0.18.7
Accelerate: 1.12.0
precision: bf16
训练数据: RoboTwin 2.0, 6,075,103 frames, 14-D action/state
图像输入: 三相机合成后的 384x320
micro batch: 1/GPU
```

每次基准测试都关闭最终 checkpoint 写盘，GPU 每 0.5 秒采样一次。完整日志、CSV 和机器可读摘要在 `runs/benchmarks/<name>/`。

## 3. 实测结果

| 测试 | 结果 | 峰值显存/卡 | 最后一步计算时间 | 稳态全局吞吐估计 |
| --- | --- | ---: | ---: | ---: |
| 全量，65 帧，4 卡 ZeRO-2，checkpointing | OOM | 38.1-39.4 GiB | 未完成 | 未完成 |
| 冻结 Video，65 帧，1 卡 ZeRO-1，checkpointing | OOM | 39.2 GiB | 未完成 | 未完成 |
| 冻结 Video，65 帧，4 卡 ZeRO-1，checkpointing | 成功 | 33.6-33.9 GiB | 0.830 s | 约 4.82 sample/s |
| 冻结 Video，65 帧，4 卡 ZeRO-1，无 checkpointing | 成功 | 33.2-34.1 GiB | 0.584 s | 约 6.85 sample/s |
| 全量，65 帧，4 卡 ZeRO-3，checkpointing | 代码不兼容 | 19.3 GiB | 未完成 | 未完成 |
| 冻结 Video，97 帧 offset，4 卡 ZeRO-2，checkpointing | 成功 | 32.5-32.7 GiB | 0.983 s | 约 4.07 sample/s |
| 冻结 Video，97 帧 offset，4 卡 ZeRO-2，无 checkpointing | 成功 | 31.2-32.8 GiB | 0.712 s | 约 5.62 sample/s |
| 冻结 Video，97 帧 offset，3 卡 ZeRO-2，无 checkpointing | 成功 | 37.75-37.79 GiB | 约 0.8-0.9 s | 约 3.2-3.8 sample/s |

“最后一步计算时间”是日志中的 `data + fwd + bwd + opt`，不含首次建立数据索引和模型装载。这里只跑了 5-6 步，history length 会随机变化，因此速度是容量规划值，不是严格论文 benchmark。

### 为什么日志里的 speed 更低

trainer 显示的累计 `step/s` 把第一个 batch 的视频解码、worker 预热和 CUDA 预热也算进去。例如最后一组第一步 `data=14.809s`，后续数据等待约为 `0.003s`。判断长训练速度应看后几步的分项时间，而不是 5 步累计平均。

### 全量 ZeRO-2 为什么 OOM

全量训练在第一次 optimizer step 初始化 Adam 状态时失败。此时每卡进程已使用约 37 GiB，又试图申请约 6.85 GiB。减小 gradient accumulation 不会减少参数和 Adam 状态；micro batch 已经是 1，因此继续调 batch 也不能解决这一处 OOM。

### 单卡冻结后为什么仍 OOM

单卡冻结 Video-DiT 后也在 Adam 状态初始化时失败，需要额外约 8.75 GiB，而当时只剩约 5.5 GiB。原因是仍训练约 2.3B 参数级别的 Action/OVCR 路径。四卡 ZeRO-1/2 将这些 optimizer state 分片后才可行。

### 三卡是否可用

三卡 ZeRO-2 已补测通过，三张训练卡峰值分别为 37.752、37.752、37.790 GiB，第四张卡保持空闲。它证明 4 卡不是算法硬性要求，但三卡只剩约 2.7 GiB 显存余量，明显小于四卡，真实 SO101 数据、较长 history 或额外临时 workspace 更容易触顶。默认累积 4 时，三卡有效 batch 为 12，四卡为 16；同配置下三卡吞吐约为 3.2-3.8 sample/s，四卡约为 5.62 sample/s。

### ZeRO-3 为什么失败

ZeRO-3 将参数本身也分片。AHA-WAM 的 MoT 前向直接切片读取 `block.modulation`；在某个 rank 上该参数局部分片长度为 0，随后与 3072 维 timestep modulation 相加，报错：

```text
The size of tensor a (0) must match the size of tensor b (3072)
```

要支持 ZeRO-3，需要在这些直接参数访问附近使用 DeepSpeed gathered-parameter 语义，并重新验证 checkpoint 保存与合并。当前不能把 ZeRO-3 当作一个可直接切换的启动选项。

## 4. 推荐训练时长

默认 `gradient_accumulation_steps=4`，所以：

```text
每个 micro step 的全局样本数 = 4 GPU x 1 = 4
每个 optimizer step 的有效 batch = 4 x 4 = 16
5000 optimizer steps 处理约 80,000 个采样窗口
```

按 97 帧无 checkpointing 的四卡稳态约 5.6 sample/s 粗略估算，5000 optimizer step 的纯训练约 4 小时；三卡约 5-6 小时。每个 optimizer step 内含 4 个 gradient-accumulation micro-batch，所以四卡每 step 处理 16 个窗口，三卡处理 12 个。真实 SO101 数据的视频编码、磁盘缓存、随机 history、checkpoint 写盘会使它变化。第一次先运行 `run_train_smoke.sh` 的 20 step，再开始 5000 step。

不要把 5000 step 理解成 5000 个 episode。一个 episode 会提供许多重叠的 97 帧训练窗口，采样器每一步取的是窗口。

## 5. 为什么先不做 LoRA

当前仓库没有 PEFT/LoRA 的注入、保存和部署加载代码。AHA-WAM 也不只是一个普通 Transformer：Video 3072 维、Action 1024 维，并通过 MoT 和逐层 KV editor 交互。LoRA 必须明确决定加在 Action-DiT、自注意力、交叉专家路径还是 KV editor，并确保 DeepSpeed checkpoint 与推理端能重建同样模块。

现有冻结方案已能在 4 卡稳定运行，所以第一轮用仓库原生路径风险最低。只有出现以下情况，才值得实现 LoRA：

- SO101 数据很少，2.3B 可训练参数明显过拟合；
- 希望降低 checkpoint 或 optimizer 状态体积；
- 以后必须在更少 GPU 上训练；
- 已有可靠验证集可以比较 LoRA 与冻结微调。

## 6. RoboCOIN 权重状态

已下载：

```text
checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt
size: 约 14 GiB
sha256: 525de83bcdfe0bd409241ec8ba760dbc06fcc0fe6a28c4ec99d2bedef236d266
```

这是在 RoboCOIN 双臂 14-D 数据上预训练的下游初始化，不是可以直接向 SO101 发送动作的 zero-shot policy。SO101 配置使用 `checkpoint_shape_adapt=true`：可复用的视觉、语言、时序和交互参数会加载；动作 encoder weight、动作 head weight 和 head bias 因 14-D/6-D 形状不同而保持新初始化，proprio encoder weight 则通过 adaptive average pooling 从 14-D 适配到 6-D。这个 shape adaptation 只解决张量装载，不会自动建立 SO101 关节语义，仍需 SO101 数据学习。

## 7. 最终建议

1. 使用已下载的 RoboCOIN AHA-WAM 权重初始化，不从纯 Wan/ActionDiT 随机任务头开始。
2. 冻结 5B Video-DiT，训练 Action-DiT、KV editor 和 SO101 输入输出相关模块。
3. 优先 4 卡 ZeRO-2、BF16、batch 1、累积 4；默认关闭 checkpointing。三卡也已实测通过，但显存余量和吞吐较差。
4. 若以后增大图像、window 或 batch 导致 OOM，把 `mot_checkpoint_mixed_attn` 改回 `true`。
5. 暂不使用 ZeRO-3，不做全量微调，不直接做真机 zero-shot。
6. 先采集少量 SO101 数据跑 20 step 管道测试，再按任务采集更完整的数据做 5000 step。

## 8. 单卡在线推理实测

使用已下载的 14-D RoboCOIN 权重，通过 `checkpoint_shape_adapt=true` 构造 6-D SO101 policy，并用临时 synthetic 6-D stats 做了不连接硬件的 dry-run。加载日志只跳过预期的三个参数：

```text
mixtures.action.action_encoder.weight  [1024,14] -> [1024,6]
mixtures.action.head.weight            [14,1024] -> [6,1024]
mixtures.action.head.bias              [14] -> [6]
```

单张 A100-40GB 结果：

| 项目 | 结果 |
| --- | ---: |
| 模型加载时间 | 347.3 s |
| 加载后 allocated | 27.886 GiB |
| 首次 Video + Action 请求 | 1.168 s |
| 首次峰值 allocated | 28.242 GiB |
| 后续 Action-only 请求 | 0.536 s |
| 后续峰值 allocated | 28.358 GiB |
| 输出 | `[16,6]`，全部 finite |

`16 / 0.536` 约为 29.9 action steps/s，这是纯 GPU action 生成吞吐，未计 TCP、Windows 图像拼接和机械臂写入。同步模式的视频更新会造成较长停顿；正式追求连续控制时，使用 `run_server_dual_gpu_async.sh` 让一张 GPU 更新视频 state、另一张 GPU 生成动作更合理。两张 GPU 会各自加载一份约 28GB 的完整 policy。

这个测试证明 40GB A100 足够进行单 policy 推理，也证明 14-D 到 6-D 的加载路径工作正常；synthetic stats 和未经过 SO101 微调的 6-D action 层使输出动作没有任务意义，不能用于真机。

## 9. 资料

- [AHA-WAM paper](https://arxiv.org/abs/2606.09811)
- [AHA-WAM repository](https://github.com/serene-sivy/AHA-WAM)
- [AHA-WAM RoboCOIN pretrained checkpoint](https://huggingface.co/SereneC/AHA-WAM-Pretrained)
- [RoboCOIN repository](https://github.com/FlagOpen/RoboCOIN)
- [Accelerate DeepSpeed guide](https://huggingface.co/docs/accelerate/main/usage_guides/deepspeed)
- [PyTorch activation checkpointing](https://docs.pytorch.org/docs/stable/checkpoint)
