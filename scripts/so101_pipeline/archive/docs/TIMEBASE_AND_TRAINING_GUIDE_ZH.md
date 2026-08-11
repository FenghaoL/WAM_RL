# AHA-WAM 时间轴、异步执行与微调策略详解

本文针对当前仓库和本机实测结果，解释 `num_frames`、FPS、offset、action chunk、模型时间轴、14-D 到 6-D 适配、多卡训练、冻结 Video-DiT、LoRA 和 DeepSpeed ZeRO。

## 1. 先给结论

1. `97 个时刻` 不等于 `97 张图全部送进 Video-DiT`。当前配置从 97 个同步时刻中每隔 8 步取一张主视频帧，因此 Video-DiT 主序列实际看到 13 帧；状态和动作仍按原始离散时间步组织。
2. 30 Hz 不是 RoboTwin 的帧率。本机 RoboTwin 2.0 的 `meta/info.json` 写的是 50 FPS；30 FPS 是当前 SO101 管道的目标数据频率。
3. “只有十几帧”必须区分两种情况：
   - 若是每秒约 15 帧，即 15 FPS，可以训练，但当前 `97/80/16/15/8` 的物理时长会全部翻倍，通常不应原样使用。
   - 若一整个 episode 总共只有十几帧，则连一个 97 时刻窗口都没有。loader 虽可在 episode 尾部 padding，但这样的监督几乎没有意义，应重新采集更长 episode。
4. offset 是训练时随机改变动作网格相对 planner 起点的相位，不是测出来的网络延迟，也不是相机和关节时间戳不同步的补丁。
5. 单张 A100 足够做当前 policy 的推理，不代表单卡足够训练。训练还要保存梯度、反向激活和 Adam optimizer state。
6. 4 卡是本机已经验证成功、速度也更好的训练配置，不是算法硬性要求。3 卡 ZeRO-2 也已实测通过；有效 batch 会从 `4 x 1 x 4 = 16` 变成 `3 x 1 x 4 = 12`，且显存余量明显变小。
7. 官方 AHA-WAM 主训练联合优化视频和动作分支。冻结 Video-DiT 是针对 SO101 小数据与 4x40GB 显存做的工程折中，不是论文唯一正确方案。
8. 当前最合理的第一阶段仍是：RoboCOIN checkpoint 初始化，冻结 Video-DiT，训练 Action-DiT、OVCR/KV editor 和 SO101 接口；建立 episode 级验证集后，再决定是否对 Video-DiT 做部分解冻或 LoRA。

## 2. frame、FPS、timestep 和 episode 分别是什么

### 2.1 timestep

一个 timestep 是数据表中的一个离散时刻。理想情况下同一个 timestep 包含：

```text
t 时刻的 fixed 图像
t 时刻的 wrist 图像
t 时刻的 6-D observation.state
t 时刻对应的 6-D action
```

这里“对应”要求采集端用同一套时间基准组织数据。offset training 不会修复摄像机晚 200 ms、关节状态早 100 ms 这类采集同步错误。

### 2.2 FPS / Hz

FPS 表示每秒记录多少个 timestep。30 FPS 约等于每 `1/30=0.0333` 秒记录一次；15 FPS 约等于每 `0.0667` 秒一次。

数据频率决定“若干离散步”对应多少真实时间：

```text
真实时长 = 离散间隔数 / FPS
```

97 个采样点之间只有 96 个间隔，所以窗口首尾时长是 `(97-1)/FPS`，不是严格的 `97/FPS`。

### 2.3 episode

一个 episode 是一次完整示范，例如“从开始位置抓起方块并放入盒子”。若 30 FPS 下录制 10 秒，则约有 300 个 timestep。episode 总帧数与 FPS 是不同概念。

## 3. 当前 97/80/16/15/8 配置到底取了什么

当前 SO101 配置为：

```yaml
num_frames: 97
action_horizon: 80
action_chunk_size: 16
max_action_offset: 15
action_video_freq_ratio: 8
```

对一个从 `t=0` 开始的数据样本：

### 3.1 原始窗口

dataset 请求 `t=0..96` 共 97 个 observation/state 时刻，以及 `t=0..95` 共 96 个 action。

### 3.2 主视频序列

Video-DiT 主序列取：

```text
0, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96
```

共 13 个视觉时刻。代码还会读取 action chunk 起点处的当前图像供 OVCR 使用，但不能把它理解成 Video-DiT 对 97 张图做完整长视频计算。

### 3.3 action horizon 与 chunk

`action_horizon=80` 表示一次训练目标包含 80 个动作，分成：

```text
chunk 0:  0..15
chunk 1: 16..31
chunk 2: 32..47
chunk 3: 48..63
chunk 4: 64..79
```

即 5 个 chunk。论文附录报告的 RoboTwin 主配置是 64 actions / 16 actions，即 4 个 chunk；仓库的 offset task 配置为了给随机偏移留空间，使用了 97 帧窗口和 80 动作目标。讨论本地运行时应以当前配置和代码为准，不能只照抄论文表格。

### 3.4 为什么需要 97

随机 offset 最大为 15，80 步动作目标最晚要从 `t=15` 开始：

```text
最晚动作范围 = 15..94
```

97 个 observation 提供了 `0..96`，96 个 action 提供了 `0..95`，因此装得下。代码明确要求：

```text
action_horizon + max_action_offset <= num_frames - 1
80 + 15 <= 96
```

它还要求 action horizon 能被 chunk size 整除，并要求视频长度满足 Wan VAE/时序 token 化约束。

## 4. 不同 FPS 下，同一组数字代表的真实时间

| 数据频率 | 97 时刻首尾 | 80 action horizon | 16-step chunk | 15-step max offset | ratio=8 后视觉频率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 FPS | 1.92 s | 1.60 s | 0.32 s | 0.30 s | 6.25 Hz |
| 30 FPS | 3.20 s | 2.67 s | 0.53 s | 0.50 s | 3.75 Hz |
| 15 FPS | 6.40 s | 5.33 s | 1.07 s | 1.00 s | 1.875 Hz |

所以 15 FPS 数据不是“不能用”，而是继续用 chunk 16 会让机器人一次开环执行约 1.07 秒，Video-DiT 也只得到约 1.875 Hz 的画面。对需要及时纠偏的 SO101，这通常太慢。

若最终实测只能稳定在 15 FPS，一个保持当前物理时长的起始候选是：

```yaml
num_frames: 49
action_horizon: 40
action_chunk_size: 8
max_action_offset: 7
action_video_freq_ratio: 4
```

这组数字满足当前代码约束，并大致保持 3.2 秒原始窗口、2.67 秒 action horizon、0.53 秒 chunk、0.47 秒 offset 和 13 个主视频时刻。但它只是合理的待测配置，不是官方公布的 SO101 最优参数。改变 chunk 和时间标定后必须重新 smoke test、微调并让部署客户端严格执行 8 步 chunk。

不要简单复制图像把 15 FPS 文件伪装成 30 FPS。重复帧不会创造新观测，只会让模型看到静止重复画面。更重要的是动作、状态、图像和部署控制频率必须一致。

## 5. offset 是什么

### 5.1 为什么固定对齐会出问题

设慢 planner 在 `t=0` 完成长视野规划。快 executor 需要持续输出短动作块。真实部署时，planner 与 executor 各自运行，planner 完成新上下文的时间不一定刚好落在 action chunk 边界。

若训练永远只让动作从 planner 的 `t=0` 开始，模型只学过一种相位关系：

```text
planner start: |0--------------------------|
action chunks: |0..15|16..31|32..47|...
```

部署时 planner 结果可能在动作周期中间到达，例如 executor 当前实际处于相对 planner 起点的第 7 步。模型若从未见过这种相位，可能错误使用 planner context。

### 5.2 当前代码怎么随机偏移

每个训练样本独立采样：

```text
delta ∈ {0, 1, ..., 15}
```

若这次 `delta=7`，80 个动作监督变成：

```text
7..22, 23..38, 39..54, 55..70, 71..86
```

对应的 chunk 起点当前观测位于：

```text
7, 23, 39, 55, 71
```

主 planner 视频窗口仍提供长时域上下文，OVCR 则使用每个 chunk 起点的最新视觉观测和 proprio，把缓存的 planner K/V 上下文改造成当前 chunk 专用的上下文。

### 5.3 offset 不是什么

- 不是“固定延迟 15 帧”。每个样本随机取 0 到 15。
- 不是把 action 标签整体永久错开。
- 不是相机和状态不同步的容错器。
- 不是在线动态选择 horizon。
- 它的单位是数据 timestep。15 step 在 30 FPS 是 0.5 秒，在 15 FPS 是 1 秒。

论文把它称为 planner-executor phase offset。因为 chunk size 为 16，覆盖 `0..15` 就覆盖了一个 chunk 周期中的所有离散相位。参见 [AHA-WAM 论文第 3.3 节](https://arxiv.org/html/2606.09811)。

## 6. 异步执行具体怎么工作

### 6.1 慢循环：Video-DiT planner

慢循环接收视觉观测和历史，运行昂贵的约 5B Video-DiT，形成每一层的 planner K/V context。rolling K/V memory 还保存最近若干次 planner refresh 的历史信息。

这个 context 不是一段已经解码出来、要在屏幕播放的视频，而是 Transformer 内部表示“场景可能如何演化”的隐藏特征。

### 6.2 快循环：Action-DiT executor

快循环在每个 action update 时取得：

```text
最新图像
最新 6-D proprio
最新可用的 planner context
文字指令 embedding
```

OVCR 先用当前图像构造少量 query，读取并修正旧 planner context；约 1B Action-DiT 再基于该 context 去噪出一个 action chunk。

### 6.3 为什么叫 asynchronous

两条循环不要求同时开始、同时结束：

```text
planner:  [------计算新 context------]       [------下一次------]
executor: [chunk0][chunk1][chunk2][chunk3][chunk4][chunk0]...
```

executor 总是消费“当前最新可用”的 planner context。新 context 尚未完成时，它继续使用上一份并由 OVCR 根据最新观测进行适配。offset training 就是在训练中模拟这种相位不固定的情况。

### 6.4 当前公开部署代码的实际代价

本项目的双 GPU 异步服务不是把 7.23B 模型一半放 GPU0、一半放 GPU1，而是两张卡各加载一份完整 policy，然后分别承担 planner 和 action 请求。优点是两条流真正解耦，缺点是约 28GB 常驻显存会在两卡各出现一次。

单 GPU 同步部署也能运行，但 Video-DiT refresh 会阻塞 action 热路径。本机公开 PyTorch 路径实测首次 Video + Action 约 1.17 秒，后续 action-only chunk 约 0.54 秒。论文的 41.37 ms 使用 TensorRT、CUDA graph 和热路径优化；当前公开脚本不能直接等同于论文优化结果。

## 7. “模型内部时间轴”包含哪几种时间

至少要分清四种完全不同的时间：

### 7.1 物理时间

墙上时钟的秒，由真实采集间隔和机器人控制周期决定。模型不会自动知道采集端是否掉帧，必须由数据与部署管道保持一致。

### 7.2 数据序列位置

这是 `t=0,1,2...` 的 timestep index。动作 token 必须知道自己代表 horizon 中第几个动作；视频 token 也要知道自己来自哪个视觉时刻。RoPE 等位置编码用于区分这些顺序。

若不提供位置，Transformer 会很难区分“先抓后抬”和“先抬后抓”，因为它看到的只是 token 集合。

### 7.3 action chunk index 与 history

推理 state 中的 `next_chunk_index` 从 0 递增。当前实现一次生成 chunk 16 后，会把 action history 长度增加 16，并把下一请求当作下一个 chunk。

因此服务端若只返回前 8 步、客户端执行 8 步并丢弃后 8 步，模型内部却已经按 16 步推进，会造成内部序列位置与真实机器人时间错位。当前 30 FPS 配置必须执行完整 16 步；若要 8 步闭环，应使用 chunk size 8 重新配置和训练。

### 7.4 diffusion / flow timestep

代码里的 `timestep_action`、`timestep_video` 还有另一层含义：它们表示去噪过程中的噪声强度，不是机器人动作发生在第几秒。

可以把它理解为“当前这份被加噪的 action/video 离干净目标还有多远”。模型在多个 flow timestep 上学习速度场，推理时再用 10 个去噪步骤逐渐得到 action。不要把 diffusion timestep 与物理 timestep 混淆。

## 8. 14-D 到 6-D shape adaptation

### 8.1 Linear 层是什么

线性层可写成：

```text
y = W x + b
```

RoboCOIN action 是 14 维，SO101 action 是 6 维，所以直接连接动作向量的矩阵形状不同。

### 8.2 实际跳过的三块张量

```text
action_encoder.weight: [1024, 14] -> [1024, 6]
action head.weight:    [14, 1024] -> [6, 1024]
action head.bias:      [14]       -> [6]
```

“跳过三个参数”是 PyTorch `state_dict` 的三个命名张量，不是只丢了三个浮点数。

- action encoder 把每步 14-D/6-D action 投影成 1024 维隐藏表示。
- action head 把 1024 维隐藏表示还原成每步 14-D/6-D action。
- head bias 是每个输出维度的偏置。

这三块是机械臂动作接口最直接的部分。SO101 版本保留新初始化值，必须靠 SO101 数据学习。

### 8.3 proprio encoder 的行为不同

proprio encoder 把 robot state 投影到 4096 维文本/上下文空间。其权重由 `[4096,14]` 变成 `[4096,6]`。当前 loader 没有跳过它，而是用 adaptive average pooling 沿 14 个输入维压缩成 6 维。

这只是数值形状适配。RoboCOIN 的“左臂 shoulder”和 SO101 的“单臂 shoulder”并没有天然一一对应，pooling 不会理解关节语义。所以即使加载日志很干净，也不能把输出直接发给真机。

### 8.4 复用了什么

绝大多数同形状参数仍被加载，包括 Video-DiT、Action-DiT 中间 Transformer 层和 OVCR/KV editor。可迁移的是视觉、语言、物体交互和时序表示；新机械臂动作接口与动力学仍需适配。

## 9. 为什么单卡可部署，而训练推荐四卡

### 9.1 推理需要什么

推理主要需要：

```text
模型权重
当前前向激活
KV/context cache
```

本机实测单 policy 约 27.89 GiB 常驻，峰值约 28.36 GiB，因此一张 40GB A100 能装下。

### 9.2 训练额外需要什么

训练还需要：

```text
反向传播所需激活
每个可训练参数的 gradient
Adam 的一阶矩 exp_avg
Adam 的二阶矩 exp_avg_sq
实现相关的 FP32 master parameter
通信 bucket 和临时 workspace
```

冻结 Video-DiT 后也仍有约 2.3B 可训练参数：约 1.02B Action-DiT，加约 1.22B rolling memory/context routing，再加 proprio 和 query/projection 模块。这不是“小动作头微调”。

本机单卡冻结方案在初始化 Adam state 时 OOM；4 卡 ZeRO-1/2 能把 optimizer state 或 gradient 分片后才通过。全量 4 卡 ZeRO-2 也在第一次 optimizer step 初始化 Adam state 时 OOM。

### 9.3 3 卡和 4 卡的区别

模型算法不要求 GPU 数必须是 2 的幂，所以 3 卡可以启动分布式训练。需要关注三件事：

1. 每卡分到的 optimizer/gradient state 比 4 卡更多，可能 OOM。
2. 默认有效 batch 从 16 变 12：`3 GPU x batch 1 x accumulation 4`。
3. 三卡总吞吐通常低于四卡；通信分片也可能不完全均匀。

本机补测的三卡 ZeRO-2、97 帧 offset、冻结 Video-DiT、无 checkpointing 成功跑完 6 个 optimizer step，峰值为 37.75-37.79 GiB/训练卡；第四卡保持空闲。后期吞吐约 3.2-3.8 sample/s。它可以用，但相比四卡约 31.2-32.8 GiB 和 5.62 sample/s，显存余量与速度都更差。

若想让 3 卡有效 batch 接近 16，可把 accumulation 设为 5，得到 15，但一次 optimizer update 要多跑一个 micro-step。无需为了整数 16 强行调整，先用验证结果判断。

部署方面：单 GPU 同步 policy 已通过；当前双流异步服务需要两 GPU，各放一份完整 policy。训练卡数与部署卡数是两套问题。

## 10. 一个 epoch 到底多久

### 10.1 epoch 的定义

一个 epoch 表示 sampler 大致遍历一次 dataset index。当前 LeRobot loader 的长度接近所选 episode 的总帧数，而不是“不重叠 97 帧窗口数”。每个 frame index 都可作为未来窗口起点；episode 尾部不够长时会 padding，`skip_padding_as_possible` 会尝试换样本，但不能把短 episode 变成长。

所以不要把一个 epoch 理解成“每个 episode 只训练一次”。同一 episode 会产生大量高度重叠的未来窗口。

### 10.2 计算公式

设训练集总 index 数为 `N`：

```text
有效 batch = GPU 数 x 每卡 batch x gradient accumulation
optimizer steps / epoch ≈ ceil(N / 有效 batch)
epoch 时间 ≈ N / 稳态全局 sample/s
```

当前 4 卡冻结 Video、97 帧、ZeRO-2、无 checkpointing 的短测约为 5.62 sample/s，因此：

- 30,000 个 frame index：一个 epoch 约 `30000/5.62=1.48` 小时。
- 15,000 个 frame index：一个 epoch 约 0.74 小时。
- 4 卡下 5000 optimizer steps、有效 batch 16：采样约 80,000 个窗口，纯训练约 4 小时。
- 3 卡下有效 batch 12、实测较慢：5000 optimizer steps 粗略约 5 到 6 小时。

启动、首批视频解码、checkpoint 写盘和真实数据 IO 会增加时间。

本机 RoboTwin 训练 split 约 601 万 index。按当前 4 卡小 batch 跑完整一 epoch 约需 12 天，因此用 4 卡复现论文的大规模 RoboTwin 训练并不现实。论文使用 global batch 512，硬件规模和吞吐假设不同。

### 10.3 推荐多少 epoch

官方 RoboTwin 主实验报告 5 epochs、global batch 512、AdamW、学习率 `1e-4`、5% warmup；RoboCOIN checkpoint model card 报告预训练 45k optimization steps。这些数字不能直接移植为小型 SO101 数据集的最优 epoch。

对你的 SO101，建议按 step 和验证结果管理：

1. 20 step smoke test：只证明数据、前向、反向、optimizer 和保存路径能走通。
2. 500 到 1000 step 初始基线：观察 loss、动作分布和是否快速过拟合。
3. 2000 到 5000 step 主实验：每 250 到 500 step 做验证并保存可比较 checkpoint。
4. 以 held-out episode 的 action loss、离线动作误差和安全真机 rollout 为停止依据，而不是机械地追求 5 或 50 epochs。

验证集必须按 episode 划分，不能把同一 episode 的重叠窗口同时放入 train 和 val，否则会数据泄漏。当前 SO101 YAML 的 `data.val=null` 适合先打通管道，不适合最终模型选择。

## 11. 官方是否冻结 Video-DiT

### 11.1 可以确认的事实

- AHA-WAM 主论文定义联合 flow-matching loss `L_action + lambda * L_video`，并说明两个分支共同优化时视频与动作损失等权。
- 论文主实验将 Video-DiT 初始化为 Wan2.2-TI2V-5B，报告约 4.99B video planner、1.02B Action-DiT 和 1.22B memory/context routing。
- 官方仓库通用训练配置默认 `freeze_video_dit: false`。
- 论文明确说在 AHA-WAM-Flash 的 ODE distillation 阶段冻结 Video-DiT；这是蒸馏，不是主模型 embodiment fine-tuning。
- RoboCOIN checkpoint model card 说明两种模型在选定 RoboCOIN 子集上训练 45k steps，但没有单独公开该次训练的 freeze flag。不能仅凭 model card 断言所有细节。

因此，主论文训练路径不是“官方默认冻结视频主干”。当前 SO101 的冻结建议是显存和小数据场景下的本地策略。

### 11.2 Fast-WAM 能告诉我们什么

Fast-WAM 消融中，去掉 video co-training 后：

```text
RoboTwin average: 91.8 -> 83.8
LIBERO average:   97.6 -> 93.5
真实毛巾折叠任务下降到 10% success
```

这强烈支持“视频预测监督对 WAM 表示学习有价值”。参见 [Fast-WAM 论文](https://arxiv.org/html/2603.16666)。

但“从头/主训练时去掉视频目标”和“已经在 Wan/RoboCOIN 上学过视频表示后，下游小数据阶段冻结 Video-DiT”不是同一个实验。冻结仍保留预训练视频表示和 planner context，只是不让约 5B 主干继续适应 SO101 相机域。

## 12. 是否值得冻结、部分解冻或 LoRA

### 12.1 第一阶段为什么建议冻结

- 当前 4xA100-40GB 的全量 ZeRO-2 已实测 OOM。
- SO101 初始数据量预计远小于 RoboTwin/RoboCOIN，全量更新 7.23B 容易过拟合和遗忘。
- 14-D 到 6-D 最急需学习的是动作接口、Action-DiT、proprio 和 OVCR 对新相机/新机械臂的映射。
- 冻结方案已经实测能训练，能最快建立一个可比较基线。

代价是 Video-DiT 无法主动适应固定相机、腕部相机、工作台背景和 SO101 运动外观。若失败主要来自视觉域差异，永久冻结可能成为上限。

### 12.2 推荐的分阶段实验

1. `freeze_video_dit=true`，训练原生 Action/OVCR 路径，建立 baseline。
2. 建立 10% 到 20% episode 级验证集，并记录真实任务成功率。
3. 若训练动作 loss 能下降但换物体、光照、位置就失败，优先尝试 Video-DiT 上层 block 的部分解冻或 LoRA。
4. 若连训练场景动作都学不会，先检查 action/state 单位、相机拼接、FPS、normalization 与数据质量，再考虑扩大模型更新范围。
5. 数据达到每任务上百条高质量且有明显视觉多样性后，才值得投入全量或大范围 Video-DiT 微调。

### 12.3 LoRA 最合理放哪里

当前仓库没有现成 LoRA。若实现，较合理的混合策略是：

- 6-D action encoder/head、proprio encoder、Action-DiT、OVCR 继续全量训练。
- 在 Video-DiT 的 attention `q/k/v/o` projection，优先上层若干 block，插入 rank 8 或 16 的 LoRA。
- VAE 和文本编码器继续冻结。

只给 action head 做 LoRA 不合适，因为 14-D 到 6-D 的输入输出层本来就是新初始化，应该直接训练。给所有 7B 模块盲目加 LoRA也不一定更省计算，且会增加 checkpoint 保存、恢复、合并和部署加载工作。

实现 LoRA 需要处理：目标层选择、PEFT 或原生模块注入、optimizer 只收 trainable 参数、DeepSpeed checkpoint、推理时重建同样结构或 merge 权重。这是中等规模工程工作，不是一个 YAML 开关。

## 13. ZeRO-1、2、3 是什么

ZeRO 的目标是把 data parallel 各 GPU 上重复保存的模型状态分片。它不会自动把算法改成另一个模型。

| 模式 | optimizer state | gradient | model parameter |
| --- | --- | --- | --- |
| 普通 DDP / ZeRO-0 | 每卡完整 | 每卡完整 | 每卡完整 |
| ZeRO-1 | 分片 | 每卡完整 | 每卡完整 |
| ZeRO-2 | 分片 | 分片 | 每卡完整 |
| ZeRO-3 | 分片 | 分片 | 分片，前向时按需 gather |

官方定义参见 [DeepSpeed ZeRO 文档](https://deepspeed.readthedocs.io/en/stable/zero3.html)。

### 13.1 ZeRO-3 能否让四卡全量训练

从内存原理和本机失败前约 19.3 GiB/卡的占用看，修复兼容性后很有希望让 4x40GB 承载全量训练。但“参数能装下”不等于完整首步一定通过，还必须实测 forward activation、backward、通信 workspace 和 optimizer step。

### 13.2 当前为什么失败

AHA-WAM MoT 的 forward 会直接读取并切片 `block.modulation` 参数。ZeRO-3 可能让当前 rank 暂时只持有该参数的局部分片，甚至当前局部 tensor 长度为 0；普通 `nn.Module.forward` 参数会被 DeepSpeed 自动 gather，但这种模块外直接参数访问不一定被正确识别，于是出现 0 维与 3072 维相加错误。

修复方向包括：

- 在直接访问处用 `deepspeed.zero.GatheredParameters`；
- 更稳妥地把直接参数运算封装进拥有该参数的 module forward；
- 审计所有类似 `.weight`、`.modulation` 的直接访问；
- 正确聚合 ZeRO-3 权重保存，否则 `state_dict` 可能只是分片 placeholder；
- 验证冻结、resume、部署单文件 checkpoint 和数值一致性。

这不需要改变 AHA-WAM 的数学算法，但也不是安全的一行修改。它属于分布式参数生命周期和 checkpoint 机制的中等工程改造。

### 13.3 其他全量方案

1. ZeRO-2 optimizer offload 到 CPU：最可能直接缓解当前 Adam 初始化 OOM，但会消耗大量主机 RAM 和 PCIe/NVLink-CPU 传输时间。本机是共享服务器，需先检查系统 RAM。
2. ZeRO-3 加 CPU/NVMe offload：显存更省，通常更慢，checkpoint/磁盘压力更大。
3. FSDP full shard：原理类似 ZeRO-3，仍需处理直接参数访问和分片 checkpoint，不是自动绕过当前问题。
4. activation checkpointing：减少 activation，但本机全量失败点是 Adam state 初始化，因此它没有解决该次 OOM。
5. 8-bit optimizer：可减少 optimizer state，需额外引入和验证 bitsandbytes/DeepSpeed 组合及数值稳定性。
6. Adafactor：optimizer state 更省，但属于优化算法变化，AHA-WAM 没有公开下游证据。
7. 部分解冻或 Video LoRA：对小型 SO101 数据通常比为全量训练投入 ZeRO-3 改造更有性价比。

## 14. 针对你的推荐顺序

### 14.1 先确认采集频率含义

查看 SO101 `meta/info.json` 的 `fps`，并检查真实 timestamp 间隔和视频 `avg_frame_rate`。先确认你说的“十几帧”是约 15 FPS，还是 episode 总长度只有十几帧。

若能稳定 30 FPS，继续使用当前 `97/80/16/15/8`。若只能稳定 15 FPS，使用第 4 节候选作为起点并重新测试，不要混合旧的 15/18/30 FPS episode。

### 14.2 训练路线

```text
RoboCOIN AHA-WAM checkpoint
  -> 6-D shape adaptation
  -> 冻结 Video-DiT 的 20-step smoke
  -> 500/1000-step baseline
  -> episode-level validation + 安全 dry-run
  -> 2000/5000-step 主实验
  -> 根据失败类型决定 Video partial unfreeze / LoRA
```

四张卡都可用时，优先沿用已验证的 4 卡 ZeRO-2。三卡也已通过短测，但显存、吞吐和有效 batch 都不如四卡。单卡用于同步推理；追求 planner/executor 真异步时，当前实现使用两卡完整副本。

### 14.3 真机前必须满足

- action/state 六维顺序、单位、正负方向一致。
- 训练和在线固定/腕部相机的 resize、拼接顺序、RGB/BGR 一致。
- stats 来自同一批 SO101 训练数据。
- 采集 FPS 与部署执行周期一致。
- chunk 完整执行，或按新 chunk size 重新训练。
- 首次部署先限速、限关节范围、空载、急停可用，并人工确认输出，不使用未微调的随机 6-D 动作层。

## 15. 主要资料

- [AHA-WAM paper](https://arxiv.org/html/2606.09811)
- [AHA-WAM RoboCOIN pretrained model card](https://huggingface.co/SereneC/AHA-WAM-Pretrained)
- [Fast-WAM paper](https://arxiv.org/html/2603.16666)
- [DeepSpeed ZeRO documentation](https://deepspeed.readthedocs.io/en/stable/zero3.html)
