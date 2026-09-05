# AHA-WAM-Flash ODE 蒸馏：从直觉、数学到本项目代码

更新日期：2026-09-03

## 1. 先给结论

当前训练做的是一种**动作生成 ODE 的少步采样蒸馏**。它属于扩散/流策略蒸馏的标准技术路线，也与 AHA-WAM 论文和仓库发布的 `ahawam_ode` 实现一致；但它不是 LLM 中最常见的 token-logit 蒸馏，也不是用 teacher 自主生成大量新机器人场景的数据蒸馏。

更准确地说：

- teacher 是已经在目标 SO101 数据上完成 SFT 的 step-5328 AHA-WAM；
- student 与 teacher 架构相同，并从 teacher 权重完整复制初始化；
- Video-DiT 冻结，只调整 student 的动作去噪路径及与它相连的可训练模块；
- 对每个数据窗口，teacher 从新的高斯动作噪声出发运行 16 步；
- student 学习从 teacher ODE 轨迹的若干中间点直接逼近 teacher 的最终动作；
- 部署时 student 用 2 步，而不是 teacher 的常规多步采样。

因此它蒸馏的是**计算过程**：把“沿动作流场走许多小步”压缩为“两次大步”。它不压缩参数量，也不旨在增加插入技能。student checkpoint 仍然很大，但每个 action chunk 的 Action-DiT 前向次数显著减少。

## 2. 为什么这与普通 SFT 不一样

### 2.1 普通动作 flow-matching SFT

令：

- $c$：图像、语言、机器人状态以及 AHA-WAM planner/OVCR context；
- $a$：demo 中的 GT action chunk；
- $\epsilon\sim\mathcal N(0,I)$：随机动作噪声；
- $x_t$：由 $a$、$\epsilon$ 和噪声时间 $t$ 构造的带噪动作。

SFT 训练的是局部流场 $v_\theta(x_t,t,c)$。监督信号来自 GT 动作和噪声所确定的 flow-matching target。模型学会在 ODE 的当前位置给出局部速度，推理时数值求解器反复调用模型：

$$
x_{k+1}=x_k+\Delta_k v_\theta(x_k,t_k,c).
$$

模型即使把局部速度学得很好，用 10 或 16 个小步积分也可能很准；但直接改成 1–2 个大步会产生明显离散化误差。普通 SFT 没有直接要求“从高噪声处一步跳到最终 teacher 样本”。

### 2.2 当前 ODE 蒸馏

teacher 从同一初始噪声 $x_T^0=\epsilon$ 出发，做 16 次确定性的 flow/ODE 更新，得到：

$$
x_T^0,x_T^1,\ldots,x_T^{16}.
$$

代码保存锚点：

$$
K=\{0,1,2,4,8,12,16\}.
$$

对 40-step 动作 horizon 的每个 8-step chunk，独立从非终点集合
$\{0,1,2,4,8,12\}$ 中采一个锚点 $k_j$。student 接收 teacher 的中间状态
$x_{T,j}^{k_j}$、对应时间和相同条件 $c$，预测 flow，再把它换算为终点动作估计：

$$
\hat x_{S,j}^{16}=x_{T,j}^{k_j}-\sigma_{k_j}
v_S(x_{T,j}^{k_j},t_{k_j},c).
$$

训练目标是：

$$
\mathcal L_{ODE}=
\frac{1}{N_{valid}}
\sum_{j,u\in valid}
\left\|\hat x_{S,j,u}^{16}-x_{T,j,u}^{16}\right\|_2^2.
$$

这里 $j$ 是动作 chunk，$u$ 是 chunk 内动作位置；episode 尾部 padding 由 GT 数据的 `action_is_pad` mask 排除。

关键点是 target 为 **teacher 的最终生成动作** $x_T^{16}$，不是 GT 动作 $a$。GT action tensor 在这条损失中主要提供 horizon、offset 对齐和 valid/padding mask；demo 的图像、历史、chunk 观测、语言及 proprio 则构成条件 $c$。

对应代码：

- teacher 16-step rollout 和新高斯噪声：[ahawam.py](../../src/ahawam/models/wan22/ahawam.py#L1904)
- 随机 chunk anchor、teacher terminal target、MSE：[ahawam_ode.py](../../src/ahawam/models/wan22/distill/ahawam_ode.py#L117)
- flow 到终点动作的换算及 1/2/4-step 路径：[base_student.py](../../src/ahawam/models/wan22/distill/base_student.py#L50)

## 3. 两步部署究竟怎样运行

发布的锚点网格是 `0,1,2,4,8,12,16`。代码把 2-step 路径固定为：

$$
0\rightarrow8\rightarrow16.
$$

第一次 Action-DiT 前向从纯噪声附近预测一个能跨越大区间的 flow，并把 latent 更新到第 8 步位置；第二次前向从第 8 步直接得到终点动作。1-step 使用 $0\rightarrow16$，4-step 使用 $0\rightarrow4\rightarrow8\rightarrow12\rightarrow16$。

训练时之所以还采 1、2、12 等位置，是为了让同一个 student 在 ODE 的更广范围内学会面向终点的流，并支持不同 NFE 的推理路径。锚点在索引上不是等间隔的，靠近 noisy end 的 `0,1,2` 更密，因此均匀采 anchor 候选本身就形成了对高噪声区域的偏置。AHA-WAM 论文也明确说其重点是让 student 能从高噪声状态做激进的大步预测。

## 4. 数据是否只是“又做了一次 SFT”

不是，但用户的质疑有一半是对的。

### 4.1 蒸馏新增了什么信息

同一个真实观察窗口每次被读取时都会重新采样：

- 初始动作高斯噪声；
- teacher 由该噪声诱导出的完整 ODE 轨迹；
- 五个 action chunks 各自的 anchor。

所以 supervision 在**动作生成噪声空间和 ODE 时间维度**上是动态、稠密的。它不是给一个 observation 永远缓存一个 teacher action。更重要的是，student 直接学习 teacher 的终点映射，而不是继续拟合只适合小步积分的局部 GT flow。

当前 SO101 训练集有 98 个 episode、17,719 帧，形成 15,984 个重叠训练窗口。5 epochs 对应 79,920 次窗口访问；每个窗口有 5 个独立采 anchor 的动作 chunk，约为 399,600 个 chunk-anchor 监督实例。后一个数字不等于独立环境状态数，但说明监督也绝不是只有 98 个固定 teacher 标签。

### 4.2 蒸馏没有新增什么信息

它没有凭空新增：

- 新插座位置；
- 新抓取失败状态；
- 新接触模式；
- teacher 从未见过的视觉/动力学状态；
- 比 teacher 更正确的插入动作。

也就是说，它只在现有 observation/support 上把 teacher 的动作生成器压缩得更快。若 teacher 在某些初始化下本来就抓偏或插不准，蒸馏没有机制知道这些行为是错的。student 偶尔可能因正则化产生小幅变化，但实验假设不应是“student 会系统性超过 teacher”。

### 4.3 为什么少量任务数据仍可能够用

student 不是随机初始化后重新学习插插头。它首先复制 teacher 的全部权重，然后只需改变动作 flow，使其允许更大的数值步长；这是一个比从零学习策略小得多的问题。因此使用 teacher 原来的任务数据是合理且常见的。

但“可能够用”不能由 train loss 单独证明。旧两卡 run 的平均蒸馏 loss 从第一 epoch 约 0.0159 降至第二 epoch 约 0.0127，说明优化有效，却不能证明两步真机成功率保持。最终必须比较固定测试分布上的 teacher 与 1/2/4-step student 成功率。

## 5. 为什么不直接自己生成无限多输入

要区分两个正交维度：

1. **如何获得条件 $c$**：真实 demo、机器人 autonomous rollout、DAgger 状态、仿真或生成视频。
2. **在给定 $c$ 上监督什么**：GT action、teacher logits、teacher action sample、teacher ODE 中间状态或 RL return。

LLM 蒸馏也不必生成新 prompt；“用模型合成大量 prompt/response”是 synthetic-data generation 加 distillation，而不是蒸馏定义本身。AHA-WAM-Flash 选择的是“真实离线 observation + 在线生成 teacher ODE 轨迹”。

随机合成 action noise 很安全，因为它属于 teacher 已定义的生成过程；随机合成视觉场景则不安全，必须保证物理一致性、相机与机器人状态对应，并且 teacher 在该状态上可信。可扩展版本可以加入 autonomous/DAgger observation，再由 teacher 生成动作 target，并做成功/置信度过滤，但那已经是新的数据蒸馏实验，不是官方 AHA-WAM-Flash 配方。

## 6. 它是不是“标准做法”

### 6.1 是标准的技术家族

扩散与 flow policy 的少步蒸馏通常利用“迭代生成过程可视为概率流 ODE”这一点，让 student 从 ODE 中间状态更快到达同一终点：

- [Consistency Models](https://openai.com/index/consistency-models/)建立了从 diffusion teacher 蒸馏单步/少步生成器的基本范式。
- [Consistency Policy（RSS 2024）](https://www.roboticsproceedings.org/rss20/p071.pdf)把一致性轨迹蒸馏用于 visuomotor policy，用原有机器人数据条件化动作生成。
- [ManiCM](https://arxiv.org/abs/2406.01586)同样在离线机器人 observation/action 数据上构造 ODE 蒸馏，并指出低维动作空间中直接预测 clean action 比预测噪声更稳定。
- [One-Step Diffusion Policy（ICML 2025）](https://proceedings.mlr.press/v267/wang25ba.html)用 diffusion-chain 上的分布/KL 蒸馏实现单步动作生成。
- [FlowPolicy](https://arxiv.org/abs/2412.04987)直接为 flow-based manipulation policy 引入 consistency flow matching。

### 6.2 但 AHA-WAM 不是上述算法的逐行复现

AHA-WAM 使用的是较直接的 **teacher trajectory endpoint regression**：采 teacher 轨迹中间状态，student 直接回归 teacher terminal action。它没有照搬 Consistency Policy 的完整 CTM 目标、ManiCM 的 online/EMA target network，也没有使用 OneDP 的 KL/score-distillation 目标。

所以准确表述应是：

> AHA-WAM-Flash 使用属于 consistency/ODE trajectory distillation 家族的简化 endpoint-regression 配方；当前实现是 AHA-WAM 官方发布的方法，而不是所有 diffusion-policy distillation 的唯一标准。

[AHA-WAM 论文](https://serene-sivy.github.io/aha-wam/static/aha-wam.pdf) Appendix D 明确给出：冻结 Video-DiT、16-step teacher、锚点 `0,1,2,4,8,12,16`、从非终点锚点回归 teacher 最终动作、重点采高噪声状态。论文报告 2-step sampler 将 CUDA 优化后的 chunk latency 从 41.37 ms 降为 17.56 ms，同时 RoboTwin 平均成功率由 AHA-WAM 的 92.80% 降到 AHA-WAM-Flash 的 90.20%。这也说明蒸馏通常以少量性能损失换速度，而非提升能力。

## 7. 当前 SO101 配方与官方配方的对应关系

| 项目 | AHA-WAM 论文 | 当前 SO101 四卡 run | 判断 |
|---|---:|---:|---|
| teacher steps | 16 | 16 | 一致 |
| anchors | 0,1,2,4,8,12,16 | 相同 | 一致 |
| student steps | 2 | 2 | 一致；仓库通用 YAML 默认 1，但论文 Flash 主结果为 2 |
| prediction | flow | flow | 一致 |
| Video-DiT | frozen | frozen | 一致 |
| learning rate | 2e-5 | 2e-5 | 一致 |
| optimizer / decay | AdamW / 1e-2 | AdamW / 1e-2 | 一致 |
| schedule | cosine | cosine + 5% warmup | 基本一致，warmup 延续仓库训练惯例 |
| epochs | 5 | 5 | 一致 |
| optimizer steps | 5000 | 4995 | 实质一致 |
| global batch | 512 | 16 | **不一致，主要规模差异** |
| local batch / grad accum | 16 / 4 | 1 / 4（4 GPU） | 显存与小数据集适配 |
| 数据 | 大规模 RoboTwin multi-task | 98 条 SO101 单任务 episode | 任务范围与多样性明显更小 |
| 分布式 | 官方示例 ZeRO-1/8 GPU | ZeRO-2 optimizer offload/4 GPU | 工程适配，不改变损失定义 |

因此，当前训练在**算法和约 5000 次参数更新的日程**上忠实于官方；不能宣称复现了官方的 batch/data scale。global batch 16 带来更高梯度方差，但若强行在 4 卡上模拟 batch 512，每个 update 需要大量累积，且五次数据遍历只剩约 160 次更新；为了兼顾 5 epochs 和约 5000 updates，当前 batch 16 是任务数据规模下更实用的选择。

## 8. 对当前插头任务的正确预期

这次 Flash 蒸馏回答的问题应是：

> 能否让 step-5328 teacher 的 task capability 在 2-step Action-DiT 中尽量保留，从而提高 action refresh rate、减少等待与卡顿？

它不回答：

> 能否仅靠蒸馏修复 teacher 的抓取偏差和最后毫米级插入失败？

后者仍需要更好的目标状态覆盖、human correction/DAgger、接触感知或后续 RL/adapter。论文中最重要的验收不应只看 `loss_ode`，而应至少报告：

1. teacher 10-step、未蒸馏直接 2-step、蒸馏 student 2-step 的真机成功率；
2. 抓取、粗对准、接触、完整插入的分阶段成功率；
3. action chunk latency、请求频率和实际控制更新频率；
4. 动作平滑性，以及相同初始化/随机种子下 student 与 teacher 的偏差；
5. 至少三个保存点的真机小样本筛选，避免只用训练 loss 选 checkpoint。

## 9. 本次四卡正式 run

启动脚本：

```bash
bash scripts/so101_pipeline/flash_distillation/start_training_4gpu.sh
```

配置：`scripts/so101_pipeline/flash_distillation/settings_4gpu.yaml`

它从 step-5328 SFT teacher 全新初始化 student，不加载旧两卡 student；四卡均为 micro-batch 1，gradient accumulation 4，effective batch 16。旧两卡 run 保留在原目录，用于比较训练曲线和诊断。
