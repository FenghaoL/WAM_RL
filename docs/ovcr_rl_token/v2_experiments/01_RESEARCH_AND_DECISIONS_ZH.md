# 研究依据与初始化选择

## 先回答初始化问题

这轮使用用户提供的 `step_005328.pt`（5328，不是 53828）。其完整路径为：

```
runs/so101_ahawam_padapter_v2/full_zero2_offload_ahawam_base/formal-padapterv2-sft-ahawambase-bs4-acc1-4ep-3gpu-20260901-001743/model/checkpoints/weights/step_005328.pt
```

它已经学会 v2 demo 的 SO101 任务表征。先继续训练的小模块、保留 demo replay、限制输出漂移，比让小模块同时重新学习机械臂适配和纠正动作更可控。每个实验**独立从同一 P0 初始化**，不串接前一个实验的权重。optimizer、学习率日程、step 均重新开始；这是 warm-start finetuning，不是恢复 P0 的 optimizer state。

用户对 π*0.6 的理解有原文依据：RECAP §V（PDF 第 7 页）说明每轮 policy 和 value 都从 pretrained checkpoint 重新 finetune，以减少多轮漂移，同时明确承认持续训练上轮模型也可能有效。这里的 pretrained checkpoint 是其预训练 VLA/RL 模型，不能直接类比为随机初始化 Wan，也不能据此认定所有 DAgger 都必须从基座重训。[RECAP 原文](https://www.pi.website/download/pistar06.pdf)

保留一个后续大预算对照：同一 AHA 预训练初始化、同一 v2 demo+纠正数据、重新完成任务适配。它回答初始化路径的问题，应匹配 demo/HUMAN 曝光量并额外报告总 GPU 小时；只给它和 adapter 一样的 2664 步，会把“尚未学会 SO101”和“纠正机制无效”混在一起。此项不进入首轮自动队列。

## 各工作的可迁移部分与边界

| 工作 | 相关证据 | 对本项目的影响 |
|---|---|---|
| DAgger | 汇总在学习者访问状态下得到的专家标签，再更新策略 | 保留 demo，逐轮累计纠正数据；不要求抛弃旧数据或固定初始化方式 |
| HG-DAgger | 人类决定何时介入，纠正学习者访问的困难状态 | 本轮 HUMAN 标签选择与之接近，但只训练截取的连续片段，不给停顿自主动作当专家标签 |
| IWR | 等量抽取 intervention / non-intervention；IWR-NB 使用不平衡的统一采样对照 | 1:1 有先例，必须同时测自然混合；其 non-intervention 含 on-policy 数据，我们只用原始 demo，因此是借鉴而非复现 |
| SIRIUS | 分别处理 demo、robot、intervention、pre-intervention；干预比例消融在其任务中以 0.5 最优 | 支持对来源加权和固定采集数据做算法消融；不意味着 AHA 上 0.5 已经最优，也不意味着所有自主帧都差 |
| RECAP / π*0.6 | 价值估计、advantage conditioning、聚合多类经验，并比较不同策略提取方法 | 若要利用成功/失败自主轨迹，需要价值/回报和完整 transition；本轮不是 RECAP |
| RLT | 将 VLA 任务表征读入 token，为小型在线 RL 策略提供输入 | 借鉴冻结大模型、训练小模块和限制策略漂移；本轮 OVCR 插件不是 RLT 的原算法 |

原始来源：[DAgger](https://proceedings.mlr.press/v15/ross11a.html)、[HG-DAgger](https://arxiv.org/abs/1810.02890)、[IWR](https://arxiv.org/html/2012.06733)、[SIRIUS](https://arxiv.org/html/2211.08416)、[RECAP](https://www.pi.website/download/pistar06.pdf)、[RLT 官方介绍](https://www.pi.website/research/rlt)。

IWR 与 SIRIUS 的消融都提醒我们：改变数据采集策略后再比较训练算法，会混入数据质量差异。因此本轮所有算法共享固定 HIL manifest。SIRIUS 的来源比例结果来自其机器人任务，不能作为 AHA 上最优参数的结论。我们的 3:1 是较温和的工程假设，用于连接自然比例与 1:1。

## OVCR 三种修改如何工作

沿用已实现的 native OVCR cache editor：video/planner 提供条件，观测编码为 chunk query，editor 改写传给 action DiT 的 K/V。模块参与训练和推理的同一条路径，具体代码见 `src/ahawam/models/wan22/ovcr_adapter.py` 与 `mot.py`。

- **V32**：在 OVCR 解码后的 value 特征加入 rank-32 零初始化残差，保持 key 不变。直觉是先改“传递的内容”，减少对注意力检索位置的扰动；不是已证实优于 Q 的结论。
- **Q32**：在观测 query 上加入 rank-32 残差，再进入原有 editor；允许改变信息路由，但对 P0 editor 的局部梯度更敏感。
- **T4**：4 个读出 token 汇聚观测 query 与路由特征，再解码回既有 K/V 槽位；可学习紧凑的纠正条件，但参数更多，不能将其收益直接归因于 token 结构。

默认作用于第 20–29 层，up projection 为零，所以初始化行为等于 P0。冻结 P0 不代表行为一定不会遗忘：adapter 仍然能够扰乱下游动作；因此需要 replay、anchor 和真机评估。

目标为 `L = L_action_flow(demo ∪ HUMAN) + λ L_anchor(demo)`。anchor 比较同一噪声、时间步下的 student 与关闭 adapter 的冻结 P0 的 action velocity，忽略 padded action。λ=1 是待验证起点，不是论文给定的 AHA 最优值。这里是 velocity MSE 正则，不是 KL，也不是环境奖励。

## 要不要 critic

本轮不需要 critic。仅靠 HUMAN BC 加一个 critic 外壳并不会成为 RL；片段末尾也不能自动视作任务终止或成功。

真正的 RLT 路线是先训练任务表征/readout，再冻结大模型和读出，在线更新小 actor/critic。要在当前项目落地，还要明确 actor 输出是动作残差还是 OVCR latent、记录实际执行动作与时间、奖励、下一观测、termination/truncation 和策略版本。RLT 官方描述的是借助 VLA 表征的小型在线 RL 方案，并非把 HUMAN-only BC 改名。[RLT](https://www.pi.website/research/rlt)

建议先把本轮最优 BC adapter 做到稳定，再采集时间一致的成功/失败轨迹；暂停与接管边界不能直接拼成固定时间间隔的 transition。若走 RECAP，学习 V 并用于 advantage 条件/权重；若走 actor-critic，可研究冻结表征上的 residual actor + Q，并添加 BC/reference 正则。具体网络、奖励和在线控制还需独立实现及真机验证，本轮脚本未包含这些功能。
