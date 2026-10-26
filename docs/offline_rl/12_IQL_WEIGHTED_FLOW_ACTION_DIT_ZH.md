# IQL 能否直接训练 Action DiT：加权 flow matching 与 VINE 的区别

日期：2026-10-25。本文讨论内部 Action DiT 路线，不改变现有四卡缓存任务或外部小 actor 的训练队列。

## 1. 判断

**可以，而且是值得尝试的内部 actor 方案。** 用 IQL 的 Q/V 在真实数据动作块上计算 detached advantage weight，再加权 Action DiT 原有 flow-matching 回归损失，不需要让 Q 的动作梯度穿过去噪链。因此，它避开了 VINE 分析的那一种 value-gradient BPTT 不稳定性。

但不能将它概括为“换成大生成模型后 IQL 就没有问题”。仍有价值误差、有效样本数、过拟合、策略提取效果，以及本项目缓存/输入协议的成本。应称为 **IQL critic＋advantage-weighted flow-matching policy extraction**；原 IQL 的显式 log-likelihood loss 被生成模型的加权回归代理目标替代，不能说是原论文 actor 代码的原样复现。

之前优先外部小 actor 的原因是小网络训练便宜、便于检查数据与 critic，并非 IQL 只能配 MLP。内部 Action DiT 路线能直接利用预训练动作生成能力，也能保留生成模型的多模态表达，是合理的重要对照。

## 2. IQL 的 actor 为什么像 BC，却仍属于 RL

IQL 首先通过 Bellman 备份与 expectile 学 Q/V，再用

\[
w(s,A)=\operatorname{stopgrad}\left[\min\{\exp(\beta[\min_iQ_i^{target}(s,A)-V(s)]),w_{max}\}\right]
\]

提取策略。这里 A 是**实际记录的完整 8 动作块**，不是 actor 刚生成的块。

原 IQL 的 actor 最大化加权数据 log-likelihood。当前外部确定性 actor 用固定方差均值形式的加权 MSE。权重若全部相同，就接近同数据 BC；权重有信息时，则优先拟合具有较高长期价值的行为。RL 的作用在于从稀疏回报与后续状态学出这些权重，而不是必须让 Q 对动作求梯度。[IQL §4.2–4.3](https://arxiv.org/html/2110.06169)

## 3. 对 Action DiT 应加权哪个损失

应保留生成模型的标准训练方式：抽取数据动作 A、噪声 z 和噪声级 sigma，构造 noisy action，预测 flow velocity。**不是先用模型生成完整动作，再对其最终动作做加权 MSE。** 后一种写法又会涉及多步生成路径，且不是本节讨论的标准加权 FM 提取。

本仓库 [scheduler_continuous.py](../../src/ahawam/models/wan22/schedulers/scheduler_continuous.py) 的约定是：

\[
x_\sigma=(1-\sigma)A+\sigma z,\qquad u_{target}=z-A.
\]

因此可用：

\[
L_{actor}=\mathbb E_{(s,A)\sim D,\sigma,z}
\left[w(s,A)\,\lambda(\sigma)\,
\|v_\theta(x_\sigma,\sigma;c_s)-(z-A)\|^2\right].
\]

lambda(sigma) 保留已有 scheduler 的 training_weight；c_s 是冻结 planner/OVCR 与当前 proprio 等条件。许多论文把时间方向反过来，写 `x_t=(1-t)z+tA`、target=`A-z`，二者约定不同，不能直接把其符号拷到本项目。

一次 actor update 可以抽一个噪声级，做一次 30 层 Action DiT 的 forward/backward；可多抽噪声级降低随机性，但**不需要训练时展开推理的 10 个去噪步骤**。执行策略时仍需原来的多步生成，除非另外做蒸馏。

简化伪代码：

```python
with torch.no_grad():
    advantage = target_q_min(frozen_state_features, data_action_chunk) - value(frozen_state_features)
    weights = torch.exp(beta * advantage).clamp(max=max_weight)

noise = torch.randn_like(data_action_chunk)
timestep = scheduler.sample_training_t(...)
noisy_action = scheduler.add_noise(data_action_chunk, noise, timestep)
target_flow = scheduler.training_target(data_action_chunk, noise, timestep)
prediction = trainable_action_dit(noisy_action, timestep, frozen_conditions)

# Preserve scheduler weighting and padding, then retain one loss per 8-action chunk.
flow_loss_per_chunk = reduce_flow_error_per_chunk(prediction, target_flow, timestep)
actor_loss = (weights.detach() * flow_loss_per_chunk).mean()
actor_loss.backward()
```

关键是：在干净数据 A 上算 w；w 不依赖当前噪声抽样、不对 actor 回传 Q 梯度；按 chunk 保留 loss 后乘权重，最后再做 batch 平均。

### 这个代理目标是否有数学依据

固定 s，在理想容量、充分数据/优化和恰当积分条件下，正权重 w 将数据动作分布变为

\[
p_w(A\mid s)=\frac{p_D(A\mid s)w(s,A)}{\mathbb E_{A\sim p_D(\cdot\mid s)}w(s,A)}.
\]

对同一噪声路径训练加权 conditional FM，就相当于对重新加权的终点分布学习生成过程。**Energy-Weighted Flow Matching，ICLR 2025，§4.1 / Theorem 4.3 / Remark 4.6**给出了相应的加权条件流与能量倾斜分布关系。[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/2d9c6cdb4cfe93869c090fea7375044b-Paper-Conference.pdf)

该依据不表示有限步 Action DiT 微调严格等价于精确的加权 log-likelihood 最大化，也不表示 IQL 权重估计已经正确。clipping 会改变所拟合的倾斜分布；有限模型、有限优化和有限推理步数也会带来近似。

## 4. VINE 分析的梯度路径与这里不同

| 方案 | actor 更新时 Q 的作用 | 是否通过多步生成链反传 Q 梯度 |
| --- | --- | --- |
| 当前外部 actor＋IQL | 数据 A 的 Q−V 变成 detached BC 权重 | 否 |
| 原 Action DiT＋IQL 加权 FM | 同样算数据 A 的权重，加权单次 noisy-input 回归 | **否** |
| 当前外部单次输出 actor＋TD3+BC | 对 Q(s,actor(s)) 的动作求梯度 | 有动作梯度，但没有多步去噪链 |
| 原 Action DiT＋朴素 TD3/DPG 风格 Q loss | Q 评价最终生成的 A_theta，再反传 | **是** |
| VINE | 改造生成路径，再做 value-gradient BPTT | 是，但专门设计其采样路径 |
| FQL | 生成 flow 学 BC；另一个 one-step actor 接收 Q 梯度 | 不对完整 flow 链做该 BPTT |

朴素生成 actor 的目标通常包含 `-Q(s,G_theta(s,z))`，G_theta 是 K 次生成更新。计算其梯度要经过 `dQ/dA × dG_theta/dtheta`，后者涉及 K 次迭代。VINE §2.3、Algorithm 1 针对这一类训练分析并修改采样路径。[VINE](https://arxiv.org/html/2607.10369v1)

加权 FM 则是 `stopgrad(w) × d(flow_loss)/dtheta`，没有这一条链。它仍有普通深 Transformer 的反向传播，但“网络 30 层”和“生成迭代 10 次的 BPTT”是两件事。

因此不能用算法名称简单分成“IQL 都没问题、TD3 都有问题”。应检查实际 actor loss：是否有 Q 对模型生成动作的梯度，以及该动作是否通过多步生成得到。当前小 actor 的 TD3+BC 并不具有 VINE 所针对的那条长生成链。

## 5. 已有工作直接测试过加权生成训练，也给出了限制

### Flow Q-Learning，ICML 2025

FQL §4.1 明确列出 weighted flow/diffusion BC 这一策略提取方式，形式正是 advantage weight 乘生成回归损失；论文也指出有效样本数与策略提取效果的限制。其另一条路线是保持 flow BC，训练一个接收 Q 梯度的 one-step policy，避免完整 flow BPTT。[FQL §3–4.1](https://arxiv.org/html/2502.02538)

这支持方案可行，但不支持“它一定与直接 Q-gradient 同样强”。其 benchmark 也不是本项目的预训练 Action DiT＋少量插入数据，不能据此直接否定复用预训练生成 actor 的收益。

### IDQL 的直接加权 DDPM 消融

IDQL **Appendix F.5 / Fig.7** 实际测试 AWR 权重乘 DDPM loss，并比较单次动作采样与64次采样后按 critic 选动作。在作者的实验中，直接加权训练通常没有优于其 IDQL 提取方案。这是与本问题很直接的经验依据。[IDQL 附录 F.5](https://arxiv.org/html/2304.10573#A6.SS5)

注意，IDQL 的主算法并不是“把扩散 BC loss 乘 IQL 权重”。它先学行为生成模型，再对生成的候选用 critic 权重进行重采样。F.5 才是直接加权训练的消融；不能把两者混称为同一种实现。DDPM 消融结果也不能被解释成所有 FM/预训练机器人模型上都会失败。

### 其他正面先例

- [Energy-Weighted Flow Matching，ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/2d9c6cdb4cfe93869c090fea7375044b-Abstract-Conference.html)：加权 flow matching 的理论与 QIPO；其策略更新、采样分布与 IQL 提取不完全相同。
- [QVPO，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/6111371a868af8dcfba0f96ad9e25ae3-Abstract-Conference.html)：Q-weighted diffusion variational objective，主要用于在线 RL，权重处理也不同；说明加权生成训练是一条已有研究路线，不是生成策略必须经 Q-gradient BPTT 才能做 RL。

## 6. 对这个数据集仍需关注的具体问题

1. **critic 错误会变成数据选择错误。** 即使不通过 Q 求动作梯度，错误的高 advantage 仍会放大错误行为；生成 actor 不会自动修复权重。
2. **有效样本数可能减少。** 数据约万级窗口但强相关，真正独立成功事件更少。指数权重若只保留少数轨迹，容易遗忘恢复行为或过拟合。监测 ESS、权重分位数和每 episode 的累计权重，而不只看总窗口数。
3. **改善能力受数据与提取方式约束。** weighted FM 主要重新分配已记录行为的概率，并可能依靠网络泛化组合更好策略；不会直接搜索任意新动作。权重接近相等时，得到的主要仍是同数据 BC。
4. **大 actor 能表示多模态，并不保证样本利用更好。** 复用预训练动作分布是优势，但不应把大模型或 FM 当作解决小数据/权重误差的保证。
5. **有遗忘风险。** 首试可冻结 planner/OVCR，只对 Action DiT 做 LoRA 或受控参数更新；保留同架构无权重 BC 对照，必要时比较与原 SFT 的混合 BC/蒸馏约束。后者是额外设计，不是 IQL 必需项。
6. **不因微调而改变 critic 表征。** 若更新 planner、OVCR 或其输入编码器，现有特征和权重未必仍匹配；简单首版应冻结这些共享条件模块。

## 7. 本仓库的实际改造点与缓存限制

| 部分 | 已核对的代码 | 后续改造含义 |
| --- | --- | --- |
| 噪声路径、target、噪声级权重 | [scheduler_continuous.py](../../src/ahawam/models/wan22/schedulers/scheduler_continuous.py) | 保留现有 sigma/noise−action 约定，再乘独立的 advantage weight |
| 当前 FM action loss | [ahawam.py::_compute_weighted_action_loss](../../src/ahawam/models/wan22/ahawam.py) | 现在的 weighted 指 scheduler 的 timestep weighting；还没有 IQL advantage weighting。需先输出 per-chunk loss，避免批量归约后才加权 |
| 原 offset 训练 | [ahawam.py::_training_loss_action_offset](../../src/ahawam/models/wan22/ahawam.py) | 现有路径包含40动作/多个chunk与视频分支；一个8动作的权重不能随意乘整段40动作 loss |
| 冻结 video state 下的 flow 预测 | [ahawam.py::_predict_action_flow_with_video_state](../../src/ahawam/models/wan22/ahawam.py) | 有可参考的 prior-only flow 预测接口，但仍需适配本轮8动作、skip_hold、anchor/offset协议 |
| Action DiT 逐层读取 KV | [mot.py::forward_action_prior_only_with_editor_cache](../../src/ahawam/models/wan22/mot.py) | 每层读取自己的 editor_cache[layer_idx]，不是所有层都读取最后一层 |
| 部署采样器 | [ahawam_chunk_base.py::infer_action_chunk](../../src/ahawam/models/wan22/ahawam_chunk_base.py) | 当前服务路径会 eval、detach、返回CPU动作；不应直接拿它做训练梯度路径 |

**当前缓存只存最后一层 K/V，足够给外部小 actor/Q 使用，不足以重建原版30层 Action DiT 的完整条件。** 直接训练原 Action DiT，需要在线重算冻结 planner 的逐层条件，或另外缓存所需各层。把同一份最后一层 KV 填到30层并不是复用原 SFT 模型的等价做法。

若原样把每观察30层 BF16 K/V全部存下，沿用当前形状约为 `30×59.99GiB≈1.76TiB`，还未计附加字段；这只是容量换算，不是建议现在扩展缓存。可先在训练时重算、局部缓存或针对性分片。训练的显存与速度还需实测，不能套用当前小 actor batch64约1.2GiB的测量。

现有最后一层缓存依然有用：IQL 的 critic 可继续在它上面训练，并计算每个真实 transition 的 Q、V、advantage 与权重。权重文件很小，可以记录 critic checkpoint hash、状态/动作索引与 feature provenance，供 Action DiT 的单独提取阶段读取。

## 8. 推荐的内部 actor 首试

1. 沿用当前数据与冻结 SFT 006572 表征，先把 IQL Q/V 训练到可评估的阶段；500步资源 pilot 不是收敛证明。
2. 固定一个 critic checkpoint，给同一批8动作数据块算 detached 权重，检查按轨迹的权重分布与 ESS。
3. 从同一 SFT Action DiT 初始化，在相同冻结条件、相同自然动作块上比较普通 FM-BC 与 advantage-weighted FM。先保持 planner/OVCR冻结，优先受控更新/LoRA。
4. 新训练入口直接对齐单个8动作块的条件与损失；不要未核对地复用原40步数据窗口与归约。保留当前 offset 的因果语义，不能让训练 planner 看见当前动作之后的画面。
5. 比较内部生成 actor 与外部小 actor，再决定是否需要带 Q-gradient 的内部路线。只有进入后者时，才重点考虑 VINE/FQL 等不同梯度路径的设计。

本轮判断是：**内部 Action DiT 的 IQL 加权 FM，是比直接对它做朴素 TD3-style 多步 value BPTT 更自然的第一项内部 RL 对照。** 它规避特定的长生成链梯度问题，但依然需要数据/价值/策略提取与成本验证。本轮仅补充设计文档，尚未切换现有 actor 或额外启动此训练。
