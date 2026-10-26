# TD3+BC：数据动作、reference 与损失的区别

核查日期：2026-10-25。本文说明现有实现，不改变训练配置或队列。

当前外部 actor 的 BC 目标是日志中实际执行的动作块，不是冻结 AHA-WAM 重新生成的 reference。这是 TD3+BC 原算法的核心机制，不是本项目自行提出的算法。

## 当前代码究竟计算什么

设 h 为冻结 KV 加本体状态，A_ref 为缓存的模型参考动作，A_data 为去掉 SYSTEM_HOLD 后保留 POLICY/HUMAN 的连续 8 个实际执行命令。actor 输出 A_pi = actor(h, A_ref)。三个块都在相同的动作归一化空间中。

对当前完整 8×6 动作块：

\[
\ell_{BC,i}=\frac1{48}\sum_{k=0}^{7}\sum_{j=1}^{6}
(A_{\pi,i,k,j}-A_{data,i,k,j})^2.
\]

TD3+BC 的 actor loss 为：

\[
L_\pi=\frac1B\sum_i\ell_{BC,i}
-\lambda\frac1B\sum_iQ_1(h_i,A_{\pi,i}),\qquad
\lambda=\operatorname{sg}\left[
\frac{\alpha}{\max(\frac1B\sum_i|Q_1(h_i,A_{\pi,i})|,10^{-3})}
\right].
\]

实现见 [learner.py](../../src/ahawam/offline_rl/learner.py)：`masked_mse`、`OfflineLearner.update` 的 `td3_bc` 分支。mask 兼容旧原型，但当前 skip_hold 主线只收完整 8 步，不做成功处 prefix 截断。按已确定的数据合同，成功后的自然动作尾部也参与 BC。

每次抽 minibatch 更新 Q；默认每两次 Q 更新一次 actor。在 actor 更新的 batch 中，每条 transition 都有 BC 项，与 Q 项相加后进行一次反向传播、一次 actor optimizer step。不是先训练若干步 Q-loss actor，再另训若干步 BC actor。纯 critic 更新没有 BC 项。初始 200 步纯 BC warmup 是本项目另加的初始化阶段，与持续存在的 `+BC` 机制不同。

当前没有对 TD3 的 BC 做 advantage 加权、成功筛选或 Q-filter：成功和失败轨迹中符合采样条件的窗口都参加。它提供行为约束，不意味着所有动作都是专家动作。失败动作也被模仿可能限制改进；Q 梯度提供改进方向，但效果仍依赖价值估计与损失平衡。不能据此保证策略不离开数据支持范围。

`reference_penalty` 是另一项可选的 `MSE(A_pi, A_ref)`，默认 **0**。reference 作为网络输入、残差输出的起点，与在损失中监督 reference 是三件不同的事。

## 论文依据与本项目适配

- [TD3+BC，NeurIPS 2021，§5 Eq.3–5](https://arxiv.org/html/2106.06860)：向 TD3 actor 目标添加数据动作的平方误差约束。[作者代码](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py) 明确是 `-lmbda * Q.mean() + F.mse_loss(pi, action)`；分母查询的是 actor 动作的 Q，本文公式按作者代码与当前实现书写。源码 blob：`36e3ffa0990d90d1e75b9066f6894f9c478f35ea`。
- [RLT，§IV-B Eq.5 与 Algorithm 1](https://arxiv.org/html/2604.23073v1)：actor 的平方距离主要锚定 VLA reference；人工接管时，将 replay 中 reference 替换成人工动作。自主 RL 动作与 VLA reference 可以不同，所以不能将其称为与 TD3+BC 相同的数据动作 BC。论文报告两次 critic 对一次 actor；这里依据公开论文，不宣称取得完整真机 learner。
- [VINE，§2.1 Eq.2、§3 与附录 A.4](https://arxiv.org/html/2607.10369v1)：明确使用数据行为约束的 actor-critic 框架，Eq.2 将行为差异写为一般函数 F；附录列出任务相关 BC 系数。其主要贡献是让价值梯度穿过多步生成时采用不同采样路径。不能仅根据一般 F 就声称其所有实验使用本项目完全相同的 8×6 endpoint MSE、Q 归一化和超参数，也不能将 flow-matching 回归与最终动作 MSE 混为同一损失。

本项目采用的是 TD3+BC 的损失机制，适配冻结 KV、reference 条件、小残差 actor、8×6 chunk 与动作 z-score；另加分母下限、可选 reference penalty、reference dropout 和 BC warmup。默认 alpha 为 0.1，原作者为 2.5：这是待比较的起始选择，不是论文最优参数的复现。按上述公式，增大 alpha 增强 Q 项的相对作用；减小 alpha 更偏 BC。这里 alpha 乘在 Q 的尺度上，与 VINE 将 alpha 写在 BC 前的约定不同。

IQL 当前也以真实数据动作作为 MSE 目标，但 actor loss 是 `mean(detached_advantage_weight * bc)`；它没有 TD3 actor 的 `-Q(h, A_pi)` 动作梯度项。两者的改进信号不同。
