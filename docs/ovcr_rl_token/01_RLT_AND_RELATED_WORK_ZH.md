# RLT 怎样做，哪些机制适合迁移

## 1. 先区分读出表征与改写动作条件

你提出的直觉值得检验：AHA 已经有一个“小观测驱动大模型上下文”的接口，可以只调整这个接口来学精细纠正。但 **RL token、OVCR query 和可训练的 correction token 是三种不同对象**。

- RL readout：从大模型读信息，提供给另一套决策器。
- OVCR query：决定最新观测如何从 planner K/V 路由信息。
- Correction token：先吸收信息，再写回 Action DiT 的条件，进而改变动作。

“都是小 token”并不意味着训练目标、梯度方向和最终 actor 相同。若实验只用了人类动作 MSE，即使模块叫 RL token，整个方法仍是监督学习。

## 2. RLT 原文的最小完整机制

RLT 在 VLA 最后层 embeddings 后加可学习读出 token，用额外 Transformer 编码，取该位置输出。因果 decoder 用它与前缀 embeddings 重建原序列，重建目标 stop-gradient；可另外做 VLA SFT。第二阶段冻结 VLA 和读出器，以 token＋本体状态为 RL 状态。小 actor 另外读取 VLA reference chunk，输出固定方差 Gaussian；critic 评价状态与动作块。actor 优化 Q，并以距离 reference 的平方惩罚约束偏离；reference dropout 防止只照抄。critic 用 chunk TD 和 target network。它不是把 residual 加进原 action expert，也不是在线 RL 更新 token。[RLT 原文 §IV](https://arxiv.org/html/2604.23073v1#S4)

用简写表示其关键接口：

\[
z_{RL}=E_\phi([\operatorname{sg}(Z),e_{RL}])_{last},\quad
\hat Z_i=D_\phi(z_{RL},\operatorname{sg}(Z_{<i})),
\]

\[
x=(z_{RL},s),\quad a\sim\pi_\theta(x,a^{ref}),\quad Q_\psi(x,a).
\]

注意这里 reference 是 **actor 的条件**；原文的 critic 数学定义并没有强制拼接 reference。此前项目材料把二者输入都笼统写成 latent/proprio/reference，需要以这个区别为准。

官方项目把它定位于通用模型精细阶段的快速专门化，展示 screw、zip tie、charger、Ethernet 等任务。官方报道的速度收益来自这些特定平台和实验条件，不能转换成 SO101 的速度或成功率预期。[官方项目](https://www.pi.website/research/rlt)

## 3. 公开代码能否直接拿来用

本轮阅读了 [rlt-openpi 仓库](https://github.com/yknxh/rlt-openpi) 及其 [readout 实现](https://github.com/yknxh/rlt-openpi/blob/main/src/rlt_openpi/models/rl_token.py)：它实现了追加 query 的 encoder、因果重建 decoder、冻结输入和 masked MSE；它依赖 OpenPI embedding/action 接口。它是独立实现，不能标注为 Physical Intelligence 官方发布的完整真机复现。

还发现 [Evo-RLT](https://github.com/MINT-SJTU/Evo-RLT) 和 [RLinf RLT](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)。前者自述为独立 LeRobot reproduction；后者提供自身的两阶段训练和环境配置。这里仅核对了它们的公开说明，没有审计其所有更新公式或在 SO101 上验证。借用它们的目录组织和接口时，应固定 commit，并单独核对实际 chunk 长度、终止语义和 reference 的生成时间。

直接复制会碰到以下仓库层面的差异：

| 项目 | AHA 中需要的修改 |
| --- | --- |
| 输入不是 VLM 最后层 prefix | 明确定义来自 query、routed tokens，还是某几层当前帧 K/V；保存 source spec |
| planner 是有历史和版本的状态 | RGB 不能唯一确定在线表征；replay 要绑定真实 cache 协议 |
| 双相机拼图 | 不应把腕部的小误差被全局图背景淹没当作 RL 算法问题 |
| 动作是 SO101 6D 绝对关节目标 | 使用原 normalization；不能复制 Franka 的 EE delta scale |
| 低控制频率与网络延迟 | 重定 C、执行周期、discount 和状态时间戳 |
| 希望保留原 Action DiT | 需要 writer 或 reward-weighted flow；单独 reader 不会改变原 DiT 动作 |

这里最容易直接迁移的是“冻结强基座、任务适配的紧凑表征、reference 约束、分阶段优化”的思路。最不能直接迁移的是 embodiment、rollout schema 与在线时序。

## 4. 哪些相关工作真正支持本次方案

| 工作／一手来源 | 可借鉴的证据 | 对当前方案的限制 |
| --- | --- | --- |
| [AHA-WAM](https://arxiv.org/abs/2606.09811) | 原方法明确设计异步双 DiT 与 observation-guided routing | adapter 的正确位置仍以本地代码为准 |
| [LoRA](https://arxiv.org/abs/2106.09685) | 冻结基座，用低秩增量减少训练参数 | 本轮的 LN–Linear–SiLU–Linear 是 bottleneck residual，不是可直接合并权重的标准 LoRA |
| [Prefix-Tuning](https://arxiv.org/abs/2101.00190) | 冻结生成器也能通过少量连续条件改变生成 | 给现有 attention 追加 token 会改变 softmax 分母；零 Value 不保证保持原输出 |
| [Perceiver](https://arxiv.org/abs/2103.03206) | 少量 latent 通过 attention 读取大量输入 | 支持 T4 的压缩读写结构直觉，不证明它保存插入的毫米级信息 |
| [FlowCorrect](https://arxiv.org/abs/2602.22056) | 以稀疏人类纠正适配生成式 flow policy，保留原任务能力 | 它的纠正接口不等于本项目连续 HUMAN 绝对关节动作；不能直接复制标签 |
| [ResiP](https://arxiv.org/abs/2407.16677) | 冻结 chunked BC，加闭环 residual RL，针对精密操作 | 我们的内部 adapter 只在 request 时读取观测，并未实现逐控制 tick 的快速反射 |
| [DSRL](https://openreview.net/pdf?id=jU7AbGq3se) | 可以把冻结 diffusion 的输入噪声当成 RL 动作空间，避免反传整个采样链 | 不等于“任意 KV 编辑向量都天然适合 RL”；新的 latent action 也需要 transition 数据 |
| [ARLI](https://arxiv.org/abs/2608.23831) | 推理延迟下增加 committed actions 和较新的中间观测，使异步 RL 更可学 | 2026-08-24 新预印本；其平台结果不能直接外推到 SO101 |
| [HIL-SERL](https://arxiv.org/abs/2410.21845) | 真机 demo、人类纠正、在线经验可以共同支持高效 RL | 机器人控制与传感条件不同；不能只复制 loss 就称完成系统 |
| [TD3](https://arxiv.org/abs/1802.09477) | twin Q、target smoothing、延迟 actor 更新控制估值误差 | 小型 critic 仍会在有限离线动作支持之外高估 |
| [TD3+BC](https://arxiv.org/abs/2106.06860) | 用 BC 正则约束离线 actor 的分布偏移 | 不能把少量恢复段自动变成可信的在线 Q 函数 |
| [IQL](https://arxiv.org/abs/2110.06169) | 从离线数据估值并做 advantage-weighted policy extraction | 将提取目标换成 flow matching 是适配方案，需要单独验证 |

此表只使用论文、作者项目或实现者自己的文档；没有把搜索聚合站的总结作为算法证据。

## 5. 几个需要纠正的研究直觉

**冻结 Action DiT 不等于它不起作用。** 本轮内部 adapter 的梯度由最终 action flow loss 产生，整个 Action DiT 仍决定条件变化怎样映射为动作。它的权重不更新，但计算图必须保留。

**参数量小不等于在线计算快。** 当前实现减少 optimizer 与 parameter-gradient 开销，仍需执行大模型。内部 adapter 不自动缩短请求延迟；外部小 actor 可以频繁更新，但还要解决何时拿到最新观测。

**低重建误差不等于好的控制表征。** 本地 readout 后续必须做 token 打乱／置零和微小偏差 probe；因果 decoder 可以借前一个 teacher-forced embedding 预测下一个 embedding，重建容易而 z 不一定有用。

**只改 V 不是数学上的稳定性保证。** 它不直接扰动本层 K 的匹配位置，但会改变本层 action hidden，继而改变后续层 query。选择 V32 是一个较克制、易解释的首试假设。

**仅添加 critic 不会自动成为 RL。** 需要奖励目标确实影响 actor 的策略改进；这一点在 [04](04_CRITIC_AND_RL_ROADMAP_ZH.md) 展开。
