# 提前停止的成功录像、零奖励数据与 critic 泛化

日期：2026-10-25。本文依据用户新说明：9 条缺少成功帧标注的录像确实来自最终成功的实验，但录像停止过早，最后画面只插入了一半。它们是**观测提前截断的轨迹**，不应改标为失败，也不需要在未完成的画面上补一个 success。

本轮是研究与方案讨论：不改变正在导出的缓存、不重写原始标注、不将这 9 条自动加入当前 pilot。当前 transition 文件仍排除它们；全部 103 条的特征导出继续。后续可以比较完整数据基线与显式处理截断的扩展版。

## 1. 结论和建议

1. 这 9 条可以有利用价值，建议作为 **partial trajectories / recording truncations** 保留。录像内已经观察到的未成功 transition，奖励仍为 0。
2. **零即时奖励，不等于零未来回报。** 提前停止录像的边界不能当成 task failure，也不能把该轨迹的 Monte Carlo return 全部定为 0。
3. 纳入 RL 时，只使用动作、结果观察均实际记录的完整窗口；不跨录像边界、不编造最后一行动作的结果。到录制边界的样本应允许 bootstrap，而不是将 discount 置零。
4. bootstrap 能否给出好答案取决于完整成功数据对末端附近状态与后续动作的覆盖。处理对了终止语义，不代表末端价值已经可靠。
5. 先保持当前不含这 9 条的基线，再做“完整轨迹＋截断轨迹”的 IQL 对照。我的建议是最终尝试纳入，而不是永久扔掉；但不建议作为零回报失败纳入。
6. critic 可以泛化，但“训练好了就必然能正确评价任意未见状态/动作”没有保证。应分别检查价值学习、actor 策略提取、部署分布泛化。

这些是结合下述论文与本地实现提出的工程判断，论文没有直接验证这个 SO101 数据集。

## 2. 先把 r、return、terminal 和 recording boundary 分开

在完整插入时只给一次 +1 的任务里，某个 transition 的 r=0，只表示**这一步尚未取得成功事件**。它可能把圆柱推进了一个很好的位置，未来仍可成功。

标准价值的含义是：

\[
Q^\pi(s,a)=\mathbb E[\sum_{k\ge0}\gamma^k r_{t+k}\mid s_t=s,a_t=a,\text{之后按}\pi\text{行动}].
\]

它与后续策略有关，不是图像的通用“好坏”或插入深度。只有一次成功奖励时，它可理解为对未来成功及等待时间的综合估计。gamma=1 且成功/失败与有限回报定义适当时可对应成功概率；gamma<1 时是折扣后的量。当前 IQL 的 V 是数据动作 Q 的上侧 expectile，更不能直接当成校准成功概率。[离线 RL 教程的价值定义](https://arxiv.org/html/2005.01643)，[IQL §4.2–4.4](https://arxiv.org/html/2110.06169)

| 录像/任务情形 | 已观察 r | 是否真的任务终止 | 值函数备份 |
| --- | --- | --- | --- |
| 已观察到完整插入的首次成功 | incoming transition 为 1 | 是，按当前单次成功目标 | 不自举 |
| 明确定义的不可恢复失败，或任务确实以该时间预算结束 | 0 | 是 | 不自举 |
| 普通进行中、尚未成功 | 0 | 否 | 自举 |
| 录像误停、之后还可能继续完成 | 0 | **否，只是数据结束** | 在实际存在的 next observation 上自举 |

若“必须在 30 秒内成功”本来就是任务目标，30 秒终止可以是环境 terminal；此时剩余时间是状态的一部分。若只是录像程序误停，不能因此把任务重新定义为“必须在录像停止前成功”。[Pardo et al., ICML 2018，§2–3](https://proceedings.mlr.press/v80/pardo18a/pardo18a.pdf)

你这 9 条属于最后一行。它们不是失败，也不是应该把成功随意前移到最后一帧的正例。若还有控制日志或额外录像，能恢复真正的完成时间和过程，那才可补充相应 transition。仅有“录像外最终成功”的说明，可以保存为 episode metadata；它不能唯一确定录像内哪一步应取得 +1。

### 当前 8 动作块的具体备份

对 8 个真实有效动作的累计奖励 R，令后继观察为 s'：

\[
y_{\mathrm{IQL}}=R+\gamma^8(1-\mathrm{terminated})V(s'),
\]
\[
y_{\mathrm{TD3+BC}}=R+\gamma^8(1-\mathrm{terminated})\min_i Q_i^{target}(s',\pi^{target}(s')+\epsilon).
\]

录像误停时 R=0，但 terminated=false。即使这个窗口结束在录制边界，也不应把上式第二项删掉。示例：如果末端 V(s')=.8，gamma=.997，则目标约为 .7810；误标失败会强制目标为 0。这个 .8 仅是解释公式的假设，不是模型已经测出的值。

必须另外保留 `episode_boundary=true`，阻止 sampler 把下一个 episode 接上来。**禁止跨录像采样**与**是否继续估计录像外未来价值**是两个不同开关。[Gymnasium 官方说明](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)

若边界前不足自然 8 动作，仍遵守用户已选择的完整块方案：舍弃不足的窗口。末行命令没有后一张结果图像，就不能伪造 next observation。保留下来的窗口即使没有恰好结束在最后一帧，其边界附近状态也仍可能缺少 outgoing action 的训练覆盖。

## 3. 为什么“在已有数据上打分”仍不保证正确

需要区分输入是否被覆盖、备份目标是否可靠、目标策略是否被覆盖。

| 查询 | 是否来自记录的输入 | 主要问题 |
| --- | --- | --- |
| Q(s, A_data)，s 与动作块均实际记录 | 是 | 值仍来自 TD target；错误的终止标签或后继价值会传回来 |
| Q(s, A_actor)，s 是同一个旧画面，动作是 actor 新生成的 | s 是，完整 (s,A) 未必是 | 条件动作分布可能偏移；48 维 chunk 的各维接近也不代表整体组合有数据支持 |
| V(s_last)，s_last 只作为截断录像的最后一张画面 | 图像在文件里 | 它可能从未作为有 outgoing action 的起始状态被训练；IQL 的 V 损失未必直接约束它 |
| Q(s_new, A_actor)，真机执行后出现的新状态 | 未必 | 状态分布与动作分布都可能偏移 |

“未精确见过”也不自动等于 OOD。连续图像、状态和动作几乎不会精确重复；来自同一受覆盖分布的新样本，通常依赖正常的统计泛化。困难是数据稀少或未覆盖的区域，而不是必须逐点查表。

**即使 Q 的左侧输入完全来自训练集，也可能被错误的右侧目标监督。** Fujimoto et al. 在 ICML 2022 §3.2 给出不完整数据集上的反例：有限数据的 Bellman error 可以为零，但数据内的 value error 仍很大；§4 还做了标准 benchmark 实验。因此不能用“训练样本 Q 值稳定、Bellman loss 很小”认证价值准确。[Why Should I Trust You, Bellman?](https://proceedings.mlr.press/v162/fujimoto22a/fujimoto22a.pdf)

### 一个对应半插录像的可识别性例子

假设只记录 `s_before --a, r=0--> s_half`，没有 s_half 之后的行动和结果。一个与记录相容的环境里，继续按压能够成功；另一个相容环境里，该位置存在未被输入区分的卡死状态。仅凭这一段记录，无法确定该给未来价值多少。更大的网络能表示两种答案，却不能凭空观测缺失的结局。

完整成功轨迹若覆盖了真正相同或有意义地相近的半插状态及完成动作，就可能提供所缺信息。这正是跨轨迹价值传播有用的地方。条件是状态表征足以区分与未来结果相关的因素；“画面看起来相近”不等于接触、朝向、卡紧程度和动力学完全相同。这里是对本任务的推论，不是说现有 KV 表征已经被实验证实缺失这些信息。

## 4. 没有负奖励，失败数据如何起作用

它仍然有监督，不是无标签数据：

- 真正 terminal failure 的 incoming transition：r=0、bootstrap mask=0，给 Q 明确的零目标。
- 中间 transition：r=0，但通过 next value 提供 TD 监督。
- success 的 incoming transition：r=1、bootstrap mask=0，给正价值锚点，再向前传播。

若某条失败录像中前半段行为本来可恢复，只是后续采取了坏动作，改善后的策略可以在同一个中间状态选其他有数据支持的动作而成功。因此，**不应把一次失败轨迹的每个中间状态/动作都硬标成“永久无价值”**。这与 IQL 可以从次优轨迹中做 multi-step stitching 的动机一致。[IQL §2、§5.1](https://arxiv.org/html/2110.06169)

反过来，如果所有记录都没有成功奖励、都被正确或错误地作为零回报 terminal 封口，在纯稀疏 TD 目标下，Q=V=0 可以成为一个自洽解。网络无法仅从“插了一半看起来更接近”自动推出任务回报；这种视觉进度先验不是当前 sparse Bellman loss 明确给出的监督。

不需要为了让失败数据有用，先加负奖励或把半插改成 .5。后一种修改改变了奖励定义；特别是需要先退出、再重新对准时，插入深度未必应单调增加。

## 5. IQL 为什么更适合作为这一步的起点

本地 [learner.py](../../src/ahawam/offline_rl/learner.py) 的 IQL：V 回归 `min target_Q(s,A_data)` 的 expectile；Q 回归 `R + discount*V(s')`；actor 用 detached advantage 加权回归数据动作。没有在 actor 更新或 Q 备份里优化新动作的 Q 值。

TD3+BC 则在 actor loss 中评价 `Q1(s,A_actor)`，还在 target 中评价带平滑噪声的 target actor 动作。即使 s 来自原数据，也不能把这些新 (s,A) 查询统一称为数据内评分。BC 约束能抑制偏移，但不是严格的动作支持集约束。[TD3+BC 原文](https://proceedings.neurips.cc/paper_files/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html)

IQL 减少了对未记录动作的价值查询，**并不保证截断末端 V(s') 正确**；它依然依赖状态覆盖、表示泛化与正确的 TD 边界。这也解释了为何“先 IQL，再 TD3+BC 对照”合理，但不能提前判断哪种最终最好。

还应避免把一切失败都归因于 critic。NeurIPS 2024 的系统研究把价值学习、策略提取和部署状态泛化分开，发现许多设置的主要瓶颈在后两项，行为约束的梯度策略提取有时比 AWR 更能利用同一 value function。因此不能从 actor 表现差直接推出 critic 没学会，也不能预断 TD3+BC 一定比 IQL 更差。[Is Value Learning Really the Main Bottleneck in Offline RL?，正式会议版 §4–5](https://proceedings.neurips.cc/paper_files/paper/2024/file/8ffb4e3118280a66b192b6f06e0e2596-Paper-Conference.pdf)

## 6. 双 Q、预训练与模型容量能解决多少

双 Q 取 min 主要缓解一部分高估，不能认证 OOD 分数正确。当前共享主干让两个 head 更可能共享表征误差；它们可能一起错，差值小也不代表可靠。

EDAC 专门分析 ensemble 的局部动作梯度相似性：即使有多个 Q，若其局部结构接近，某些数据外方向上的方差仍会很小；作者提出梯度多样化改善这个问题。它支持“别把一致性当正确性”的担忧，但没有直接实验比较本项目的共享 KV decoder 与独立 decoder。[EDAC，NeurIPS 2021，§3–4 / Fig.3–4](https://proceedings.neurips.cc/paper/2021/file/3d3d286a8d153a4a58156d0e02d8570c-Paper.pdf)

目前建议保留共享主干基线，同时用已有 `--independent-q` 做消融、检查跨 seed 稳定性。无需立即移植 EDAC 的大 ensemble。独立双 Q 也不提供普遍的准确性保证。

冻结 SFT WAM 提供了视觉/任务先验，可能降低小样本学习难度。但这个 checkpoint 原来优化的是行为/生成目标，没有直接被训练为任意新动作输出可靠回报；复用预训练权重不等于已经预训练好 critic。容量不够、缺少后续成功覆盖、表征不能区分接触状态，是三类需要分开诊断的问题。

## 7. 建议的实验顺序与判断依据

### A. 当前完整数据基线

保持正在准备的基线：94 条标注/结局就绪的 episode，排除这 9 条。保留 `_val` 整 episode 隔离。500-step pilot 主要验证资源和链路，不能据此判断 critic 的最终价值泛化。

### B. 完整数据＋明确截断数据

为 9 条设置独立的 `recording_end_reason=early_recording_stop`：录像内未成功的 r=0，真正任务 terminated=false。保留完整可观察自然窗口的 nonzero discount；不生成录像外动作，不填 success=1，不对整条 return 标零。8 条 train 候选与 1 条 `_val` 候选仍遵守原目录划分，不把该 val episode 加进训练。

前提是附近完整成功数据提供一定后续覆盖。没有覆盖时，靠末端 bootstrap 得到的高值或低值都不应单独被当成事实。若 B 明显不稳定，可继续排除这些段作 RL，或仅将其中可用行为用于明确分开的 BC 消融；混合 policy/HUMAN 的所有动作也不自动都是专家动作。

### C. 验证价值与策略时分清真值

1. **受控的人工截断对照**：在一部分完整训练 episode 中隐藏尾部，分别比较排除、错误 terminal=0、正确 truncation bootstrap。选中的 episode 不能同时把完整未来也放进同次训练，避免假装模型“推断了”其实已经见过的结局。此试验检测缺尾部处理，不替代独立 episode 泛化测试。
2. **独立 episode 测试**：冻结表征与 head，在未参与训练的完整录像上检查 Q/V 行为及策略动作误差；不要随机拆重叠窗口。半插、正确对准、偏斜、退出重试等可作为审阅分组，不能先把插入深度排序强行当成价值真值。
3. **MC 的适用范围**：完整录像上的实际 return 属于记录行为策略（含 HUMAN 接管），不是当前 actor 的 Q 真值。IQL 的上侧 expectile 与当前 actor 也不完全相同。可另训行为回报预测器作表征诊断，但不能把其分类准确率当作 actor-Q 校准证据。
4. **新动作诊断**：在相同 s 下比较数据动作、reference、actor 动作及小扰动的 Q 与分歧，查看是否出现离数据更远反而分数突增。距离/分歧只是诊断，不是真实 OOD return 标签。
5. **最终评价**：固定策略的独立真实 rollout，才直接给该策略的结果证据。FQE 等离线策略评估也依赖覆盖与表示假设，并不自动消除未知后续。先做至少几个 seed 的对照，再把改变归因给数据处理或架构。

### 对当前实现的注意点

现有 `data.py` 中 `pending_success_annotation` 会让这 9 条不产生训练样本，因此当前没有“末端强行归零”的错误。缓存与奖励注释独立，无需重导 103 条 KV。

将来纳入时不能只把 status 改 ready：现有 `first_success=None` 分支会把录像尾部视为 failure end，需要新增明确的 ending type。旧 `truncated` 字段还用于 intervention censor，旧分支会将 discount 置零；它不等同于 Gymnasium 的“录制截断仍自举”。应分别记录 `episode_boundary`、`terminated`、`recording_truncated` 和最终用于备份的 discount/mask。上述 B 是实施建议，本轮没有悄悄改变训练数据。

对原先 13 条 manifest failure，也要最终确认结局语义：若是真正失败/明确任务预算结束，可以 terminal；若只是人工停止但任务仍可继续，也应按同样逻辑审计，不能只因 manifest 写 failure 就推导吸收态。

## 8. 论文依据、质量与适用边界

这里优先使用正式 ICML / NeurIPS / ICLR / COLT 会议来源。正式发表是筛选依据，不是对任一工程设置的保证。主要结论已读到相应方法、定理或实验段落；补充理论论文仅作为能力假设边界，不声称其最坏情况就是本任务。

| 工作 / 发表 | 本问题相关证据 | 适用边界 |
| --- | --- | --- |
| [Time Limits in Reinforcement Learning，ICML 2018](https://proceedings.mlr.press/v80/pardo18a.html) | §3、Eq.5–6：partial-episode bootstrapping；§2：真实有限时间任务需要时间信息 | 录像误停属于外部边界是本任务判断；论文不保证离线末端估计准确 |
| [BCQ，ICML 2019](https://proceedings.mlr.press/v97/fujimoto19a.html) | §3：extrapolation error；约束候选动作贴近数据 | 不说明一切新动作必错，而是无数据覆盖时常规 off-policy 估计会失效 |
| [BEAR，NeurIPS 2019](https://proceedings.neurips.cc/paper_files/paper/2019/hash/c2073ffa77b5357a498057413bb09d3a-Abstract.html) | 数据外动作的 bootstrap error 可经备份传播，提出支持约束 | 不是本项目自然动作块边界的具体实现 |
| [CQL，NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html) | §3.1 / Thm3.1–3.2：保守价值与分布偏移 | 常用 tightened 目标主要保证条件下的策略期望下界，不是任意神经 Q 的逐点准确或成功概率校准 |
| [PEVI，ICML 2021](https://proceedings.mlr.press/v139/jin21e.html) | §3–4：数据内在不确定性、uncertainty penalty；线性 MDP 的信息下界 | 需要有效 uncertainty quantifier；不能把普通两 Q 的差值直接代进就继承定理 |
| [TD3+BC，NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/hash/a8166da05c5a094f7dc03724b41886e5-Abstract.html) | 行为正则抑制 actor 利用不可靠的价值方向 | BC 是软约束，不能保证每个 actor action 在条件数据支持内 |
| [EDAC，NeurIPS 2021](https://papers.neurips.cc/paper_files/paper/2021/hash/3d3d286a8d153a4a58156d0e02d8570c-Abstract.html) | §4：Q 梯度对齐会削弱 ensemble 的局部 OOD 不确定性 | 不足以断言两个独立 Q 一定比共享主干效果好 |
| [IQL，ICLR 2022](https://arxiv.org/html/2110.06169) | §4：只查询数据动作；§5.1：次优轨迹 stitching 实验 | 避免数据外动作查询，不等于避免末端状态泛化 |
| [Why Should I Trust You, Bellman?，ICML 2022](https://proceedings.mlr.press/v162/fujimoto22a.html) | §3.2 / Corollary1、§4：缺失后续可使数据内 Bellman loss 很小但值误差很大 | 区分经验 Bellman loss 与真实价值准确性，不能从训练 loss 认证 critic |
| [Fundamental Barriers for Value Function Approximation，COLT 2022](https://proceedings.mlr.press/v178/foster22a.html) | 表示能容纳真 Q 与一般覆盖条件，并不足以在所有问题上保证高效离线学习 | 理论的最坏情况限制，不是对这 103 条数据的性能预测 |
| [Reining Generalization via Representation Distinction，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/802a4350ca4fced76b13b8b320af1543-Paper-Conference.pdf) | 从 representation 区分数据内/策略生成 state-action，分析过度泛化 | D4RL 实验，不是冻结 WAM KV 接触任务的直接证据 |
| [Is Value Learning Really the Main Bottleneck?，NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/8ffb4e3118280a66b192b6f06e0e2596-Paper-Conference.pdf) | 价值、策略提取、部署泛化的分离实验；数据覆盖可比单纯提高最优性更有帮助 | 不能据此保证本任务 critic 已足够好，也不能推导特定算法必胜 |
| [The Generalization Gap in Offline RL，ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/5c1ddd2e59df46fd2aa85c833b1b36ed-Abstract-Conference.html) | Procgen / WebShop 的新关卡、新指令泛化与数据多样性实验 | 是跨环境/任务泛化，比本问题的同状态新动作更远，不能混为一个 OOD 定义 |
| [Role of State Aggregation and Trajectory Data，COLT 2024](https://proceedings.mlr.press/v247/jia24a.html) | value realizability 下，轨迹数据与 coverage/representation 条件仍存在理论障碍 | 作为“网络有能力表达正确答案并不自动意味着可从有限数据学出”的补充 |

RLT、HALO-WA 的在线系统结果支持冻结表征＋小 actor-critic 的工程路线；VINE 研究生成策略的优化稳定性。但它们不能替代上述关于缺失后续、离线分布支持和价值可识别性的论证，也没有被本轮当作“能正确推断截早录像结局”的依据。

## 9. 作者代码审计

本轮通过 GitHub 连接读取原文件，以下是文件 blob SHA（不是仓库 commit）：

| 代码 | 实查行为 | blob SHA |
| --- | --- | --- |
| [IQL critic.py](https://github.com/ikostrikov/implicit_q_learning/blob/master/critic.py) | V 用 data actions；Q target 为 rewards＋discount×masks×next_v | `a49d2b1ed3dcc12e15b726cf1d4f592a955725f7` |
| [IQL actor.py](https://github.com/ikostrikov/implicit_q_learning/blob/master/actor.py) | 以 data action 的 Q−V 权重回归数据 log_prob，不对 actor action 做 Q 梯度优化 | `7c73f10b454c38f3af0346cb0783054deb15f850` |
| [IQL dataset_utils.py](https://github.com/ikostrikov/implicit_q_learning/blob/master/dataset_utils.py) | trajectory dones_float 与 backup masks 分开；masks=1−terminals | `adc607fd01dbf54b41344647b79c662f8bacf35f` |
| [D4RL qlearning_dataset](https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/__init__.py) | 默认 terminate_on_end=False 跳过 timeout transition；上游 loader 也影响边界，不可仅看 learner mask | `721a6d95805092406cdce793d548c1d7972fefe1` |
| [TD3_BC.py](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py) | target 查询平滑后的 target actor 动作；actor 查询当前 Q1(state,pi) | `36e3ffa0990d90d1e75b9066f6894f9c478f35ea` |
| [TD3+BC utils.py](https://github.com/sfujim/TD3_BC/blob/main/utils.py) | 导入数据时 not_done=1−terminals | `46676820cb63b4a1c9fdb61466cbe6693348ab48` |

这份审计说明“数据边界、任务终止、价值备份 mask”应分离，也说明 IQL/TD3+BC 对新动作的查询不同。它不表示应原样复制 D4RL 的数据丢弃策略到本项目。

已将9条的用户说明、episode identity与来源hash单独记录为 [录制截断清单](early_truncation_cases_20261025.json)，该文件目前仅作审计，不被训练程序自动读取。
