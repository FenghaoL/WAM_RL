# 成功附近怎样取动作块：终止、自举、跨成功与尾部补齐

核查日期：2026-10-24。针对用户新增问题核查论文和作者代码。本轮补充研究说明，没有改变现有训练/数据构造的候选策略，也没有裁剪原视频。

## 1. 先给针对本项目的建议

建议保留原视频与标注，以“首次满足最终成功判据”作为 **RL 的逻辑终止**。成功之前的样本都可以采，不仅保留唯一一个恰好提前 C 步的完整块；离成功不足 C 步的起点也值得保留，用正确的终止前缀和固定形状表示处理。

不建议为了凑足 C 个真实动作而必须读取成功后的撤退、松手、人工复位动作。如果动作块原本是在成功前一次提交的完整计划，则可以保存完整计划，实际执行到成功便终止；它与事后把跨请求/跨接管的动作拼成一个新计划不是一回事。

**成功后几帧的观察是否影响 target，取决于 bootstrap mask；不取决于画面离成功有多近。** 成功终止时 mask=0，下一状态的 Q/V 数值应被排除；若 mask=1，仅仅认为“反正主要是传播 r”是不正确的。

以下用连续 tick 的 C=8 解释。若选择命令块＋hold 宏动作，终止原则相同，但奖励偏移和 discount 应按包含 hold 的实际时间计算，不把“成功前 8 个 tick”和“成功前 8 个有效命令”混为一谈。

## 2. 成功、录像结束、bootstrap 必须分开

| 概念 | 含义 | 本项目建议 |
| --- | --- | --- |
| success observation | 第一次观察到任务完成的状态 s_T | 人工标注的最终判据 |
| task termination | 成功以后不再属于本次优化目标 | 首次成功后逻辑终止 |
| recording boundary | 视频或采集 episode 在哪里结束 | 可以比 T 晚，原始数据保留 |
| bootstrap mask | 是否继续估计未来回报 | 成功终止取 0；其他情况按任务定义 |
| action padding mask | 哪些动作位置有实际监督 | 只控制有效前缀，不能替代 bootstrap mask |

若任务要求“插入后持续稳定 1 秒”或“插入并松手”，刚碰到插入状态就不该当最终成功。应先统一判据，把 T 移到条件真正完成处，再讨论是否跨 T；不能靠保留不一致的成功后尾段隐式学习这个条件。

## 3. 正确 target 与用户提出的 next observation 问题

一行数据包含观察 s_t 和随后执行的 a_t。成功观察为 s_T，则奖励属于进入它的转移：r_(T-1)=1，而非 s_T 后发出的动作。

从 t 出发，计划 C 步，实际到终止或块末经历 m 步：

$$
y_t=\sum_{j=0}^{m-1}\gamma^j r_{t+j}
   +\gamma^m b_t\,U(s_{t+m}),
$$

U 在 IQL 中是 V，在 TD3 风格更新中是目标双 Q 的最小值；成功终止 b_t=0。

例如 T-t=3，C=8，首次成功后不再给奖励：

$$
y_t=\gamma^2.
$$

此时无论结构体保存 next_obs=s_T 还是同一录制轨迹的 s_(t+8)，**只要后者的 value 被严格 mask 掉**，它都不出现在标量 target 里。成功处 gamma^3 或 nominal gamma^8 乘上零在这一项也相同；终止前奖励的折扣位置 gamma^2 仍然必须正确。

如果错误地继续 bootstrap，gamma=.99、估计的 V(s_(t+8))=.8，则 target 变成约 1.7183，而正确值是 .9801。它会把不应存在的未来价值也传播回去，并不是无关紧要的几帧偏移。

工程上仍建议 next_obs 保存 s_T：方便一致审计、避免不必要的未来特征计算。mask=0 并不允许输入 NaN 后指望 `0*NaN` 变成零，也不能忽略 batch-coupled 网络或有状态特征提取器的副作用。若成功并不终止、b_t=1，则 next_obs 必须对应动作定义下真正的块末状态，不能随便改成 s_T。

## 4. 论文与代码分别怎么做

### RLT：确认有滑窗与终止信号，没有确认到尾部策略

[原文 §IV–V](https://arxiv.org/html/2604.23073v1) 明确：critic 接收 chunk，使用块内折扣奖励；中间观察可生成 stride=2 的重叠窗口；操作者提供成功/失败的 terminal signal。

但公开正文、算法与所核查附录没有明确说明“最后不足 C 步的窗口是否用于 Q”“怎样 pad”“next_obs 到底放在哪里”。公式也省略了常见 terminal mask。**因此不能把我们现在的 padding 说成 RLT 原版的确定实现。** 独立社区复现也不能替代作者对此细节的证据。

### HALO-WA：有明确的终止块保留、repeat-last padding、关闭 bootstrap

核查本地干净 checkout `611008193fd3438bf99cc8f8377858fbdfcb9555`，并读取对应远端文件：

- [`fsdp_gigawa_policy_worker.py`](https://github.com/YeanRoot/HALO-WA/blob/611008193fd3438bf99cc8f8377858fbdfcb9555/rlinf/rlinf/workers/actor/fsdp_gigawa_policy_worker.py#L1819)：`_find_first_terminal_position` 优先检查 termination/done，后备检查首个非零 reward；`_sanitize_single_sample_trajectory` 保留包含终止的最后一个 chunk、去掉后续 chunk。
- 同一函数将终止槽之后的 action/ref_action 重复为终止槽动作，终止之后的 rewards 设零，termination 从终止槽起设 true。这是在块内处理尾部，**不是仅留下刚好一个满 C 步到终点的样本**。
- 可选 sliding-window 路径使用 `valid_len`，不足 C 的动作重复最后值补齐；scalar reward 补 0；不同配置控制标志 padding。是否启用滑窗须看配置，不能说所有实验一定全滑窗。
- `_compute_critic_outputs` 使用 `terminations.any(-1)`，[`gigawa_losses.py`](https://github.com/YeanRoot/HALO-WA/blob/611008193fd3438bf99cc8f8377858fbdfcb9555/rlinf/rlinf/algorithms/gigawa_losses.py#L73) 的 TD target 乘 `1-done_mask`。块末 next_obs 即使晚于实际终止，也不通过 value 延续。

这是公开 RoboTwin worker 的实现证据，不能外推到论文没有公开的全部真机数据处理。它的 chunk reward 直接求和，也不能照搬为本项目按 tick 定义 gamma 的折扣和。首个非零 reward 后备规则只适合其相应奖励合同，不适合所有含负奖励任务。

### QC / QC-FQL：必须同时检查 terminals 与 masks

[作者 `utils/datasets.py::sample_sequence`](https://github.com/ColinQiyangLi/qc/blob/main/utils/datasets.py#L114) 分别计算：累计 reward、累计最小 bootstrap mask、累计 episode terminal，以及 `valid_i=1-terminal_(i-1)`。

[`agents/acfql.py::critic_loss`](https://github.com/ColinQiyangLi/qc/blob/main/agents/acfql.py#L22) 用 `valid[..., -1]` 乘整个 Q loss；[`agents/acrlpd.py`](https://github.com/ColinQiyangLi/qc/blob/main/agents/acrlpd.py#L41) 有同样的最后位置权重。因此：

| 所采窗口情况 | 最后 valid | Q loss 是否使用 |
| --- | ---: | --- |
| 尚未到录制 episode 边界 | 1 | 使用 |
| 最后一步恰好是 episode 终止 | 1 | 使用，bootstrap 看 mask |
| episode 在块中间已经终止 | 0 | 该窗口不贡献 Q loss |

所以若人为把“首次成功”同时设成 episode boundary，固定 C 的 Q 数据会只使用恰好结束于边界的完整窗口，较晚起点的短窗口不贡献 Q loss。flow BC 路径还有逐位置 valid，不能由 Q loss 的结论推断这些起点在所有 loss 中都被丢弃。

**但 QC 的 OGBench 离线数据不必在成功处截断录像。** [OGBench `relabel_utils.py`](https://github.com/seohongpark/ogbench/blob/master/ogbench/relabel_utils.py#L4) 重算 reward/masks，在成功处 mask=0；它没有把所有成功位置写入原 `terminals`。后者仍界定原采集 episode。[`ogbench/utils.py`](https://github.com/seohongpark/ogbench/blob/master/ogbench/utils.py#L16) 还明确排除跨 reset 的伪转移。

因此 QC 可以出现“跨成功、未跨原 episode boundary、Q loss 有效、bootstrap=0”的窗口。其具体奖励含未成功的负奖励，sampler 的 reward 累计也没有自动在第一个 mask=0 后停止；不能把它当作与本项目首次成功单次 +1 完全相同的目标。

### DSRL：有执行完整块、最后才汇总 done 的公开例子

[`ajwagen/dsrl/env_utils.py::ActionChunkWrapper.step`](https://github.com/ajwagen/dsrl/blob/main/env_utils.py#L139) 对整个块逐动作调用环境，没有在第一个 done 时 break；返回最后观察、奖励求和，以及 `max(done_)`。这说明**实际代码确实可能返回成功点之后的块末观察**。

这是该 wrapper 的事实，不代表所有 DSRL 平台，也不说明每个 Gym 环境都允许在 done 后继续 step。它的 reward 累加、底层成功判定、done→learner 的语义需配套核查；如果成功状态持续给 1，简单求和会奖励成功持续时间，与本项目单次成功事件不同。

### IQL、TD3+BC、FQL、ConRFT、HIL-SERL、V-GPS

这些工作的已核查更新/采集代码提供了共同的 bootstrap 证据，而非统一的 chunk 尾部策略：

| 工作 | 实际证据 | 不能从这里推出什么 |
| --- | --- | --- |
| [IQL critic](https://github.com/ikostrikov/implicit_q_learning/blob/master/critic.py) | `reward + discount * masks * next_v`；[数据](https://github.com/ikostrikov/implicit_q_learning/blob/master/dataset_utils.py) 区分 episode 分割与 bootstrap mask | 原始单步算法没有规定本项目 C=8 的 padding |
| [TD3+BC](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py#L106) | target 乘 not_done | 不等于只要双 Q 就可以忽略终止 |
| [FQL](https://github.com/seohongpark/fql/blob/main/agents/fql.py#L18) | target 乘 masks | 原 FQL 与 QC-FQL 的序列过滤不是同一层代码 |
| [ConRFT collector](https://github.com/cccedric/conrft/blob/main/examples/train_conrft_octo.py#L207) | 存 `masks=1-done`；[critic](https://github.com/cccedric/conrft/blob/main/serl_launcher/serl_launcher/agents/continuous/conrft_single_octo_cp.py#L263) 使用它 | 没有证据说明它原样处理 AHA 的 8 步 terminal prefix |
| [HIL-SERL collector](https://github.com/rail-berkeley/hil-serl/blob/main/examples/train_rlpd.py#L189) | 保存 `masks=1-done`，done/truncated 后 reset | 不能把其单步经验说成 chunk padding 的直接实验支持 |
| [V-GPS IQL 分支](https://github.com/nakamotoo/V-GPS/blob/main/jaxrl_m/agents/continuous/iql.py#L177) | target 乘 masks；不同 agent/数据分支要分别看 | 该分支不是对全部 V-GPS 实验的统一说明 |

[VINE 原文及附录](https://arxiv.org/html/2607.10369v1) 没有给出足以确认本问题的 C=8 success-tail 数据配方；其主要离线证据来自状态输入 OGBench，公开仓库未提供完整训练代码。它的生成采样链设计不决定 RL episode 是否应在成功处终止。

## 5. “只保留提前 8 步”与“保留成功附近前缀”的区别

以下 T 是首次成功观察，奖励在 T-1 的转移，C=8：

| 起点 | 实际用到的动作 | next_obs 建议 | target |
| --- | --- | --- | --- |
| T-12 | 8 步，到 T-4 | s_(T-4) | gamma^8 U(s_(T-4)) |
| T-8 | 8 步，恰到 T | s_T | gamma^7 |
| T-3 | 3 步，终止前缀 | s_T | gamma^2 |
| T-1 | 1 步，终止前缀 | s_T | 1 |
| T 及以后 | 不再作为本任务的新起点 | — | — |

仅使用完整窗口是一个可成立的保守基线，QC 的某些 boundary 定义就会产生这种效果。但小数据下，只保留 T-8 会让最接近成功的状态缺少直接 Q/优势提取监督；在 IQL 中这也影响这些状态的 V 拟合覆盖。保留终止前缀有数据利用率上的理由，不必因为无法执行满 8 步就丢弃。

还有一个采样细节：若只按起点 `0,2,4,...` 取窗口，T-8 不一定落在网格上。选择“只保留完整终止窗”的对照时，应显式补入合法的 T-8 anchor，否则可能把这一轨迹唯一的完整 terminal-success 样本也漏掉。当前实现没有单独做此 anchor 策略，不能把它说成已完成的 QC 精确复现。

重叠窗口仍是同一次成功的相关样本；可适量采样或按 episode/终止事件平衡，不能当成新增独立成功，也不能随机拆到训练/验证两侧。

## 6. Q 输入、actor 监督与 padding 是另一层问题

关闭 bootstrap 只解决 target 的价值延续，不自动解决 Q 输入的完整 C×6 动作向量如何构造。

1. **完整计划在终止前已经提交**：可输入该计划，环境在成功后吸收终止，记录真实 executed prefix。尾部不作为实际执行事实，不把之后人工接管的结果归给它。
2. **只有实际执行前缀**：需要 padding 或显式的 prefix-value 模型。当前候选重复最后一个真实关节目标补齐，不把 padding 当 actor 的真实监督；Q 接收固定形状向量。这是有先例的工程适配，仍需检查尾部依赖。
3. **读取成功后的真实动作凑足 C**：同一录制 episode 内可构造数组，但若优化的是首次成功，后缀对本次回报没有贡献；它还可能包含下一次请求、人工复位或撤退。不能因为数组有真实值就把它们当成通向成功的必要专家动作。

理论上的 full-command Q 评价完整计划；一个显式依赖“计划执行长度”的 Q 是另一种函数定义。若长度是事先选择的决策变量，它可以是合法输入；若把事后才知道的成功时刻/valid length 当作部署时可用条件，会有标签泄漏。不能把所有长度 mask 都笼统称为错误，也不能把未来终止长度伪装成当前已知量。

当前 IQL 的 actor 加权 MSE 只用有效前缀，Q 不直接输入 action_valid。当前 TD3+BC 对照还会把 actor 的终止尾部 canonicalize 后再查询 Q。这些是本项目的候选适配，不能当作所有论文的原版配方；在有限数据下仍要比较 padding、完整计划和完整窗口基线的差异。

## 7. 原视频该不该截断

建议：**原视频和标注保留；RL 视图在 T 逻辑终止；成功附近的短前缀也保留。** 因而“截断 RL episode”和“网络必须接受一个固定长度 tensor”没有矛盾。

原始尾段可用于审核 success 判据、训练成功分类器、判断是否稳定完成；这些用途与是否允许从那里继续 bootstrap 不同。如果研究目标改成持续稳定、释放或安全撤退，就重新定义任务终点和奖励，不能只改为跨成功取块而沿用旧的首次成功目标。

我们当前 raw 数据没有被物理截断。当前 `data.py` 已经保留被选中起点的 terminal prefixes，并设置 discount=0；tick 默认 stride=2 不表示每个临近起点都采。macro 路径按命令边界取样，不应为了增加样本而随意把每一帧当成真实宏决策点。是否对 Q 做 RLT 式滑窗增强，应与 hold/动作空间的选择一起确定。

## 8. 本轮执行验证与源码标识

本轮没有运行外部仓库完整训练。使用 AST 提取作者原函数，并以 NumPy/简单 stub 调用，避免 JAX/Gym/大模型依赖：

- QC，C=4，episode terminal 在 transition 5：起点 2/3/4/5 的 Q-loss 权重分别为 `[1,0,0,0]`。
- 同样 success mask 在 transition 5，但 recording terminal 放在 15：上述起点的 Q-loss 权重为 `[1,1,1,1]`，bootstrap masks 均为 0。验证了成功和录制边界的区别。
- DSRL 原 wrapper，stub 在第 3 步报告一次 reward=1/done=true：实际仍调用环境 8 次，返回第 8 步观察、reward=1、done=true。该测试只证明 wrapper 控制流，不证明真实环境允许终止后执行或训练效果。

源码指纹：QC datasets blob `b457145703cfc9fe208b7722af523cdb01cd8fa1`；QC-FQL blob `291acd830f4a652e8c2d566449e5eb42673412ae`；OGBench relabel blob `4088a8a9bb642ec51ae957e4098fddc1aecee641`；DSRL env_utils blob `7ddbacc221023a3b3adcceb8c6a53998659fab33`；HALO worker 本地 blob `cbfb0d7933345ad1c6f2b26b6b2534dee6b41224`。

外部代码仅用于本轮只读审计/临时函数验证，未复制进本项目训练依赖。原视频、标签、候选 learner 和数据采样行为均保持原状。
