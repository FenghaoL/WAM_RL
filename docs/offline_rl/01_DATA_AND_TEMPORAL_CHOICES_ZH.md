# 真实数据、成功标签和动作块时间语义

本轮只检查 `data/ahawam_hil_raw/so101_ahawam_cylinder_hil_v2/20260911_183057`。源 Parquet、视频和标注 JSON 均未改写。

## 1. 实测内容

共 2,941 帧：POLICY 1,228、HUMAN 234、SYSTEM_HOLD 1,479。保存了 7 条 episode，其中 6 条成功、1 条失败；这里的成功包含人工恢复后的完成。

| episode_index | 原始帧数 | 首个 success frame | 成功观察所在来源 | 备注 |
| --- | ---: | ---: | --- | --- |
| 0 | 389 | 无 | — | manifest 为 failure，整条可用 |
| 1 | 404 | 399 | POLICY | 保留至 frame 399 的观察 |
| 2 | 436 | 419 | HOLD | 不能把 hold 删除后丢掉奖励 |
| 3 | 507 | 494 | POLICY | 接管后完成 |
| 4 | 342 | 317 | HOLD | 本条无 HUMAN，仍有执行等待 |
| 5 | 417 | 393 | HOLD | 等待中观察到完成 |
| 6 | 446 | 437 | POLICY | 接管后完成 |

`action` 与 `executed_action_arm` 在这批数据逐值相同；训练以实际下发关节目标为准，而非 `policy_action`。后者在 HUMAN/hold 处不能充当已执行动作。

JSON 正例目前是单帧区间，6 条均未填写 `reviewed_labels=success`。本轮根据用户意图将首个正标记视作首次成功并截断；不将它转换成已审阅的逐帧分类数据。每份标注都校验 episode、logical ID、长度、Parquet 路径和 SHA256，源文件变化即拒绝复用。

其他采集批次：manifest failure 可作为全零奖励；manifest success 没有 success 正标记时为 pending；未知 outcome 也不自动标负。通过 `--roots` 显式加入多个目录，使用 logical episode＋run 作为全局键，禁止跨目录仅凭 episode_index 合并。

## 2. 行、动作和奖励的方向

原数据一行表示 `o_t, q_t, executed_target_t`。构造 `(o_t,A,R,o_(t+m))` 时，`A` 来自实际执行区间 `[t,t+m)`；`o_(t+m)` 不再算一次动作。成功观察 k 的奖励属于 `k-1 → k`。

成功后不再构造转移；末行目标没有后继图像，也不额外构造一个虚拟后继。原 `timestamp=frame_index/15` 是保存帧的名义时间，不能用它掩盖被删掉的控制 tick；必须检查 `tick_index` 和 `monotonic_ns`。

当前有 16 处 tick 缺口，都在接管进入或退出处；最大相邻记录 tick 差为 126，即中间缺少 125 个控制 tick（名义 15 Hz 下约 8.3 秒）。两种构造器都拒绝跨缺口/`dropped_tick` 的动作块，不把它们前后拼接。保留原时间戳供检查，折扣候选当前按实际连续 tick 计；真实单步周期偏差通过 `duration_seconds` 保留。若后续周期波动明显，应改为 `exp(-lambda * dt)`，同时对齐 chunk 内每个奖励时刻。

这会影响稀疏信用传播：reward 不能沿着伪造的相邻行穿过缺口传播。合法转移的 next observation 即使位于一个记录段的末尾，也仍可作为 V 的输入，但该处可能缺少后续行为样本，依赖表征泛化，不能声称有完整的 Monte Carlo 监督。

## 3. A：连续 tick 动作块

以 C=8 为例：从连续记录中取 8 个实际关节目标，POLICY/HUMAN/hold 都是可观察到的行为动作；next state 是 8 tick 后的观察。

```text
tick         0    1    2    3    4    5    6    7   | 8
来源         P    P    P    P    HLD  HLD  HLD  HLD | P
RL action    <----------- 8 个实际目标 ----------> | next obs
```

优点：最接近逐步日志；不需要把延迟归因给另一种环境协议；奖励在 hold 中也不会丢失。C=1 时尤其便于检查因果方向。

代价：会训练到大量保持动作，actor 每次预测的 8 tick 并不等于原 WAM 一次返回的 8 个有效目标。若实际部署又在该块后额外等待一次 WAM 推理，就改变了训练动作的时间含义。若仍采用 sync-hold，必须把新增等待也记入 RL 环境，或设计能按此动作定义运行的本地执行器。

连续 tick-C8 可跨控制来源。在离线 chunk 学习中，人类后续几个动作可能依赖中间观察；把它们视作起点的一次开环宏动作是近似，会受到隐藏历史和行为选择偏差影响。Q-chunking 的优点不意味着任意切片都自动满足其假设。[Q-chunking 论文](https://arxiv.org/html/2507.07969v4)

本轮 C=8、stride=2 得到 1,358 个窗口；24 个 terminal-success 窗口来自 6 个独立成功事件。它们不能跨训练/验证集，不能当 24 次自主成功。

## 4. B：原命令块＋等待的宏动作

POLICY 使用同一 `policy_chunk_id`、从 `action_index_in_chunk=0` 起的实际执行目标；HUMAN 使用最多 C 个连续目标。执行后的连续 SYSTEM_HOLD 归入环境耗时，下一次非 hold 观察或终点作为 next state。

```text
          <--- actor 控制的 C 个目标 ---><---- 执行器等待 ---->
tick       0 1 2 3 4 5 6 7              8 9 ... 18       | 19
来源       P P P P P P P P              hold...         | P/H
RL action  A=(a0,...,a7)                                | next obs
RL duration m=19，discount=gamma^19；等待中仍可收到成功奖励
```

优点：actor 仍输出现有 AHA-WAM 的命令块，hold 不直接成为要模仿的一串动作；总等待时间和成功奖励均保留。对于当前 sync-hold 部署，我更倾向于把它作为主实验候选。

代价：把等待视为环境动力学，就必须保证新策略执行器的 hold 行为一致。缓存化/本地小 actor 更新可能大幅缩短等待，新的环境不再等于旧数据。耗时是随机结果，不应把未来实际 duration 放入 actor 或 Q 的起始状态。本轮 duration 只用于 reward/discount 和日志。

当前构造器只纳入合法完整块，以及到成功/失败终点的执行前缀；短的非终止接管段或取消块会被排除，不为它们虚构尾部。hold 的实际目标必须等于前一个命令的最后目标，否则拒绝这条宏转移。首段没有前序命令的 leading hold 不会被随意附到下一个动作上。

本轮得到 168 条宏转移，包含全部 6 个成功事件；比 tick 窗口少是语义和切片方式造成的，不是数据突然变少六七倍。宏版本当前忽略 `stride`，按实际块边界遍历；不能把 stride=2 理解为宏样本重叠。

## 5. C=1 的意义

得到 2,829 条真实相邻转移、6 个成功事件。它适合验证 reward、缺口和 critic 数据接线。长时间的稀疏信用传播更困难，而且把现有网络远程推理频率改成每 tick 会超出计算预算；因此不能因数据最简单就默认部署 C=1。

## 6. 接管信用的两个不同目标

`mixed`：所有来源进入价值学习。成功回报可以传播到接管前状态，说明“数据中的后续恢复可使这个状态成功”。好处是充分利用人类纠正，IQL 可以提取高价值动作；限制是得到的 Q 不能直接解释为当前自主策略的成功率。

`censor`：POLICY/hold 即将进入 HUMAN 时，提前结束该样本并令 bootstrap=0；HUMAN 段仍然可以训练。这能检查对人类救援的依赖，但把某些尚可恢复的状态当成零延续价值，是改变目标的保守选择。它不是恢复“无接管反事实结果”的统计方法，也没有把接管变成负 reward。

在线后应保存自主/辅助成功、planned chunk、实际 executed prefix、取消原因、版本和状态时间戳。训练能学习人类动作，与评估允许人类帮助，是两件不同的事。

本批真实数据的接管边界伴有时序缺口，过滤已去除跨越这些边界的块；实测 mixed/censor 的窗口数量与终止原因计数相同，不能据此认定接管信用问题不存在。连续接管边界的 censor 行为由专门的合成时序测试覆盖。

## 7. 终止、padding 与有限时限

2026-10-24 补充：针对成功附近是否跨终点取块、只留完整窗口还是保留短前缀，见 [成功边界专项核查](05_SUCCESS_BOUNDARY_AND_TERMINAL_CHUNKS_ZH.md)。HALO、QC 和 DSRL 的具体实现不同，不应由简写 TD 公式推断统一做法。

成功终止：bootstrap=0；失败录像末尾：本候选把它解释为有限时限失败，bootstrap=0。若实际 outcome=failure 仅代表操作者提前中断而非任务失败，则应改为可辨认的截断/未知延续；当前 manifest 的 `termination` 为空，不能假装知道其物理失败原因。这是正式协议需确认的内容。

成功发生在块中途时，保留实际执行前缀。固定 tensor 的尾部重复最后一个真实目标，`action_valid` 只用于 actor loss，不把未来 valid length/成功/来源作为 policy state。Q 始终接收同样形状的完整向量，避免直接把“这个块只执行了 3 步所以成功了”作为输入标签。

这仍是 **terminal-prefix 的工程适配**：没有观察到的尾部不是实际执行事实。需要在 GPU/部署前检查 Q 对 terminal 后不同 padding 的敏感度；更严格的 POLICY 宏动作版本可以从 ledger 还原当时提交的完整 chunk，把环境提前终止作为动力学。接管取消后的未执行尾部则不能这样赋予人的成功结果。当前实现不声称已经完成 planned-command MDP 的精确复现。

## 8. 折扣、历史与延迟的选择

候选默认 gamma_tick=0.997，只是起始参数。在 15 Hz 下，回报半衰期约 15.4 秒，30 秒后的成功折扣约 0.259；gamma=0.999 则约 0.638。若全程 pickup→insert，过强折扣可能让前段 Q 几乎为零，应该比较实际秒尺度，而不是直接复制 HALO YAML。

旧 collector 保存了一些请求哈希和动作 ledger，但没有完整 KV/所有请求原始图像；不能声称仅凭视频已经恢复原 rollout hidden state。当前 exporter 提供新的确定性因果协议：每个查询状态独立 reset，可选择回放过去 H 个观察后再取当前 post-OVCR V 和 reference；不读取未来真实帧，不复用旧 `policy_action` 作为新基座的 reference。

这不是精确在线缓存重放，也不是免费的计算加速：H>0 会重复大量 WAM 推理。对照必须同时评估 **同协议下的 frozen-reference baseline**，否则无法区分历史协议变化和 RL 的收益。实际长期部署仍需实现相同观测/历史构造，或者重新记录真实特征和状态后做在线适配。
