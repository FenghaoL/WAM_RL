# AHA-WAM 在线 RL 可行性与实施方案

2026-09-18 更新：[VINE与生成策略RL重新评估](05_VINE_AND_GENERATIVE_RL_REASSESSMENT_ZH.md)。补充直接更新diffusion/flow生成器的路线：VINE、QAM、DPPO、ReinFlow、FPO及SAC Flow；区分“快速实机闭环”与“WAM内部奖励适配”的研究优先级。

2026-09-11 补充：在用户明确的 9 月 15 日投稿约束下，先完成局部恢复数据与无记忆监督对照，在线 RL 降为有条件扩展；见 [截止前实验与投稿决策](../cylinder_experiments/04_ICRA_RECOVERY_AND_MEMORY_PLAN_ZH.md)。下文排序针对长期技术路线。

**结论：可以做在线 RL，优先冻结已经适配 SO101 的 AHA-WAM，在动作输出端训练小 actor/critic；第二条路线做 DSRL-SAC 式噪声控制，反演验证通过后再加入人工噪声监督。暂不优先对 V32/Q32/T4 权重直接做在线 RL。**

现有 DAgger 效果弱，说明目前这些监督更新没有解决部署误差；它既不能证明 RL 一定有效，也不能证明已有数据达到了任务上限。第一轮 RL 应回答一个明确问题：在固定基座、同一接触阶段和同一交互预算下，奖励驱动的小策略能否提高独立成功率。

## 阅读顺序

| 文档 | 内容 |
|---|---|
| [01：框架与梯度](01_FRAMEWORK_AND_GRADIENTS_ZH.md) | 现有代码接口、OVCR 是否需要反传 DiT、计算开销与可行性 |
| [02：方法与已核对参数](02_METHODS_AND_VERIFIED_DEFAULTS_ZH.md) | RLT、RLinf RLT 实现、HIL-SERL、Policy Decorator、DSRL、FlowDAgger、UniSteer、DPPO |
| [03：推荐训练协议](03_RECOMMENDED_PROTOCOL_ZH.md) | 更新公式、batch、学习率、奖励、探索、接管、噪声反演 |
| [04：实现与实验计划](04_IMPLEMENTATION_AND_EXPERIMENTS_ZH.md) | 应改哪些接口、三卡分工、验收条件、科学消融与停止条件 |
| [来源与代码版本](SOURCES_ZH.md) | 一手来源、公开实现 commit、证据边界 |

## 路线排序

| 优先级 | 路线 | learner 是否运行 AHA DiT | 人工数据利用 | 主要代价 |
|---|---|---|---|---|
| P0，必须 | 完整 transition、成功标注、延迟审计和冻结基线 | 不训练模型 | 审计旧数据可用范围 | 数据与控制闭环 |
| P1，主线 | 输出端 bounded residual SAC，小 actor + 双 Q | 不需要，表征在采集时固定并保存 | 动作级 transition 可直接用；缺奖励的纠正仅做 BC | 新在线 rollout、可靠奖励 |
| P1，对照 | RLT 风格 reference-conditioned actor | 不需要 | 动作级监督/奖励 | readout 是否保留精度信息；公开复现与原文须区分 |
| P2，第二主线 | DSRL-SAC：噪声 actor + 噪声 Q | 不需要 | 不能把关节动作直接当噪声 | 噪声探索是否能改变插入结果 |
| P2，条件成立后 | 噪声反演 BC → DSRL-SAC / UniSteer 风格混合 | BC/RL 更新不需要；反演需要冻结 DiT 前向 | 通过反演构造噪声标签 | 反演误差、缓存一致性、生成动作与真实动作的差异 |
| P3 | DSRL-NA | 需要大量冻结 DiT 前向，无 DiT 反向 | action critic 可用旧动作 transition | 大模型 batch 推理可能很慢 |
| P3 | OVCR 动态 latent controller | 噪声/latent Q 路线不需要；动作 Q 路线需要 | 没有现成的 OVCR latent 专家标签 | latent 的可控性尚未验证 |
| P4 | OVCR 权重 / action DiT / 全模型直接 RL | 通常需要大网络训练计算 | 取决于算法 | 实机数据需求、内存和迭代延迟 |

这里的“快”指小模型的 **learner 更新可以与大模型解耦**，不是已经测得 SO101 成功率或控制频率提高。AHA-WAM 的图像编码、Video DiT、10 步 Action DiT 和跨服务器通信仍在采集链路中。

旧power-adapter HIL日志中，1835次请求的模型延迟p50约538ms、往返p50约631ms，超过8步动作在15Hz下覆盖的约533ms；SYSTEM_HOLD有11,385/26,658帧。70条episode中66条为接管后成功，仅1条标为自主成功。见[数据快照](audit_snapshot.json)。这不是当前服务测速，但说明连续反馈和奖励归因必须在启动RL前核对。

## 必须区分的三件事

1. **RL Token 是供小策略读取的状态表征。** 之前 T4 是写回 OVCR K/V 的监督 adapter，不是已经完成的 RLT actor/critic。
2. **冻结参数不等于无需反向传播。** 用动作 Q 更新 DiT 前面的模块，仍需对 DiT 的输入求导；用噪声 Q 学一个噪声策略则不需要。
3. **在线 RL 需要新交互，但不要求无人值守。** 可以保留人工接管和人工成功标注；自主成功率评估时必须关闭接管，接管率另外报告。

方法依据包括 [RLT](https://www.pi.website/research/rlt)、[DSRL 官方实现](https://github.com/ajwagen/dsrl)、[HIL-SERL 官方实现](https://github.com/rail-berkeley/hil-serl)、[FlowDAgger 官方参考实现](https://github.com/microsoft/FlowDAgger)。这些工作支持方法组成，不构成 AHA-WAM/SO101 上的效果保证。

## 本次交付边界

这是基于仓库代码和一手论文/公开实现的研究与实施规格，检索和代码审计截至 2026-09-10。没有启动实机在线 RL，没有新增一个声称已经验证成功的 learner，也没有修改部署参数、基座 checkpoint 或既有训练任务。

参数表明确分成“原论文/公开代码里的值”和“AHA-WAM 的建议起点”。新文档补充此前的研究结论：噪声空间的人类动作标签可以通过近似反演构造；AHA 当前子类关闭 action-history KV，不能将父类的缓存写入逻辑误认为当前缺陷。
