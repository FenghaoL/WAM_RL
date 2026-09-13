# 在线 RL 的实现分工与实验计划

## 1. 实现顺序

目前的 SFT/OVCR suite 脚本适合监督loss、固定epoch和大模型checkpoint。真正的在线RL要管理不断增长的replay、环境reward、target网络和actor发布，不能仅将原 `trainer.py` 的flow loss换成SAC loss便完成。

保持现有任务checkpoint和部署入口作为基础服务。第一版在旁边增加小策略与learner；对现有推理代码只增加明确的特征/噪声接口。下表文件名是 **建议新增位置，尚未实现**。

| 模块 | 建议位置 | 要解决的问题 |
|---|---|---|
| transition定义和校验 | `scripts/so101_pipeline/online_rl/transition.py` | 动作、来源、reward、时间和终止语义 |
| replay | `scripts/so101_pipeline/online_rl/replay.py` | demo/online分池、去重、版本检查、采样覆盖 |
| 小actor/双Q | `src/ahawam/rl/actor_critic.py` | 明确动作Q或噪声Q，不复用Q32名字 |
| SAC learner | `scripts/so101_pipeline/online_rl/learner.py` | TD、target、温度、更新节流和断点 |
| 冻结特征读出 | `src/ahawam/rl/features.py` | 固定pooling与RLTokenReadout对照 |
| 采集桥接 | `scripts/so101_pipeline/online_rl/collector_bridge.py` | 返回实际执行结果与reward；HUMAN不改写成policy |
| 噪声入口 | 现有 `ahawam_chunk_base.py` 与 `ahawam_policy.py` | prepare condition / decode noise分离 |
| 反演工具 | `scripts/so101_pipeline/online_rl/invert_actions.py` | 复用精确schedule、冻结条件、误差筛选 |
| 独立评估 | `scripts/so101_pipeline/online_rl/evaluate.py` | 冻结actor、无接管、固定初始分布 |

依赖尽量保持当前PyTorch环境。官方JAX HIL-SERL/FlowDAgger适合作为算法规范与对照，直接将其机器人驱动、Franka动作和JAX依赖堆进本仓库会增加不必要耦合。

## 2. 数据契约

每条transition至少需要：

| 类别 | 字段 |
|---|---|
| 身份 | episode_id、transition_id、task_id、control_source、policy_version |
| 模型与坐标 | base_ckpt_hash、readout_hash、normalizer_hash、sampler_config_hash |
| 观测 | 当前/下一冻结feature、proprio、观测时间；关联原始双相机帧 |
| 动作 | reference、nominal、实际下发动作、实际执行mask、夹爪语义 |
| RL | reward序列或已定义的块奖励、terminated、truncated、终止原因、bootstrap discount |
| 时序 | decision/command/next-observation时间、实际执行步数、dt、reference/cache年龄 |
| 分块 | chunk_index、offset、generation/request/chunk id、discarded tail |
| 干预 | takeover起止、逐tick来源、clamp/reject标记 |
| 噪声路径 | 实际注入w、完整decoder条件版本、反演标签是否近似及误差 |

对输出actor，保存冻结feature和reference通常足够训练小网络，无需每次重新运行AHA。对DSRL-SAC，保存对应feature和w同理。

对DSRL-NA或反演，仅feature不足以重新生成同一条件的动作：需要与原调用一致的video/history信息及chunk条件。保存每条全量GPU KV通常很大；优先保留有序原始观测、帧索引与可复现的reset/prefill协议，重建成本另计。

推理cache不是环境reward，也不是未来真值。训练、验证、评估的episode划分必须发生在导出feature与反演之前，防止同一rollout的邻近帧跨集合。

## 3. 三张GPU怎样使用

对小actor/critic，三卡数据并行通常没有必要；多卡同步可能比MLP更新本身更贵。建议按功能分配，物理卡号依当前占用选择：

| 卡 | 工作 | 原因 |
|---|---|---|
| GPU A | 冻结AHA推理服务 | 采集延迟优先，不与大batch learner争显存 |
| GPU B | 小actor/双Q learner | FP32、小batch；异步接收experience与发布权重 |
| GPU C | readout离线训练、反演或评估副本 | 不阻塞控制服务；算法稳定后也可闲置 |

小actor执行端优先靠近机械臂；learner留服务器。跨隧道传递经验和版本化小权重可以异步进行，但逐tick动作不可每次等待网络往返。

如果当前基础模型必须占多卡才能满足显存，则先实测可行放置，再决定learner是否CPU或共享GPU。三卡“都忙”不应成为牺牲控制实时性的目标。本轮没有测定当前空闲显存或启动这套进程。

## 4. 启动前的可执行验收条件

这些是拟实现的测试与实测项，**不是本轮已通过的在线RL测试**。

| 验收项 | 应观察到的结果 |
|---|---|
| 零residual等价 | 同一ckpt、stats、seed、条件下，关闭小策略与原输出一致 |
| 连续噪声注入 | 显式w改变当前块；固定w可重现；不污染下一块噪声 |
| 参数冻结 | learner.step后AHA/readout权重及梯度状态不变 |
| Bellman手算 | terminal无bootstrap；真实truncation用final observation；C_eff折扣正确 |
| 干预标签 | Q收到实际执行动作，不是未执行的policy proposal |
| cache/时序 | 多候选decode不推进真实state；真实执行只提交一次位置更新 |
| replay一致性 | 旧base/readout/stats版本混用被拒绝；next_obs不是reset后的obs |
| 熵与动作单位 | tanh logdet正确；b的单位一致；夹爪不混入旋转关节单位 |
| 异步发布 | 同一动作块/控制tick使用一个版本，加载不会阻塞控制 |
| 恢复 | actor/Q/targets/optimizers/α/RNG/replay索引和版本均恢复 |
| 控制耗时 | 独立记录模型、通信、queue wait、feature年龄，均报告p50/p95/p99 |
| 奖励质量 | 随机抽检成功/失败标签，误报成功不可被反复刷分 |

CPU合成MDP应先验证learner能学到已知更优动作，避免把SAC实现bug带上实机。真实模型仅先做不执行机械臂的decode测试；通过后再做受控局部rollout。

对中途停止的块，不能为了凑固定shape而让未执行尾部参与TD。C=8版本若不支持严格的可变时长动作表示，应将混合/中断块留作逐步数据，完整块实验不使用它们作为普通48维动作样本。

## 5. 最值得先做的实验

### E0：冻结基线与数据/延迟诊断，最高优先

选择一个任务、一份基础checkpoint和固定stats。按局部孔口分布采集20–30个评估episode，禁用训练中的人工救援计分；同时保留可中止能力。

测量当前服务的请求往返、推理拆分、队列空闲占比、reference年龄、clamp率，以及失败发生在抓取、接近、对准、接触还是完全就位。旧HIL延迟仅作为诊断线索，不代替此步骤。

若主要失败是执行等待或过时图像，先完成逐步小策略/异步参考的连续执行验证；不在这种情况下盲目启动高UTD噪声RL。

### E1：输出residual，仅BC，与在线RL配对

先训练同结构的BC小actor：只用有效demo/HUMAN纠正，基础AHA冻结。对照SAC版本使用相同初始actor/readout、控制模式和旧数据权限，随后只在SAC支路用新reward更新。

比较回答“提升来自小输出头本身，还是来自在线奖励”。若同时增加新接管数据，BC对照也应获得等量的新人工数据；否则不能把数据差异归因于RL。

C=8的块级对照先保持现有8步结构；若改用C=1逐步控制，BC、SAC和零residual基线一起改，并单列实验。新鲜局部图像编码器与旧readout也必须同样提供给这些对照。

### E2：reference-conditioned RLT风格动作actor

使用相同表征与控制节奏，比较：

- 有界residual SAC。
- 外部完整动作actor，reference conditioning + reference/人工BC约束 + TD3风格Q目标。

第二项可沿RLinf公开实现的loss建立可追溯版本，但σ、BC强度要按SO101坐标校准。它改变了参数化和RL目标，属于路线比较；进一步判断某个组件贡献时，再逐项做reference dropout、BC项、残差边界消融。

### E3：DSRL-SAC噪声策略

固定AHA条件与采样步数，先做32个独立状态、每状态16组噪声的离线输出诊断。统计动作覆盖、纠正方向和噪声敏感性，不把任何候选在真实机械臂上同时执行。

若具有有意义的动作变化，再使用与E1相同的局部初始分布、真实交互预算进行噪声RL。噪声范围先比较b=1.0/1.5，属于参考DSRL的探索范围消融，不是机器人关节幅度。

标准Gaussian基础噪声与tanh bounded噪声不是相同分布。必须单独评估“未训练噪声actor”的基础性能，防止第一步更换噪声分布就改变基线。不能用noise actor的均值w=0代替原基础Gaussian采样而称为公平P0。

### E4：反演BC → DSRL-SAC

沿E3条件，对HUMAN全长有效块先做动作反演，记录接受率与耗时。分三组：

| 组别 | 训练内容 | 回答的问题 |
|---|---|---|
| N-BC | 反演标签，仅噪声BC | 不用reward能否改善 |
| N-RL | 从同一个未做反演BC的初始化做DSRL-SAC | 纯自主噪声探索如何 |
| N-BC-RL | 反演BC初始化，再做同预算DSRL-SAC | 人工先验是否改善探索 |

三组都要披露总人工数据和预处理成本。若需要严格排除额外数据优势，可再给N-RL相同人工数据，但以单独定义的合法方式利用；不能将未反演动作硬塞进噪声Q凑“公平”。

UniSteer式“人工近似latent transition进TD”作为第四组，比较它与仅actor-BC利用人工标签的差异。其主要风险指标是解码误差、Q校准和真实自主成功率。

### E5：OVCR latent研究，后置

只在前面证明reward与控制闭环可学后再做。冻结writer、定义小u并确认其能改变动作；先比latent Q与输出Q的梯度成本，再研究writer结构。不要直接把V32/Q32/T4现有监督权重命名为RL actor。

## 6. 公平预算与统计

建议每条在线路线先做 **30分钟有效机器人交互** 的pilot，包含warmup，另外记录复位、人类指导、反演和总墙钟时间。等交互时间并不等于等transition数量：15Hz下30分钟约27,000个控制tick，C=8完整块约3375条；以上是理论上限，实际按日志扣除缺失和不可用块。

pilot只用于发现不可学习/不稳定路线，不据此宣布最优。进入正式比较后，每种方法使用相同任务初始分布、固定总交互预算、相同评估频率，至少3个独立训练seed/采集session；预算不足时如实标为单seed探索。

最终固定checkpoint，对共同的保留初始条件进行50–100次无接管评估，报告成功率及Wilson区间、成功所需时间、接管/中止率、限幅率。20次评估每次相差5个百分点，小幅改善不能仅凭几次成功认定成立。

同日评估交错方法顺序，控制机械臂发热、物体摩擦和孔位漂移。训练过程中人工复位分布必须记录，不能让某一方法总从更容易的孔口状态开始。

主要指标使用自主成功率和成功/真实总分钟；训练loss、Q值和MSE是诊断项。局部插入成功后，再单独测试完整任务；不把局部reset带来的提升算成全任务提升。

## 7. 何时停止或转向

| 观察 | 合理下一步 |
|---|---|
| 无成功覆盖，actor/Q都缺少正反馈 | 更短局部任务、少量成功示教、保留人工；不先提高UTD |
| 有成功标签但Q无法区分成功/失败状态 | 检查reward时刻、观测历史、表征；比较新鲜双RGB |
| Q持续上涨，真实成功不涨 | 检查OOD动作、replay版本、clip语义；降低UTD/范围 |
| residual频繁达到边界 | 测b是否过小或reference有系统误差，不盲目扩大 |
| 反演噪声很大或重建误差高 | 拒绝该标签；增加迭代/改诊断，必要时放弃噪声路线 |
| learner很快但旧动作仍持续半秒以上 | 优先逐步反馈/异步参考，增加卡数无直接帮助 |
| 旧readout重建好但控制差 | 用zero/shuffled-token消融检查信息是否真正有用 |
| 静态孔口成功，AHA前段接过来失败 | 从基础策略真实到达状态继续训练局部策略 |

在线RL没有通用的“只要加critic就有效”保证。先验证可观测、可控制、可奖励、可重放，才能把实机失败归因到算法本身。

## 来源

算法与默认参数依据见 [方法审计](02_METHODS_AND_VERIFIED_DEFAULTS_ZH.md)；本页GPU分工、30分钟pilot、实验顺序和验收项是针对AHA的工程/实验设计建议，不是某篇论文声称已验证的SO101配方。现有训练与数据背景见 [B方案记录](../v2_experiments/06_B_PRETRAINED_NATURAL_4EPOCH_ZH.md)；本地日志统计见 [audit_snapshot.json](audit_snapshot.json)。

