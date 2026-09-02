# AHA-WAM + SO101 插入任务：WAM、VLA、Diffusion Policy 与强化学习文献全景和技术路线

> 检索截止：2026-08-23
> 目标系统：AHA-WAM；SO101；全局 RGB + 腕部 RGB；约 15 Hz；6 维状态/动作；计划中的动作块为 8 步；当前不假定具有腕部六维力/力矩或触觉。
> 配套文档：已有的 HALO-WA 逐代码分析见 [HALO_WA_RL_AND_AHAWAM_MIGRATION_GUIDE_ZH.md](./HALO_WA_RL_AND_AHAWAM_MIGRATION_GUIDE_ZH.md)。针对 π*0.6/π0.7、隐藏在“自我改进/经验学习”名义下的 VLA-RL，以及“外挂 actor、ActionDiT、VideoDiT、整套 WAM 到底更新哪一层”的重审，见 [AHAWAM_RL_SCOPE_AND_VIDEO_ACTION_COTRAINING_REVIEW_ZH.md](./AHAWAM_RL_SCOPE_AND_VIDEO_ACTION_COTRAINING_REVIEW_ZH.md)。本文不重复它们的全部代码细节，而是把视野扩展到 WAM、VLA、diffusion/flow policy、真实机器人 RL、精密插入与接触控制。

---

## 0. 先给结论

### 0.1 最推荐的总架构

对当前硬件，最稳妥、最有论文依据、也最有研究价值的结构不是“直接对整个 AHA-WAM 做 PPO”，而是：

```text
全局相机 + 腕部相机 + proprio
                │
                ▼
        冻结或慢更新的 AHA-WAM
        长时程理解、靠近、粗对准
        输出 nominal action chunk
                │
                ├──────────► 供 critic 判断的参考动作/latent
                │
                ▼
  接触阶段的小型闭环 adapter / residual policy
  每个真实控制步重新观察，只允许毫米级/小角度修正
                │
                ▼
  独立安全层：限位、限速、限加速度、超时、回退、接管
                │
                ▼
              SO101
```

这不是一个孤立猜想，而是多条独立研究线共同指向的交集：

- [ResiP](https://arxiv.org/abs/2407.16677) 把 action-chunk BC/diffusion policy 当“轨迹规划器”，冻结基座，用逐控制步 PPO residual 补足局部闭环；这是与 AHA 插入最直接的算法模板。
- [RLT](https://arxiv.org/abs/2604.23073) 从冻结 VLA 学专门的 RL token，小 actor/双 Q 同时读取 latent、本体状态和 VLA 参考动作块；真机任务直接包含 Ethernet 和充电器插入。
- [HIL-SERL](https://arxiv.org/abs/2410.21845) 与 [SiLRI](https://arxiv.org/abs/2512.24288) 给出少量示教、双 replay、人类接管、成功分类器和实机 off-policy RL 的成熟训练闭环。
- [RLDG](https://www.roboticsproceedings.org/rss21/p028.html) 表明可以先让窄任务 RL 专家学会连接器插入，再把高质量 RL 轨迹蒸馏回 OpenVLA/Octo；这给出了“RL 不必直接改大模型”的另一条稳健路径。
- [FAWAM](https://arxiv.org/abs/2606.08555)、[VT-WAM](https://arxiv.org/abs/2607.02503) 和 [Tactile-WAM](https://arxiv.org/abs/2606.26663) 一致说明：最后几毫米的接触、卡滞、滑移和力并不总能从 RGB 中判断。它们不是当前无触觉 SO101 可直接复现的方案，却明确揭示了纯视觉系统的上限。

### 0.2 方法优先级

| 优先级 | 路线 | 为什么适合当前 AHA-WAM | 现在是否建议做 |
|---|---|---|---|
| A0 | HG-DAgger/IWR + 完整 transition 记录 | 先让系统安全地产生成功、失败、纠正数据；也是后续 critic 的地基 | **立即做** |
| A1 | ResiP 式逐步视觉 residual | 与动作块 WAM 的结构最匹配，模块小、可限幅、可回滚 | **第一个 RL 主线** |
| A2 | RLT/HALO 式 latent + ref-action actor/critic | 保留 WAM 语义先验，直接利用动作块和失败数据 | **与 A1 对照** |
| A3 | HIL-SERL/RLPD 或 SiLRI 训练系统 | 解决真机样本效率、人类接管和稀疏奖励 | **作为训练骨架** |
| B1 | Q-chunk / Max-Q / IDQL 候选排序 | 原生对应 AHA 的 `8×6` chunk，不必先改大模型 | A 线稳定后做 |
| B2 | RLDG/PLD 式 RL 轨迹蒸馏 | 把恢复和精细动作最终吸收到 AHA 中，部署简单 | A 线稳定后做 |
| B3 | EXPO-FT 多候选 + edit actor + Q | 利用 diffusion 多样性，但 AHA 多次采样可能太慢 | 作为增强实验 |
| C1 | DSRL 噪声 steering | 冻结 ActionDiT，仅学噪声；仍在生成分布内探索 | 论文研究支线 |
| C2 | DPPO/FlowDPG/ALOE 直接更新生成策略 | 理论完整，但实机样本、显存、概率计算与稳定性成本高 | **不做第一版** |
| C3 | WMPO/WoVR/World4RL imagined RL | 需要可按候选动作预测后果的世界模型 | 当前 AHA **不能直接做** |
| D | 联合更新 VideoDiT + ActionDiT 的 WAM-RL | 对长时程有研究价值，但最贵、最不稳，且不是插入首要瓶颈 | 长期方向 |

### 0.3 最关键的三个现实判断

1. **AHA 当前不是可直接拿来做 MPC/RL rollout 的 action-conditioned simulator。** 当前配置 [`configs/model/ahawam.yaml`](../../configs/model/ahawam.yaml) 明确是 `video_dit_config.action_conditioned: false`。代码虽然预留了 action embedding，但当前配置、checkpoint 和部署路径没有证明其学会了 `候选动作 → 对应未来`。所以不能把“会生成未来视频”自动等同于“可反事实评估任意动作”。
2. **SO101 与论文常用 Franka/KUKA 并不只差价格。** 大量插入 RL 使用任务空间速度/位姿增量、腕部 F/T 和 1 kHz 阻抗控制；SO101 常见接口是有齿隙和延迟的位置伺服。成功率不能横向照搬，探索边界、安全层和被动柔顺反而是第一等设计变量。
3. **插入要分阶段。** 自由空间靠近是视觉和长时程规划问题；接触后的搜索/对准/压入是高频、部分可观测、受动力学支配的问题。用一个固定 8 步 chunk 同时解决两者，通常不是最好的控制尺度。

---

## 1. 零基础概念：这些论文究竟在优化什么

### 1.1 模仿学习、DAgger 与强化学习

设机器人在时刻 (t) 看到状态或观测 (o_t)，输出动作 (a_t)，环境转移到 (o_{t+1})，并给奖励 (r_t)。

- **行为克隆（BC）**：拿人类的 `(观察 → 动作)` 配对训练。优点是稳定；缺点是只学到数据覆盖的状态，一旦偏离示教便容易累积错误。
- **DAgger**：让当前策略上机，人在它即将犯错时给正确动作，把这些“策略真正会到达的坏状态”加入数据再训练。
- **HG-DAgger**：由人主动接管。它首先是交互式模仿学习，不等于 RL，因为 actor 不一定通过 reward 或 Q 更新。
- **RL**：不要求每一步都有专家动作，而是利用结果好坏学习。插入任务常只有最终成功 `1`、失败 `0`，因此奖励很稀疏。
- **HIL-RL**：把人类接管同时当成安全机制和学习数据。人类不一定是完美专家，所以 SiLRI、Max-Q 等方法会判断“这次接管是否真的更好”。

### 1.2 actor、critic、Q 值

- **actor** 是动作策略：输入观察，输出动作。
- **critic** 判断“在这个状态执行这段动作，以后大概能获得多少累计奖励”。常写作 (Q(s,a))。
- **双 Q** 同时训练两个 critic，并取较小值，目的是减轻少量数据下的过度乐观。
- **off-policy RL** 能反复使用旧示教、失败和接管数据；对昂贵真机通常比只用一次数据的 on-policy PPO 更合适。

### 1.3 action chunk 为什么既有用又会伤害插入

AHA 一次输出未来 8 步：

\[
A_t=[a_t,a_{t+1},\ldots,a_{t+7}].
\]

它降低了长时程预测难度，让动作更连贯，也减少了大模型调用次数。但如果 8 步全部开环执行，机器人在第 2 步碰到孔边后，第 3–8 步仍可能继续向错误方向推。

因此有三种 RL 粒度：

- **chunk-level RL**：把整个 `8×6` 当一个宏动作，critic 评价整块。HALO、RLT、Q-chunk 属于这一类。
- **step-level residual RL**：AHA 仍给 8 步计划，但每个真实步都加一个小修正。ResiP 属于这一类。
- **adaptive chunk**：自由空间用 8 步，接触前缩成 2–4 步，接触后逐步闭环。它最符合插入的多时间尺度本质。

### 1.4 residual policy 是什么

若 AHA 的当前参考动作为 (a_t^{base})，小策略输出修正 Δa：

\[
a_t^{exec}=\operatorname{SafetyFilter}⁡\left(a_t^{base}+\alpha_t\,\operatorname{clip}(\Delta a_t,-b,b)\right).
\]

其中：

- (b) 是每个关节/夹爪的最大修正；
- α 可以是接触阶段 gate；
- SafetyFilter 是独立硬安全层，不由 critic 自己“学会安全”。

这类方法的价值是：RL 只需学“基策略差在哪里”，不必重新发现完整任务。

### 1.5 WAM 与 action-conditioned world model 不是同一个概念

“世界—动作模型”常把未来视觉表示与动作生成放进同一系统，但存在两个不同方向：

```text
AHA 当前主要信息方向：  图像/未来视觉上下文 ──► 动作

MPC/RL simulator 要求： 当前图像 + 候选动作 ──► 候选动作导致的未来图像/状态
```

第二种才允许 CEM、MPC、WMPO 反复尝试候选动作。AHA 源码的 `WanVideoDiT` 有 action-conditioned 代码路径，但当前配置关闭；这意味着“未来视觉 latent 可以给 actor 用”，不意味着“它能正确预测反事实动作后果”。

---

## 2. 本次检索如何分层

“所有相关论文”在快速增长的 2025–2026 机器人 RL 领域无法按字面保证穷尽。本文采用可复查的边界：

- 时间：公开至 2026-08-23；
- 证据：论文原文、会议页面、作者项目页、官方代码；
- 任务：插入、连接器、peg-in-hole、装配、接触操作；
- 模型：WAM、VLA、diffusion/flow action policy、视觉/触觉 world model；
- 学习：在线 RL、离线 RL、offline-to-online、residual RL、HIL-RL、RL 轨迹蒸馏、价值引导推理；
- 排除：只有营销描述且没有论文细节的项目，不作为主要证据；把“偏好训练/蒸馏/规划”与真正 policy-gradient/actor-critic RL 分开标记。

文献按迁移距离分为：

1. **S 级：直接回答 AHA + 插入 + RL 的核心文献**；
2. **A 级：直接回答其中两个维度**；
3. **B 级：提供重要组件、反例或未来硬件方向**；
4. **C 级：概念相关，但当前硬件/代码边界使其不宜立即实施**。

---

## 3. S 级核心必读：最直接决定方案的工作

| 方法 | 学习方式 | 更新什么 | 插入/装配证据 | 对 AHA 的用途 | 最重要限制 |
|---|---|---|---|---|---|
| [ResiP](https://arxiv.org/abs/2407.16677) | on-policy residual RL | 冻结 chunked BC；PPO 更新逐步 residual | 0.2 mm peg；多件装配 | **第一 RL 主线** | RL 主体在仿真；SO101 无阻抗/F/T |
| [RLT](https://arxiv.org/abs/2604.23073) | off-policy actor-critic | 冻结 VLA；RL token、小 actor/双 Q | Ethernet、充电器、螺丝、扎带 | **latent/ref-action 主线** | 论文平台 50 Hz、多相机、强硬件 |
| [HALO-WA](https://arxiv.org/abs/2607.04265) | 在线 actor-critic | 冻结 WAM；latent-guided adapter | 论文精密对准/插入 | AHA 最近邻 | 公开代码与论文真机路径要分别审计 |
| [HIL-SERL](https://arxiv.org/abs/2410.21845) | HIL off-policy RL | 小视觉 actor/critic | RAM、SSD、USB、皮带等 | **训练系统骨架** | 依赖 Franka/阻抗/持续人工监控 |
| [SiLRI](https://arxiv.org/abs/2512.24288) | HIL off-policy RL | state-wise Lagrange 权衡 RL/模仿 | Insert USB | 处理非完美接管 | 新预印本；策略不是 WAM |
| [RLDG](https://www.roboticsproceedings.org/rss21/p028.html) | RL 教师 + SFT 蒸馏 | specialist 用 RL；VLA 用 SFT | 多种连接器 | **将 RL 能力吸回 AHA** | 需单独训练 specialist |
| [Max-Q Selective Imitation](https://arxiv.org/abs/2608.15088) | MC Q + 选择性模仿 | chunk critic 和 ACT/flow actor | USB pick-and-insert | 与 `8×6` chunk 极匹配 | 截止检索日只发布约一周 |
| [SPARR](https://arxiv.org/abs/2602.23253) | 仿真 base + 真机 residual RL | 只在真机学视觉 residual | 10 类两件式装配 | 有仿真时的强路线 | 工业臂/仿真资产与 SO101 差异大 |
| [ViTaL](https://arxiv.org/abs/2506.13762) | BC encoder + residual RL | 局部视触觉策略 | USB、插头、钥匙、刷卡 | 支持“全局到达 + 局部接触策略” | 需要触觉；不是 WAM 本体 RL |

### 3.1 ResiP：为什么它几乎就是 AHA 插入问题的抽象

ResiP 的核心观察是：现代 action-chunk 策略很擅长产生一段形状合理的轨迹，但不像真正的反馈控制器。它比较了 diffusion policy、ACT 和 MLP，发现单纯继续增加 BC 数据会饱和；甚至 DAgger 也未必补足 chunk 内反应性。[论文与官方项目页](https://residual-assembly.github.io/)报告，冻结基策略后加入每步 residual，可把多类精密装配从约 5–50% 提到 95% 以上，其中 0.2 mm peg 任务从约 5% 到 99%。

迁移到 AHA 时可写成：

```text
每 8 步：AHA 生成 nominal chunk
每 1 步：residual 读取最新腕部图像、qpos、nominal 当前步、chunk phase
每 1 步：输出有界 joint delta
若接管/安全触发：立即废弃旧 chunk，不继续执行
```

需要正确理解其证据边界：ResiP 的 residual RL 主要在仿真；RGB 真机部署使用 RL 轨迹蒸馏、视觉随机化和少量真机示教。它证明了架构合理性，不等于已经证明“低成本位置伺服臂上直接在线 PPO 安全有效”。

### 3.2 RLT：最值得迁移的不是 TD3 名字，而是 RL 接口

RLT 的关键不是“又一个 TD3”，而是给大 VLA 建立一个小而稳定的 RL 状态：

```text
VLA 内部 token ─► encoder ─► 低维 RL token
                                      │
proprio ──────────────────────────────┼─► actor ─► refined action chunk
VLA reference action chunk ───────────┘
                                      │
                                      └─► twin-Q critic
```

[RLT 原论文](https://arxiv.org/abs/2604.23073)使用长度 10 的动作块、50 Hz 控制；actor/critic 很小，并通过 reference-action regularization 和 reference dropout 平衡“不要忘掉 VLA”与“不能永远照抄 VLA”。论文使用人类成功标注与干预，可先只训练关键精细阶段，再扩展完整任务。

对 AHA 的真正启示：

- RL feature 不能随便从巨大 KV cache 平均一下就算完成。RLT 先用任务示教训练 encoder-decoder，让 RL token保留可重建、可控制信息。
- AHA 的 VideoDiT/OVCR 是有状态的；rollout 时应保存抽出的 feature、planner age、phase、reference chunk 和模型版本。只保存一张 RGB，日后离线重算未必得到相同 latent。
- actor 可以输出完整 `8×6` refined chunk，但接触阶段还应允许执行 1–2 步后重新规划；否则仍有 chunk 内开环问题。

### 3.3 HIL-SERL、SiLRI：它们解决的是“怎样真的在机器人上学”

[SERL](https://arxiv.org/abs/2401.16013) 和 HIL-SERL 的主要贡献包括：

- 20–30 条左右示教给出非零成功和安全起点；
- demo buffer 与在线 buffer 分开，训练时常等量取样；
- 成功分类器把相机观测变成稀疏 reward；
- actor 和 learner 异步，机器人不必等每次梯度更新；
- 人类接管既阻止损坏，也产生状态分布内的纠正 transition。

SiLRI 指出一个更细的现实：人类接管不总是最优。它学习状态相关 Lagrange 权重；在人类动作一致、可信的状态更重视模仿，在人类操作分散的状态更信 RL 目标。其论文包含 Franka 的 USB 插入，并报告达到 90% 所需时间相对 HIL-SERL 至少减少一半；但这是 2025/2026 的新预印本，仍应以复现为准。

对数据记录的硬要求是：

```text
obs_t
nominal_action_t       # AHA 原始动作
policy_action_t        # adapter 建议动作
executed_action_t      # 经过接管、clip、安全层后实际执行的动作
reward_t
obs_t+1
terminated / truncated
intervention_mask
policy_version / feature_version / planner_phase
```

critic 必须评价 `executed_action_t`。若人接管后救回成功，却把成功回报错误归给接管前的坏动作，就产生 intervention credit leakage。

### 3.4 RLDG：专用 RL 教师再反哺大模型

[RLDG](https://arxiv.org/abs/2412.09858) 在 VGA、Type-C、DisplayPort、XLR 等连接器上先训练窄任务 specialist RL，再把其高质量成功轨迹用于 OpenVLA/Octo 微调。论文报告相对仅用人类示教的 generalist，部分任务最多提高约 40 个百分点。

对你尤其有吸引力，因为它把风险隔离开：

1. AHA 先负责抓取、靠近和进入 pre-insertion 区域；
2. 小型 specialist 只学最后的搜索/插入；
3. 收集 specialist 的成功、恢复、快速插入轨迹；
4. 重新训练 AHA ActionDiT，使最终部署可选择保留或去掉 specialist。

这叫“RL-generated data / policy distillation”，不能误写成“RL 直接更新了 VLA”。

### 3.5 Max-Q：动作块时代的 HIL 学习

[Max-Q Selective Imitation](https://arxiv.org/abs/2608.15088) 不让 actor 无条件模仿 replay 中的人类动作，而是训练 Monte-Carlo Q-chunk critic，对“当前策略 chunk”和“buffer/intervention chunk”评分，硬选择更高 Q 的一个作为模仿目标。论文在真实 USB pick-and-insertion 上报告 ACT 版本用 20 demos、30 分钟达到 99%，flow 版本约 0.8 小时达到 96%。

它与 AHA 的匹配非常直接，但证据要降权：该论文在检索截止日前刚发布，尚缺独立复现。安全的研究做法是：

- 使用 critic ensemble；
- 只有当 Q 差异超过保守 margin 才切换 target；
- 在 held-out 成功/失败/near-miss 上校准排名；
- 不让它替代 ResiP/HIL-SERL 基线，而是作为第二阶段对照。

---

## 4. WAM + RL：真正直接的工作与概念边界

### 4.1 HALO-WA

[HALO-WA](https://arxiv.org/abs/2607.04265) 是冻结 WAM、读取 latent 与动作参考、在线训练轻量 actor-critic 的直接代表。论文报告四个真实精密任务平均从 26.4% 提升到 87.1%，每任务在线训练约 45–75 分钟。它与 AHA 迁移的详细 tensor、replay、target network、BC 正则、代码/论文差异，已经在配套 HALO 指南中逐项分析。

本文只强调三点：

- 它证明“冻结 WAM + 小 RL adapter”是合理主线；
- 其 chunk refiner 不自动解决接触阶段的逐步闭环，应与 ResiP 的时间尺度思想结合；
- 公开仓库中的仿真任务、reward 和训练路径不能被当成论文真机插入的完整复现。

### 4.2 WAM-RL：联合优化世界与动作分支

[WAM-RL](https://arxiv.org/abs/2606.17906) 做两件事：

1. 成功 online rollout 用于 VideoDiT 的视频 SFT，并以 KL 约束新旧 latent；
2. actor 根据“世界模型想象的未来”与“执行后的实际未来”之间的重建差异获得 dense reward。

论文在 LIBERO-Object 从 68% 到 82%，RLBench Water Plants 从 19% 到 22%；只有仿真实验，没有插入真机。作者观察到 actor-only RL 对短任务有效，但长时程需要世界分支一起适配。

它对 AHA 的价值主要是长期研究问题，而不是第一实现：

- RGB 重建误差不等于插入质量；“看起来贴合”可能仍在卡孔。
- 如果世界模型本来就想象错了，actor 被奖励去实现错误想象，目标会自洽但不正确。
- 在线更新 VideoDiT 会改变 adapter 输入分布，使 replay 中旧 latent 和当前模型不一致。
- AHA 的 5B VideoDiT 比小 residual 难以用少量 SO101 数据稳定优化。

### 4.3 WAM + tactile/force：目前最相关的 2026 分支

| 方法 | 核心设计 | 实证 | 对当前 SO101 的判断 |
|---|---|---|---|
| [VT-WAM](https://arxiv.org/abs/2607.02503) | 视觉、触觉、动作联合 flow matching；接触 gate | 6 个真机接触任务，均值 71.67%，比 Fast-WAM +26.67 pp | 证明纯 RGB 的缺口；需 Xense 类触觉 |
| [FAWAM](https://arxiv.org/abs/2606.08555) | 历史 6D wrench；预测动作和未来 wrench；10 Hz residual | 4 个真机任务均值 85% | 架构极可借鉴，但当前缺 F/T |
| [Tactile-WAM](https://arxiv.org/abs/2606.26663) | VideoClean 非对称注意，动作可读触觉、视频被保护 | ManiFeel 接触任务明显提升 | 适合未来触觉版 AHA；非 RL |
| [N0-TWAM](https://arxiv.org/abs/2607.23783) | 6 embodiments、450 tasks 的触觉原生 WAM；NeoForce、触觉事件阶段 | 仿真/真机接触基准 | 很新，代码/权重仍承诺未来发布 |
| [N0-VTLA](https://arxiv.org/abs/2607.23782) | latent tactile token + ALTER 优势条件离线改进 | 9 个真机任务；长时程 75–95% | 可借鉴 advantage-conditioned 数据更新；需触觉 |
| [Dream-Tac](https://arxiv.org/abs/2606.08737) | 统一未来视觉/触觉/动作，contact gate | 6 个真机任务均值 83.3%，但 USB 仅 35% | 很好的反例：有触觉 WAM 也不保证连接器成功 |
| [TacWAM](https://arxiv.org/abs/2607.28391) | 触觉外观、稠密力、形变；future tactile 只作训练监督 | 4 个任务均值 75% | “未来目标不得泄漏给部署动作”值得迁移 |
| [OmniVTA](https://arxiv.org/abs/2603.19201) | 21k+ 轨迹、86 任务；60 Hz 触觉反射控制器 | 多类真机接触 | 强烈支持慢规划/快反射的多速率结构 |

FAWAM 的时间尺度尤其值得注意：其 Force-Envisioned Action Model 约 1 Hz，Force-Guided Residual Corrector 约 10 Hz；训练只用 20 个额外人类干预 episode。它不是 RL，而是 gated residual imitation，但其“预测参考 wrench 与实时 wrench 的偏差驱动纠正”正是接触闭环的清晰物理接口。

### 4.4 这一组论文给出的共同结论

这些触觉/力 WAM 不是要求你立刻买同款传感器。它们共同证明：

- 自由空间视觉占主导，接触后触觉/力占主导；
- 未来视觉预测对对象位置和语义有效，对卡滞/滑移/压力可能不充分；
- 低频大模型 + 高频局部 corrector 是重复出现的结构；
- 若未来加入触觉，优先把它接入小 adapter/critic，没必要第一天就重训整套 VideoDiT。

---

## 5. 世界模型作为环境、规划器或价值评估器

### 5.1 三类方法必须分清

| 类别 | 学到什么 | 怎样改策略 | 典型方法 | 当前 AHA 可否直接用 |
|---|---|---|---|---|
| 生成式 WAM | 未来视觉/动作的联合表示 | 动作读未来 latent，或小 adapter RL | AHA、HALO、WAM-RL | AHA 已有；HALO 式可用 |
| action-conditioned simulator | `当前状态 + 候选动作 → 未来` | 在想象 rollout 中 PPO/GRPO | WMPO、WoVR、World4RL | **不可直接用** |
| latent MBRL | `z,a→z'` + reward/value | MPPI/MPC 或 latent RL | SimDist、TD-MPC2 | 需另建小世界模型 |

### 5.2 与插入最接近的 action-conditioned 工作

#### WorldSample

[WorldSample](https://arxiv.org/abs/2607.02431) 以真实 rollout 锚定 action-conditioned video world model，在真实动作附近作局部扰动，合成 counterfactual future，再把 demo、真实、合成 transition 放进 HIL-SERL/RLPD。其 Policy-Paced Learning 用 Q 选择和熵调度控制合成数据比例，避免 critic 被幻觉样本带偏。

论文的真实 Insertion 使用侧视 + 腕部 D435i、7D 末端 delta action，报告成功率 63→95、训练步数 40k→10k、时间 60→30 分钟。对 AHA 可迁移的是“从少量合成开始、按 Q 平衡正负样本、真实闭环持续校准”；不可迁移的是把当前 AHA VideoDiT 直接当其 world model。

#### WMPO

[WMPO](https://arxiv.org/abs/2511.09515) 对像素级 action-conditioned world model 做行为对齐，并在完整 imagined trial 上用 VideoMAE reward 和 GRPO 优化离散 VLA。其真机 square-in-stick 间隙为 5 mm，200 条专家数据加 128 条 policy rollout，30 次评估中 base 53%、DPO 60%、WMPO 70%。

论文也承认世界模型会漏判“部件已卡住”等细微物理状态。对 SO101，这不是小瑕疵，而是决定是否能安全用 imagined reward 的核心风险。

#### SimDist

[Simulation Distillation](https://arxiv.org/abs/2603.15759) 不是视频 WAM，而是更务实的小 latent world model：

1. 仿真中用 RL expert、中间 checkpoint 和动作扰动产生 100k 多质量轨迹；
2. 预训练 encoder、latent dynamics、reward、value 和 BC prior；
3. 真机冻结 encoder/reward/value/policy，只在线微调 dynamics prediction；
4. MPPI 在 latent model 中规划。

其 UR5e 真机包含 Peg Insertion 和 Table Leg threading，用约 20 demos、15–30 分钟真实数据快速适配。对 AHA 的潜在路线是：保留 AHA 作先验/候选生成器，另建一个小 action-conditioned latent dynamics 做短 horizon 排序，而不是改造整个 5B VideoDiT。

#### ViTaL、FeelWorld 与 TACO

不要混淆两个同名 ViTaL 工作：

- [Touch begins where vision ends / VisuoTactile Local](https://arxiv.org/abs/2506.13762)：全局 VLM 定位 + 局部视触觉 residual RL，含 USB/插头/钥匙任务。
- [Inference-time Policy Steering via Vision and Touch](https://arxiv.org/abs/2606.14981)：冻结 diffusion policy，长时程视觉 world-model 选行为 mode，短时程触觉 latent reward 编辑动作；1 mm peg 插入中视觉 base 40→80%。

[FeelWorld](https://arxiv.org/abs/2607.24267) 联合预测视觉 latent、接触、3D tactile latent 和 slip，并用 contact-aware CEM 做短视野规划；USB 插入中视觉 CEM 37.5%，加入接触状态后 75%。它在每次规划中采样 400 个候选、视野 6 步、执行 2 步再规划，论文也承认计算不适合高频控制。

[TACO](https://arxiv.org/abs/2607.02840) 则不是标准在线 actor-critic。它从真实失败 rollout 找 failure-adjacent anchor，用视触觉 WAM 想象局部纠正段，再由 progress-action model 标动作和 binary advantage，最后做 advantage-conditioned flow matching 后训练。Insert Flower 从 50→93%，六任务均值 38→82%。其最可迁移部分是“只围绕失败邻域合成纠正”，而不是漫无边际地生成完整插入。

### 5.3 其他重要 imagined-RL / world-model 路线

| 方法 | 世界模型怎样用 | 策略更新 | 证据与迁移判断 |
|---|---|---|---|
| [WoVR](https://arxiv.org/abs/2602.13977) | Wan2.2 action-conditioned simulator；从失败关键帧开始短 rollout；WM/policy 共演化 | GRPO | KIR 很适合 pre-insertion 失败帧；需另训 WM |
| [World4RL](https://arxiv.org/abs/2509.19080) | 冻结 diffusion WM + success classifier | imagined PPO | 证明需 expert/policy/random 覆盖；无插入 |
| [World-Gymnast](https://arxiv.org/abs/2602.02454) | video WM + VLM binary reward | model-based GRPO | 需要 H200 级计算；VLM 不适合判断卡滞 |
| [ProphRL](https://arxiv.org/abs/2511.20633) | 大规模 action-to-video Prophet | 面向 flow VLA 的 FA-GRPO | 数学上贴近 ActionDiT；64 H200 预训练、无插入 |
| [RISE](https://arxiv.org/abs/2602.11075) | 多视角 dynamics + progress/value | imagined PPO/优势训练 | task progress 分解很有用；仍有不可能 transition |
| [PlayWorld](https://arxiv.org/abs/2603.09030) | 30h autonomous play 训练 WM | DSRL 噪声策略 | 长时间训练会 exploit WM error，是重要负面证据 |
| [World-VLA-Loop](https://arxiv.org/abs/2602.06508) | success + near-success 数据训练 WM/reward | GRPO，真实 rollout 反哺 | “差几毫米/半插入”的数据设计极值得迁移 |
| [VLA-MBPO](https://arxiv.org/abs/2603.20607) | unified manipulation model，chunk branched rollout | PPO 类 | 概念相关；真实性与算力需单独验证 |
| [SAILOR](https://arxiv.org/abs/2506.05294) | latent WM + reward/value | MPPI 搜索冻结 BC 附近 residual plan | 不需在线更新策略；适合接触前 chunk，未证明毫米接触 |
| [DREAMSTEER](https://arxiv.org/abs/2607.02865) | latent WM rollout + language value 排名 | 部署时 steering，不训练 | OOD 实机显著提升；15 Hz AHA 的候选延迟需实测 |
| [Latent Policy Steering](https://arxiv.org/abs/2507.13340) | optical-flow 跨 embodiment world model | BC 周围 latent search | 适合视觉运动约束，弱于接触状态 |
| [IRASim](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhu_IRASim_A_Fine-Grained_World_Model_for_Robot_Manipulation_ICCV_2025_paper.pdf) | action-conditioned video WM | planning | 可作为 world model 基准，非 AHA 即插即用 |
| [ViTacWorld](https://arxiv.org/abs/2607.22530) | action-conditioned 视触觉 rollout 与 policy evaluation | 过滤后作 policy 数据扩增 | charger/U-block/cuboid insertion；需触觉和人工过滤 |

### 5.4 imagined RL 的准入测试

只有通过以下测试，才允许把一个 world model 用于插入策略优化：

1. **反事实动作区分**：同一 pre-contact 帧，向左/右/下的候选动作必须产生不同且方向正确的未来。
2. **near-miss 区分**：成功、偏 1–5 mm、孔边接触、半插入、卡住必须能排序。
3. **接触一致性**：未来视频不能出现物体穿透、瞬移或“视觉上已经就位但真实仍在孔外”。
4. **多步校准**：报告 candidate-action success ranking/AUC，而不只看 FVD、PSNR、LPIPS。
5. **不确定性**：对训练分布外几何或动作能给高 uncertainty，并截断 rollout。
6. **真实闭环再校准**：policy 更新后，必须用新 policy 的真实失败反哺 WM；不能永远相信旧策略分布的模型。

当前 AHA 尚未通过第 1 项，因为其 action-conditioned 路径没有启用/训练证据。因此本文把 imagined RL 放在长期线，而不是主线。

---

## 6. VLA + RL 与 action-chunk RL 全景

### 6.1 冻结大模型，只训练小模块

| 方法 | 小模块/更新目标 | 数据与任务 | AHA 适配度 |
|---|---|---|---|
| [RLT](https://arxiv.org/abs/2604.23073) | RL token + chunk actor + twin Q | 真机 Ethernet/charger/screw/zip tie | **最高** |
| [EXPO-FT](https://arxiv.org/abs/2605.25477) | 多个 VLA 候选 + Gaussian edit actor + Q 排序 | 真机插电、插花、击球，平均 19.1 min，30/30 | 高，但多次 AHA 推理可能慢 |
| [EXPO](https://arxiv.org/abs/2507.07986) | expressive base + light edit policy | 仿真/离线到在线 | 高，EXPO-FT 的算法基础 |
| [PLD](https://arxiv.org/abs/2511.00091) | Gaussian residual specialist，之后蒸馏 | VLA/DP 多平台 | 高，兼顾在线 adapter 与最终部署 |
| [Policy Decorator](https://arxiv.org/abs/2412.13630) | 有界 residual + progressive exploration | diffusion/BeT 仿真 | 高，适合 ResiP 对照 |
| [Object-Centric Residual RL](https://arxiv.org/abs/2606.18953) | 冻结 VLA + object-pose residual | 仿真训练、零样本真机 | 中；需稳定物体位姿估计 |
| [TORL-VLA](https://arxiv.org/abs/2606.09337) | wrench-aware reference + 小 actor/critic，干预 credit censor | 接触真机 | 若有 F/T/可靠电流则高 |
| [E2HiL](https://arxiv.org/abs/2601.19969) | 按对 policy entropy 的影响筛 intervention | SO101 双相机等真机 | 适合作 replay sampler，而非主模型 |

EXPO-FT 对 AHA 有一个很自然的接口：ActionDiT 用不同噪声种子产生多个 `8×6` 候选，edit actor 给小 residual，critic 排序。但多候选会成倍增加推理延迟；第一版可只做 `base + 1 edited` 两个候选，而不是一开始 best-of-N 很大。

### 6.2 Q-chunk、offline-to-online 与策略无关的改进

| 方法 | 核心思想 | 对 AHA 的直接用法 | 风险 |
|---|---|---|---|
| [Reinforcement Learning with Action Chunking](https://arxiv.org/abs/2507.07969) | 整个 chunk 是宏动作；chunk 边界 Bellman backup | `Q(s, 8×6 chunk)` | 接触中开环假设弱；接管会切断 chunk |
| [Adaptive Q-Chunking](https://arxiv.org/abs/2605.05544) | 多 horizon critic 自适应选长短 | 自由空间 8；接近 2–4；接触 1 | 多 horizon 增加数据需求 |
| [Max-Q](https://arxiv.org/abs/2608.15088) | 比较 policy 与 buffer chunk，择优模仿 | AHA chunk vs 人类 chunk | 新、hard max 放大 critic 错误 |
| [ALOE](https://arxiv.org/abs/2602.12691) | action-level off-policy evaluation + pessimistic ensemble + AWR | advantage-weighted diffusion loss 更新 ActionDiT | 改动/稳定性成本高 |
| [PA-RL](https://arxiv.org/abs/2412.06685) | Q rerank、action gradient 局部优化，再监督蒸馏 | 不需 ActionDiT likelihood | Q 可能制造不平滑 OOD chunk |
| [IDQL](https://arxiv.org/abs/2304.10573) | diffusion 学行为分布，IQL critic 重采样候选 | HG-DAgger 后离线 best-of-N | 推理慢；Q 分辨 near-miss 需数据 |
| [WSRL](https://arxiv.org/abs/2412.07762) | Cal-QL 离线预训 + 短 warmup + 在线高 UTD | 有较多 HG transition 后的 critic warm-start | 离线 Q 质量决定安全性 |
| [ConRFT](https://arxiv.org/abs/2502.05450) | Octo + consistency policy Q-learning，offline→online | 先把 ActionDiT 蒸馏成 few-step actor | 工程量高、原文多为单步 action |

对于插入短关键阶段，Monte-Carlo return 有一个现实优势：episode 很短，最终成败可以直接回传，无需多次 bootstrap。代价是方差大。因此建议同时保留：

- MC Q-chunk 作为离线排序/Max-Q 实验；
- twin TD critic 作为在线 RLT/HALO 实验；
- 在相同 replay 和固定评估初态上比较 calibration，而不只看训练曲线。

### 6.3 直接/间接更新完整 VLA 的路线

| 方法 | 是否真正 RL 更新 VLA | 方法摘要 | 当前适用性 |
|---|---|---|---|
| [π0.6 / RECAP](https://arxiv.org/abs/2511.14759) | 是，但以价值/优势条件 supervised flow extraction | demos + rollout + intervention；distributional value；迭代提取高优势策略 | 思路强，单 SO101 系统规模大 |
| [iRe-VLA](https://arxiv.org/abs/2501.16664) | RL head 与 SFT 交替 | SACfD 更新 action head，成功在线轨迹再 SFT VLM | 可借“RL→成功数据→SFT”循环 |
| [RIPT-VLA](https://arxiv.org/abs/2505.17016) | critic-free on-policy RL | 同初态 K rollout，binary success，leave-one-out advantage | 仿真可用；真机难精确复位 |
| [SimpleVLA-RL](https://arxiv.org/abs/2509.09674) | GRPO/PPO 类 | 大规模并行仿真、action-token logprob | 不适合直接套连续 diffusion AHA |
| [VLA-RL](https://arxiv.org/abs/2505.18719) | trajectory PPO/actor-critic | 自回归 OpenVLA + process reward model | 可借 reward；更新方式不匹配 |
| [RLinf-VLA](https://arxiv.org/abs/2510.06710) | 基础设施 | PPO/GRPO 分布式训练 | 仿真研究工具，不是插入算法本身 |
| [TwinRL-VLA](https://arxiv.org/abs/2602.09023) | 数字孪生 RL + 定向真机 HIL | 找 failure-prone config，再 real refine | 能建 SO101 twin 时有价值 |
| [RECAP](https://www.pi.website/blog/pistar06) | 离线 RL 风格价值提取 | 真实长时程家庭任务 | 权重/完整训练闭源，难直接复制 |

这一组的共同门槛是：自回归 token VLA 能方便计算 log-probability，而 AHA ActionDiT 是 diffusion/flow 风格的连续动作生成器。不能把 GRPO loss 名字抄过来就认为接口成立。

---

## 7. Diffusion / flow policy 与 RL

### 7.1 主要方法地图

| 方法 | RL 在哪里作用 | 是否冻结生成基座 | AHA 迁移判断 |
|---|---|---|---|
| [DPPO](https://arxiv.org/abs/2409.00588) | 把去噪链视作内部 MDP，对后段 denoising PPO | 部分层可冻结 | 理论直接，真机首版成本过高 |
| [DSRL](https://arxiv.org/abs/2506.15799) | 小 RL policy 输出初始 diffusion noise | **是** | 很有研究价值；安全边界不如动作 residual 直观 |
| [Diffusion-QL](https://arxiv.org/abs/2208.06193) | diffusion BC loss + `-Q`，穿过采样链反传 | 否 | 适合离线基准，易 Q exploitation |
| [IDQL](https://arxiv.org/abs/2304.10573) | diffusion 提案，IQL Q 重采样 | 是 | 最保守的 offline 候选排序 |
| [EDP](https://arxiv.org/abs/2305.20081) | 高效 diffusion actor 兼容 TD3/CRR/IQL | 否 | 可先蒸馏/近似 ActionDiT，优先级低 |
| [Flow Q-Learning](https://arxiv.org/abs/2502.02538) | flow 学行为分布，one-step actor 最大化 Q | flow prior 固定/稳定 | 可训练快速 `8×6` actor，但损失多模态 |
| [SAC-Flow](https://arxiv.org/abs/2509.25756) | flow velocity 的 off-policy SAC 重参数化 | 否 | 数学接近 flow ActionDiT，改动大 |
| [FlowDPG](https://arxiv.org/abs/2606.22303) | critic gradient 蒸馏进 flow velocity | 否 | 论文含 AirPods 装配；AHA 架构适配仍大 |
| [Value-Guided Denoising](https://openreview.net/forum?id=gKZtkg9k3I) | 推理时用 Q gradient 引导 frozen diffusion | 是 | 可作 test-time 分支；Q 梯度安全性需限幅 |

### 7.2 为什么先做 residual，后做 DPPO

DPPO 的优点是探索仍遵循 diffusion policy 的结构化分布，不像简单 Gaussian actor 那样容易产生锯齿动作。困难是：

- 必须记录每个 denoising step 的分布/旧策略 ratio；
- 有效 horizon 变成“环境步 × 去噪步”；
- PPO 是 on-policy，昂贵真机数据难反复复用；
- 对大 ActionDiT 反传的显存和 wall-clock 成本远高于小 residual；
- 一旦同时更新 VideoDiT，replay 与特征版本问题更严重。

如果以后做 DPPO，建议只在仿真中更新 ActionDiT 的 LoRA 或最后若干去噪层，并以冻结 AHA、相同数据预算的 ResiP/RLT 作为基线。

### 7.3 DSRL 为什么值得作为论文支线

DSRL 的 RL action 不是关节动作，而是 diffusion 的初始 noise/latent。AHA 仍负责把 latent 解码为动作块，因此探索更可能留在其动作流形附近。可定义：

```text
z_noise = small_policy(adapter_state)
A_chunk = FrozenActionDiT(obs, proprio, z_noise)
```

它的风险也很独特：noise 到最终 chunk 是高度非线性的，难以给每个关节直接设“residual 不超过多少”。因此应在 ActionDiT 输出后仍通过硬安全过滤，并先离线检查 noise 扰动对应的动作分布。

---

## 8. 精密插入、装配与 residual RL 文献

### 8.1 真机 RL 与示教增强

| 工作 | 任务/模态 | 方法 | 主要价值 | 对 SO101 的限制 |
|---|---|---|---|---|
| [Residual RL for Robot Control](https://arxiv.org/abs/1812.03201) | Sawyer block assembly；视觉位姿 + F/T | 手工 controller + TD3 residual | residual RL 的经典实证 | 精确跟踪、柔顺与 F/T |
| [EDRIAD / Practical Insertion](https://arxiv.org/abs/1810.01531) | variable socket；raw image；peg/柔性 clip | DDPGfD + 20–30 成功/失败示教 | 几分钟真机交互；正负数据都重要 | 老式 DDPG、自动 reset/治具 |
| [Industrial Insertion with Natural Rewards](https://arxiv.org/abs/1906.05841) | USB 等连接器；视觉/状态/力 | SAC/TD3；示教与经典 controller residual | 支持 sparse/goal-image reward 比较 | 32×32 图像与专用实验平台 |
| [rLfD](https://arxiv.org/abs/2008.07682) | peg、gear、plug | DMP base + residual RL | 稳定先验 + task-space residual | DMP 表达力低，但原则有效 |
| [DDPGfD](https://arxiv.org/abs/1707.08817) | 仿真插入 + 真机柔性 clip | demo prioritized replay、n-step return | HG 数据可进入 critic，不只做 BC | 不是现代视觉大模型 |
| [SHIELD / Robust Multi-Modal Assembly](https://arxiv.org/abs/2103.11512) | NIST USB/HDMI/齿轮/钥匙；视觉+F/T | DDPGfD、示教、on-policy 人工纠正 | 相对坐标、课程、干预数据 | KUKA iiwa、定制阻抗 |
| [SERL](https://arxiv.org/abs/2401.16013) | PCB、线缆等 | 少量 demo + 高 UTD off-policy RL | 真机样本效率/基础设施强 | 1 kHz impedance，RL 约 10 Hz |
| [HIL-SERL](https://arxiv.org/abs/2410.21845) | RAM/SSD/USB 等 | RLPD + 双 buffer + 接管 | 最成熟实机闭环 | 需要人监控与强控制器 |
| [SPARR](https://arxiv.org/abs/2602.23253) | 10 个 AutoMate 两件式装配 | 仿真 state base + 真机视觉 residual | 约 0.5h、95–100%，无人接管 | 新预印本、工业平台 |
| [AutoSERL](https://arxiv.org/abs/2607.01651) | 6 个接触任务 | 单 demo、滑窗自动 intervention/recovery | 降低长期人工看守 | 对固定轨迹和任务结构依赖高 |
| [WSRL](https://arxiv.org/abs/2412.07762) | Franka peg insertion | Cal-QL + warmup + online high UTD | 有历史数据时很强 | warmup 和 offline Q 校准关键 |

### 8.2 大规模仿真、sim-to-real 与装配泛化

| 工作 | 核心方法 | 对 AHA/SO101 的可迁移部分 | 不可直接迁移部分 |
|---|---|---|---|
| [Factory](https://www.roboticsproceedings.org/rss18/p035.html) | GPU 高保真接触仿真 | 建插入仿真的底层能力 | 它是 simulator/benchmark，不是策略答案 |
| [IndustReal](https://arxiv.org/abs/2305.17110) | SDF reward、sampling curriculum、simulation-aware update | 几何 reward、curriculum、action integrator | CAD/接触仿真/工业控制器 |
| [AutoMate](https://arxiv.org/abs/2407.08028) | 100 assemblies；assembly-by-disassembly；RL+imitation | 反向拆卸轨迹、任务族、蒸馏 | 数据/资产规模远超当前任务 |
| [FORGE](https://arxiv.org/abs/2408.04587) | force-threshold conditioned policy、DR、success prediction | 把最大允许力作为 condition 的安全思想 | 无可信 F/T 时不能做 |
| [Fabrica](https://fabrica.csail.mit.edu/) | 几何 planner + residual RL，跨几何/方向/抓取 | 规划器给强 prior，RL 只处理接触误差 | 依赖 CAD/几何与仿真 |
| [TRANSIC](https://arxiv.org/abs/2405.10315) | sim base + 真机人工 correction residual/gate | AHA base + human-corrected residual gate | correction imitation，不是 reward RL |
| [Parameterized Manipulation Primitives](https://arxiv.org/abs/2306.06679) | RL 选/参数化“下探、搜索、插入”等 primitive | 大幅缩小低成本臂探索空间 | 先要实现可靠 primitive/接触检测 |
| [Meta-RL for Industrial Insertion](https://arxiv.org/abs/2004.14404) | PEARL task latent，任务族少试验适配 | 多插头/孔型阶段可用 | 单固定任务不值得先做 |
| [Offline Meta-RL for Insertion](https://arxiv.org/abs/2110.04276) | 历史任务离线 meta-train，再少量适配 | 数据积累后的跨连接器泛化 | 第一任务数据不够时优先级低 |
| [Generalizable Insertion with World Models](https://openreview.net/pdf?id=DR3n6IqGKI) | TD-MPC2；100 plug/socket，90 train/10 holdout | 多几何 task family 与 latent dynamics | 全仿真、非生成式 WAM |

### 8.3 不用 RL 但不能忽略的强基线

强化学习不是插入的默认最优答案。以下工作应该作为工程 baseline 或 safety primitive：

- [Vision-driven Compliant Manipulation](https://arxiv.org/abs/2106.14070)：RGB-D 6D 跟踪 + 被动柔顺手，无 F/T 和在线 RL，实现 0.25 mm 容差。它说明低成本硬件上，柔性腕、RCC、倒角和治具可能比更复杂网络更值钱。
- [Robust Peg-in-Hole via Interactive Contact Funnel](https://www.roboticsproceedings.org/rss21/p060.html)：学习无关的主动接触搜索和环境约束“漏斗”，可作为卡住后的 recovery primitive。
- [InsertionNet 1.0](https://arxiv.org/abs/2104.14223) / [2.0](https://arxiv.org/abs/2203.01153)：把插入看成视觉+力回归/关系学习，16 类任务超过 97.5%。如果能采纠偏标签，小 supervised corrector 可能比 RL 更安全。
- [Learning to Fill the Seam](https://arxiv.org/abs/2204.07776)：用 seam segmentation 中间表示和仿真 RL；表明局部孔/边缘 mask 或 keypoint 比让 critic 从完整图像自己发现毫米级关系更数据高效。
- [Safe Self-Supervised Visuo-Tactile Insertion](https://arxiv.org/abs/2210.01340)：NIST USB 中结构化视触觉方案 45/45，而 50-demo BC 为 1/45、直接真机 TD3 为 0/45。它是“状态表示和安全采集比裸 RL 更重要”的强反例。
- [Tactile Active Inference](https://arxiv.org/abs/2309.15681)：小于 0.1 mm clearance，加入触觉从 5% 到 90%；说明视觉无法替代所有接触信息。
- [Fine Manipulation without F/T](https://arxiv.org/abs/2301.13413)：用内部机器人信号估计 wrench，实现 100 μm 插入。但只有在电流、负载、实际位置/速度可稳定读取并标定时才可参考；普通舵机 `load` 不能未经验证直接当真实力。

### 8.4 插入任务的正确分解

建议把 episode 显式拆成：

| 阶段 | 主要感知 | 控制策略 | 允许探索 |
|---|---|---|---|
| 抓取/搬运 | 全局 + 腕部 RGB | AHA 长 chunk | 低；沿示教分布 |
| pre-insertion 到达 | 腕部 RGB、qpos | AHA 8 步或 4 步 | 小范围 |
| 视觉精对准 | 腕部局部 crop/keypoint | AHA + residual | 毫米级平面/姿态修正 |
| 首次接触 | 跟踪误差、电流/F/T（若有） | 1–2 步 micro-chunk | 极小；低速 |
| 搜索/插入 | 接触代理 + 腕部视觉 | 高频 residual / primitive | 受限螺旋/摆动/回退 |
| 就位确认 | 深度、视觉、电气连通/开关、保持时间 | 零动作/轻微保持 | 不探索 |
| 卡死恢复 | 接触阈值、无进度、超时 | 后退到已知安全点再重规划 | 固定 recovery 优先 |

对于第一篇论文，最干净的实验边界是：先让插头已经抓好，并从固定或小扰动的 pre-insertion pose 开始。否则“抓取失败、到达失败、插入失败”混在一个 sparse return 中，难以判断 RL 学到了什么。

---

## 9. Reward、状态与人类接管

### 9.1 插入 reward 应该怎样设计

最稳健的主 reward 是真正的成功判定：

\[
r_t = \mathbf{1}[\text{fully seated and held}] - c_{time} - c_{unsafe}.
\]

推荐信号优先级：

1. 电气连通、机械开关、插入深度/限位等物理成功信号；
2. 双相机成功分类器，并要求连续保持若干帧；
3. 人工 episode 成功标签；
4. 视觉进度、keypoint/seam 误差作为小的 shaping；
5. VLM/视频重建相似度只能作辅助，不能作唯一成功信号。

可加入很小的正则：

\[
r_t^{aux}=-\lambda_\Delta\|\Delta a_t\|^2-\lambda_j\|a_t-a_{t-1}\|^2-c_{time}.
\]

但安全阈值不能只当软惩罚，因为策略可能为了 `+1` 成功接受损坏风险。过流、越界、跟踪误差、持续推压、视觉丢失应由硬层截断。

### 9.2 可参考的 reward 表征工作

- [VIP](https://arxiv.org/abs/2210.00030)：视觉预训练价值表示，可作 goal-conditioned progress。
- [LIV](https://arxiv.org/abs/2306.00958)：语言—图像价值表征，适合语义进度，不适合单独判断接触。
- [RoboCLIP](https://arxiv.org/abs/2310.07899)：视频—语言 reward，可用于较粗任务阶段。
- [VLAC](https://arxiv.org/abs/2509.15937)：pairwise observation + language 的通用 process reward 和分级 HIL。
- [DEMO³](https://openreview.net/forum?id=Bv7LUUYOiq)：由示教得到多阶段 dense reward；插入中比通用 VLM 更可信。
- [World-VLA-Loop](https://arxiv.org/abs/2602.06508)：success + near-success 数据专门训练 dynamics/reward，适合“差几毫米”的关键分布。

### 9.3 接管数据如何用才不出错

人类接管至少有四种不同语义：

- **纠正动作**：当前策略将失败，人把它拉回；
- **安全停止**：不是专家动作，只是避免损坏；
- **任务恢复**：回到已知安全状态，未必是最短路径；
- **奖励纠正**：视觉 classifier 判断错，人工修改成功/失败标签。

建议分别保存标志，不能都作为“最优专家动作”。

相关工作形成一条清晰演进：

- [IWR](https://arxiv.org/abs/2012.06733)：intervention 与普通数据分池，平衡抽样，避免关键纠正被淹没。
- [ThriftyDAgger](https://arxiv.org/abs/2109.08273)：依据 novelty 与失败风险主动请求人。
- [Runtime Monitoring + Interactive IL](https://arxiv.org/abs/2310.17552)：latent dynamics 预演短未来，联合 OOD/failure classifier 提前接管。
- HIL-SERL：把 intervention transition 同时用于安全和 critic。
- SiLRI：按人类动作不确定性调整模仿约束。
- Max-Q：只有当 intervention chunk 的估值更高时才模仿。
- TORL-VLA：接管后的成功不应反向奖励接管前的错误策略动作。

### 9.4 AHA replay 的状态重建问题

AHA 的动作依赖连续的 VideoDiT/OVCR/KV 状态。若只存 `image_t`，以后用新模型或不同 cache 重算，得到的 latent 可能不是 rollout 时 actor 看见的 latent。

第一版推荐直接存：

```text
pooled_ovcr_feature
current_image_feature
normalized_proprio
nominal_chunk
executed_micro_actions
chunk_index
planner_age / phase
policy_version / feature_encoder_hash
latency and timestamp
```

完整 KV cache 太大，不适合 replay；固定、冻结、带 shape contract 的 pooled OVCR feature 更现实。也可同时保存原始双相机帧，供未来重新训练 encoder，但 critic 训练应优先使用 rollout 时真实 feature。

---

## 10. 迁移到 AHA-WAM + SO101 的逐项判断

### 10.1 AHA 当前结构与论文假设的对照

| 维度 | 当前 AHA/SO101 | 文献常见假设 | 影响 |
|---|---|---|---|
| 世界分支 | VideoDiT → ActionDiT；当前 action-conditioned=false | candidate action → future simulator | imagined RL 不能直接接 |
| 模型状态 | OVCR/KV 有历史与异步 phase | 单帧 Markov latent | replay 要保存 feature/phase/version |
| 动作 | 6D SO101 关节/夹爪；计划 chunk 8 | 6D EE twist/delta pose，或 torque | residual 不能照抄动作尺度 |
| 频率 | 约 15 Hz | RLT 50 Hz；部分接触 reflex 60 Hz；底层 impedance 1 kHz | 高频论文结果不可直接复现 |
| 感知 | 全局 + 腕部 RGB | 多相机 + F/T/触觉/深度 | 接触状态部分不可观 |
| 执行器 | 低成本位置舵机、齿隙/延迟/柔顺未知 | Franka/KUKA 工业级控制 | 探索更小，需被动柔顺/治具 |
| 数据 | SO101 自有任务数据 | 大规模 robot data + 自动 reset | 训练预算与重置是瓶颈 |

仓库默认 [`configs/model/ahawam.yaml`](../../configs/model/ahawam.yaml) 仍是 `action_horizon: 64`、`action_chunk_size: 16`；先前 SO101 部署方案是 horizon 40、chunk 8。实验时必须以实际启动配置为准，并把 config snapshot 写进每个 rollout，避免文档参数与运行参数不一致。

### 10.2 推荐的两条并行 baseline

#### Baseline R：逐步 residual（ResiP 型）

```text
input:
  wrist_feature
  optional_global_feature
  normalized_qpos
  current_nominal_action
  remaining_nominal_chunk summary
  chunk_phase / planner_age
  previous residual

network:
  2–3 layer MLP or small transformer
  stochastic actor during training; deterministic mean at evaluation

output:
  bounded 6D joint residual

critic:
  Q(state, executed_single_step_action)
```

优点：接触后每步反应，限幅直观，最容易审计。缺点：如果视觉 feature 不能判断接触，critic 会把不同物理状态混成同一观测。

#### Baseline C：chunk refiner（RLT/HALO 型）

```text
input:
  learned_rl_token
  normalized_qpos
  flattened AHA 8×6 reference chunk
  planner metadata

actor:
  output refined 8×6 chunk

critic:
  twin Q(state, actual valid chunk)

execution:
  free space: execute 4–8 steps
  near contact: execute 1–2 steps, then re-observe/replan
```

优点：直接利用 AHA 长时程结构，和 HALO/RLT 最接近。缺点：critic 输入维度大，接触时完整 chunk 的 credit assignment 较难。

### 10.3 为什么建议二者都做

这是一个有价值且干净的研究问题：

> 对动作分块 WAM 的精密插入，RL 应该重写整个动作块，还是仅在执行时逐步修正？

统一数据、统一 AHA checkpoint、统一安全边界，比较：

- 成功率；
- 首次接触后的失败率；
- 每次成功所需接管步数；
- 最大跟踪误差/电流代理；
- cycle time；
- 对孔位、相机、光照扰动的泛化；
- critic 的成功/near-miss 排名 AUC。

这比只报告“加 RL 后成功率提升”更能产生可解释贡献。

### 10.4 动作空间建议

如果 AHA 当前输出绝对关节位置，residual 不能无限累积。可采用：

```text
delta_norm = tanh(actor_output)
delta_joint = delta_norm * per_joint_bound
target = nominal_target + delta_joint
target = clamp_position_limits(target)
target = slew_rate_limit(target, previous_target)
target = acceleration_limit(target_history)
```

接触阶段最好在 task-space 对齐误差上学习，但 SO101 若没有稳定正逆运动学和标定，不应为了论文形式强行切 task-space。先建立可靠的 joint-space 有界 residual，随后再比较 task-space delta。

### 10.5 多时间尺度建议

理想结构：

| 层 | 频率 | 职责 |
|---|---:|---|
| AHA VideoDiT planner | 低频/异步 | 长时程世界上下文 |
| AHA ActionDiT | ~15 Hz query | nominal action chunk |
| residual/contact adapter | 每个可用控制步 | 局部修正、gate、风险估计 |
| servo/safety loop | 硬件可达到的最高稳定频率 | 限速、跟踪、停止、回退 |

不要凭论文把 residual 标成 50/60 Hz；先测 SO101 client、TCP/ROS、图像采集、推理与舵机总延迟。若实际稳定端到端只有 15 Hz，论文中“高频 corrector”的优势只能部分复现。

---

## 11. 推荐实验路线图

### 阶段 0：把任务和安全系统做成可学习的环境

完成条件：

- 插头预抓好，从 pre-insertion pose 开始；
- 自动/半自动 reset 可重复；
- 明确 `success`、`timeout`、`unsafe`；
- residual 全为 0 时，系统与原 AHA 行为逐数值一致；
- 人接管时旧 chunk 立即取消；
- 有工作空间、关节、速度、加速度、连续推压和超时限制；
- 至少有一种物理成功确认，不只看单帧 RGB；
- 每次失败可以快速更换耗材或确认零件未损坏。

### 阶段 1：建立非 RL 基线

1. AHA-WAM 原始策略；
2. AHA + 更短执行 chunk/receding horizon；
3. AHA + HG-DAgger/IWR；
4. AHA + supervised residual BC；
5. 视觉 keypoint/seam corrector；
6. 被动柔顺/RCC/倒角或 contact-funnel recovery。

这一阶段很重要：如果简单缩短 chunk 或加一个柔性结构就解决了绝大多数失败，RL 的论文问题应该聚焦在剩余 recovery/generalization，而不是把机械问题包装成 RL 成果。

### 阶段 2：离线 critic sanity check

从 demo、AHA rollout、HG 接管中构建成功/near-success/失败数据。先不在线探索，训练 IQL/Cal-QL/MC Q，然后检查：

- 成功 chunk 的 Q 是否显著高于失败；
- 同一状态中，向正确对准方向的小偏移 Q 是否上升；
- 接管前错误动作是否因接管后成功获得虚高值；
- Q 是否主要记住背景、光照或 episode 编号；
- 在新孔位/新相机曝光上是否仍能排序。

若这些检查不通过，在线 actor 只会更快利用 critic 的错误。

### 阶段 3：有限真机 RL

推荐从接触前 1–2 cm 的短 episode 开始：

```text
initial data:
  20–50 successful demonstrations
  HG-DAgger corrections
  deliberate but safe near-miss / failure trials

buffers:
  demo/intervention buffer
  online buffer

sampling:
  start 50% / 50%
  later ablate adaptive ratios

reward:
  binary physical success
  small time cost
  small residual/jerk regularization

exploration:
  only in residual space
  per-joint bound
  reduce bound after first contact
```

比较 A1 `step residual` 与 A2 `chunk refiner`。在线训练达到预设 interaction budget 或连续若干窗口无改善即停止，不让机器人无期限探索。

### 阶段 4：动作块价值与蒸馏

- 加 Q-chunk / Max-Q；
- 让 ActionDiT 采 2–4 个候选，由 IDQL/EXPO critic 选；
- 用成功 RL 与 recovery 轨迹微调 AHA ActionDiT；
- 比较“保留 adapter”与“蒸馏后移除 adapter”；
- 检查蒸馏是否损失 specialist 的快速反应。

### 阶段 5：触觉/接触代理

若前四阶段的主要错误集中在“看不见卡住/插到底”，增加：

- 腕部 3D/6D F/T；或
- 视觉触觉传感器；或
- 可标定电机电流、负载、实际—目标位置误差、速度历史；或
- 被动柔顺腕/RCC 与机械限力。

先把新信号输入小 adapter/critic，比较视觉-only；只有数据量和效果证明必要后，再做 FAWAM/VT-WAM 风格的 joint future contact prediction。

### 阶段 6：world-model 研究分支

1. 单独训练短 horizon `p(z'|z,a)`，不要先改 AHA VideoDiT；
2. 用 AHA 候选 chunk warm-start MPPI/CEM；
3. 只对 pre-contact/刚接触关键帧做 2–6 步 rollout；
4. 加 ensemble uncertainty 和 OOD 截断；
5. 真实 rollout 持续校准；
6. 通过准入测试后，再比较 SAILOR/SimDist/WorldSample/WoVR 路线。

---

## 12. 建议的论文实验矩阵

| 编号 | 模型 | 是否 RL | 粒度 | 人类数据 | 目的 |
|---|---|---|---|---|---|
| B0 | 原始 AHA | 否 | chunk 8 | demos | 基准 |
| B1 | AHA + receding horizon | 否 | 执行 1/2/4/8 | demos | 分离 chunk 开环问题 |
| B2 | AHA + HG-DAgger | 否 | chunk/step | intervention | 聚合分布外状态 |
| B3 | AHA + supervised residual | 否 | step | intervention | 判断 RL 是否必要 |
| R1 | ResiP-style residual | 是 | step | demo + online | 主方法 1 |
| R2 | RLT/HALO refiner | 是 | chunk | demo + online | 主方法 2 |
| R3 | Adaptive chunk | 是 | 8→2→1 | demo + online | 多时间尺度贡献 |
| Q1 | MC Max-Q | critic + imitation | chunk | intervention | 避免无条件模仿 |
| Q2 | IDQL/EXPO selection | offline/online Q | candidate chunk | mixed | 利用 ActionDiT 多样性 |
| D1 | RLDG/PLD distillation | RL teacher + SFT | trajectory | RL success | 能力吸回 AHA |
| Z1 | DSRL noise steering | 是 | diffusion latent | mixed | 生成流形内探索 |
| T1 | R1 + contact proxy/tactile | 是 | step | mixed | 测接触可观测性 |

至少报告：

- 每任务总真实交互分钟与 env steps；
- demo 数、接管时间、接管步比例；
- 训练期间损坏/紧急停止次数；
- 20–50 个固定 test seeds 的 Wilson confidence interval；
- success、cycle time、插入深度、首次接触后恢复率；
- 对孔位、角度、光照、相机轻微移动、不同零件的泛化；
- 各失败类型的 confusion matrix，而不是只报总成功率。

### 12.1 建议的失败标签

```text
F1: 抓取姿态错误
F2: 未到 pre-insertion 区域
F3: 视觉对准错误
F4: 首次接触方向错误
F5: 卡在孔边/楔住
F6: 部分插入后停止
F7: 插入成功但 classifier 未识别
F8: classifier 假成功
F9: 安全层截断
F10: 通信/延迟/丢帧
```

如果 RL 只改善 F3 而 F5/F6 不变，说明问题是视觉对准而非接触策略；如果加入触觉只改善 F5/F6，便形成清楚的因果证据。

---

## 13. 每类方法到底能不能迁移

### 13.1 可几乎直接迁移的组件

- 冻结 AHA；导出稳定的 OVCR/current-observation feature。
- actor 同时读取 `feature + proprio + nominal action`。
- demo/intervention buffer 与 online buffer 分开。
- twin-Q、target network、delayed actor update、BC/reference regularization。
- 人接管立即取消旧 chunk。
- step residual 硬限幅、gate、slew-rate limit。
- success classifier + 人工 reward correction。
- failure-adjacent / near-success 数据专门采集。
- specialist RL 成功轨迹重新加入 AHA SFT。

### 13.2 可以迁移，但必须改接口的组件

| 论文组件 | 原接口 | AHA 需要的改法 |
|---|---|---|
| RLT RL token | π 系列 VLA embedding | 训练 AHA-specific encoder；保存 OVCR phase/history |
| HALO chunk actor | GigaWA latent/ref chunk | 改为 AHA pooled OVCR + `8×6` nominal chunk |
| ResiP residual | task-space/end-effector action | 先做 per-joint bounded residual；后续才做 task-space |
| HIL-SERL SAC | 单步 Gaussian 6D twist | 用 residual actor，或定义 Q-chunk transition |
| Max-Q | ACT/flow chunk | 处理接管导致的 variable-length chunk/mask |
| EXPO-FT | 快速 VLA 多候选 | 限制候选数；缓存 AHA context；严格测延迟 |
| DSRL | standard diffusion noise | 明确 AHA ActionDiT 初始噪声注入与可复现 seed 接口 |
| RECAP/AWR | flow-matching VLA loss | 对 ActionDiT 实现 advantage-weighted denoising/flow loss |
| FAWAM | 6D wrench | 以后接 F/T；或先用经标定 contact proxy，但不得等同真实 wrench |

### 13.3 当前不能直接迁移的组件

- 把 AHA 当前 VideoDiT 当候选动作 simulator；
- 在 AHA 内做 CEM/MPPI，除非重新训练 action-conditioned dynamics；
- 照搬 Franka 1 kHz impedance 的探索幅度/接触速度；
- 把 RGB video reconstruction 当插入主 reward；
- 把论文的 50/60 Hz controller 数字写进 SO101 系统而不测真实端到端频率；
- 把所有 human takeover 都标成 optimal action；
- 在只有成功示教、没有失败/near-miss 的数据上信任 offline Q；
- 直接对整套 VideoDiT + ActionDiT 做真机 PPO/GRPO。

---

## 14. 最有价值的潜在研究创新点

### 14.1 WAM planner + adaptive contact RL

把 AHA 固定 8 步动作块改成阶段自适应执行：

```text
free-space:       Q evaluates 8-step chunk, execute 4–8
pre-contact:      evaluate 4-step chunk, execute 2
contact:          step residual, execute 1
stuck/recovery:   fixed safe retreat, invalidate planner cache as needed
```

这把 Adaptive Q-Chunk、ResiP 和 AHA 异步 planner/executor 统一起来，且问题与插入任务高度匹配。

### 14.2 Intervention-censored WAM critic

AHA 的长动作块容易跨越人类接管边界。可研究：

- 在 intervention 开始处截断 return；
- `valid_action_mask` 排除未执行的 nominal steps；
- critic 只对实际 micro-action 序列估值；
- 分别预测 autonomous success 与 “允许人类救援后的 success”。

这比简单复制 HALO replay 更符合真实 HIL。

### 14.3 接触可观测性消融

比较同一 residual/critic：

1. 双 RGB + qpos；
2. 加 commanded-vs-measured joint error；
3. 加舵机电流/负载历史；
4. 加视觉触觉或 F/T。

这能回答“WAM latent 是否已经包含足够接触信息”，而不是预先假定有或没有。

### 14.4 RL specialist 到 WAM 的双向闭环

不只做一次 RLDG 蒸馏，而是：

```text
AHA rollout → failure-adjacent states
      ↓
residual RL specialist learns recovery
      ↓
successful/recovery trajectories → ActionDiT SFT
      ↓
new AHA rollout → harder failures
```

每轮保留旧任务 rehearsal，测是否灾难性遗忘。这条线比一次性 full-model RL 更容易控制。

### 14.5 AHA-specific RL token

RLT 的 token 目标可以扩展为多任务约束：

- 重建当前 observation feature；
- 预测 AHA nominal chunk；
- 预测阶段/进度；
- 对 success/near-miss/failure 做 contrastive separation；
- 可选预测短期跟踪误差/contact proxy。

然后消融“简单 pooled OVCR”与“任务训练 RL token”，检验 latent 是否真的需要专门压缩。

---

## 15. 推荐阅读顺序

### 第一组：先理解“为什么 chunk 需要局部闭环”

1. [ResiP](https://arxiv.org/abs/2407.16677)
2. [Residual RL for Robot Control](https://arxiv.org/abs/1812.03201)
3. [Policy Decorator](https://arxiv.org/abs/2412.13630)
4. [Vision-driven Compliant Manipulation](https://arxiv.org/abs/2106.14070)

读完应能回答：AHA 是 planner 还是 controller；residual 加在哪里；机械柔顺能解决什么。

### 第二组：理解实机 RL 数据闭环

1. [SERL](https://arxiv.org/abs/2401.16013)
2. [HIL-SERL](https://arxiv.org/abs/2410.21845)
3. [IWR](https://arxiv.org/abs/2012.06733)
4. [SiLRI](https://arxiv.org/abs/2512.24288)
5. [Max-Q](https://arxiv.org/abs/2608.15088)

读完应能回答：demo/online/intervention 如何存、如何抽样、怎样避免接管 credit 污染。

### 第三组：理解大策略如何接 RL

1. [RLT](https://arxiv.org/abs/2604.23073)
2. [HALO-WA](https://arxiv.org/abs/2607.04265)
3. [EXPO-FT](https://arxiv.org/abs/2605.25477)
4. [RLDG](https://arxiv.org/abs/2412.09858)
5. [PA-RL](https://arxiv.org/abs/2412.06685)

读完应能比较：latent adapter、动作 residual、多候选排序、RL-data distillation。

### 第四组：理解接触为什么需要新模态

1. [ViTaL Local](https://arxiv.org/abs/2506.13762)
2. [VT-WAM](https://arxiv.org/abs/2607.02503)
3. [FAWAM](https://arxiv.org/abs/2606.08555)
4. [FeelWorld](https://arxiv.org/abs/2607.24267)
5. [TACO](https://arxiv.org/abs/2607.02840)

读完应能回答：视觉在哪些阶段够用、哪些状态发生 aliasing、触觉应进入 planner 还是 fast corrector。

### 第五组：最后再读 imagined RL

1. [SimDist](https://arxiv.org/abs/2603.15759)
2. [WMPO](https://arxiv.org/abs/2511.09515)
3. [WoVR](https://arxiv.org/abs/2602.13977)
4. [WorldSample](https://arxiv.org/abs/2607.02431)
5. [WAM-RL](https://arxiv.org/abs/2606.17906)

读完应能清楚区分：世界模型作 feature、作 reward、作 simulator、作 MPC dynamics，以及每一种对 action conditioning 的要求。

---

## 16. 扩展论文索引

下面列出正文未展开或只简述、但与路线有关的工作，方便继续追踪。

### 16.1 基础 RL / offline-to-online

- [SAC](https://arxiv.org/abs/1801.01290)：随机 off-policy actor-critic；适合逐步 residual。
- [TD3](https://arxiv.org/abs/1802.09477)：确定性 actor + twin Q + delayed update；HALO/RLT 类基础。
- [IQL](https://arxiv.org/abs/2110.06169)：dataset-only value/Q + advantage-weighted extraction。
- [CQL](https://arxiv.org/abs/2006.04779)：保守压低 OOD action Q。
- [Cal-QL](https://arxiv.org/abs/2303.05479)：校准 CQL value scale，便于 offline→online。
- [RLPD](https://arxiv.org/abs/2302.02948)：高 UTD、prior data 与 online replay 混合；HIL-SERL 基础。
- [AWAC](https://arxiv.org/abs/2006.09359)：优势加权监督式 policy extraction。
- [AW-Opt](https://arxiv.org/abs/2111.05424)：value learning + advantage-weighted imitation，支持 heterogeneous robot data。
- [Q-Transformer](https://arxiv.org/abs/2309.10150)：高容量 autoregressive Q；不建议直接离散化整个 AHA 连续 chunk，但可参考 critic 设计。
- [CRSfD](https://proceedings.mlr.press/v205/guo23a.html)：从相似但不完全匹配的示教保守 shaping；多插头共享数据时有用。

### 16.2 Diffusion/flow 和策略 steering

- [DTQL](https://arxiv.org/abs/2405.19690)：diffusion behavior trust region + one-step actor。
- [RL for Flow-Matching Policies](https://arxiv.org/abs/2507.15073)：RWFM/GRPO；主要是 toy dynamics，机器人外推证据弱。
- [Flow Policy Gradients](https://arxiv.org/abs/2602.02481)：flow policy gradient，覆盖多类控制，迁移工程量高。
- [RL-100](https://arxiv.org/abs/2510.14830)：iterative offline/online RL + consistency distillation；强但很新，需复现。
- [Hume](https://arxiv.org/abs/2505.21432)：低频 System-2 候选/value selection + 高频 System-1 denoising。
- [VGAS](https://arxiv.org/abs/2602.07399)：Q-Chunk-Former + 几何 regularization 排序 near-miss chunk。
- [VLS](https://arxiv.org/abs/2602.03973)：VLM 合成 differentiable reward，在推理时引导 diffusion/flow；不等于 RL fine-tuning。
- [V-VLAPS](https://arxiv.org/abs/2601.00969)：冻结 Octo + MCTS/value；需要可靠 forward model。
- [GRAPE](https://arxiv.org/abs/2411.19309)：从成功/失败和 VLM keypoint constraints 做 trajectory preference optimization，不是标准 actor-critic。
- [ReinboT](https://arxiv.org/abs/2505.07395)：return-conditioned supervised VLA，属于 RL-inspired 而非清晰在线 RL。

### 16.3 世界表征和评估

- [FLARE](https://arxiv.org/abs/2505.15659)：让未来 latent 与未来观测对齐；可借鉴表征目标，非 RL。
- [WorldGym](https://openreview.net/forum?id=9dSs5m9mB7)：评估 world model 能否可靠评价 policy；重要性高于单纯视频画质。
- [Ctrl-World](https://arxiv.org/abs/2510.10125)：policy-in-loop world model 和成功 imagined trajectory SFT。
- [RWM-U](https://arxiv.org/abs/2504.16680)：ensemble epistemic uncertainty 和 imagined transition penalty。
- [DEMO³](https://openreview.net/forum?id=Bv7LUUYOiq)：示教增强 reward/policy/world model，多阶段 manipulation。
- [World-Env](https://arxiv.org/abs/2509.24948)：video simulator + VLM reflector reward/termination。
- [VLA-RFT](https://arxiv.org/abs/2510.00406)：action-conditioned WM + goal-reference trajectory reward。
- [AtomVLA](https://arxiv.org/abs/2603.08519)：atomic subtask + V-JEPA2 latent WM + offline GRPO。

### 16.4 传感、接触与 compliance

- [InsertionNet 1.0](https://arxiv.org/abs/2104.14223) / [2.0](https://arxiv.org/abs/2203.01153)
- [Safe Self-Supervised Visuo-Tactile Insertion](https://arxiv.org/abs/2210.01340)
- [Tactile Active Inference](https://arxiv.org/abs/2309.15681)
- [Learning Tactile Insertion in the Real World](https://arxiv.org/abs/2405.00383)：Dreamer-v3 从触觉图像学习；结果初步。
- [Variable Compliance Control](https://arxiv.org/abs/2008.10224)：RL 学位置控制机械臂的可变柔顺参数；当前 SO101 难直接复现。
- [Eye-in-Hand Sim2Real Peg-Hole](https://arxiv.org/abs/2005.14401)：RGB-D + joints 的仿真 RL 零微调部署。
- [Visual/Force Residual Assembly](https://doi.org/10.1016/j.jmsy.2023.11.008)：把 reaching/searching/insertion 分阶段，视觉负责搜索、力负责交互。
- [FIRe](https://rcilab.github.io/FIRe/)：冻结 VLA chunk + force-informed residual；论文/代码截至检索日仍不完整，故只作观察项。

---

## 17. 一份可执行的最终建议

如果现在就开始，我会按以下顺序：

1. **把任务裁成 pre-insertion 短 episode**，先避免抓取/搬运干扰 credit。
2. **实现 HG-DAgger/IWR 数据记录**，完整保存实际动作、接管、reward、下一观测与 AHA feature/version。
3. **让 chunk 可中断并做 receding horizon 对照**；先确认开环是多大比例的失败原因。
4. **训练 supervised step residual**，证明视觉 feature 能否在不靠 reward 的情况下学到局部修正。
5. **以 HIL-SERL/RLPD 骨架训练有界 ResiP-style residual**，AHA 全冻结。
6. **并行训练 RLT/HALO-style chunk refiner**，统一数据预算比较 step 与 chunk RL。
7. **对 intervention credit 做 censoring**，再试 SiLRI/Max-Q。
8. **把 RL 成功与恢复轨迹蒸馏回 ActionDiT**，做 RLDG/PLD 对照。
9. **如果失败仍是卡滞/半插入，增加 contact proxy、F/T 或触觉**；优先接小 adapter。
10. **只有在另建 action-conditioned latent dynamics 并通过准入测试后，才做 imagined RL/MPC**。

最可能形成一篇扎实工作的主题不是“把某个 VLA RL 算法搬到 WAM”，而是：

> **面向低成本位置伺服机械臂的多时间尺度 WAM 强化学习：长动作块视觉规划、接触阶段自适应短视野、受限 residual、干预去污染，以及 RL 轨迹再蒸馏。**

这条路线同时尊重 AHA 的优势、插入的物理本质和 SO101 的安全边界，也有明确而公平的 baseline。

---

## 18. 证据强度与阅读注意

- SERL、HIL-SERL、RLDG、Factory、IndustReal、AutoMate 等已有正式会议/较成熟代码，适合作基础线。
- 2026 年 WAM/VLA-RL 论文大量是同期 arXiv 预印本。RLT、HALO-WA、WAM-RL、EXPO-FT、WorldSample、TACO、VT-WAM、Max-Q 等很相关，但不同论文的机器人、容差、reset、trial 数和成功定义不一致，成功率不能直接排名。
- 真机评估常只有每任务 10–40 次；本文引用数字用于理解量级，不代表在 SO101 上的预期结果。
- 视频生成指标不等于接触因果正确性。所有 imagined-RL 结论必须用真实插入的 candidate ranking 和闭环成功验证。
- 本文“可迁移”表示算法接口和问题结构兼容，不表示公开代码可以不修改直接运行。

---

## 19. 插座/锁孔到底应固定还是移动：证据、因果混淆与首轮采数方案

### 19.1 先给直接答案

**不建议把正式训练数据中的插座或锁孔全部严格固定在一个位置。** 对当前 AHA-WAM + SO101，最合适的第一版不是一开始覆盖整个桌面，也不是收完一大批完全固定位置数据，而是：

> **先定义一个中心工位，在中心附近做小范围、分层且可复现的位姿扰动；目标在每条 episode 内保持静止，在 episode 之间移动。先只变化平面位置，再逐渐加入小角度。**

这样同时满足两个目标：

1. 遥操作员仍能稳定产生较高质量的成功示教；
2. 同一机器人起始状态会对应多个不同目标位置和不同动作，模型不能仅凭 `qpos + 时间` 复现一条固定轨迹，必须从图像中区分孔在哪里。

但“目标不固定”仍不足以自动证明视觉闭环。必须把**目标位姿、插头抓取位姿、机器人起始位姿**独立变化，并准备视觉消融和 held-out 位置评测。否则仍可能存在捷径。

### 19.2 为什么固定位置确实可能让模型绕开视觉

设插座永远固定在世界坐标 `p_socket`，机械臂每次也从近似同一 `q_start` 开始。训练集近似变成：

```text
固定 q_start + 固定时序阶段  ──► 近似固定的关节轨迹
```

即便网络输入了两路 RGB，也可能主要依靠：

- 当前关节位置；
- episode 已经过了多久；
- 固定背景或治具边缘；
- action chunk 的历史延续；
- AHA 预测的平均未来运动；

而不是实时估计“钥匙相对锁孔偏左还是偏右”。训练成功率很高也不能排除这种捷径。

固定目标并不等于毫无价值。如果目标是工业固定工位、只希望处理抓取误差、舵机回差和接触扰动，固定母口是合理的窄任务定义。问题只在于：**不能把这种结果解释成视觉定位或空间泛化能力。**

### 19.3 文献里两种做法都存在，并不是“大家都随机”

| 工作 | 任务和训练分布 | 目标是否移动 | 能支持什么结论 |
|---|---|---|---|
| [ACT / ALOHA](https://arxiv.org/pdf/2304.13705) | 8 个任务都随机初始物体位置；真实任务沿 15 cm 参考线变化，仿真任务在二维区域均匀变化。电池任务中遥控器沿 15 cm 线移动，电池在大致同一区域但旋转变化 | **移动** | 纯示教学习也常从第一批数据就加入受控位置变化；不是只能先学固定轨迹 |
| [EDRIAD](https://arxiv.org/pdf/1810.01531) | 每回合用 pan-tilt 装置均匀采样插座姿态，对应开口移动约 2–4 cm；20–30 条成功和同量失败 demo，再做真机 RL | **移动** | 论文直接发现：固定版本有无视觉都能解决，随机插座版本必须依赖视觉；这是“固定位置会掩盖视觉无效”的强证据 |
| [ViSk](https://arxiv.org/pdf/2410.17246) | 插头任务把插线板随机放在 `20 cm × 7 cm`；USB hub 随机放在 `20 cm × 15 cm`；约 96/98 条 demo；评测使用训练未见的 10 个固定位置 | **移动且留出未见位置** | 明确研究空间泛化；训练位置和评测位置分开比随机混合后再随机切帧严谨得多 |
| [RLDG](https://arxiv.org/abs/2412.09858) | Connector Insertion：母口治具固定，已抓住的公头从母口上方 `10 cm × 10 cm` 平面、约 5 cm 高处开始；FMB Insertion：装配板在 `35 cm × 35 cm`、`±15°` 内随机 | **同一论文两种设置都有** | 固定母口可用于窄域“最后一段”RL specialist；要研究通用装配板定位时则大幅随机目标。它否定了“论文一定采用同一种设置” |
| [SHIELD](https://www.roboticsproceedings.org/rss17/p088.pdf) | HDMI 从无随机、静态目标开始，课程式扩展到初始位置 `±6 cm` 和移动目标；钥匙任务从零偏差逐渐增到 x/y `±5 mm` | **课程式扩大** | 对困难接触任务，先固定/小扰动学会，再逐步扩大是有强实机依据的做法；不必第一天就覆盖最大工作区 |
| [Safe Self-Supervised USB Insertion](https://arxiv.org/abs/2210.01340) | 固定 NIST 任务板，但评测 45 个不同初始抓取位姿；50 条人类插入 demo 的 BC 仅 1/45，自监督视触觉方法 45/45 | **母口可固定，插头—夹爪关系变化** | 只移动母口并非唯一重要因素；抓取误差会改变真正的相对插入几何，也必须覆盖 |

几个必须批判性理解的细节：

- ACT 的“随机物体位置”并不等于锁孔式亚毫米插入已解决；其仿真双臂插入间隙约 5 mm，电池任务还有另一只手固定遥控器。
- ViSk 的大范围泛化依赖三台固定相机、一台腕部相机和触觉；不能把其 `20 cm` 范围直接当作 SO101 首轮目标。
- RLDG 的连接器母口确实固定，但公头初始位置变化，而且其目标是把高质量窄域 RL 数据蒸馏到 generalist。这是合理实验，不是证明固定目标也能学到视觉空间泛化。
- SHIELD 的钥匙任务甚至不用视觉，靠 F/T 搜索；它支持的是“逐步增加初始误差”的课程设计，而不是证明 RGB 足以完成你的钥匙任务。

### 19.4 本地 HALO-WA 代码实际随机了什么

你本地 `~/workspace/HALO-WA` 中的 `realworld_charger_sac_cnn_async.yaml` 很容易被误读：

```yaml
target_ee_pose: TARGET_EE_POSE
random_xy_range: 0.02
random_rz_range: 0.35
```

结合 [`franka_env.py`](/home/fenghao/workspace/HALO-WA/rlinf/rlinf/envs/realworld/franka/franka_env.py) 的 reset 实现可知，代码随机的是**机器人末端 reset pose**：x/y 各均匀扰动 `±2 cm`、yaw 约 `±0.35 rad`；`target_ee_pose` 仍是一个固定世界坐标。也就是说，它并没有在每个 episode 里物理移动充电口。

这有两个含义：

1. 它能训练“从不同插头初始姿态到固定工位”的局部策略；
2. 因为绝对 TCP/qpos 与固定目标存在确定关系，它不能单独证明策略通过视觉找到了移动后的插座。

而且这份配置的 actor 是 CNN + SAC 真机示例，位于 HALO-WA 所包含的 RLinf 基础设施中；**不能把它等同为 HALO-WA 的 WAM 本体已经在移动充电口上完成了验证。**

### 19.5 真正避免固定轨迹捷径：三个变量必须独立

定义：

- `T`：插座/锁的世界位姿；
- `G`：插头在夹爪中的抓取偏差；
- `R`：机械臂 episode 起始位姿。

第一版数据不能只令 `T=R+常数`，也不能每次移动插座后又把机械臂放回完全相同的“相对插座位姿”。推荐一个小型析因设计：

```text
目标位置 T：左 / 中 / 右 × 前 / 中 / 后（9 个小区域）
起始位姿 R：至少 3 种独立偏差
抓取偏差 G：自然产生的小偏心；记录而不是强求完全一致
```

不必采满 `9 × 3 × 所有角度` 的笛卡尔积，但要保证：

- 相同或非常接近的 `R` 下出现不同 `T`；
- 同一个 `T` 下也出现不同 `R/G`；
- 每个位置桶都有多条成功 demo，而不是某个位置只由某一天、某个操作者或某种光照产生；
- 数据划分按**位置桶/episode**做，不按帧随机切分。

其中第一条最重要：当 qpos 相同而插座在左、右两个位置时，正确的最初横向动作方向相反；模型若还能选对，才出现较强的视觉因果证据。

### 19.6 适合 SO101 的第一轮范围

下面是起始建议，不是假装具有普适性的“论文标准值”。范围应按你的遥操作成功率、相机像素分辨率和机构回差调整。

#### 第 0 阶段：只用于打通流程

- 10–20 条中心位姿或极小扰动 demo；
- 充电插头实验使用断电、确认无残余电压的插座或机械 dummy receptacle，不能让探索/接管训练直接面对带电市电；
- 插头/钥匙预先抓好，从孔前 3–5 cm 的 pre-insertion pose 开始；
- 目标是验证录制、时间同步、动作方向、相机曝光和安全停止；
- 这批数据可以保留，但不要让它占正式训练集的大多数。

#### 第 1 阶段：第一次正式 SFT

- 锁孔：先尝试 x/y `±3–5 mm`，yaw `±2–3°`；
- 尺寸更大的插头/插座：可从 x/y `±5–10 mm`，yaw `±3–5°` 开始；
- z、高低、roll/pitch 第一轮先固定，避免一次把六个自由度全打开；
- 约 60–100 条高质量短 episode 比大量包含长时间来回寻找的轨迹更有价值；
- 建议 60–70% 来自中心小扰动区，20–30% 来自边缘，10% 专门覆盖可恢复的轻微误对准，而不是全用中心位置。

选择下限时还应看网络真正输入的缩放后图像。当前 SO101 配置会把两路相机处理到约 `240×320` 并拼入 WAM；如果一次目标移动在模型输入上只有 1–2 像素，网络很容易忽略。小扰动至少要在全局相机中形成稳定可分辨的位移；可先录几个候选位置并测缩放后的孔中心像素差。

#### 第 2 阶段：根据第一次结果扩范围

- 若中心成功率高、边缘失败主要是到达误差，把平移范围逐轮扩大到 `±10–20 mm`；
- 若平移已稳，再单独增加 yaw；
- 若一扩大范围，遥操作 demo 出现大量来回试探，先改善相机视角、治具、控制增益或拆成“粗对准/局部插入”两段，不要用低质量轨迹硬填范围；
- HG-DAgger 应重点补当前策略会到达的边缘、偏转和接触恢复状态，而不是机械地继续均匀加中心成功样本。

因此，用户提出的“第一次先大概放在一个位置，但不严格固定，每次随机动一点，先看 SFT”是推荐方案；应进一步改成**事先定义的小范围分层采样**，而不是人随手移动，因为随手移动很容易导致位置分布、难度和示教质量相互相关。

### 19.7 训练前就要冻结的测试集

至少预留四组测试：

| 测试组 | 初态 | 回答的问题 |
|---|---|---|
| A：中心 seen | 训练高密度中心区域 | 基本能力是否学会 |
| B：范围内 held-out | 训练网格之间、训练中未用的精确位置 | 是否会插值，而非记住离散轨迹 |
| C：轻度 OOD | 比训练边界再外扩约 25–50% | 泛化边界在哪里 |
| D：视觉依赖诊断 | 固定相同 `R`，只把目标放左/右/前/后 | 动作是否因图像中的目标变化而改变 |

每组尽量使用固定的物理位置清单，而不是每次临时随手摆；至少 20–30 次，正式论文最好 50 次左右并报告置信区间。训练过程中不要不断把失败的 held-out 位置偷偷加入训练，否则它们已不再是测试集。

### 19.8 怎样证明它真的在视觉纠错

按证据强弱排列：

1. **反事实位置对**：机器人保持同一 qpos，把插座放左/右两个位置，检查最初修正动作是否随目标方向改变。
2. **目标在 episode 中被轻微扰动**：只在已具备安全闭环后测试；策略若能改变轨迹重新对准，是很强的在线视觉证据。第一轮不必做动态目标。
3. **分相机消融**：分别去掉全局相机、腕部相机，判断粗定位与近接触各依赖哪一路。
4. **proprio-only 重训基线**：去掉图像重新训练同规模模型。它比部署时突然喂黑图更公平。
5. **图像冻结/跨 episode 打乱**：可作快速诊断，但这会制造训练分布外输入；性能下降或不下降都不能单独作最终结论。
6. **固定轨迹或最近邻轨迹基线**：如果它与 AHA 成功率相近，就还没有证明 WAM 的视觉价值。

还应看动作而不只看最终成功率：目标横向移动 `Δx` 后，pre-contact 阶段的横向修正是否方向正确、幅值是否随 `Δx` 单调变化；接触失败后是否会先撤回再重新对准，而不是重复同一开环动作块。

### 19.9 这对 AHA-WAM 的 VideoDiT 尤其重要

目标固定时，VideoDiT 很容易学到“固定背景 + 机械臂沿常见路径移动”的平均未来；ActionDiT 也可能主要读 proprio 和动作历史。目标位置变化后，正确未来视频和动作都必须以当前图像中的孔位为条件，才有机会形成有意义的视觉—动作耦合。

但不要在第一轮同时随机：目标大范围位置、相机位置、背景、光照、插头类型和六维姿态。样本量有限时，这会让 VideoDiT 先忙于解释外观变化，而不是学习毫米级相对几何。建议顺序是：

```text
小范围 x/y → 加小 yaw → 扩平移范围 → 加抓取变化
→ 最后再做相机/背景/不同插头外观泛化
```

目标在一条 episode 内保持静止并不会削弱 AHA 的闭环意义。只要 episode 间位姿变化、执行时持续重新观测，并在接触前后以 receding horizon 更新动作，模型仍必须利用视觉决定当前该往哪里修正。

### 19.10 可以直接查看的论文、视频和数据

- [RLDG 官方 RL 轨迹数据](https://huggingface.co/datasets/charlesxu0124/RLDG)：含 USB-A、Ethernet、VGA connector insertion 和 FMB insertion，约 6.9 GB；可直接检查每条轨迹中母口是否固定、公头起始分布和 RL 恢复动作。
- [LeRobot ALOHA 电池插入数据](https://huggingface.co/datasets/lerobot/aloha_static_battery)：49 条、约 29,400 帧，可看真实示教中遥控器位置和电池姿态如何变化。
- [LeRobot ALOHA 人类仿真插入数据](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_human)：适合与 scripted 版本比较人类轨迹的变化和纠错。
- [LeRobot ALOHA scripted 插入数据](https://huggingface.co/datasets/lerobot/aloha_sim_insertion_scripted)：可结合 [ACT reset 源码](https://github.com/tonyzhaozh/act/blob/main/sim_env.py) 和位姿采样脚本核对二维随机初始化。
- [SHIELD / NIST 项目](https://sites.google.com/view/shield-nist)：论文报告开放约 50 小时插入交互数据，含多种模态和 reward 标签；先看项目视频中的新位置、移动目标和钥匙搜索实验。
- [ViSk 项目页](https://visuoskin.github.io/)：插头、USB、刷卡等任务的 rollout 视频最适合直观看空间随机化与触觉介入阶段。
- [RLDG 项目页](https://generalist-distillation.github.io/)：对照 RL 轨迹蒸馏后的 OpenVLA/Octo 与人类 demo 微调版本，重点观察卡在端口表面后的恢复差异。

最终推荐可以压缩成一句话：

> **首轮可以围绕一个名义工位采数据，但不要严格固定；让插座/锁孔在一个小而明确的二维区域内独立、分层变化，并从第一天就留出未见位置。先证明模型会随视觉改变修正方向，再扩大范围和做 RL。**
