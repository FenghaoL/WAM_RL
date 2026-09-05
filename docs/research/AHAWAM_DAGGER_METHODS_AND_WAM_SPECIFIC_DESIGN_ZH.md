# 从 DAgger 到 WAM-specific Interactive Correction：文献、设计与 AHA-WAM 迁移判断

> 更新日期：2026-09-02  
> 任务背景：双相机 SO-101，AHA-WAM，抓取插头并插入排插；已有 task-SFT、在线人工接管和 HUMAN-only/mixed 训练链路。  
> 本文目标：回答“还能从哪些 VLA/生成式策略的 DAgger 改进中借方法”“怎样做出真正针对 WAM 的改进”“video/action 两个分支究竟该怎样训练”。
> 对应部署端的逐字段实现契约见 [`SO101_AHAWAM_IG_OVCR_RICH_V4_COLLECTION_AND_REALTIME_EXECUTION_HANDOFF_ZH.md`](../hil/SO101_AHAWAM_IG_OVCR_RICH_V4_COLLECTION_AND_REALTIME_EXECUTION_HANDOFF_ZH.md)。

## 0. 先给结论

### 0.1 不建议把“慢速未来 latent + 快速当前 latent”作为主贡献

这句话有两个问题。

第一，它与 [HALO-WA](https://arxiv.org/abs/2607.04265) 的距离不够大。HALO-WA 已把 WA latent、WA reference action 和 robot state 一起交给 hybrid-attention actor–critic；论文还专门用 `RL-token-like` 比较“压成一个 token”和“保留分布式 latent”。如果只是把两个表示拼成 RL state，审稿人很容易把它理解成另一种 latent adapter。

第二，它没有准确描述当前 AHA-WAM 的普通部署路径。当前配置是 `action_video_read_mode: current_only`。在线生成每个动作块时，并不会先把未来视频完整去噪出来；Video DiT prefill 得到的是**当前帧、历史和文本条件下的逐层 planner K/V**，随后 OVCR 用最新双相机观测生成 query，改写第一帧 K/V，再供 Action DiT 读取。显式生成未来 13 帧只存在于 preview/诊断路径。参见：

- [`settings.yaml`](../../scripts/so101_pipeline/online_deploy/settings.yaml)：`current_only`、40-step horizon、8-step chunk；
- [`mot.py`](../../src/ahawam/models/wan22/mot.py)：`LayerwiseChunkKVCacheEditor`；
- [`AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md`](../architecture/AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md)：完整缓存与时间关系。

因此更准确的研究问题是：

> **在人类接管揭示局部失败以后，应当如何只改写 WAM 的“当前观测 → predictive planner K/V → action flow”接口，使模型获得精密纠错能力，同时保持原有长程行为？**

这与 HALO-WA 的区别是：HALO-WA 冻结 WAM，另训通用 actor–critic；这里研究的是**干预标签如何监督 WAM 内部的 observation-to-planner routing / action-flow correction**，第一版完全不需要 value、reward 或 TD learning。

### 0.2 推荐的两级路线

**低风险、已有论文验证的版本：**

1. 使用 IWR/SIRIUS 的数据分层与重加权；
2. 使用 Diff-DAgger 的生成流不确定性做自动接管提示；
3. 使用 FlowCorrect/CR-DAgger 的 gated residual 或低秩 action-flow adapter；
4. 冻结 Video DiT，以 demo/成功 rollout 做 anchor，避免 HUMAN-only 灾难性漂移。

**更像论文主方法的 WAM-specific 版本：**

> **Intervention-Gated OVCR Correction（暂称 IG-OVCR）**：冻结原始 WAM，在 OVCR query/KV editor 处插入小型低秩纠正 adapter；接管动作只训练 adapter，接管边界训练 gate，成功数据约束 gate 关闭时严格复现基础 WAM。

它比输出端 residual 更能说明“为什么要用 AHA-WAM”，也比 RL token 更直接利用 AHA 的异步 observation–planner 接口。

### 0.3 当前最重要的数据事实

现有 `ahawam_hil_raw` 在 HUMAN 控制期间 `policy_action_valid=false`。所以现有数据可以直接做：

- HUMAN action flow-matching；
- intervention/gate 标签；
- 接管前风险标签（只在 `policy_action_valid=true` 的行）；
- demo/成功段 anchor。

但它不能直接复现 FlowCorrect 的成对 residual target，因为 HUMAN 时刻没有同一状态下有效的 base-policy action，也没有保存 action-flow 初始噪声。要做严格 residual，需要离线重放基础策略，或下一轮在 HUMAN 控制时让策略继续 shadow inference 并保存 proposed chunk、noise seed、cache id/age。

---

## 1. 评分规则：不要把“新论文”误当成“权威论文”

下文采用四个独立指标，均为 1–5 分：

| 指标 | 5 分含义 |
|---|---|
| 影响力 | 奠基性工作、顶会/顶刊且已形成后续研究线；不是按当前引用数机械排序 |
| 证据可靠性 | 同行评审、真机、多任务、足够评测次数、公平基线和代码细节都较强 |
| AHA 适配度 | 能直接利用现有双相机、action chunk、flow matching、HIL schema 或 OVCR |
| 实现难度 | 1 最容易，5 最难；高分不是优点 |

2026 年论文即使实验漂亮，也不能因为“更新”就获得高影响力或高可靠性。预印本会明确标注。

---

## 2. 代表性方法全景

| 工作 | 核心改动 | 影响力 | 可靠性 | AHA 适配 | 难度 | 结论 |
|---|---|---:|---:|---:|---:|---|
| [DAgger](https://proceedings.mlr.press/v15/ross11a.html), AISTATS 2011 | learner rollout 状态由专家重标注并累计训练 | 5.0 | 5.0 | 3.5 | 3 | 理论基线；真人不可能给每个状态完整标签 |
| [HG-DAgger](https://ieeexplore.ieee.org/document/8793698), ICRA 2019 | 人判断何时 takeover，只收关键专家动作 | 4.8 | 4.3 | 4.5 | 2 | 当前采集范式的直接来源，但没有解决数据比例、坏动作和大模型遗忘 |
| [IWR](https://arxiv.org/abs/2012.06733), 2020/2021 | intervention/non-intervention 分池，50:50 抽样 | 4.3 | 4.2 | 5.0 | 2 | 当前最应该补齐的强基线之一 |
| [SIRIUS](https://arxiv.org/abs/2211.08416), IJRR | demo/intv/robot/pre-intv 四类加权；接管前低质量动作权重为零 | 4.4 | 4.6 | 5.0 | 2 | 比“只取人工段”更充分利用接管结构 |
| [ThriftyDAgger](https://arxiv.org/abs/2109.08273) | novelty + failure risk，预算感知 robot-gated query | 4.1 | 4.1 | 4.0 | 3 | 自动接管 gate 的经典参考 |
| [MILE](https://arxiv.org/abs/2502.13519), ICRA 2025 | 显式建模“人为何接管/不接管”的干预模型 | 3.7 | 4.0 | 4.0 | 3 | 可把没接管也当信息，适合小 gate |
| [RLIF](https://arxiv.org/abs/2311.12996), ICLR 2024 | 把 intervention 信号转成 reward，用 off-policy RL 学习 | 4.2 | 4.2 | 3.0 | 5 | 能超越次优专家，但当前周期风险高 |
| [Diff-DAgger](https://arxiv.org/abs/2410.14868), ICRA 2025 | 用 diffusion denoising loss 检测 OOD 并自动请求专家 | 4.1 | 4.2 | 4.8 | 3 | 最适合迁移为 Action DiT flow-loss gate |
| [Uncertainty Comes for Free](https://arxiv.org/abs/2503.01876) | 用 diffusion vector field 的跨模态/模态内分歧估计不确定性 | 2.8 | 3.2 | 3.8 | 3 | 可参考，但其 task-space pose 假设与 SO-101 joint action 不完全一致 |
| [Compliant Residual DAgger](https://arxiv.org/abs/2506.16685), NeurIPS 2025 | base 继续运行，人给 delta；小型高频 residual；接触/力控制 | 4.3 | 4.5 | 4.2 | 4 | residual 与 anchor 很有价值；SO-101 缺力控，不能照搬 compliant 部分 |
| [FlowCorrect](https://arxiv.org/abs/2602.22056), 2026 预印本 | 冻结 flow policy，约 10k 参数 LoRA 修正 flow velocity，再训 local gate | 2.3 | 3.2 | 5.0 | 3 | 与 Action DiT 最贴近，但当前数据缺 shadow action/noise |
| [APO](https://proceedings.neurips.cc/paper_files/paper/2025/hash/347110fb894281e5e937f6ccd998a6eb-Abstract-Conference.html), NeurIPS 2025 | demo/正常动作正标签，接管前动作负标签，KTO 式 action preference | 4.3 | 4.5 | 4.0 | 4 | 标签设计强；原 loss 针对离散 autoregressive action token，不能原样套 flow |
| [Diffusion Meets DAgger](https://arxiv.org/abs/2402.17768), RSS 2024 | 图像 diffusion 合成 eye-in-hand OOD/recovery 视觉状态 | 4.0 | 4.1 | 2.5 | 4 | 可做未来工作；合成视频与物理动作一致性是大风险 |
| [Sirius-Fleet](https://arxiv.org/abs/2410.22689), CoRL 2024 | visual world model 想象未来，异常模型决定请求人 | 4.0 | 4.1 | 4.2 | 4 | “WAM 帮 gate”最强先例之一，但不是纠正 WAM 内部 |
| [WM-DAgger](https://arxiv.org/abs/2604.11351), 2026 预印本 | action-conditioned world model 合成 OOD recovery，再做一致性过滤 | 2.0 | 2.8 | 4.0 | 5 | 概念相近；AHA 当前 video branch 不以 action 为条件，短期不建议 |
| [AutoIntervene](https://arxiv.org/abs/2608.07065), 2026 预印本 | successful support memory、phase-local threshold、切入/退出滞回 | 1.8 | 2.6 | 4.5 | 3 | 很新但对 action chunk gate 的工程细节很有启发 |
| [DRC](https://arxiv.org/abs/2503.15368), IROS 2025 | 遥操作相对修正逐渐衰减，平滑回归基础策略 | 3.3 | 3.8 | 4.0 | 3 | 下一版采集接口可用；当前 absolute takeover 数据无法恢复成真正的在线 relative control |
| [Policy Decorator](https://openreview.net/forum?id=e5jGTEiJMT), ICLR 2025 | 冻结大策略，用 model-agnostic residual RL 做在线 refinement | 4.1 | 4.0 | 3.2 | 5 | 证明 frozen-base residual 有效，但真机时间和 RL 工程负担较大 |
| [HIL-SERL](https://www.science.org/doi/10.1126/scirobotics.ads5033), Science Robotics 2025 | 人工 demo/intervention + sparse reward classifier + sample-efficient RL | 4.8 | 4.8 | 3.2 | 5 | 强 RL 对照思想；不是简单 DAgger，也不是必须的第一步 |
| [RLinf-USER](https://arxiv.org/abs/2602.07837), 2026 技术报告 | 统一真机异步采集/训练，支持 π0 + HG-DAgger 和多类 RL | 2.5 | 3.3 | 4.3 | 3 | 系统复用价值高；方法新意和完整 HIL recipe 证据仍弱于经典工作 |
| [HALO-WA](https://arxiv.org/abs/2607.04265), 2026 预印本 | 冻结 WA，distributed latent + reference action 的 actor–critic adapter | 2.4 | 3.3 | 5.0 | 4 | 最接近任务，但更说明我们必须把贡献放在 WAM 内部干预学习接口上 |
| [Hand-in-the-Loop](https://arxiv.org/abs/2605.15157), 2026 预印本 | 手—臂 VLA 无缝接管和相对 retargeting，减少切换抖动 | 1.8 | 2.8 | 3.0 | 4 | 对 dexterous hand 有价值；SO-101 单夹爪主要借平滑交接思想 |
| [TORL-VLA](https://arxiv.org/abs/2606.09337), 2026 预印本 | tactile-guided VLA online RL；intervention-censored critic | 1.8 | 2.8 | 2.8 | 5 | “被人救回的成功不能奖励之前坏动作”很重要，但当前无触觉/critic |
| [ORPA](https://arxiv.org/abs/2608.17323), 2026 预印本 | ACT 上的在线 residual policy adaptation | 1.4 | 2.4 | 3.5 | 3 | 进一步支持小 residual；时间太新，不宜作为权威支柱 |

这些不是所有含“intervention”的机器人论文，而是与当前决策真正相关的代表集。其余纯自动驾驶 DAgger、只做语言反馈、没有连续控制或没有真机证据的工作，没有因为关键词相似而塞进主表。

### 2.1 还有哪些“邻近但不应成为主线”的工作

- [Probe-Learn-Distill / residual RL data generation](https://arxiv.org/abs/2511.00091) 是 HALO-WA 的 residual baseline 来源之一，说明可先用小 residual learner 探索，再把改进轨迹蒸馏回大策略。它需要 reward/RL，短期复杂度高于直接 correction adapter。
- [RoboCopilot](https://arxiv.org/abs/2503.07771) 强调双边遥操作和策略之间的 compliant seamless switching，价值主要在接口与控制安全，不是新的 WAM 学习目标。影响力 2.4、可靠性 3.2、AHA 适配 2.8、难度 4。
- [Diffusion Meets DAgger](https://arxiv.org/abs/2402.17768) 与 [WM-DAgger](https://arxiv.org/abs/2604.11351) 都试图用生成模型扩大 OOD/recovery 数据；前者主要生成视觉状态，后者要求 action-conditioned world model。它们适合作为“第二阶段数据增广”，不适合在现有 HIL 数据尚未稳定利用前优先实现。
- [TORL-VLA](https://arxiv.org/abs/2606.09337) 的 `intervention-censored critic` 提醒：一次 episode 最终在 HUMAN 帮助下成功，不等于接管前 robot action 获得正回报。即便不做 RL，也应避免把 rescued rollout 的所有 POLICY 段当作成功正样本。

---

## 3. 第一条路线：把别的策略上的已验证 DAgger 改进迁到 AHA-WAM

### 3.1 IWR：解决 correction 被 demo/robot 数据淹没

[IWR](https://arxiv.org/abs/2012.06733) 的关键不是换 backbone，而是把 intervention 和 non-intervention 分池，并在 minibatch 中各占 50%。这是重要的**采样分布**设计，不是说物理数据必须复制或丢弃一半：少数类会重复采样，但所有样本都仍留在数据集中。

对当前 AHA：

- `demo`：保持抓取、搬运、粗对准能力；
- `human correction`：学习孔口附近和抓偏后的动作；
- 第一版可做 50% demo、50% correction；
- 更保守的 75% demo、25% correction 用于检查遗忘；
- 不要再跑 correction-only 全量微调并把它当 DAgger 的代表结果。

为什么 correction-only 一轮就可能崩：9,158 个人工帧高度集中在失败状态和接管后的局部恢复，而且控制者、动作平滑、episode 开头历史、视频监督分布都与完整 demo 不同。对 6B 联合模型做全量梯度更新，相当于用一个狭窄而重复的条件分布重写通用动作和视觉路由，而不是“只补几个错误”。

### 3.2 SIRIUS：把接管时间结构变成四类监督

[SIRIUS](https://arxiv.org/abs/2211.08416) 把样本分为：

$$
c_t\in\{\text{demo},\text{intervention},\text{robot},\text{pre-intervention}\}.
$$

它用目标类别分布 $P^*(c)$ 做 importance weight：

$$
w(s,a,c)=\frac{P^*(c)}{P(c)},\qquad
\mathcal L=-\mathbb E[w(s,a,c)\log\pi(a\mid s)].
$$

论文设置 intervention 目标比例为 50%，保持 demo 的原始比例，并将 `pre-intervention` 的目标比例设为 0。接管前窗口取约 2 秒、15 个动作，理由是人类反应延迟。需要注意：

- 它没有对接管前错误动作做“负 BC”；只是让这些动作不再被正向模仿；
- 它仍保留正常 autonomous robot data，因为未接管通常也是“人默许”的弱正反馈；
- 预窗口长度不能机械照抄 15。你的系统 15 Hz，但 chunk、网络延迟和人在 leader 上的反应不同，应通过接管视频估计。

对当前数据的严格实现：

1. HUMAN 段：action positive；
2. HUMAN onset 前 $K$ 步且 `policy_action_valid=true`：`pre_intv=true`；
3. SYSTEM_HOLD、无效 reply、队列切换帧：全部 mask；
4. 远离接管且最终成功的 POLICY：弱 positive/anchor；
5. 失败 rollout 中远离接管的 POLICY 不能一律视为正样本，应低权或只用于 gate。

### 3.3 APO：把“接管前动作不好”变成显式 preference

[APO](https://proceedings.neurips.cc/paper_files/paper/2025/file/347110fb894281e5e937f6ccd998a6eb-Paper-Conference.pdf) 在 OpenVLA 和 π0-FAST 上使用：

- demo 与正常 policy action：desirable；
- HUMAN correction：desirable；
- takeover 前 $K=10$ 个 policy action：undesirable；
- batch 比例 50% expert、25% correction、25% failure action。

它的 real-world square insertion 报告了 DAgger 可能不改善甚至在某些 OOD 条件退化，而 APO 的偏好训练更稳。这是非常适合写进论文动机的同行评审证据。

但是 APO 的 KTO/偏好目标建立在离散 autoregressive action token 的 log probability 上。AHA 的动作是 flow matching 连续生成，不能把 token log-probability 公式直接复制。可迁移的是：

- 正/负标签定义；
- source-balanced batch；
- reference-policy regularization；
- 不把所有 autonomous action 当作专家标签。

flow 版本可以先采用 pairwise energy / reconstruction preference，但这会增加算法风险。短期更稳的是让 pre-intervention 只监督风险 gate，而不进入 action flow loss。

### 3.4 Diff-DAgger：无需 value 的生成式策略自动接管

[Diff-DAgger](https://arxiv.org/abs/2410.14868) 的洞见是：diffusion policy 的训练 loss 本身可以作为动作是否处于训练分布的分数。对模型生成的 action 重新加噪并计算 expected denoising loss；如果连续 $K$ 次超过训练集分位数阈值，就请求专家。它在实验中包含 charger plugging，且是 ICRA 2025 工作。

对 AHA action flow，定义基础模型生成动作块 $\hat a$，随机采样 flow time $\tau_j$ 与噪声 $\epsilon_j$：

$$
x_{\tau_j}=\alpha(\tau_j)\hat a+\sigma(\tau_j)\epsilon_j,
$$

再用训练时相同的 target convention 计算：

$$
U_{\mathrm{flow}}(o_t)=
\frac1{J}\sum_{j=1}^J
\left\|v_{P_0}(x_{\tau_j},\tau_j,c_t)-v_j^*\right\|_2^2.
$$

必须直接复用仓库 scheduler 的 `add_noise` 和 target 代码，避免把 $\epsilon-a$ 与 $a-\epsilon$ 符号写反。

原论文用的大批量重复评估对 6B AHA 太贵。建议：

- 固定 Video DiT prefill/KV，不重复视频前向；
- $J=4\sim16$ 作为初始范围，而不是 512；
- 只在每个 8-step chunk 开头计算一次；
- 在成功 demo/成功 autonomous rollout 上估计 phase-local 95%/99% 分位阈值；
- 连续 2 个 chunk 或使用滞回再触发，避免单帧抖动。

这可以作为一个很清楚、易解释的 **robot-gated DAgger** 增量，即使还没做 IG-OVCR，也足以比纯人工主观接管更完整。

### 3.5 CR-DAgger / FlowCorrect：只学“基策略不会的修正”

[Compliant Residual DAgger](https://arxiv.org/abs/2506.16685) 冻结低频 base diffusion policy，用小型高频 residual policy 学人的 delta。论文的一个关键消融是：把**接管之前**的机器人动作当成 residual 的零标签会降低成功率，因为这些恰好是导致接管的坏动作；相反，接管开始后的短窗口应被过采样。

[FlowCorrect](https://arxiv.org/abs/2602.22056) 更直接作用于 flow policy：冻结 base，在 transformer MLP 中放约 10k 参数 LoRA，修正 flow velocity，并训练 local gate。其 target 大意是从相同噪声状态把流导向 corrected action；同时用少量成功 rollout 做 anchor，防止 easy/ID 状态退化。

两者共同支持以下原则：

1. 冻结 base，避免少量纠正破坏长程技能；
2. 纠正模块学习 delta，而不是重新学完整动作；
3. gate 只在局部困难状态开启；
4. 需要成功 base rollout/demo 作为“不要改”的 anchor；
5. takeover 前坏动作适合风险标签，不适合正 BC 或 zero residual。

但当前数据不是 compliant relative correction，而是 leader absolute takeover；HUMAN 时 policy shadow action 无效。所以可以先做“expert-conditioned flow adapter”，不能声称复现了严格的 paired residual correction。

### 3.6 MILE、ThriftyDAgger、AutoIntervene：让 gate 不只靠人的感觉

三类信息互补：

- ThriftyDAgger：风险 + novelty，只有既危险又值得花预算时才问人；
- MILE：接管和“不接管”都反映操作者的隐含判断，应学习 human intervention model；
- AutoIntervene：action chunk 需要 phase-local support、不同的切入/退出阈值和 hysteresis。

对插头任务，统一 gate 可以写为：

$$
U_t = z(U_{\mathrm{flow},t})
+\lambda z(U_{\mathrm{route},t})
+\eta z(\Delta t_{\mathrm{cache}}),
$$

其中：

- $U_{\mathrm{flow}}$：Action DiT 对自身生成 chunk 的 flow reconstruction/OOD 分数；
- $U_{\mathrm{route}}$：当前 observation 引起的 OVCR K/V delta 与成功数据分布的距离；
- $\Delta t_{\mathrm{cache}}$：当前 planner cache 的年龄或 generation lag。

不要假定 $\|\Delta KV\|$ 大就一定失败；正常任务阶段变化也可能产生大改写。阈值必须按抓取、运输、粗对准、插入等 phase 校准，或者用一个小型 risk head 学习：

$$
p_\psi(I_{t:t+H}=1\mid U_t,h_t),
$$

positive 是未来一个 action horizon 内发生 takeover，negative 来自成功且远离 takeover 的 chunk。采用 class-balanced BCE/focal loss，并单独评估 precision、recall、false alarm/minute。

---

## 4. 第二条路线：真正为 WAM 设计 DAgger

## 4.1 WAM-DAgger 与普通 VLA-DAgger 的痛点不同

普通 policy 通常近似 $a_t\sim\pi(o_t)$。AHA 的在线动作更接近：

$$
M_t=\mathrm{VideoDiT\_prefill}(o_{\le t_0},p),
$$

$$
Q_t=f_{obs}(o_t),\qquad
\widetilde M_t=\mathrm{OVCR}(M_t,Q_t),
$$

$$
a_{t:t+C-1}=\mathrm{FlowActionDiT}(x_0,\widetilde M_t,s_t,p).
$$

因此同一张图像并不唯一决定动作：planner cache 的版本、历史、年龄、action-flow noise、chunk index 都会改变输出。由此产生八个 WAM-specific 问题：

1. HUMAN correction 究竟是在修正 planner、当前观察路由，还是 action generator？
2. 离线只重放 RGB，能否重现 rollout 时的缓存条件？
3. HUMAN 段的未来视频是“专家导致的未来”，而 rollout 开头缓存是“旧 policy 历史”；混合是否形成反事实不一致？
4. 一个 takeover 往往发生在 action chunk 中途，旧 chunk 未执行部分如何标注？
5. 人接管后的 abrupt absolute action 是否应直接当 flow target？
6. 如果只更新 Action DiT，固定 Video DiT 的 feature 是否仍足够区分毫米级误差？
7. 如果更新 Video DiT，少量 correction video 会不会破坏原先世界表示和抓取阶段？
8. planner 更新滞后本身是否是失败源，而非 action mapping 不准？

当前 rich-v3 schema 已解决第 4 个问题的大部分数据工程：旧队列失效、SYSTEM_HOLD 和 action supervision mask。下一步应把 cache generation/age 与 shadow policy chunk 也纳入日志。

## 4.2 三个可选注入位置

| 位置 | 做法 | 优点 | 缺点 | 论文区分度 |
|---|---|---|---|---|
| 输出 action residual | $a'=a_0+g\delta a$ | 最快、最稳、易限幅 | 很像 CR-DAgger/Policy Decorator/HALO residual | 低 |
| Action DiT LoRA / flow velocity | $v'=v_0+g\Delta v_\phi$ | 符合生成式动作头，参考 FlowCorrect | 仍是通用 flow policy 方法 | 中 |
| OVCR query/KV correction | $Q'=Q+g\Delta Q_\phi$ 或 $M'=M+g\Delta M_\phi$ | 直接作用于 AHA 独有的快观测—慢 planner 接口 | 需要细心改模型和导出中间量 | 高 |

推荐先把 Action DiT LoRA 做成 sanity baseline，再把 OVCR correction 作为主方法。这样即使新结构暂时不增益，也有经过验证的稳定结果，而不是把所有实验押在一个结构上。

## 4.3 主方法草案：Intervention-Gated OVCR Correction

AHA 已有的 `LayerwiseChunkKVCacheEditor` 每层执行两次 attention：

1. 当前 observation query 从第一帧 planner K/V 中读取 routed feature；
2. routed feature 解码回所有第一帧 token，得到 $\Delta K_l,\Delta V_l$；
3. 初始化很小的 gate 控制 delta，再把更新后的 K/V 供 Action DiT 使用。

这已经天然是“当前观测纠正慢缓存”的位置。新增模块可保持很小：

$$
\Delta Q_{t,l}=A_lB_l\,h_\phi(Q_t,s_t,e_t),
$$

$$
Q'_{t,l}=Q_{t,l}+g_t\Delta Q_{t,l},
$$

再沿用原有 KV editor：

$$
(K'_{t,l},V'_{t,l})=mathrm{OVCR}_l(K_{t,l},V_{t,l},Q'_{t,l}).
$$

这里 $e_t$ 可包含 cache age、chunk index 和 flow uncertainty。与“另挂 actor”相比，最终动作仍由原 Action DiT 生成；adapter 只改变 Action DiT 从 WAM predictive prior 中读取到什么。

### 参数冻结建议

第一版：

- Video VAE、text encoder：冻结；
- Video DiT：冻结；
- 原 OVCR/KV editor：冻结；
- 原 Action DiT：冻结；
- 只训练低秩 $\Delta Q$/$\Delta KV$ adapter、gate，必要时再打开 Action DiT 输出层 LoRA。

原因不是“Video DiT 永远不该调”，而是当前 correction 数据少且窄，普通部署又不显式生成未来视频。先证明 action correction 能否通过已有 predictive representation 完成，因果解释最干净、训练成本最低。

### 三个训练目标

**1. HUMAN correction action flow loss**

$$
\mathcal L_{\mathrm{corr}}
=\mathbb E_{(o,a^H),\tau,\epsilon}
\left[m^H\left\|v_{\phi}(x_\tau,\tau,c)-v^*\right\|^2\right].
$$

`m^H` 必须只覆盖同一连续 HUMAN 段和有效动作，不能跨 SYSTEM_HOLD、释放边界或下一个 episode。

**2. Base-preservation anchor**

在 demo、成功 autonomous 且远离 takeover 的状态，令 gate 为零并保持基础动作分布：

$$
\mathcal L_{\mathrm{anchor}}
=\|v_\phi-v_{P_0}\|_2^2+\beta\,\|\Delta Q\|_2^2.
$$

若使用 GT demo flow loss，也要保留 reference distillation；仅用 demo GT 不保证新模型在同一噪声下保持 P0 的生成路径。

**3. Intervention gate loss**

$$
y_t=\mathbf 1[\exists k\in[0,H_g): I_{t+k}=1],
$$

$$
\mathcal L_{gate}=\mathrm{BCE}(g_t,y_t)
+\rho\,\mathbb E_{\mathrm{safe}}[g_t].
$$

第二项惩罚在安全状态乱开 gate。切换执行时用两阈值：`enter_threshold > exit_threshold`，并要求连续若干 chunk 满足，避免抖动。

总目标：

$$
\mathcal L=\mathcal L_{corr}
+\lambda_a\mathcal L_{anchor}
+\lambda_g\mathcal L_{gate}.
$$

这仍然是交互式模仿/监督后训练，不应在论文里叫 RL。未来加 sparse success reward 或 critic 才是 RL extension。

## 4.4 为什么这个方法比 RL token 更像 WAM 贡献

| 问题 | RL token / HALO-like adapter | IG-OVCR |
|---|---|---|
| WAM backbone | 主要作为 feature provider/reference action generator | correction 直接发生在 WAM 的 observation-to-planner coupling |
| 表征压缩 | 通常压成 token 或另做 latent reader | 保留逐层、逐 token planner K/V |
| 动作生成 | 外部 actor 可能取代 Action DiT | 原 Action DiT/flow generator 仍执行动作 |
| 人类信号 | 放进 replay 做 BC/RL | 明确监督 gate、routing correction、action flow 三种语义 |
| 长程技能保持 | 靠 BC regularization/reference action | 结构上冻结 base，safe 状态 gate=0 |
| 主要主张 | “WAM latent 是好 RL state” | “干预应局部改写 WAM 的闭环路由，而非重训整个 world/action model” |

真正需要实验验证的不是“听起来更 WAM”，而是：同等参数量、同等接管数据下，OVCR adapter 是否优于 output residual 和 Action DiT LoRA。

---

## 5. Video DiT 和 Action DiT 到底怎样更新

## 5.1 四种方案不是同一个问题

| 方案 | Video loss | Video DiT 梯度 | Action loss | 判断 |
|---|---:|---:|---:|---|
| A. joint full fine-tune | 有 | 有 | 有 | 当前 correction-only 已显示高遗忘风险，不宜做主方法 |
| B. 不算 video loss，但不冻结 Video DiT | 无直接 loss | **仍可能通过 action attention 反传** | 有 | 容易被误称为“只调 action”；实际上世界表征会被 action loss 改写 |
| C. 冻结 Video DiT，训练整个 Action DiT/OVCR | 无 | 无 | 有 | 比 B 干净，但小数据仍可能重写 Action DiT |
| D. 冻结 base，只训 adapter/gate | 无 | 无 | 有 | 第一版首选，最适合小量 HIL |

用户提出的“不要 video loss，但不冻结 Video DiT”完全可以做，却不是中性选择。Action DiT 每层读取 video K/V，action loss 会穿过 mixed attention/OVCR 回到 Video DiT 或编辑器；这可能让 latent 更适合动作，也可能在没有未来视频约束时丢掉预测语义。必须通过梯度路径检查和消融明确命名为 `action-loss co-adaptation`。

## 5.2 什么时候才值得重新打开 video loss

只有当以下诊断至少一个成立时再做：

1. 同一孔位的 1–5 mm/1–5° 微扰在 VAE/OVCR 表示中不可分；
2. probe 能从当前 RGB 预测偏差，但不能从冻结 latent 预测，说明表征丢细节；
3. 人工 correction 动作在给定冻结 latent 时不可拟合，而更直接的视觉 encoder 可以拟合；
4. 明确生成的 future-video/latent 对接触结果判断错误，并与 action failure 相关。

若打开：

- 只用长度足够、时间连续的 HUMAN 段计算 video loss；
- demo/原始数据至少 1:1 replay；
- 加 $\|Z-Z_{P_0}\|$ 或 K/V distillation；
- Video DiT 只用 LoRA/末层小学习率，不做 5B 全量 correction-only；
- 报告抓取、运输等旧阶段是否退化。

不要把“未来 HUMAN 视频真实存在”误解为“联合 video loss 一定有益”：这些帧是人已经接管后的结果，它未必教会自主策略在 takeover 前怎样避免错误。

---

## 6. 当前数据如何分配监督

| 时间区域 | Action flow/BC | Gate/risk | Video loss | 说明 |
|---|---|---|---|---|
| 初始 demo | 正样本/anchor | negative | 是 | 保持完整任务能力 |
| 成功 rollout 中正常 POLICY | 弱正样本或 P0 distill | negative | 可选 | 人未接管是弱认可，但不是完美专家 |
| takeover 前有效 POLICY | **不要正向模仿** | positive/undesirable | 否 | APO/SIRIUS/CR-DAgger 一致启发 |
| HUMAN correction | expert target | positive | 长连续段可选 | 当前最重要的 correction 数据 |
| HUMAN release 后短暂恢复 | grace/mask 或低权 | 退出标签需滞回 | 否 | 避免把切换瞬间误当稳定 safe |
| SYSTEM_HOLD/无效 reply | mask | mask | mask | 不含主动控制语义 |
| 失败 episode 其余 POLICY | 不做默认正样本 | 可作弱风险/unknown | 否 | 不能因“没来得及接管”就视为正确 |

### 建议 batch 结构

初始可用：

- 50% demo/success anchor；
- 25% HUMAN correction；
- 25% pre-intervention/safe gate 样本，其中正负在 gate loss 内平衡。

Action loss 只作用前两类；第三类主要训练 gate。不要为了“每类都利用”而让 pre-intervention 动作进入 action BC。

---

## 7. HALO-WA 的 HG-DAgger 对照究竟说明了什么

HALO-WA v1 论文给出的 real-world 对照为：WA-base 26.4%、Probe-Learn-Distill 52.1%、HG-DAgger 62.9%、RL-token-like 67.9%、HALO-WA 87.1% 的四任务平均成功率。每任务用 100 条真机 demo 训练 task-specific GigaWorld Policy，并用 35 次评估；主方法在线 45–75 分钟。论文明确说明：

- 人发现错误/潜在失败时实时接管；
- 接管动作写入 replay；
- 接管结束后 WA 重新推理 reference chunk，补齐统一 chunk 结构；
- 在线时 WAM backbone 冻结，只更新 adapter；
- HG-DAgger baseline 用在线 human intervention data 做 supervised imitation。

但补充材料没有给 HG-DAgger 的参数更新对象、demo/correction 采样比例、epoch/update 数、是否有 anchor、接管前窗口处理等细节；公开任务脚本也主要对应 HALO-WA 的 online RL，RLinf 中通用 DAgger 实现是另外一套支持 Pi0/MLP 的框架。故不能得出“HG-DAgger 天生只有 62.9%”或“你现在 correction-only 全量微调就是论文的精确复现”。更合理的解释是：它证明**纯接管模仿是重要强基线，但经过 value feedback 的轻量 adapter 在其设置中更强**。

这反而给当前工作留下空间：HALO-WA 没有研究 DAgger 信号应怎样监督 WA 内部的逐层 routing/action flow，也没有充分展开 pre-intervention negative、shadow action pairing、video/action selective loss。

---

## 8. 需要新增的日志，否则很多“高级 DAgger”无法严格实现

下一轮 rich-v4 建议每个控制 step/chunk 保存：

1. `policy_shadow_action_chunk`：即使 HUMAN 控制，基础 policy 仍后台推理；
2. `policy_shadow_valid`、模型 checkpoint hash；
3. `action_flow_seed` 或初始噪声 hash/张量；
4. `planner_cache_id/generation`、`planner_cache_age_ms`；
5. `chunk_request_id`、`action_index_in_chunk`、被取消的剩余 chunk；
6. OVCR query 的小型 pooled summary，而非完整 30 层 K/V；
7. `u_flow`、`u_route`、gate score 和触发原因；
8. HUMAN action、实际 executed action、takeover onset/release 的单调时钟；
9. episode success、阶段标签或至少事件标签（grasp/contact/insert）；
10. 人工对该 intervention 的原因：抓偏、目标偏差、姿态错、接触卡住、安全。

完整 K/V 太大，不宜直接入 parquet；原始图像、checkpoint/version、cache metadata 和固定维 pooled feature 已足够做多数重放与诊断。

---

## 9. 实验矩阵：怎样证明不是“换了个 adapter”

### 9.1 主表最低配置

| ID | 方法 | base 是否冻结 | 使用的 HIL 信息 | 目的 |
|---|---|---:|---|---|
| B0 | task-SFT P0 | — | 无 | 原始能力 |
| B1 | naive correction-only continuation | 否 | HUMAN action | 保留当前失败结果，显示简单堆 correction 的风险 |
| B2 | HG/IWR mixed SFT | 否或冻结 Video | demo + HUMAN | 公平强 DAgger baseline |
| B3 | SIRIUS-style weighted BC | 建议冻结 Video | demo/HUMAN/robot/pre-intv | 验证结构化接管标签价值 |
| B4 | output residual / Action-LoRA | 是 | HUMAN + anchors | 通用 generative-policy correction baseline |
| Ours | IG-OVCR | 是 | HUMAN + anchors + boundary/gate | 验证 WAM-specific 注入位置 |

如果实现 robot gate，再把 human-gated 与 auto-gated 的 autonomous success、人类时间和误报率分开报告。自动 gate 不应偷用评测时的成功标签。

### 9.2 关键消融

1. output residual vs Action DiT LoRA vs OVCR adapter；
2. correction-only vs +demo anchor vs +success-rollout anchor；
3. 无 gate vs human gate vs learned gate；
4. `U_flow` only vs `U_route` only vs combined + cache age；
5. 全局阈值 vs phase-local threshold；
6. pre-intervention 当正 BC、mask、只做 risk label 三种；
7. Video frozen vs action-loss co-adaptation vs selective long-HUMAN video loss；
8. 无 shadow paired residual vs offline reconstructed shadow vs 在线真实 shadow；
9. 适配参数量和推理延迟匹配，避免 OVCR 方案只是参数更多。

### 9.3 指标

- 完整任务 autonomous success；
- grasp、coarse align、contact、insert 分阶段成功率；
- 每 episode intervention 次数、总 HUMAN 控制秒数；
- autonomous recovery success 和 recovery time；
- robot gate precision/recall/F1、false alarms/minute、漏报导致失败率；
- 旧初始化/容易场景上的回归；
- 最终位置/姿态误差，若能可靠测量；
- 平均动作 jerk、takeover/release discontinuity；
- adapter 参数量、AHA 推理延迟和显存；
- 数据量曲线：5/10/20/全部 correction episodes。

每个任务只报 10 次非常不稳。插入成功率若是主结果，推荐至少 30 次/方法/任务；HALO-WA 是 35 次。若资源不足，主方法和最强基线做 30–50 次，其余消融 15–20 次并报告 Wilson 区间。

---

## 10. 面向当前周期的执行优先级

### P0：两天内必须拿到的稳结果

1. 固定 P0 和评测初始化；
2. 重新跑一个 Video DiT frozen 的 IWR 50:50 demo/correction；
3. 实现 SIRIUS 四类 mask：pre-intv 不进入 action loss；
4. 保留 correction-only 崩溃结果作为 naive baseline，不继续在其上堆 epoch；
5. 评测至少分抓取、粗对准、插入三阶段。

### P1：最值得实现的算法增量

1. Action DiT flow-loss uncertainty，先离线画 success/failure/takeover 前后的 ROC；
2. 小型 action-flow LoRA 或 residual adapter + safe anchor；
3. 学一个 intervention gate，采用 horizon label 和 hysteresis；
4. 下一轮采集打开 shadow policy logging。

### P2：论文主贡献候选

1. 把同等参数量 adapter 从 Action DiT 输出位置移到 OVCR query/KV editor；
2. 加入 route disagreement 与 cache age；
3. 对比 output residual、Action-LoRA 和 IG-OVCR；
4. 只有冻结 latent 明显缺精细视觉时，才开放 selective video adaptation。

### 暂缓

- 从零训练大 value model；
- 对 5B Video DiT 做 correction-only RL/full fine-tune；
- 用 AHA 当前非 action-conditioned Video DiT 大规模合成 WM-DAgger recovery；
- 同时实现 tactile、force control 和高频 compliant residual；
- 把所有想法一次叠满，导致无法做因果消融。

---

## 11. 最可能成立与最可能失败的研究假设

### 最可能成立

1. mixed/anchor training 显著优于 correction-only；
2. pre-intervention 只做 risk label 优于把它当 BC 正样本；
3. 冻结大 base、小 adapter 能显著减少抓取/运输退化；
4. Action flow self-loss 在接管前或失败状态上统计性升高；
5. OVCR adapter 对需要视觉纠偏的插入，比纯 action residual 更数据高效。

### 高风险假设

1. OVCR delta 大等于失败：它也可能只是正常阶段变化；
2. 当前视觉分辨率足够表示毫米级孔口偏差；
3. absolute takeover 可无损转换成 residual；
4. correction-only 视频能改善 planner 而不遗忘；
5. 一个插头任务足以证明通用 WAM-specific 方法；
6. 自动 gate 会自然减少人力：误报过多时可能比人工主观接管更累。

这些风险都应先用离线 probe/ROC 和小样本 ablation 检查，而不是直接进行昂贵联合训练。

---

## 12. 论文定位建议

如果最终只有 mixed HG-DAgger + 小 adapter，最诚实的定位是系统/应用型：

> A human-in-the-loop adaptation pipeline for asynchronous world-action models in contact-rich insertion.

如果 IG-OVCR 在至少三个接触任务上稳定优于同参数量 output/action adapter，可以升级为方法型：

> Intervention-gated predictive-context correction for world-action models.

贡献点应写成：

1. 揭示普通 DAgger 对异步 WAM 的 cache/chunk/counterfactual label 问题；
2. 提出利用 intervention boundary 的 WAM 内部局部 routing correction；
3. 提出 action-flow uncertainty + predictive-route disagreement 的 robot gate；
4. 在多种精密接触任务中，以相同 HIL 数据量和相同 adapter 参数量验证。

不要把 DAgger 强行叫 RL。可以在讨论中说明它与 HALO-WA/HIL-SERL 的在线 RL 互补：IG-OVCR 先把 correction 学成安全初始化，后续 critic/reward 只优化小 adapter。

---

## 13. 主要资料入口

### 奠基与干预学习

- [DAgger, AISTATS 2011](https://proceedings.mlr.press/v15/ross11a.html)
- [HG-DAgger, ICRA 2019](https://ieeexplore.ieee.org/document/8793698)
- [IWR: Human-in-the-Loop Imitation Learning using Remote Teleoperation](https://arxiv.org/abs/2012.06733)
- [SIRIUS: Robot Learning on the Job](https://arxiv.org/abs/2211.08416)
- [ThriftyDAgger](https://arxiv.org/abs/2109.08273)
- [MILE](https://arxiv.org/abs/2502.13519)
- [RLIF](https://arxiv.org/abs/2311.12996)

### 生成式动作、VLA 与 residual correction

- [Diff-DAgger, ICRA 2025](https://arxiv.org/abs/2410.14868)
- [Uncertainty Comes for Free](https://arxiv.org/abs/2503.01876)
- [Compliant Residual DAgger, NeurIPS 2025](https://arxiv.org/abs/2506.16685)
- [FlowCorrect](https://arxiv.org/abs/2602.22056)
- [Action Preference Optimization, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/347110fb894281e5e937f6ccd998a6eb-Abstract-Conference.html)
- [Decaying Relative Correction, IROS 2025](https://arxiv.org/abs/2503.15368)
- [Policy Decorator, ICLR 2025](https://openreview.net/forum?id=e5jGTEiJMT)
- [HIL-SERL, Science Robotics 2025](https://www.science.org/doi/10.1126/scirobotics.ads5033)

### World model / WAM 相关

- [Sirius-Fleet: Multi-Task Interactive Robot Fleet Learning with Visual World Models, CoRL 2024](https://arxiv.org/abs/2410.22689)
- [Diffusion Meets DAgger, RSS 2024](https://arxiv.org/abs/2402.17768)
- [WM-DAgger](https://arxiv.org/abs/2604.11351)
- [AutoIntervene](https://arxiv.org/abs/2608.07065)
- [HALO-WA](https://arxiv.org/abs/2607.04265)

---

## 14. 一句话决策

**先把当前 DAgger 从“只模仿 HUMAN 段”升级为“SIRIUS 式结构化标签 + demo/success anchor”，再实现 Diff-DAgger 式 flow uncertainty；若要做出区别于 HALO-WA 的方法贡献，把小型 gated correction 放到 AHA-WAM 的 OVCR query/KV 路由，而不是再造一个读取 latent 的外部 actor–critic。**
