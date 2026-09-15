# 视觉证据与专家动作分开使用：WAM纠正后训练的论证

## 1. 推荐的方法定位


推荐题目方向：**Learning from Human Corrections and Interaction Video in Memory-Based World–Action Models**。方法名称可用 **Context-Preserving Human-Gated Post-Training**，不必再造容易掩盖内容的缩写。

核心洞察是：**一次动作不值得模仿，不代表它产生的观测不值得利用；控制权切换的时刻，也不应自动成为模型历史和视觉目标的边界。** 对普通行为克隆，一段交互主要提供观测—动作标签；对同时使用记忆与视频预测目标的WAM，同一段交互还提供状态上下文与时序预测目标。因此，“是否采用这段数据”应细化为“它参与哪个条件或损失”。

这是对工程问题的合理方法化：明确一个建模假设为何不成立，定义可复用的数据算子，再做单因素对照。它不需要把工程事实改写成未经验证的认知机制。“加深理解历史如何引入操作”可以改成“在策略实际访问的状态与历史条件下学习专家纠正，并以实际视觉后续作为辅助预测目标”。

最合适的总括是 **human-supervised action learning with self-supervised visual prediction on interaction trajectories**。真正需要验证的增量，不是联合动作/视频loss本身，而是短接管数据如何在这些已有目标下保持合适的上下文与目标范围。

## 2. 原设想中需要修正的四点

### 2.1 历史条件不等于自监督目标

当前历史图像在无梯度前向中重建K/V，再作为当前分支的条件。历史会改变动作与视频预测，也改变当前参数收到的梯度，但没有单独的“重建历史”损失，梯度不通过历史编码路径回传。因而不宜统称“通过过去和未来视频进行自监督训练”。准确说法是“保留过去作为上下文，以未来作为自监督预测目标”。

### 2.2 自主rollout视频是真实测量，不是模型自产标签

这里的未来图像来自真实机器人交互，而非世界模型生成的想象视频。对真实图像进行遮噪/流预测，目标来自数据本身，可以称为自监督；但接管动作和成功筛选仍含人工信息。因此整个管线不能称为完全自监督，也不是self-distillation。

当前未来延伸只针对满足条件的成功混合控制轨迹，起点仍来自人类纠正段；并未对所有纯自主起点进行全轨迹自监督后训练。历史及延伸区间可能含更早/再次人类接管，不能全部称为“纯自主视频”。

### 2.3 人类动作纠正的是行为目标，不是自动消除world-model bias

人类动作在困难状态提供可执行的纠正方向，避免把这些位置的原策略动作当作专家标签。这可以改变训练所强调的状态—动作分布。但专家接管具有选择性，成功筛选也改变未来分布；两者没有使数据无偏。

尤其当前 `action_conditioned=false`，不能把视觉目标写成“如果执行任意候选动作，会发生什么”的因果动力学。它是在记录中的观测/历史/任务条件下拟合实际未来，未来依赖当时的人机混合控制及轨迹选择。精确的专家动作并不自动校准所有视觉预测。

### 2.4 可推广的是问题结构，不是所有WAM必需记忆

AHA-WAM本身使用滚动K/V；其论文的组件消融支持记忆有帮助，但不支持“所有WAM去掉记忆都会严重失效”。Fast-WAM具有从当前观测读取表征的直接策略接口；DIM-WAM则专门研究更长历史。模型、任务是否部分可观测、是否使用持久状态都会影响结论。[1](https://arxiv.org/html/2606.09811v1) [2](https://arxiv.org/html/2603.16666v1) [3](https://arxiv.org/abs/2606.27677)

适用范围宜写成“带时间上下文及未来预测目标的WAM”，或更宽泛地称“带状态的预测式策略”。在AHA-WAM上验证一种实例，不等于已经验证跨WAM泛化。Stage 1无历史表现较差仍需给出对应checkpoint与正式评估；不能与Stage 2接管内历史截断混为一个对照。

## 3. 三种可供论文选择的切入点

| 切入点 | 研究问题与方法 | 与当前实现贴合度 | 最关键的新证据 |
|---|---|---|---|
| A. 接管边界之外的上下文与监督 | 控制来源只决定专家动作标签，不统一裁剪历史与视觉目标；分别定义三种时间范围 | 最高，建议主线 | 同数据、同骨干的接管内历史对照；基本full与vidext |
| B. 行为可模仿性与视觉证据有效性分离 | 自主动作可能次优，但其真实视觉后果仍可用于表征学习；人类标签提供局部行为约束 | 高，但需谨慎限定成功混合轨迹 | video loss开关；有效视频监督覆盖；独立恢复指标 |
| C. 部署分布上的联合后训练 | 原demo回放保留正常技能，交互中的纠正动作与视觉预测适配策略访问状态 | 中高，联合训练已有先例 | 正常任务与局部恢复两种指标；配比敏感性与第二任务 |

推荐A作为方法定义，B作为洞察，C作为训练协议与实验分析。它们是同一方法的三个视角，不宜拆成三个重复贡献。不要把“只在WAM上做”作为唯一差异；说明WAM增加了哪些需要特别处理的输入和目标，再用实验回答这些处理是否有用。

## 4. 数学表述：无需假装有新的RL目标

### 4.1 交互与选择算子

设一次真实交互为

$$
\tau=\{(o_t,a_t,c_t)\}_{t=0}^{T},\quad
c_t\in\{\mathrm H,\mathrm P,\mathrm{HOLD}\},\quad y_\tau\in\{0,1\}.
$$

$o_t$包含双相机图像与机器人状态；H/P表示人类/策略控制，HOLD表示系统保持。$y_\tau$为完整轨迹结果。采集行为策略$\mu$由人类门控与当前策略共同组成；它不是纯自主策略$\pi$。纠正起点集合$\mathcal I_H$只含有效HUMAN位置。

对于$t\in[s,e]\subset\mathcal I_H$，分别定义：

- $C_t^-$：同一轨迹中、严格早于当前观测的历史帧，可跨越s；不跨轨迹。
- $m^a_{tj}$：第j个动作目标是否仍属于当前有效HUMAN段；超出段末的padding不监督。
- $C_t^+$：从t开始的实际视频时间线。基本版本在e结束；vidext在成功轨迹上追加e之后非HOLD执行帧，按固定49位置窗口查询。
- $m_t^v$：未来视频查询是否具备完整有效目标；延伸失败或不足不以重复帧冒充完整监督。

这不是给任意旧策略动作赋予“可信度分数”，也没有引入未实现的价值估计。它是已有控制来源和有效性记录所定义的确定性数据构造。

### 4.2 带停止梯度历史的生成式学习

令历史状态、当前共享视觉表征分别为

$$
h_t=\operatorname{sg}\!\left(E_\phi(C_t^-)\right),\qquad
z_t=F_\phi(o_t,h_t,\ell),
$$

其中$\operatorname{sg}$表示stop-gradient，$\ell$为任务指令；表达式省略视频潜变量噪声及chunk对齐细节。动作flow模型通过原生路由使用$z_t$，视频分支拟合来自$C_t^+$的未来潜变量。没有推理时读取未知未来的操作。

对动作目标a和高斯噪声$\epsilon$，构造

$$
x_\sigma=(1-\sigma)a+\sigma\epsilon,\quad
v^*=\epsilon-a,
$$

$$
\ell_i^a=\frac{\sum_jm^a_{ij}w_{ij}^a
\|v_\theta(x_{\sigma,ij};o_i,h_i,\ell)-v^*_{ij}\|^2}
{d_a\max(1,\sum_jm^a_{ij})}.
$$

视频在VAE潜空间使用基础模型的生成式损失$\ell_i^v$。设$q_r=rD+(1-r)H$为起点采样分布，r为demo份额，概念上有

$$
\mathcal J_r=\mathbb E_{q_r}$$ell^a]
+\lambda_v\frac{\mathbb E_{q_r}[m^v w^v\ell^v]}{\mathbb E_{q_r}[m^v]}.
$$

该式是总体分布下的解释，不是小batch实现的逐步等式。代码中视频loss按**每卡有效样本数**归一化，再由DDP平均：

$$
\widehat{\mathcal J}=\frac1R\sum_{b=1}^{R}
\left$$frac1{|B_b|}\sum_{i\in B_b}\ell_i^a+
\lambda_v\frac{\sum_{i\in B_b}m_i^vw_i^v\ell_i^v}
{\max(1,\sum_{i\in B_b}m_i^v)}\right].
$$

所以有限batch中的分母随机性和跨卡来源组成也影响实际更新。不能把整个算法写成固定全局系数$r\ell_D^a+(1-r)\ell_H^a+\lambda(r\ell_D^v+(1-r)\ell_H^v)$而忽略mask。

共享视觉参数同时接收当前动作与视频路径的梯度，但历史编码路径被截断。联合梯度可能协同，也可能冲突；当前没有梯度对齐约束，不保证视频loss越低，策略就越好。

### 4.3 如何解释上下文截断的偏差

接管内裁剪改变的是输入条件分布，即从$(o_t,h_t^{\mathrm{rollout}},a_t^H)$变成$(o_t,h_t^{\mathrm{segment}},a_t^H)$。这不是只有“少了几帧”的存储问题：在相同当前图像下，历史可能提供此前接近方向、物体状态或阶段信息。

一个可选的说明性界限是：对固定参数，若单样本损失对历史表征h为L-Lipschitz，则在同一批起点、当前观测与标签配对下

$$
|\mathbb E\ell(o,h^{\mathrm{segment}},a)
-\mathbb E\ell(o,h^{\mathrm{rollout}},a)|
\le L\mathbb E\|h^{\mathrm{segment}}-h^{\mathrm{rollout}}\|.
$$

它只说明上下文差异可影响风险，不证明跨界历史必然更好；L未估计，有限前缀重算也不等价于持续部署缓存。建议将这一式作为解释或附录，不包装成理论保证。主要正文用条件分布差异与数据审计即可。

此外，$d_\mu(o,h\mid c=H)$和最终$d_\pi(o,h)$仍不同；只在接管状态训练不能直接继承标准DAgger的完整专家标注/无遗憾保证。[4](https://arxiv.org/html/1810.02890v1)

### 4.4 自监督未来到底学到什么

当前视觉监督拟合记录分布中的$p_\mu(V^+\mid o,h,\ell,\text{selection})$。对于成功筛选的延伸部分，selection包括$y_\tau=1$；这个条件描述训练数据选择，不一定作为显式输入传入网络。它不是$p(V^+\mid o,h,\operatorname{do}(a))$，也不是专家始终控制的未来。

因此可说“利用真实交互的视觉后续塑造预测表征”。不可说“人类消除了模型对自主rollout的偏见”“学到了任意动作的因果后果”或“自监督证明了恢复动力学”。是否从这些目标受益，需要同数据video-loss-off和vidext对照回答。

## 5. 最接近的相关工作与区别

| 工作 | 与本项目最相关的已有内容 | 应明确的区别 |
|---|---|---|
| HG-DAgger，ICRA 2019 [4] | 人类决定接管并提供纠正；另有不确定性风险阈值研究 | 当前使用人类门控采集思想，没有复现其风险估计；研究重点在WAM输入/目标构造 |
| IWR [5] | 干预与非干预等比例采样，并对照自然采样和追加完整示范 | 当前非HUMAN动作不自动监督，原demo回放也不等同于IWR的自主动作池 |
| SIRIUS，RSS 2023 [6] | 依据人类信任重加权混合部署数据 | 不新发明“按来源使用数据”；本项目区分同一来源的视觉与动作用途 |
| Sirius-Fleet，CoRL 2024 [7] | 世界模型辅助异常监测与交互部署 | 本项目不是新异常监测器，而是生成式WAM骨干的纠正后训练 |
| τ₀-WM，2026 [8] | 异构视频/动作/rollout数据的分模态监督掩码 | “视频可监督而动作不可监督”已有近邻；本项目的具体问题是短接管边界与历史一致性 |
| WAM-TTT，2026 [9] | 通过视频预测适配记忆，另经元训练使其对动作有用 | 本项目是真实机器人交互和直接专家动作监督，非人类视频测试时记忆适配 |
| Hi-WM，2026 [10] | 在学习的世界模型中接管、回滚、分支生成纠正 | 当前是在实机上采集真实后续，不在模型想象环境中收集纠正 |
| Fast-WAM [2] | 区分训练视频共训与测试时未来生成，并做video-loss对照 | 支持将视觉目标视为训练辅助，但不能重复主张“无需测试时想象”是新发现 |

这些近邻意味着题目和贡献应避开“首次自监督WAM结合人类干预”。区别可以具体、可检验，但仅有区别不等于达到强方法创新；还需展示短接管导致的问题及所提构造的实测作用。

## 6. 最小方法消融与可证伪预测

| 对照 | 固定项 | 若出现改善可以支持什么 | 若无差异如何表述 |
|---|---|---|---|
| 段内历史 vs 跨接管历史 | 同SFT、HIL、视频规则、步数、部署history6 | 上下文构造对该记忆WAM有价值 | 保留为兼容性设计，不能作为性能增益来源 |
| 基本full vs vidext | 人类动作目标与更新范围相同 | 延伸实际视频目标的整体效应 | 现有5/13 vs6/13不够证明明确优势 |
| vidext vs 相同配方video-loss-off | 历史、动作标签、数据与训练预算 | 显式视觉目标是否提供额外收益 | 不能声称自监督是主收益来源 |
| 原SFT vs 整体纠正 | 同部署条件与初始化 | 整套后训练协议的总效果 | 无法拆分数据、额外更新、历史等单项贡献 |

“history6 SFT部署关memory”只测试部署条件敏感性，不等价于Stage 1重新训练无memory。跨WAM普适性则需要第二个模型或至少第二种时序结构，当前不宜在截止前承诺。

## 7. 可直接用于论文的段落

**问题与洞察（英文参考）**

Human intervention boundaries specify when expert action labels are available, but do not necessarily delimit the visual evidence relevant to learning. For a memory-based world–action model, observations preceding a correction provide decision context, while subsequent recorded frames can supply a predictive training target even when the corresponding autonomous actions are excluded from imitation. We therefore construct context and supervision separately, rather than treating each intervention as an isolated demonstration.

**方法（英文参考）**

Our post-training protocol retains within-rollout visual history across takeover boundaries and applies action supervision only to valid human-controlled intervals. An optional visual continuation objective extends video targets into subsequent executed observations of successful mixed-control rollouts, without extending the expert action labels. We combine these correction samples with the original demonstrations. This constitutes supervised action learning with an auxiliary self-supervised visual objective; it does not assume that autonomous actions are expert labels or that recorded futures are counterfactual expert outcomes.

**贡献建议（根据实际完成的结果收缩）**

1. Formulate and implement context-preserving corrective post-training for memory-based WAMs, separating intervention labels from the temporal scope of visual evidence.
2. Analyze how intervention truncation and demonstration replay affect available action and video supervision, with controlled comparisons under a fixed optimization budget.
3. Evaluate task completion and recovery behavior on low-cost real-robot insertion tasks, including matched initialization conditions and, when completed, controlled offset or latency tests.

如果只完成两项机制对照，第二项就只写对应范围。无需让第三项必须是一个新网络或新加速算法。第三贡献采用实证发现是正常论文结构，但未完成的power或延迟结果不能先写成已经支持的结论。

## 8. 当前代码的核对入口

| 文件 | 对应事实 |
|---|---|
| [hil_human_only_dataset.py](../../../scripts/so101_pipeline/hil_human_only_dataset.py) | HUMAN段、跨界历史、未来重映射、监督mask |
| [ahawam.py](../../../src/ahawam/models/wan22/ahawam.py) | 历史stop-gradient、共享当前分支、按卡video-loss归一化 |
| [prepare_hil_v2_full.py](../../../scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/prepare_hil_v2_full.py) | vidext只延伸视频，demo2to1不延伸 |
| [samplers.py](../../../src/ahawam/utils/samplers.py) | 来源采样与跨卡分配 |

工程数据与已测权重见[交接表](../01_WRITER_HANDOFF_ZH.md)。上述数学表达是对实现的抽象；额外的因果模型、风险估计器或表征对齐loss均未实现。

## 文献与来源

1. Cai et al. [AHA-WAM: Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing](https://arxiv.org/html/2606.09811v1), arXiv, 2026，§3.4、§3.5、§4.3。用于记忆结构与已有推理优化边界。
2. [Fast-WAM: Do World Action Models Need Test-time Future Imagination?](https://arxiv.org/html/2603.16666v1), arXiv, 2026，§3。用于视频共训与测试未来生成的区分。
3. [DIM-WAM: World-Action Modeling with Diverse Historical Event Memory](https://arxiv.org/abs/2606.27677), arXiv, 2026。仅据摘要确认多尺度历史机制，不引用未核实实验数字。
4. Kelly et al. [HG-DAgger: Interactive Imitation Learning with Human Experts](https://arxiv.org/html/1810.02890v1), ICRA 2019；预印本2018。用于采集与标注机制。
5. Mandlekar et al. [Human-in-the-Loop Imitation Learning using Remote Teleoperation](https://arxiv.org/html/2012.06733v1), CoRL 2020，§IV–V。用于IWR、等比例和无平衡对照；该文实验为模拟任务，不将其当成本项目真机配置依据。
6. Liu et al. [Robot Learning on the Job](https://ut-austin-rpl.github.io/sirius/), RSS 2023。用于信任加权部署学习。
7. Liu et al. [Multi-Task Interactive Robot Fleet Learning with Visual World Models](https://ut-austin-rpl.github.io/sirius-fleet/), CoRL 2024。用于世界模型辅助监测的近邻关系。
8. Zhou et al. [τ₀-WM: A Unified Video-Action World Model for Robotic Manipulation](https://arxiv.org/html/2606.01027v1), arXiv, 2026，§III–IV。用于分模态监督及动作条件模拟接口的区别。
9. Feng et al. [WAM-TTT: Steering World-Action Models by Watching Human Play at Test Time](https://arxiv.org/html/2607.06988v1), arXiv, 2026，§3、附录E.4。用于视频预测适配与数据预算研究的近邻关系。
10. Li et al. [Hi-WM: Human-in-the-World-Model for Scalable Robot Post-Training](https://hi-wm.github.io/), arXiv:2604.21741, 2026。用于真实交互与世界模型内交互的区别。
