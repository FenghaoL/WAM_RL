# 从 π*0.6 / RECAP 到 SO101 插头插入：训练轮次、HG-DAgger 与下一阶段实验方案

> 检索与本地代码审计日期：2026-08-27  
> 对应本地 AHA-WAM 分支：`ahawam-code`，提交 `6af7e85`  
> 目标：回答 π*0.6 到底怎样训练、每轮从哪个 checkpoint 开始、使用什么数据，并为当前 SO101“抓起插头并插入插线板”任务制定立即可执行、同时适合论文消融的数据闭环。

配套阅读：

- [AHA-WAM 强化学习范围与联合优化综述](./AHAWAM_RL_SCOPE_AND_VIDEO_ACTION_COTRAINING_REVIEW_ZH.md)
- [AHA-WAM 插入与 RL 文献综述](./AHAWAM_INSERTION_RL_LITERATURE_REVIEW_ZH.md)
- [AHA-WAM 架构全解](../architecture/AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md)

---

## 0. 先直接回答当前最重要的决策

### 0.1 可以现在开始 HG-DAgger，但要先满足三个条件

你的当前策略已经能够：

- 在部分初始化下抓到插头；
- 把插头移动到插线板附近；
- 暴露出可重复观察的抓取偏差、粗对准偏差和孔口附近失败。

这已经足够开始 learner-on-policy intervention。HG-DAgger 并不要求初始 SFT 策略接近成功；它的目的恰恰是访问 SFT 自己会进入的错误状态，再由人从这些状态恢复。

但今天开始之前必须做到：

1. **建立固定评测。**至少先用统一初始化分层快速比较 2 epoch 与 4 epoch checkpoint，选择数据采集 policy，并保存未经后续数据污染的 SFT 成功率。
2. **接管必须真正实时。**接管时清空未执行的 AHA action queue；否则人的第一步之后仍可能继续执行旧模型动作。
3. **逐步保存控制来源。**至少记录 `policy_action`、`human_action`、`executed_action`、`intervention_mask`、takeover/release 时刻和 episode outcome。当前工作区尚未发现这些 HG-DAgger 字段或采集实现，不能把普通人工录制直接改名为 HG-DAgger。

### 0.2 不建议先暂停一切，只追加大量“插头已在插座上方”的完整人工 demo

更好的做法是一个混合采集日程：

```text
少量局部插入校准/补覆盖
        +
从完整初始化分布开始的在线 HG-DAgger
        +
保留无人接管的自主成功和自主失败
```

局部插入 episode 有价值，但其作用是：

- 验证你的遥操作在孔口附近足够精确；
- 给终端瓶颈提供最低覆盖；
- 建立“只补 demo 能走多远”的 SFT 消融；
- 诊断长任务失败究竟来自上游抓取，还是插入技能本身。

它不应替代 on-policy intervention，因为从人工摆出的标准上方状态开始，覆盖不到模型真实产生的歪抓、偏航、遮挡、过冲和错误接触状态。

### 0.3 当前建议的第一次增量数据预算

这不是论文定律，而是根据你已有 73 条训练 episode、当前失败模式和单台 SO101 成本给出的第一轮预算：

| 数据 | 建议数量 | 目的 |
|---|---:|---|
| 局部插入校准/演示 | 8–12 episodes | 覆盖孔口附近不同平移和角度误差；建立 SFT-only 增量基线 |
| 完整初始化 HG-DAgger | 30–40 episodes | 访问当前模型真实失败分布；允许一条 episode 多次接管 |
| 其中无需接管的完整自主 rollout | 不设成功配额，全部保留 | 估计当前策略分布，并供 IWR/RECAP value 使用 |
| 正式评测 | 每方法至少 40 个分层 trial，论文最好 50–100 | 给成功率置信区间，不用 5–10 次主观体验代替评测 |

HG 数据最好按“人类控制的时间步/干预段数”计预算，而不是只按 episode 数。第一轮可以要求至少覆盖：

- 8–12 个抓取纠正段；
- 6–10 个移动/粗对准纠正段；
- 20–30 个精对准、接触或插入纠正段；
- 若出现滑脱、卡边、错孔或插线板被推动，分别保留恢复段。

一条 episode 可以同时贡献多个类别，因此这些数字不能直接相加成 episode 数。

### 0.4 不要删原始数据；筛选应通过 manifest 完成

- 灯光略有变化通常是有价值的部署变化，不应因为“不整齐”而删除。
- 成功但有少量真实纠偏的数据可能非常有价值。
- 明显不同步、相机丢帧、控制映射错误、外部人员碰动场景、危险动作或任务标注错误，应从训练视图排除。
- 长时间无目的抖动、遥操作犹豫和重复探孔，不应直接混入“全部都是高质量”的 SFT；保留原文件，并标为 `quality=low`、降低采样权重或用于 negative/低优势条件。

论文更关心你是否透明报告数据来源、人工时间、机器人交互量和筛选规则，而不是是否把 demo 数压到一个漂亮的小数字。

---

## 1. 先把名字说准确：π0.6、π*0.6 与 RECAP

这三个词不是同一个东西：

| 名称 | 含义 |
|---|---|
| π0.6 | Physical Intelligence 的基础 VLA；约 5B VLM backbone 加 action expert，支持更加异构的条件输入 |
| RECAP | `RL with Experience and Corrections via Advantage-conditioned Policies`，一套从 demo、reward、autonomous rollout 和 intervention 学习的算法 |
| π*0.6 | 在 π0.6 上加入 advantage conditioning，并通过 RECAP 训练得到的 RL 版本 |

因此“走 π0.6 方案”在当前项目中真正对应的是：

> 借鉴 RECAP 的数据聚合、value/advantage 标签和条件策略训练，而不是把 AHA-WAM 换成 π0.6，也不是直接复制一套已经开源的 π0.6 训练命令。

[π*0.6 论文](https://www.pi.website/download/pistar06.pdf)给出了算法和实验，但截至本次检查，Physical Intelligence 官方 [openpi 的 RECAP 支持请求](https://github.com/Physical-Intelligence/openpi/issues/857)仍指出官方训练代码缺少 value training、advantage computation、advantage-conditioned policy、在线循环和 human correction integration。网上存在若干社区实现，但不能把它们的超参数当成论文官方配置。

---

## 2. π*0.6 的完整训练骨架

### 2.1 第一层：通用 RL 预训练

RECAP 先在多任务、多机器人示范集合 $D_{demo}$ 上训练：

```text
多任务 demonstration data
        │
        ├──> 训练通用 value function V_pre
        │
        └──> value 估计 advantage
                         │
                         ▼
               训练 advantage-conditioned π_pre
```

得到的 $\pi_{\mathrm{pre}}$ 不是某个插入任务的 SFT 模型，而是已经学会接收 positive/negative improvement indicator 的通用 π*0.6 预训练模型。

这点无法在当前 AHA 项目中一比一复制：你的 AHA-WAM 公开预训练 checkpoint 没有 RECAP value model，也没有 advantage condition 输入。你能迁移的是下游循环，而不是假定当前 AHA checkpoint 已经完成同样的 RL 预训练。

### 2.2 第二层：每个下游任务单独建立 iteration-0 SFT

对任务 $\ell$，先用该任务的 demonstrations 初始化累计数据集：

$$
D_\ell \leftarrow D_\ell^{demo}.
$$

然后：

1. value function 从 $V_{\mathrm{pre}}$ 开始，在任务数据上微调得到 $V_\ell^0$；
2. policy 从 $\pi_{\mathrm{pre}}$ 开始，在任务 demo 上微调得到 $\pi_\ell^0$；
3. 这一阶段所有 demonstration 的 improvement indicator 固定为 `True`，因此本质上是任务 SFT。

$\pi_\ell^0$ 才是第一轮真实机器人数据采集 policy。

### 2.3 第三层：批量采集—离线更新的循环

第 $k$ 轮：

```text
部署上一轮策略 π^(k-1)
        │
        ├── autonomous success
        ├── autonomous failure
        └── expert intervention episode
                        │
                        ▼
追加到累计任务数据 D_ℓ
                        │
          ┌─────────────┴─────────────┐
          ▼                           ▼
从 V_pre 重新微调 V^k       从 π_pre 重新微调 π^k
          │                           ▲
          └── advantage / I_t ────────┘
```

它是 iterated batch/offline update，而不是机器人执行一步、模型立刻做一次梯度更新。论文也明确把“完全并发的 online RL”留作未来工作。

---

## 3. “每一轮重新训练”到底是什么意思

### 3.1 正确答案

论文的 Algorithm 1 和实现说明是：

- 第 $k$ 轮数据由上一轮策略 $\pi_\ell^{k-1}$ 采集；
- 数据追加到累计数据集 $D_\ell$，旧 demo、旧 rollout 和新 rollout 都保留；
- 训练第 $k$ 轮 policy 时，参数重新从通用 $\pi_{\mathrm{pre}}$ 初始化；
- value 同样从通用 $V_{\mathrm{pre}}$ 初始化；
- **不是**从上一轮 $\pi_\ell^{k-1}$ 接着训练；
- **也不是**从随机参数完全从头训练。

作者给出的理由是减少多轮连续微调导致的 drift。他们同时承认，从上一轮继续训练也可能有效，但不是论文采用的默认做法。

### 3.2 它也不是“每轮只训练新数据”

虽然参数回到同一个通用 checkpoint，数据却是累计的：

$$
D_\ell^k
=D_\ell^{demo}\cup D_\ell^1\cup\cdots\cup D_\ell^k.
$$

所以更准确的表述是：

> 每轮从共同通用初始化 checkpoint，对不断增大的累计任务数据重新做一次 task fine-tuning。

### 3.3 对当前 AHA-WAM 应怎样做

应区分“快速工程迭代”和“正式实验”：

| 用途 | 初始化建议 | 原因 |
|---|---|---|
| 今天的快速 smoke test | 可以从当前 2-epoch SFT 继续训少量步数 | 尽快判断新 intervention 数据有没有信号 |
| 正式 SFT/HG/IWR 消融 | 全部从同一公开 AHA-WAM pretrained checkpoint 开始 | 公平比较，避免不同历史 checkpoint 混入结论 |
| 第 2 轮及以后 RECAP-like | 每轮从固定 AHA base 或固定 advantage-capable base 开始，在累计数据上训练 | 最接近 RECAP 的抗漂移设计 |

若你先在 AHA 中增加 advantage-conditioning adapter，那么应保存一个“只有结构改造、尚未做目标任务训练”的固定 base，之后每轮从这个 base 重训。

---

## 4. positive / negative 不是人工给每条动作打“好/坏”标签

### 4.1 reward 先来自 episode 结果与速度

RECAP 对每条 episode 使用成功/失败标签，并引入每步时间代价：

$$
r_t=
\begin{cases}
0,&t=T\ \text{且成功},\\
-C_{fail},&t=T\ \text{且失败},\\
-1,&\text{其他时间步}.
\end{cases}
$$

因此成功越快，return 越高；失败获得很低的终局值。value function 学到的不只是“最终会不会成功”，还近似表示距离成功还需要多少步。

### 4.2 value 是 201-bin distributional classifier

经验 return 被离散成 201 个 bins，value model 输出

$$
p_\phi(V=b\mid o_t,\ell).
$$

用交叉熵训练，再通过各 bin 对应值的期望得到连续 value：

$$
V^{\pi_{ref}}(o_t,\ell)
=\sum_b p_\phi(V=b\mid o_t,\ell)v(b).
$$

这不是一个简单的 episode 成功分类器。它要对同一 episode 中“还很远”“接近完成”“卡住退步”等状态给出不同进展估计。

### 4.3 advantage 再转成二值 indicator

下游 post-training 使用约 50 步 lookahead 的 value bootstrap 估计 advantage：

$$
A_t=
\sum_{j=t}^{t+N-1}r_j
+V(o_{t+N})-V(o_t),\qquad N=50.
$$

然后按任务阈值 $\epsilon_\ell$ 二值化：

$$
I_t=\mathbf 1[A_t>\epsilon_\ell].
$$

论文在 fine-tuning 中通常调阈值，使一轮 evaluation rollout 约 40% 的数据为 positive；某些已经很成功但速度慢的 laundry 任务只保留约 10% positive，以更强地偏向快速行为。这不是固定的 `advantage > 0` 通用规则。

### 4.4 人工 intervention 有一个特殊规则

人工实际控制期间的 correction actions 被强制设置为 $I_t=True$。其假设是人类纠正总体上提供了有用的恢复动作。

论文也明确批评了这个假设的局限：接管是破坏性的切换，人的纠正不一定平滑、快速或全局最优。因此 intervention 主要解决大错误和探索瓶颈，速度与细微质量仍依赖 reward/value 从自主经验中优化。

### 4.5 negative 数据不是被反向惩罚

RECAP 的关键不是在 negative action 上乘一个负 loss，把网络“推离”该动作。它让同一个 policy 学习两个条件分布：

$$
\pi_\theta(a\mid o, I=1),
\qquad
\pi_\theta(a\mid o, I=0).
$$

训练时模型在 negative 条件下仍然拟合 negative 样本；部署时只请求 $I=True$ 的分布。换言之，negative 数据告诉模型“这种行为属于低优势条件”，而不是让普通 BC loss 把坏动作当好动作，也不是做简单的动作反分类。

论文还随机丢弃 advantage condition 30% 的时间，学习 unconditional 分支，并可在推理时做类似 classifier-free guidance 的条件强化。

### 4.6 正确的数据顺序

所以真实顺序不是：

```text
先 HG-DAgger -> 人给 episode 打 positive/negative -> SFT
```

而是：

```text
demo SFT policy
    -> autonomous + intervention rollout
    -> episode success/failure 和时间代价
    -> 训练 value
    -> 计算每个时间步 advantage
    -> 自动生成 positive/negative indicator
    -> correction 时间步强制 positive
    -> advantage-conditioned flow/action training
```

---

## 5. π*0.6 做了几轮、每轮多少数据

### 5.1 论文没有一个所有任务共用的固定轮数

论文说可以按需要做多轮，而且一次迭代经常已经带来显著提升。公开实验包括一轮和两轮，并非统一做三轮或固定 epoch 数。

| 任务/实验 | 迭代 | 公开的每轮经验量 | 是否 intervention |
|---|---:|---:|---|
| T-shirt/shorts laundry | 2 | 每轮 300 autonomous trajectories，4 台机器人 | 否 |
| Box assembly | 2 | 正文写每轮 600 autonomous + 360 intervention trials | 是 |
| Cafe/espresso | 1 | 附录写 414 autonomous + 429 correction episodes | 是 |
| Diverse laundry | 文中未清楚统一为几轮 | 450 evaluation + 287 correction episodes | 是 |
| Targeted laundry failure removal | 2 | 正文写每轮 600 trajectories | 正文称无 intervention |

需要保留一个批判性备注：π*0.6 论文正文与附录的数据措辞并不完全一致。

- Box 正文写 600 autonomous trials，附录写 600 demonstrations，再加 360 correction episodes。
- Targeted failure removal 正文称没有 intervention data，但附录又写约 1000 autonomous 和 `280 + 378` correction episodes。

因此可以可靠引用“数量级为每轮数百条、任务间预算不同”，但若要复现某个精确 mixture，应向作者确认，而不能把矛盾行自行拼成一个确定答案。

### 5.2 目标任务初始 SFT demonstration 数没有完整公开

论文说明不同任务初始 demo 数不同，却没有为所有目标任务提供一张清晰的初始 SFT demonstration 数量表，也没有公开每轮 policy/value 的完整 epoch、batch 和 optimizer schedule。

所以不能从论文推出“π0.6 就是 20 或 30 条 demo”。常见的 20–30 条来自其他真机工作：

- [IWR](https://arxiv.org/abs/2012.06733)：30 条初始 demonstration，之后做 3 轮 intervention collection；
- [HIL-SERL](https://arxiv.org/abs/2410.21845)：大多数任务 20 条初始 offline demonstrations；
- [RLDG](https://arxiv.org/abs/2412.09858)：connector/insertion specialist 也按 HIL-SERL recipe 用 20 条 demo 初始化 RL buffer。

这些是不同算法、不同机器人和不同控制空间，不能用来判定你的 73 条“过多”。

### 5.3 是否每个任务都额外训练

是。π*0.6 先有共享的通用 $\pi_{\mathrm{pre}},V_{\mathrm{pre}}$，然后每个目标 task/skill $\ell$ 有：

- 自己的 demonstration 集；
- 自己的 iteration-0 SFT policy；
- 自己的 rollout/intervention 数据；
- 自己的 advantage threshold；
- 自己的 specialist fine-tuning 循环。

这和“一个最终 generalist 自动不经训练掌握所有新任务”不同。π*0.6 的实验结果是 task-specialized models；跨任务共享主要来自通用预训练。

---

## 6. HG-DAgger 原文究竟怎样接管

### 6.1 不是先 rollout 一次，再第二次提前一秒复现

[HG-DAgger 原文](https://arxiv.org/abs/1810.02890)采用实时 human gate：

1. 每条 rollout 从 novice/model 控制开始；
2. 人持续观察；
3. 当人判断模型进入不允许继续自主运行的状态时，立即拿到完整控制权；
4. 人连续控制一段时间，把系统带回安全、稳定、模型可能继续完成任务的区域；
5. 人主动把控制权交还模型；
6. 原始 HG-DAgger 只把这段 human recovery trajectory 加入监督数据；
7. 下一轮在累计 BC + intervention 数据上重新训练 novice。

第二次复现同一 rollout 通常既不必要也不可靠：初始物体姿态、抓取微差、相机延迟和生成式 policy 随机性都会让“出错前一秒”难以重现。

### 6.2 是否接管确实依赖人的判断

在原始 HG-DAgger 中，是的。人就是 gating function。它甚至把允许模型运行的状态集合称为 permitted set：

$$
g(x_t)=\mathbf 1[x_t\notin\mathcal P].
$$

这不等于完全随意。实验前应给操作者一个书面的 intervention policy，减少同一个人今天和明天标准漂移。

### 6.3 对插头任务的接管标准

建议在出现以下任一条件时接管：

**抓取阶段**

- 夹爪中心明显偏离插头可抓区域；
- 当前闭合会造成空抓、斜抓或夹住电线；
- 插头已滑动到训练分布外；
- 当前抓姿即使拿起，也几乎不可能完成后续插入。

**移动与粗对准**

- 去向错误插孔、插线板错误区域或明显错误高度；
- wrist view 长时间完全遮挡目标且策略仍继续接近；
- 轨迹即将碰撞插线板、桌面或拉扯电线；
- 已经大幅越过可恢复的 approach corridor。

**精对准与接触**

- 插头横向或角度误差明显超过孔口的捕获范围；
- 接触后连续若干控制步没有插入深度进展；
- 出现重复左右探孔、卡边、推走插线板或插头从夹爪内转动；
- 继续执行当前 action chunk 很可能放大错误。

不要因为动作“不像你”就接管。只在安全、成功概率、恢复性或明显效率问题上接管，否则数据会把模型压回操作者的个人风格，而不是修复任务失败。

### 6.4 接管后控制到哪里再交还

原始 HG-DAgger 强调连续恢复，不是只给一个瞬时动作。对当前任务：

- 若抓取失败，至少控制到稳定重新抓住，必要时继续移动到模型熟悉的 transport state；
- 若粗对准错误，控制到相机能清楚看到孔口、插头位于可插入走廊；
- 第一轮数据中，孔口附近建议人直接完成精对准和插入，不要每次都在“刚刚对准”时交还给一个尚不会插入的策略；
- 随着新策略获得插入能力，再逐渐更早交还，让模型承担更多终端动作。

这形成自然课程，但课程来自模型能力变化，不需要硬编码一个复杂 phase controller。

---

## 7. HG-DAgger、IWR 与 RECAP 对同一条人机轨迹怎么用

设一条 episode 的控制来源为：

```text
robot robot robot | human human human | robot robot | human human | robot
```

三种方法的训练视图不同：

| 方法 | robot 段 | human 段 | outcome/失败 | 训练目标 |
|---|---|---|---|---|
| 原始 HG-DAgger | 丢弃，不当监督样本 | 加入 expert dataset | 通常不用 reward | 在 intervention states 上 BC/flow matching |
| IWR | 保留为 robot dataset | 保留为 intervention dataset | 不必有显式 reward | 两类样本 50/50 平衡采样做 BC |
| RECAP | 整条 episode 均可保留 | 保留且强制 positive | 成败与时间用于 value | 全数据 advantage-conditioned policy training |

[IWR](https://arxiv.org/html/2012.06733v1)专门指出：只在 intervention 段训练可能使下一轮 policy 大幅漂移并产生新的错误分布。因此它把 robot samples 和 intervention samples 分开保存，每个 batch 各采一半：

$$
\mathcal L_{IWR}
=\mathbb E_{D_I}[\ell(\pi(o),a_H)]
+\mathbb E_{D_R}[\ell(\pi(o),a_R)].
$$

其中 $a_R$ 是机器人当时实际执行的 policy action；它不是被当作绝对正确，而是起到靠近上一轮行为分布的 regularization 作用。

对 AHA-WAM，$\ell$ 应换成 action flow-matching loss，而不是普通单步 MSE，但数据采样思想不变。

### 7.1 接管前后哪些数据都要保存

建议原始 recorder 全部保存：

- 接管前至少 0.5–1.0 秒的 observation、policy proposal 和 executed action；
- 接管期间 human action；
- 交还后的模型行为；
- episode 最终成功、失败或 timeout。

但“保存”不等于“全部当正向 SFT target”：

- 接管前 robot action 可用于 value、negative condition、IWR regularizer 和 failure analysis；
- 不能把它伪标为 expert action；
- intervention 段可作为 HG/IWR expert target；
- 交还后的成功自主段可作为高价值 autonomous experience；
- 若交还后再次失败，应保留第二次 intervention，而不是裁掉让 episode 看起来完美。

---

## 8. 权威工作与代码到底开放到什么程度

### 8.1 π*0.6 / RECAP

开放内容：

- 论文算法、value/reward 公式、advantage threshold、若干数据规模；
- 项目页面和长时间运行视频；
- π 系列基础 openpi 代码。

未完整开放：

- 论文内部的 RECAP value + advantage + on-robot collection 完整实现；
- 实际 intervention UI、低层控制和数据 schema；
- 每个目标任务的完整初始 demo 数、policy/value train schedule；
- 可直接复现论文结果的 checkpoint/data mixture。

因此不能声称“按 openpi 官方代码复现 π0.6 HG-DAgger”。

### 8.2 原始 HG-DAgger

原文清楚说明了实时接管、连续恢复、只聚合 intervention labels 和迭代训练，但实验是自动驾驶，不是 6-DoF 机械臂。它提供的是算法定义，不是适配 SO101/AHA 的现成采集栈。

### 8.3 IWR

IWR 的论文和项目页公开了：

- 浏览器视频监控；
- 手机 6-DoF 相对位姿控制；
- intervention button；
- robot/intervention 分开保存和等比例采样；
- threading 和 coffee-pod insertion 实验。

这是解释“插入瓶颈应该如何收人机数据”的经典直接证据，但实验在 MuJoCo 中完成，不能把其成功率当真机 SO101 数字。

### 8.4 HIL-SERL 与 LeRobot 的两条 HIL 路径

[HIL-SERL 论文](https://arxiv.org/abs/2410.21845)及其[代码](https://github.com/rail-berkeley/hil-serl)公开了真机 actor/learner、demonstration buffer、intervention 和 reward classifier 路线，但原版 robot infra 针对 Franka、SpaceMouse 和末端增量控制，不能作为 SO101/AHA-WAM 的直接 drop-in。

这里还要更正一个容易由旧文档造成的误解：旧 LeRobot `rl.gym_manipulator` / HIL-SERL 路径虽然曾把 SO101 leader 写成支持设备，但截至 2026-08-27，[issue #2952](https://github.com/huggingface/lerobot/issues/2952)仍显示 `SOLeader` 缺少该路径要求的 `get_teleop_events()`；相应的[修复 PR #3086](https://github.com/huggingface/lerobot/pull/3086)仍未合并。因此不能直接按旧 HIL-SERL 文档启动。

真正值得采用的是 LeRobot v0.6 新增的 [`lerobot-rollout --strategy.type=dagger`](https://github.com/huggingface/lerobot/blob/v0.6.0/docs/source/hil_data_collection.mdx)。这是一条独立的新实现，明确支持 `so_leader`，提供：

- `AUTONOMOUS → PAUSED → CORRECTING` 状态机；
- keyboard/foot-pedal 控制；
- leader 与 follower 的平滑姿态同步；
- leader torque enable/disable 与 `send_feedback()`；
- policy pause/reset/resume；
- 每帧 intervention 标记。

它的 policy engine 和默认 recorder 仍不是为 AHA-WAM 设计：AHA 一次返回 8-step chunk，接管时需要清空本地 queue、使在途 TCP reply 失效，交还前还要 hard reset 服务端历史。LeRobot v0.6 的 correction-only 与按文件大小自动轮转的 continuous mode 也都不能原样替代当前插头任务的完整逻辑 episode/outcome。具体复用边界、schema 和部署端实现见[SO101 + AHA-WAM 在线接管数据采集交接](../hil/SO101_AHAWAM_HIL_DATA_COLLECTION_IMPLEMENTATION_HANDOFF_ZH.md)。

### 8.5 RLDG

[RLDG](https://generalist-distillation.github.io/static/high_performance_generalist.pdf)对当前“是否单独收最后插入段”尤其有价值：

- connector insertion 从插头已经被抓住、位于插孔上方 5 cm、在 $10\text{ cm}\times10\text{ cm}$ 平面内随机的位置开始；
- FMB insertion 同样从预抓取状态开始，公差约 $\pm1.5$ mm；
- 对完整 FMB assembly，抓取/运输用 human demos，只有 insertion bottleneck 用 RL 数据；
- 最终把两个阶段的数据合并微调 generalist。

这证明“局部技能 episode + 完整任务 episode”是文献中合理的分解，不会自动让论文故事变差。真正需要避免的是只在一个完全固定姿态上训练，却宣称解决了完整可变初始化插入。

---

## 9. 相关插入工作中的 demonstration 与纯模仿成功率

### 9.1 不是所有纯 demo baseline 都是 0

[IWR](https://arxiv.org/html/2012.06733v1)从每个 operator 的 30 条初始 demo 开始。在模拟的 Threading/Coffee Machine：

| 方法 | Threading | Coffee Machine |
|---|---:|---:|
| 30-demo base | 58.0% | 52.0% |
| 额外等量 Full Demos | 76.7% | 64.9% |
| 最终 HG-DAgger | 75.3% | 69.6% |
| 最终 IWR | 87.3% | 87.5% |

这说明继续增加人工 demo 可能明显提升，不应预设 SFT 上限就是当前表现；也说明 on-policy intervention 与正确的数据加权可以进一步提高。

### 9.2 也存在 BC 完全失败的精密/长程设置

[ThriftyDAgger](https://arxiv.org/abs/2109.08273)的真实 cable-routing 实验用 25 条初始 demo，文中报告 BC 从未完整成功，而 HG-DAgger/ThriftyDAgger 能达到较高自主成功。这不是插头任务，但说明 episode 数相近时，状态分布与瓶颈覆盖可以使结果从 0 到可用相差很大。

### 9.3 真机 USB 插入：20 条 demo 只是 RL 启动数据，不代表 20 条足以 SFT

HIL-SERL 的 USB grasp-insertion 使用：

- 20 条 initial offline demonstrations；
- 2 cm 平移、10° 旋转随机化；
- 50,000 RL transitions；
- camera + TCP pose/velocity/force-torque/gripper observations；
- 10 Hz task-space twist 和底层控制；
- binary reward classifier；
- 在线人类 interventions。

论文的模仿式 HG-DAgger 对照在 USB 任务约 26%，HIL-SERL 达到 100%。注意这个 26% 对照使用了与 RL 匹配的人类示范/干预数据，并非“只用 20 条 demo 的纯 BC”。因此它只能说明模仿式利用同样的人类数据仍可能远弱于 RL，不能作为 20-demo BC 的精确成功率。

### 9.4 RL 生成数据为什么在插入中常胜过人工数据

RLDG 的 connector/FMB 实验显示：

- OpenVLA 在 seen VGA 上用 45 条 RL rollout 达到 100%，human data 约需 300 条；
- unseen Type-C 上，150 条多 connector RL rollout 可达 100%，human demos 增至 900 条仍约 90%；
- 分析认为 RL actions 更集中在真正促进插入的方向，并且 RL rollout 产生不同的状态覆盖。

这不意味着应该删除你的人工 demo，而是解释了后续 RL/advantage optimization 的科学价值：它可能优化人工遥操作的犹豫、绕行、探孔和不稳定动作分布。

---

## 10. 当前 SO101 数据与训练状态的本地审计

### 10.1 磁盘中的实际训练数据

当前 [`settings_plug_all.yaml`](../../scripts/so101_pipeline/full_finetune_zero2_offload/settings_plug_all.yaml)列出的训练集为：

| 类型 | episodes | frames | 每条长度范围 | 平均长度 |
|---|---:|---:|---:|---:|
| `02_grasp_move_and_align` | 32 | 9,351 | 151–368 | 292.2 frames |
| `04_full_remove_and_insert` | 41 | 14,596 | 260–473 | 356.0 frames |
| 合计 | **73** | **23,947** | — | — |

另外有 4 条 held-out full-task recordings，共 1,150 frames。

所以你口述的“32 + 40”与当前 manifest 差 1 条。正式报告应从不可变的 dataset manifest 自动生成，不要手填。

### 10.2 当前 checkpoint 的真实来源

训练日志记录：

```text
train_episodes=73
train_frames=23947
steps_per_epoch=1996
epochs=4
total_steps=7984
init=checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt
learning_rate=1e-5
```

checkpoint 对应：

| checkpoint | epoch |
|---|---:|
| `step_001996.pt` | 1 |
| `step_003992.pt` | 2 |
| `step_005988.pt` | 3 |
| `step_007984.pt` | 4 |

当前部署 yaml 指向 `step_003992.pt`，也就是你实际测试的 2-epoch checkpoint。4 epoch 训练已经完成且 checkpoint 存在，只是尚未被你充分真机评测。

### 10.3 这不是 π*0.6 的 iteration

当前 4 个 epoch 是同一次 SFT/full-WAM fine-tuning 对同一数据的重复遍历，不是四轮：

```text
rollout -> 新数据 -> value -> advantage -> 重训 policy
```

所以“训练了 4 epoch”不能称为“做了 4 轮 RECAP”。RECAP 的 round 以真实机器人数据集是否更新来划分，不以训练 epoch 划分。

### 10.4 开始新数据前要消除一个部署混杂因素

训练配置使用：

```yaml
num_history_frames: 6
```

而当前 [`online_deploy/settings.yaml`](../../scripts/so101_pipeline/online_deploy/settings.yaml)写的是：

```yaml
num_history_frames: 0 #6
```

这会造成训练—部署 history 条件不一致。它未必是当前插入失败的主要原因，但在没有 A/B 测试前，不能把全部偏差都归因于 demonstration 或 RL 缺失。

建议在固定的快速调参初始化集上比较：

```text
2ep, history=0
2ep, history=6
4ep, history=0
4ep, history=6
```

若 history=6 的实时性能或延迟不可接受，应把它作为明确的 deployment design choice 报告，并在下一次训练时让训练条件匹配部署，而不是静默保留 mismatch。

---

## 11. 怎样解释你现在的失败模式

### 11.1 抓取不准

这说明策略访问到了 demo 中覆盖不足的初始条件或视觉/动作误差。它是最典型的 HG-DAgger 适用场景：模型先造成偏差，人从模型自己的偏差状态重新定位并完成抓取。

如果只追加从标准初始位置开始的完美完整 demo，新增数据仍主要处在人类状态分布，未必覆盖模型抓偏之后看到的腕部视角。

### 11.2 能到插线板附近，但存在大偏差

可能同时包含：

- 目标在合成图中的像素尺度不足；
- 训练数据中的 approach corridor 过窄；
- 抓姿变化导致“同一末端关节动作”对应不同插头姿态；
- `chunk=8`、15 Hz 下约 0.53 秒的执行块对近接触过长；
- 关节空间遥操作轨迹本身在孔口附近不够一致；
- history mismatch 或真实控制延迟。

不要把它简单解释为“还差几条 demo”或“必须上 RL”。先通过分阶段成功率和 intervention 位置分布确定主要瓶颈。

### 11.3 完全插不进孔

如果 73 条数据里的“全流程成功”在孔口附近动作质量差异大，普通 SFT 会学习一个多模态、平均或反复探孔的生成分布。继续加高质量局部 demo 可能提升，但 RL/advantage 的价值在于用结果与速度区分：

- 真正推进插入的微动作；
- 无进展的横向抖动；
- 卡边后有效退出重对准；
- 虽最终成功但耗时很长的探孔轨迹。

---

## 12. 今天具体怎样采：建议协议

### 12.1 Step 0：冻结当前 baseline

保存：

- AHA base checkpoint hash；
- 1/2/3/4 epoch checkpoint；
- resolved train/deploy config；
- dataset manifest；
- normalization stats；
- 相机标定、裁剪和安装照片；
- 控制频率与真实 action latency。

### 12.2 Step 1：快速选择 collector checkpoint

使用 12–20 个不进入最终 test 的调参 trial，均匀覆盖：

- 插头平移左/中/右；
- 插头朝向小/中/大偏差；
- 插线板位置的训练内小扰动；
- 至少两个光照条件。

比较 2ep/4ep，若可行再比较 history 0/6。选择成功率更高、危险动作更少的版本作为 collector。不要仅因为 4ep loss 更低就默认 4ep 更好。

### 12.3 Step 2：8–12 条局部插入校准数据

建议：

- 插头可以预先抓住，或者从稳定抓取状态开始；
- 起点在孔口上方约 2–5 cm；
- 系统改变横向、纵向和 yaw 误差，不要每次完全相同；
- 包含从小误差直接插入，以及从较大误差退出—重对准—插入；
- 尽量完成成功，但保留原始重录和异常 manifest；
- 记录实际初始 bin，不要只在文件夹名里写“局部插入”。

这些数据既是 teleop/recorder 校准，也是 `SFT + local insertion demos` 的独立 ablation 数据，不能和 HG intervention 文件无法区分地混在一起。

### 12.4 Step 3：30–40 条完整初始化 HG-DAgger

每条：

1. 按预先抽样的 initialization bin 摆放插头和插线板；
2. model 开始执行；
3. 人按书面标准实时接管；
4. 接管瞬间清空旧 action chunk/queue；
5. 人连续恢复或穿越当前 bottleneck；
6. 交还前刷新最新 observation；
7. 必要时同一 episode 再次接管；
8. 最后人工确认 `success/failure/timeout` 和 failure mode；
9. 不因为 episode 失败就删除；不因为没有接管就删除。

### 12.5 最低 recorder schema

每个时间步至少：

```text
episode_id
timestamp_monotonic
policy_checkpoint_id
policy_action
human_action                 # 无人控制时可为空
executed_action
control_source               # policy / human / safety
intervention_id
intervention_mask
action_chunk_id
queued_action_index
fixed_image_timestamp
wrist_image_timestamp
joint_command
joint_feedback
gripper/current if available
phase                        # 可离线标注
```

每个 episode 至少：

```text
task
initialization_bin
success / failure / timeout
failure_mode
num_interventions
human_control_steps
autonomous_steps
completion_time
unsafe_event
operator_id
lighting_bin
notes
```

对于 AHA-WAM，额外保存 planner refresh 时间、cache age 和 action inference latency 会很有价值，因为粗对准失败可能与异步 planner/action 时延有关。

### 12.6 接管切换的工程细节

- human takeover 优先级必须高于 model action queue；
- takeover 时不要继续执行已经预测出的剩余 8-step chunk；
- leader 与 follower 姿态必须先同步或平滑接入，避免接管瞬间跳变；
- release 时丢弃 takeover 前的旧 proposal，重新用最新图像请求动作；
- 安全限位、速度限制和急停独立于模型与人工控制；
- 如果切换延迟大到无法在碰撞前接管，先修系统，不应靠操作者“更早猜测”。

---

## 13. 第一轮数据收完以后怎么训练

### 13.1 先建立四个共享数据的 baseline

所有正式模型从同一 AHA base checkpoint 初始化，训练步数和有效采样预算尽可能匹配：

| 编号 | 数据/训练 | 回答的问题 |
|---|---|---|
| S0 | 原始 73 条 SFT | 当前基线 |
| S1 | 原始 + 8–12 局部插入 demo，普通 SFT | 只补 terminal demo 能否解决 |
| H1 | 原始 + HG human segments，普通 flow SFT | 原始 HG-DAgger 是否有效 |
| I1 | 原始 + robot/human 段，IWR 50/50 sampling | 保留 rollout regularization 是否比只学接管段稳定 |

这组结果非常重要。若 S1 已经把成功率从低水平推到很高，说明主要是 terminal coverage 不足；这不会毁掉论文，反而给出清晰的数据诊断。之后 RL 的贡献可以转向更大初始化范围、恢复能力、速度和更少人工干预。

### 13.2 然后做 RECAP-lite / advantage flow

第一版不必立即复刻 670M VLM value。可以：

1. 用 frozen AHA visual/planner feature + proprio 训练较小的 progress/value model；
2. target 使用 success/failure 和 remaining-time return；
3. 在 held-out episode 上检查 value 是否随接近、插入而单调提高，卡边/滑脱时下降；
4. 根据 advantage 给 action windows 标 positive/negative；
5. correction windows 默认 positive，但保留一个“value 校验人工纠正”的消融；
6. 给 ActionDiT/OVCR 加 advantage condition 或使用 clipped advantage-weighted flow matching；
7. VideoDiT 初期冻结或小学习率监督更新，不让稀疏 reward 无约束更新整个 5B 分支。

### 13.3 第二轮数据只针对第一轮残余失败

训练并评测 S1/H1/I1/R1 后，再决定第二轮：

- 若抓取仍是主要失败，增加抓偏后的 on-policy recovery；
- 若近孔横向偏差主要，扩大局部 offset 分布；
- 若接触后卡住，采退出—旋转—重插的 recovery；
- 若成功率已高但很慢，减少随意接管，更多保留自主经验，让时间 reward 优化吞吐量；
- 若新策略产生新的失败类型，第二轮 HG 专门覆盖它们。

这就是 DAgger/RECAP 的意义：下一轮数据分布由上一轮策略的真实残余错误决定，而不是一开始主观列出所有可能场景。

---

## 14. 数据是否太多、是否应该筛到 20–30 条

### 14.1 73 条对 AHA-WAM 并不多

73 条 episode 只有约 23,947 个 15 Hz frames，且来自两种不同长度/阶段的轨迹。对于包含 5B VideoDiT、1B ActionDiT 和大规模 K/V editor 的模型，这是小规模 task fine-tuning，不是“大数据”。预训练降低了适配所需数据，但没有让几十条轨迹自动覆盖所有精密接触状态。

### 14.2 “20 条 demo”通常只是 RL bootstrap

HIL-SERL 的 20 demos 后面还有 18k–108k 真实 RL transitions；USB 是 50k transitions。只引用“20 demos”会严重低估真实交互数据。

π*0.6 下游每轮是数百条经验；RLDG 还展示了 45–900 条 fine-tuning trajectory 的 scaling。因此你的 73 条既不异常，也不会让工作失去研究意义。

### 14.3 论文应报告什么

至少报告：

- initial demonstrations：episode、frame、小时；
- partial/full-task 构成；
- autonomous rollouts：成功/失败数量；
- intervention episodes、segments、steps 和人工控制秒数；
- reset 人工时间；
- 被排除的数据及确定性规则；
- 每轮累计数据量；
- 每种方法实际看到的 sample mixture；
- 真机训练总步数和 wall-clock robot time；
- evaluation trials 与置信区间。

审稿人通常不会因为你用了 73 而不是 20 条就否定方法，但会质疑：

- 只报告“demo 数”却不报告长度；
- 删除失败或抖动轨迹而没有规则；
- 不同 baseline 使用不同人类数据预算；
- 在 5 个 trial 上宣称显著提升；
- 用训练初始化做测试；
- 把人为分阶段的容易设置和完整任务混为同一成功率。

---

## 15. 推荐的论文故事和消融结构

### 15.1 不建议的故事

```text
SFT 不好 -> 收很多 HG 数据 -> RL 好了
```

它缺少可验证的因果问题：是多了数据、数据来自 on-policy、加权方式、reward，还是模型参数更新方式起作用？

### 15.2 更强的故事

```text
AHA-WAM 在低成本机械臂的长程插头任务上
能学习抓取和粗移动，但在 learner-induced errors 与插入瓶颈失效
                     │
                     ▼
统一记录 autonomous / intervention / outcome 数据
                     │
       ┌─────────────┼─────────────┐
       ▼             ▼             ▼
local demo SFT   HG/IWR flow SFT   advantage-conditioned flow
       │             │             │
       └─────────────┴─────────────┘
                     │
                     ▼
比较成功率、恢复、速度、人工负担和更新模块
```

它能回答：

1. terminal demo coverage 是否已经足够？
2. 相同人工动作预算下，完整 demo 还是 on-policy correction 更有效？
3. 只用 intervention 还是保留 robot rollout 更稳定？
4. reward/advantage 是否超过普通数据聚合？
5. ActionDiT/OVCR 更新是否足够，是否需要 VideoDiT 联动？
6. 方法提升的是 nominal success，还是对抓偏、卡边和初始位置变化的恢复？

### 15.3 建议的正式表格

| 方法 | 初始 demo | local demo | autonomous data | intervention | reward/value | 更新参数 |
|---|---:|---:|---:|---:|---:|---|
| AHA-SFT | 73 | 0 | 0 | 0 | 否 | full WAM |
| +Local-SFT | 73 | matched | 0 | 0 | 否 | full/action-side |
| HG-DAgger | 73 | 0 | 丢弃训练 | matched | 否 | action-side |
| IWR | 73 | 0 | 保留 | matched | 否 | action-side |
| RECAP-like | 73 | 0 | 保留 | matched | 是 | ActionDiT/OVCR |
| Joint-WAM | 73 | 0 | 保留 | matched | 是 + video SFT | ActionDiT/OVCR + VideoDiT |

其中 `matched` 最好按人类控制动作数或人工秒数匹配，而不只是 episode 数。

---

## 16. 所有合适方案都必须建立在 HG-DAgger 上吗

不是。

- π*0.6 的简单 laundry 实验只用 autonomous data，也能通过两轮 RL 改善。
- targeted failure removal 正文也主张可以没有 intervention，只用 reward 修复具体失败模式，尽管其附录数据描述存在矛盾。
- HIL-SERL、residual RL、DSRL、πRL 等可以从 demo + autonomous reward 开始。
- RLDG 可先训练 task-specific RL specialist，再把成功轨迹蒸馏回 VLA。
- 纯 SFT + 局部技能数据也是合法且必须建立的 baseline。

但对你当前低成功率、真实机器人、稀疏终局奖励和易碰撞插线板的状态，HG-DAgger 是很实用的第一批 on-policy 数据来源，因为它同时：

- 防止模型反复把硬件带入危险状态；
- 给当前策略无法探索到的恢复路径；
- 精确标出瓶颈发生在哪里；
- 为后续 IWR、value、RECAP、residual RL 共用同一份经验。

所以结论是：

> HG-DAgger 不是所有算法的逻辑前提，但在你当前阶段是最值得优先建设的数据基础设施。基础设施应该支持“无人干预 rollout”同样被完整记录，不能设计成只有按下接管按钮才保存数据。

---

## 17. 最终行动清单

### 今天开始采集前

- [ ] 自动导出当前 73 条 dataset manifest；
- [ ] 快速比较 2ep 与 4ep；
- [ ] 测试 train history=6 / deploy history=0 的影响；
- [ ] 明确接管按钮、交还按钮和急停；
- [ ] takeover 会清空 action queue；
- [ ] recorder 保存三种 action 和 intervention mask；
- [ ] 写好 initialization bins 与 intervention 标准；
- [ ] 保留固定、不参与调参的最终 test 分布。

### 第一轮采集

- [ ] 8–12 条局部插入校准/演示；
- [ ] 30–40 条完整分布 HG rollout；
- [ ] 自主成功、失败、无接管 episode 全保留；
- [ ] 每种主要失败至少有若干恢复段；
- [ ] 不删除原始数据，只生成训练 manifest；
- [ ] 记录人类控制时间，而不只记录 episode 数。

### 第一轮训练

- [ ] S0 原始 SFT；
- [ ] S1 local-demo SFT；
- [ ] H1 intervention-only HG；
- [ ] I1 robot/intervention balanced IWR；
- [ ] 所有正式模型从共同 base 初始化；
- [ ] 快速 continued-SFT 只作为工程 smoke test，不混入正式消融。

### 评测与是否进入 RL

- [ ] 分阶段报告 grasp / approach / align / insert / final success；
- [ ] 统计 intervention-free autonomous success；
- [ ] 报告 completion time、碰撞、重复探孔和 action jerk；
- [ ] 至少 40 个分层 trial，最好 50–100；
- [ ] 若 IWR/HG 已高成功，RL 主攻速度、恢复和扩大初始化；
- [ ] 若插入仍低，训练 progress/value，再做 ActionDiT advantage flow；
- [ ] 只有表征诊断表明视觉是瓶颈，才加入 VideoDiT 更新。

---

## 18. 核心资料

1. [π*0.6: a VLA That Learns From Experience](https://www.pi.website/download/pistar06.pdf)：RECAP 算法、checkpoint 重启方式、value/advantage、迭代数据量与实验。
2. [Physical Intelligence π*0.6 项目页](https://www.pi.website/blog/pistar06)：方法阶段和长时间真实部署结果。
3. [HG-DAgger](https://arxiv.org/abs/1810.02890)：human gate、连续接管恢复、intervention-only aggregation 的原始定义。
4. [IWR / Human-in-the-Loop Imitation Learning](https://arxiv.org/html/2012.06733v1)：插入瓶颈、robot/intervention 平衡采样和三轮训练。
5. [ThriftyDAgger](https://arxiv.org/abs/2109.08273)：novelty/risk gating、人类负担和 BC 失败案例。
6. [HIL-SERL](https://arxiv.org/abs/2410.21845)：20-demo bootstrap、真机 USB/RAM/assembly、RL transitions 和 HG 模仿对照。
7. [HIL-SERL 官方项目与代码](https://hil-serl.github.io/)：actor/learner、reward classifier、intervention 流程。
8. [LeRobot v0.6 HIL data collection](https://github.com/huggingface/lerobot/blob/v0.6.0/docs/source/hil_data_collection.mdx)：新 `lerobot-rollout` DAgger 路径、SO101 leader 平滑交接和 intervention recorder；不要与旧 `rl.gym_manipulator` 路径混淆。
9. [RLDG](https://generalist-distillation.github.io/static/high_performance_generalist.pdf)：局部 insertion specialist、完整 assembly 分阶段数据和 RL-to-VLA distillation。

---

## 19. 一句话收束

你不需要把 demonstration-only 路线推到接近 100% 才能开始 HG-DAgger；当前策略已经能产生有价值的 learner-induced failures。最稳且最有论文价值的下一步，是在保留 73-demo SFT 基线和少量局部插入 SFT 消融的同时，立即建设可区分 robot/human/executed action 的在线接管数据闭环，再用同一批数据依次比较 HG-DAgger、IWR 和 RECAP-like advantage flow。这样既能尽快推进，也不会牺牲消融的可信度。
