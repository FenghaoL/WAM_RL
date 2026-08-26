# AHA-WAM 强化学习范围重审：隐藏的 VLA-RL、ActionDiT、视频分支与联合优化

> 检索截止：2026-08-24  
> 目标：重新检查“只搜 WAM/VLA + RL”造成的文献遗漏，并回答 AHA-WAM 在 SO101 插入任务上应当更新外挂 actor、ActionDiT、VideoDiT，还是联合更新。  
> 配套阅读：[插入 RL 文献全景](./AHAWAM_INSERTION_RL_LITERATURE_REVIEW_ZH.md)、[HALO-WA 迁移指南](./HALO_WA_RL_AND_AHAWAM_MIGRATION_GUIDE_ZH.md)。

---

## 0. 结论先行

### 0.1 上一次检索确实存在关键词偏置

如果只检索标题中的 `VLA + RL`、`WAM + RL` 或 `Diffusion Policy + RL`，会漏掉至少四类重要工作：

1. 标题强调“从经验学习”“自我改进”“交互式后训练”，方法内部实际训练 value/advantage 并进行 RL policy improvement；
2. 基础模型本身不用 RL loss，但训练数据来自 RL 专家、自主 rollout、失败和人工纠正；
3. 论文不更新整套 VLA，而是在 action expert、diffusion noise、flow velocity 或 VLA latent 上做 RL；
4. 论文把 RL 写成 advantage-conditioned/weighted flow matching，看起来仍是监督 loss，但数据权重和策略改进来自 reward/value。

最重要的漏项是 Physical Intelligence 的 [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759)。它不是标准 PPO，而是训练分布式 value function、估计每步 advantage、把动作标成 `Advantage: positive/negative`，再用条件化的离散动作 loss 与 flow-matching loss更新 VLA。它确实是 RL，并且直接使用示范、自主 rollout 和 HG-DAgger 式人工纠正。

用户提到的 [π0.7](https://www.pi.website/download/pi07.pdf) 也和 RL 有关系，但关系不同：π0.7 主要使用带质量、速度、长度、策略和视觉子目标等上下文的统一训练；它吸收了 π*0.6 等 RL specialist 产生的自主数据、失败和高质量轨迹，最终以一个 generalist 模型复现甚至超过部分 RL specialist。因而它是“**由 RL 经验供数的统一模型**”，不能直接等价为“π0.7 本身用 PPO 端到端训练”。

### 0.2 直接更新 AHA-WAM 参数不是不可行

之前优先推荐外挂 actor–critic，不是因为 ActionDiT 或整套 WAM 不能做 RL，而是因为：

- 真机数据少，外挂 residual/adapter 更容易保证探索范围；
- off-policy 小 actor/critic 可以高频更新，回滚简单；
- flow/diffusion action head 没有简单的解析 log-probability，直接 PPO 需要 FPO、Flow-SDE、noise-MDP、AWR/RECAP 等专门算法；
- AHA 不是一个小模型：当前代码审计约为 5.00B VideoDiT、1.02B ActionDiT、1.23B chunk-KV editor，联合优化成本和非平稳性都显著高于普通 VLA action head；
- SO101 插入的回报数据量不足以可靠地给数十亿参数提供无约束 critic gradient。

新的、更准确结论是：

> **外挂 actor–critic 是安全的在线基线；ActionDiT 的 advantage-conditioned/weighted flow fine-tuning 是应当并行建立的“直接改 AHA 参数”主线；VideoDiT 应先用真实 rollout/HG-DAgger 视频做带正则的监督适配，而不是直接吃稀疏 RL reward；整套 WAM 联合 RL 放在证据表明视觉预测/表示确实是瓶颈之后。**

### 0.3 推荐优先级已调整

| 优先级 | 路线 | 更新范围 | 对当前插入任务的判断 |
|---|---|---|---|
| A0 | HG-DAgger/IWR 数据闭环 | 先不限定 | 必须先有成功、失败、纠正和严格时间对齐的数据 |
| A1 | ActionDiT-only SFT 基线 | ActionDiT + OVCR/editor + proprio/视觉投影 | 先验证冻结视频表示是否足够 |
| A2 | RECAP/AWR 式 ActionDiT offline RL | 外部 value/critic；更新 ActionDiT/条件路径 | **最值得新增的主方法**；是真正修改 AHA 本体参数 |
| A3 | RLT/HALO/ResiP 式小 actor–critic | 冻结 WAM；更新小 actor/critic | 安全、样本高效；也是 A2 的可靠对照和数据生成器 |
| B1 | RL specialist → 蒸馏回 ActionDiT | 先小 RL，再 SFT ActionDiT | PLD、RLDG、π0.7 路线；部署时可移除外挂 actor |
| B2 | Video online SFT + latent anchor | VideoDiT 小学习率/LoRA；ActionDiT 联动 | 仅当近接触视觉预测或表示被证明确实不足 |
| B3 | 分层联合优化 | 视频 SFT；ActionDiT RL；两条 loss 分开 | 比“reward 一把反传整个 WAM”稳得多 |
| C | 全 WAM policy-gradient RL | VideoDiT + ActionDiT + editor | 原理可行，当前单 SO101 项目不应作为第一版 |

---

## 1. 判断一篇工作“是否用了 RL”要看什么

不能只看标题。应当逐项检查：

- 是否存在环境 reward、人工结果标签或 learned reward；
- 是否训练 value/Q/advantage；
- policy update 是否显式偏向高回报动作；
- 更新的是基础模型、action head、latent/noise adapter，还是另一个 specialist；
- RL 只负责产生数据，最终模型是否仍用普通 SFT；
- 测试时是否仍需要 critic、world model 或外挂 actor。

根据这几项，当前文献可分成五层。

### 1.1 直接 policy-gradient/actor–critic

模型或其 action expert 直接通过 PPO、Q gradient、SAC/TD3、Flow-SDE 等优化 reward：

- [πRL](https://arxiv.org/abs/2510.25889)：把 flow denoising 变成 noise-MDP 或 Flow-SDE，主要冻结 VLM、只更新 action expert；
- [Self-Improving Embodied Foundation Models](https://arxiv.org/abs/2509.15155)：标题没有出现 RL，但第二阶段明确使用在线 REINFORCE 自我改进；实验还包含与本项目高度相关的 ALOHA 单孔插入；
- [FPO](https://arxiv.org/abs/2510.09976)：用 conditional flow-matching objective 的变化近似 policy ratio；
- [Flow Policy Gradients](https://arxiv.org/abs/2602.02481)：绕过 flow policy 的显式 likelihood；
- [RLDT](https://arxiv.org/abs/2606.08602)：把最大熵策略改进写成 action density transport；
- [ConRFT](https://arxiv.org/abs/2502.05450)：冻结 encoder/backbone，以 consistency action head + Q loss 做 offline-to-online 真机微调；
- [RLT](https://arxiv.org/abs/2604.23073)、[HALO-WA](https://arxiv.org/abs/2607.04265)：冻结大模型，训练小 actor/critic。

### 1.2 “RL as supervised learning”

表面仍在最小化 flow/diffusion/BC loss，但 value/advantage 决定条件或权重：

- [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759)：优势条件策略；所有数据可保留，高低 advantage 通过条件区分；
- [VLAW](https://arxiv.org/abs/2602.12063)：保留 reward model 判定的成功 real/imagined trajectory，用加权 flow matching 更新 π0.5；
- [IG-RFT](https://arxiv.org/abs/2602.20715)：interaction-guided AWR，依据是否发生交互调节探索和加权强度；
- [CO-RFT](https://arxiv.org/abs/2508.02219)：chunked Cal-QL critic，以 action-chunk Q 优化策略；但其当前 action head 是确定性头，不是 flow head；
- [ALOE](https://arxiv.org/abs/2602.12691)：用 pessimistic action-level evaluation 和 AWR 更新生成策略。

这类方法尤其适合 AHA，因为不要求对整个多步去噪链计算精确 log-probability，也可以利用旧 rollout、HG-DAgger 和失败数据。

### 1.3 RL 更新 latent/noise，而不是基础动作网络

- [DSRL](https://arxiv.org/abs/2506.15799)：冻结 diffusion policy，在其初始噪声空间训练小 RL policy；
- [GR-RL](https://arxiv.org/abs/2512.01801)：先以 offline Q 过滤低质量示范，再训练 latent noise predictor；真机鞋带穿孔要求长时程和毫米级精度；
- [RL²-VLA](https://arxiv.org/abs/2607.26991)：离线 RL 学 VLA action-expert latent 上的额外 flow velocity，且只在预测基础策略可能失败时启用；
- [Align-Then-stEer](https://arxiv.org/abs/2509.02055)：统一 latent 后以 guidance steer diffusion/flow VLA，重点是轻量跨任务/跨本体适配。

这类方法不等于“外挂一个完全独立的动作策略”：基础生成过程仍决定动作流形，小模块只改变 noise 或 velocity。

### 1.4 RL 专家负责采数，再蒸馏回 generalist

- [PLD：Self-Improving VLA via Residual RL](https://arxiv.org/abs/2511.00091)：residual RL 探测失败区域、生成恢复轨迹，再用标准 SFT 蒸馏回 VLA；
- [RLDG](https://arxiv.org/abs/2412.09858)：专用 RL policy 生成连接器插入等高质量轨迹，再训练 OpenVLA/Octo；
- π0.7 吸收 RL specialist 的自主数据并用质量/策略上下文统一训练。

这条路线能同时满足两个目标：在线阶段保留小策略的安全和效率，最终又把能力吸回 AHA-WAM，而不是永久依赖外挂 actor。

### 1.5 自主数据/self-improvement，但严格说不是 RL

必须避免把所有 self-improvement 都叫 RL：

- [RoboCat](https://arxiv.org/abs/2306.11706) 会自主生成下一轮训练数据，但主要是成功过滤和迭代 imitation；
- [Octo](https://arxiv.org/abs/2405.12213) 使用 diffusion action head，但其基础训练是大规模行为克隆；
- [GR00T N1](https://arxiv.org/abs/2503.14734) 的 System 1 是 diffusion transformer，和 VLM 紧密联合训练，但论文主体是 heterogeneous-data imitation，不是 RL；
- [RT-X](https://arxiv.org/abs/2310.08864) 汇聚的底层数据可能由不同控制器、脚本或 RL 专家产生，RT-X 模型本身仍主要做监督策略学习。

“数据由 RL policy 产生”和“当前模型由 RL objective 更新”对计算、稳定性和可复现性的含义完全不同。

---

## 2. π*0.6、π0.7 对 AHA-WAM 的真正启示

### 2.1 π*0.6/RECAP 不是传统 PPO

RECAP 每一轮执行：

```text
示范 + 历史 rollout + 当前 rollout + 人工纠正
                    │
                    ▼
训练 distributional value function
                    │
                    ▼
计算每步 advantage，并二值化为 positive / negative
                    │
                    ▼
用 advantage 条件训练离散动作和 flow action expert
                    │
                    ▼
部署时请求 positive 条件，重新收集真实数据
```

它有几个对 AHA 很关键的设计：

1. 使用所有历史数据，而不是只留下成功 episode；
2. 人类纠正默认强制标成 positive，但论文也承认纠正不总是平滑或最优；
3. reward 是普适的终局成功/失败与完成时间，不要求每个像素都有人打标签；
4. 每轮重新从预训练 checkpoint 微调，而不是无限接着上一轮参数继续漂移；
5. 对 flow action expert 用原生 flow-matching objective，不需要把动作离散化成 PPO token；
6. π0.6 使用 Knowledge Insulation：continuous action expert 可以读取主干激活，但 action flow loss 对主干 stop-gradient，减少动作训练破坏通用视觉/语言表示。

第 6 点和 AHA 的“视频—动作是否要一起动”高度相关：紧密条件并不要求所有梯度都无约束地穿透所有模块。

### 2.2 π0.7 是 RL 经验的统一蒸馏，而不是每个任务继续挂 specialist

[π0.7 原文](https://www.pi.website/download/pi07.pdf)明确包含：

- demonstration data；
- RL post-trained agent 的 autonomous data；
- autonomous failures；
- episode quality/length/strategy metadata；
- subgoal image、语言和不同控制模态。

如果简单混合成功、失败和不同策略，模型会平均互相矛盾的模式。π0.7 的关键是给数据加“如何做、做得多好”的上下文，在测试时请求期望的质量和策略。

迁移到 AHA 的对应设计不是复制一个巨大文本 prompt，而是至少给 replay/sample 添加：

```text
source = demo / autonomous / intervention / residual-RL
outcome = success / failure / timeout
quality = high / medium / low
phase = approach / align / contact / insert / verify
intervention = none / recovery / safety-stop
speed_bin / path_length / max_tracking_error
```

否则把 HG-DAgger 恢复视频和自主失败视频直接混到 non-action-conditioned video model 中，模型可能学到模糊、多模态的平均未来。

### 2.3 Self-Improving Embodied Foundation Models：标题不写 RL，却直接做到插入任务

[Self-Improving Embodied Foundation Models](https://arxiv.org/abs/2509.15155) 是本次补查中最值得单独强调的“隐藏 RL”案例之一。它采用两阶段流程：

1. **Stage 1：监督学习。**模型除了预测动作，还预测当前观察到任务完成大约需要多少步，即 `steps-to-go`；
2. **Stage 2：在线自我改进。**冻结一份 Stage-1 模型作为进度/成功判别器，用相邻时刻预测的 steps-to-go 差构造稠密奖励，再以 Monte-Carlo return 和 on-policy REINFORCE 更新正在执行的策略；
3. 论文不仅研究通用操作，还包含模拟 ALOHA Single Insertion，以及附录中的真实 ALOHA 单孔插入；腕部相机中的细微滑移也会反映在 steps-to-go 预测中。

这篇工作的价值不只是“又一个 VLA+RL”，而是为插入提供了一个比二值成功奖励更实用的方向：**从示范和 rollout 学习视觉进度函数，再把它作为稠密奖励。**例如可将插入的连续进展理解为“接近—对准—接触—修正—进入—到底”，让进度头对未对准、卡边和滑脱表现出倒退，而不是直到 episode 结束才获得 0/1。

但它的策略梯度公式不能原样搬到当前 AHA-WAM。论文的策略更新需要动作对数概率，而 AHA-WAM 的 flow-matching ActionDiT 没有一个可直接、低方差使用的常规 `log π(a|o)`。更稳妥的迁移方式是：

- 保留“steps-to-go/视觉进度模型产生稠密奖励”；
- 用该奖励训练外挂 chunk critic；
- 再通过 RECAP 式优劣条件、AWR/advantage-weighted flow matching，或专门的 flow policy-gradient 方法更新 ActionDiT；
- 用人工成功标签、接触/插深传感器校准和不确定性拒判约束进度模型，防止策略钻视觉奖励漏洞。

因此，这篇论文同时支持两点：直接更新基础策略并非不可行；但“奖励如何构造”和“flow actor 如何更新”应分别选择方法。

---

## 3. Diffusion/flow 动作 RL 不只属于“Diffusion Policy”

用户的判断完全正确：方法边界应按动作生成数学形式划分，不应按模型名字划分。

以下都属于生成式连续动作策略：

| 模型 | 动作生成 | 是否基础训练就用 RL | 其 RL 方法能否启发 AHA |
|---|---|---:|---|
| Diffusion Policy | diffusion action chunk | 否 | 是，DPPO、IDQL、DSRL、Q-guidance |
| Octo | diffusion action head | 否 | 是，但视觉/语言接口更轻 |
| π0/π0.5/π0.6 | flow-matching action expert | π*0.6 才明确加入 RECAP RL | **很高**，与 AHA ActionDiT 最接近 |
| GR00T N1 | diffusion transformer action system | 否 | action expert-only RL 与 latent steering 可迁移 |
| AHA-WAM | flow/diffusion 式 ActionDiT + video-derived K/V | 当前是 joint supervised flow | 需要专门适配 flow loss、chunk 与 OVCR |

对应的 RL 接口有五种：

1. **去噪链当 MDP**：πRL Flow-Noise、DPPO；
2. **ODE 转 SDE 得到 likelihood**：πRL Flow-SDE、WAM-RL actor；
3. **不算 likelihood，直接 Q gradient**：Diffusion-QL、ConRFT、部分 ALOE；
4. **用 advantage 加权/条件化原生生成 loss**：RECAP、AWR、VLAW；
5. **冻结生成器，改 noise/velocity/candidate selection**：DSRL、GR-RL、RL²-VLA、IDQL。

对真机 SO101，推荐顺序是 `4 → 5/外挂 residual → 1/2`。直接把完整去噪过程放进 on-policy PPO，计算和交互成本最高，而且最难限制探索。

---

## 4. AHA-WAM 代码中的真实视频—动作耦合

### 4.1 当前组件和参数规模

本地审计记录给出的 bf16 模块规模约为：

| 组件 | 参数量 | 作用 |
|---|---:|---|
| VideoDiT | 5.00B | 视频/世界表示与未来生成 |
| ActionDiT | 1.02B | 生成连续 action chunk |
| chunk-KV editor | 1.23B | 将 chunk observation/query 写入视频 K/V，再供动作分支读取 |
| VAE | 0.71B，冻结 | 图像/视频 latent 编解码 |

来源见 [`baseline_video_experiments/README.md`](../scripts/so101_pipeline/baseline_video_experiments/README.md)。因此“只更新动作侧”在 AHA 中也可能是 2B 级别，并不等于训练一个几百万参数的小 head。

### 4.2 ActionDiT 不是和视频解耦的独立策略

在 [`mot.py`](../src/ahawam/models/wan22/mot.py) 的 `forward_prior_action_with_chunk_updated_kv` 中，每一层都会：

1. 用 VideoDiT block 产生 video K/V；
2. 用 chunk query 和首帧 video K/V 构造 updated cache；
3. 让同层 ActionDiT 的 attention 读取 updated K/V；
4. 再输出 action token。

因此冻结 VideoDiT 后，ActionDiT 仍然：

- 读取视频派生的多层表示；
- 使用 chunk observation query；
- 使用 proprio 和语言 context；
- 可以改变“如何把视频表示翻译为精细动作”。

这更像 π0.6 中“action expert attend to backbone activations”，而不是把视频完全扔掉。

### 4.3 当前视频还不是反事实 simulator

当前 [`configs/model/ahawam.yaml`](../configs/model/ahawam.yaml) 明确设置：

```yaml
video_dit_config:
  action_conditioned: false
```

所以不能直接输入多个候选动作，让视频模型分别预测“这个动作会造成什么后果”。它当前更接近：

```text
当前图像、历史、语言、行为分布先验 → 可能的未来视频/latent
```

而 VLAW、WMPO、VLA-RFT 等 imagined RL 需要：

```text
当前图像 + 候选动作 → 该动作导致的未来
```

在 action-conditioned 能力没有经过专门训练和 counterfactual 验证前，不能用 AHA 视频端给候选动作打因果分。

### 4.4 现有训练器已经支持一个有意义的 action-side baseline

[`trainer.py`](../src/ahawam/trainer.py) 的 `freeze_video_dit=true` 会冻结 VideoDiT，但仍训练 MoT 中未冻结的 action/editor 路径，以及 proprio、视觉投影和 chunk query encoder。这个 baseline 是：

> 固定世界表示，学习新的动作解码与视频—动作接口。

它正好应当作为直接 ActionDiT RL 前的 SFT sanity check。

---

## 5. 四种参数更新路线的可行性

### 5.1 路线 A：冻结 WAM，外挂 actor–critic

```text
AHA latent/ref action ──► 小 actor ──► 修正动作
                      └─► critic
```

优点：

- 最容易做 off-policy 真机 RL；
- residual 可以限幅并随时关闭；
- WAM 参数不漂移，replay 中旧 latent 更稳定；
- 可高频读取最新腕部图像，补 chunk 内闭环；
- RLT、HALO、ResiP、HIL-SERL 已给出多种实证模板。

缺点：

- 最终部署多一个模型和控制路径；
- actor 可能逐渐忽略 AHA；
- 如果不蒸馏，AHA 本体没有真正学会恢复和精插入。

结论：仍是必要 baseline，但不应再被描述为唯一建议。

### 5.2 路线 B：外置 critic，直接更新 ActionDiT/动作侧

这里 actor 就是 AHA ActionDiT 或其 LoRA/adapter，critic 仍可独立。`actor–critic` 并不等于 actor 一定是外挂网络。

最稳妥的目标不是直接 PPO，而是：

#### 方案 B1：advantage-weighted flow matching

对 replay 中每个 action chunk 估计 advantage：

\[
w_t=\operatorname{clip}\left(\exp(A_t/\beta), w_{\min}, w_{\max}\right),
\]

然后：

\[
L_{\text{actor}}
=
\mathbb E[w_t L_{\text{flow}}(o_t,a_{t:t+H})]
+\lambda_{\text{anchor}}L_{\text{base/rehearsal}}.
\]

优点是兼容原 ActionDiT 训练接口；缺点是低 advantage 数据会被弱化，数据效率取决于 critic 校准。

#### 方案 B2：RECAP 式 advantage conditioning

给 ActionDiT 增加一个 positive/negative embedding 或 condition token：

\[
I_t=\mathbb 1[A_t>\epsilon].
\]

训练时保留全部数据，测试时固定 `I=positive`。这比简单成功过滤更能利用失败数据，也更适合将 intervention 强制标为 positive。

#### 方案 B3：Q-filtered target

比较当前策略 chunk、WAM 原始 chunk、人类 intervention chunk、residual-RL chunk：

\[
a^*=\arg\max_{a\in\mathcal C}Q(o,a),
\]

仅当 Q margin 足够可信时才以 (a^*) 为 flow/BC target。它接近 Max-Q、IDQL 和 candidate ranking。

结论：**这条路线对你的研究价值最高**。它直接改 AHA 参数，又不要求 reward gradient 穿过 5B VideoDiT。

### 5.3 路线 C：只更新 VideoDiT

如果目标是“提高视频生成精度”，最合理的训练信号通常不是 RL reward，而是在线真实未来视频：

\[
L_{\text{video}}
=L_{\text{flow/reconstruction}}(\hat{x}_{t+1:t+H},x_{t+1:t+H})
+\lambda_{\text{latent}}\|z_{\text{new}}-z_{\text{base}}\|^2.
\]

[WAM-RL](https://arxiv.org/abs/2606.17906) 的世界模型也是在成功在线轨迹上做 video SFT，并用 latent KL anchor 保持 actor 接口稳定；reward/policy gradient 主要优化 action model，而不是直接用成功奖励推 VideoDiT 像素。

[VLAW](https://arxiv.org/abs/2602.12063) 则用真实 policy rollout——包括失败——按标准 action-conditioned diffusion objective 更新 world model，并混入原 DROID 数据防止过拟合。

对 AHA 可以迁移为：

- 成功自主视频：训练准确的正常局部未来；
- intervention 恢复视频：训练恢复模式，但必须带 `source/intervention` 条件或谨慎混采；
- 失败视频：训练“会失败的真实动力学”，不能把失败未来假装成正向计划；
- 原始 SO101 示范：作为 anchor/rehearsal；
- 旧 VideoDiT latent：feature distillation/KL anchor，避免 ActionDiT 看到的表示突然漂移。

结论：VideoDiT 可更新，但应叫“RL 交互数据驱动的 online video SFT”，而不是“直接对视频做 RL”。

### 5.4 路线 D：联合整个 WAM

联合训练原理上可行，[WAM-RL](https://arxiv.org/abs/2606.17906) 甚至报告：actor-only 在短任务有提升，但在长任务上受 world-model 累积误差限制；world + actor 联合才改善长时程任务。

但其真正做法是分层损失：

```text
VideoDiT：online video SFT + latent KL
Action model：flow-SDE policy gradient + reconstruction reward
```

不是：

```text
终局成功 reward → 一个 critic gradient → 无约束穿透整套视频和动作网络
```

而且 WAM-RL 当前证据来自 LIBERO/RLBench 仿真，使用 8 张 A800 训练约 8 小时；它很新，不能把结论无条件外推到单台 SO101。

对于 AHA，更合理的联合更新是“双优化器/双时间尺度”：

- ActionDiT/OVCR：较快更新，来自 advantage-conditioned/weighted action flow loss；
- VideoDiT：较慢更新，来自真实未来视频 loss + frozen-base feature anchor；
- 禁止未经控制的 action RL gradient 直接改写 VideoDiT；
- VideoDiT 每更新一轮，要重新校准 critic 或重新抽取 replay feature。

---

## 6. 用户提出的几种设计逐一判断

### 6.1 “视频负责大概方向，action 独立负责短程精密”是否合理？

合理，而且并不等于失去 AHA-WAM 的意义。

AHA 仍提供：

- 任务阶段和长时程视觉先验；
- 全局靠近、粗对准和恢复方向；
- 多层 video latent；
- nominal action chunk 和动作流形；
- 异步世界预测与较快动作执行的计算分工。

局部精密模块只负责最后几毫米内的闭环误差修正。ResiP、RLT、GR-RL、RL²-VLA 都支持“保留强生成式先验，只对局部控制接口做 RL”的思路。

风险在于：如果冻结视频 latent 根本看不见微小横向/角度误差，那么 ActionDiT 或外挂 actor 都无法凭空恢复信息。应先做表征诊断，而不是凭可视化视频好不好猜测。

### 6.2 “只做 HG-DAgger，视频端自然会学会正确判断”是否足够？

可能足够，但不是必然。

它成立需要：

1. 训练确实解冻或适配 VideoDiT，而不是只训练 ActionDiT；
2. 数据包含接近失败边界、恢复和成功重试，不只是完美示范；
3. 视频 loss 对小位置/姿态误差敏感；
4. 腕部相机分辨率、曝光、延迟足以观察误差；
5. intervention 之后的人类未来不会和自主失败未来在无条件模型中被平均；
6. ActionDiT 读取的 latent 改进，而不仅是最终解码视频看起来更清晰。

HG-DAgger 主要解决 learner-induced state coverage，并不自动保证视频生成器学到接触动力学或毫米级几何。

### 6.3 “精密阶段缩短 video 预测长度”是否是好主意？

先不要把它作为第一项结构改动。必须区分三个量：

- video prediction horizon：世界分支预测多远；
- action chunk horizon：ActionDiT 一次生成多长；
- execution horizon：机器人实际执行几步后重新观察/规划。

插入时最需要先缩短的是 **execution horizon**，不是一定要缩短 video prediction horizon。可以让视频线程继续低频预测较长趋势，同时动作线程每 1–2 步用新腕部图像闭环；这样仍然保留异步价值。

只有当实验显示“长视频预测导致 near-contact latent 变旧、错误累积或拖慢动作更新”时，再引入：

- contact-aware video refresh；
- uncertainty-triggered refresh；
- 多尺度视频 horizon；
- 自由空间长 horizon、接触阶段短 horizon。

这会成为有意义的研究消融，而不是预先硬编码的复杂规则。

### 6.4 “视频生成画面不好，是否一定要对视频做 RL？”

不一定。需要区分：

1. 解码后像素不漂亮；
2. latent 无法表达控制相关误差；
3. 未来预测方向错误；
4. world model 不能区分候选动作后果。

第 1 项未必影响控制；第 2/3 项才可能要求表示适配；第 4 项要求 action-conditioned world model，单纯多做 video SFT 不一定补齐。

视频质量应使用任务指标，而不是只看 FVD/PSNR：

- 孔中心/插头尖端 keypoint error；
- 轴线角度误差；
- 深度/接触阶段分类；
- 对 near-miss 与 success 的可分性；
- 预测未来中的插入进度与真实进度误差；
- frozen latent 上简单 probe 的预测精度。

---

## 7. 一个能回答“到底该更新哪里”的诊断矩阵

在写 RL 训练代码前，先做四个 probe。

### 7.1 Frozen-latent geometry probe

冻结 AHA，抽取接触前后 video/OVCR feature，训练小 probe 预测：

- 左右/前后偏差；
- 插入轴角度；
- 进度 phase；
- success probability；
- 是否将在未来 0.3–1.0 秒触发接管。

若 probe 准确，说明视频表示已经包含精密信息，ActionDiT-only RL 很有希望。

### 7.2 Action sensitivity test

构造位置和角度逐渐变化的小扰动，检查 ActionDiT 输出是否随误差方向单调变化。若 latent probe 准但动作不敏感，瓶颈在 ActionDiT/动作解码，而不是 VideoDiT。

### 7.3 Raw-wrist vs AHA-latent probe

用相同小模型比较：

```text
raw wrist RGB → error/success
AHA latent    → error/success
```

若 raw RGB 明显更好，适合在 residual actor 增加局部腕部 encoder，或部分适配 video/OVCR；若两者都差，优先修相机、视角、标定、照明和数据分辨率。

### 7.4 Video future task-metric test

在同一批 near-contact clip 上评价基础、HG-DAgger 微调后和部分联合微调后的：

- future keypoint/pose error；
- 接触/插入阶段准确率；
- recovery 预测召回率；
- latent drift；
- 最终自主成功率。

最终按下面的矩阵决策：

| 观察 | 判断 | 下一步 |
|---|---|---|
| latent 好，action 差 | 动作映射瓶颈 | ActionDiT-only SFT/RL |
| latent 好，chunk 内反应慢 | 控制时间尺度瓶颈 | step residual 或缩短 execution horizon |
| latent 差，raw wrist 好 | AHA 表示/接口瓶颈 | 局部视觉 adapter；再考虑部分 VideoDiT |
| video 像素差，latent/动作好 | 解码画质不是瓶颈 | 不动 VideoDiT |
| future/latent 都差，且 rollout 分布偏移 | world representation 瓶颈 | online video SFT + anchor |
| 候选动作后果不可区分 | 缺 action conditioning | 不做 imagined RL；先训练 action-conditioned WM |

---

## 8. 推荐的实验主线与消融

### 8.1 共享数据闭环

所有方法使用同一份数据预算：

```text
初始示范
+ 自主成功
+ 自主失败
+ HG-DAgger intervention
+ 接管前 warning window
+ 接管后恢复与重新交还
```

每步保存 `policy_action`、`human_action`、`executed_action`、`intervention_mask`、reward、next observation、phase、模型版本、video/cache age 和时间戳。

### 8.2 必做 baseline

| 编号 | 方法 | 更新参数 | 目的 |
|---|---|---|---|
| B0 | 当前 joint SFT | VideoDiT + ActionDiT + editor | 现有完整微调基线 |
| B1 | action-side SFT | 冻 VideoDiT；训 ActionDiT/editor | 判断视频表示是否足够 |
| B2 | HG-DAgger action-side SFT | 同 B1 | 交互数据是否已能解决失败 |
| R1 | residual off-policy RL | 冻 AHA；小 actor/critic | 最安全的真机 RL 基线 |
| R2 | ActionDiT advantage-weighted flow | 外部 critic；训动作侧 | 直接改 AHA 参数的主方法 |
| R3 | RECAP-style condition | value + positive/negative ActionDiT | 保留失败数据并可控采样 |
| R4 | R1 轨迹蒸馏回 ActionDiT | 最终只部署 AHA | 检查外挂是否可移除 |
| W1 | online video SFT + anchor | 小步更新 VideoDiT | 仅在视觉诊断失败后做 |
| W2 | 分层 joint | R2 + W1 | 检验 WAM-RL 式联合收益 |

### 8.3 成败判定不能只看 success rate

还应记录：

- autonomous success；
- intervention rate 与人工控制时间；
- 完成时间/成功吞吐量；
- 插入前最大横向/角度误差；
- 接触后无进展时间；
- 轨迹长度、动作 jerk、关节跟踪误差；
- 卡滞、碰撞、工件移动等安全事件；
- seen 与 held-out 初始偏置；
- VideoDiT task metric 与 latent probe；
- 参数量、训练 GPU-hours、真实机器人交互分钟数。

π*0.6 强调 throughput 而不只 success，这对插入也重要：一个策略可能靠反复试探最终成功，但速度和接触风险很差。

---

## 9. 对当前项目最具体的算法建议

### 9.1 第一版“直接修改 AHA 参数”的 RL

建议从 ActionDiT advantage-weighted flow 开始，而不是全模型 PPO：

1. 冻结 VideoDiT；
2. 保持 ActionDiT、chunk-KV editor、chunk query、visual/proprio projection 可训练；
3. 用 frozen AHA feature + executed action chunk 训练 conservative chunk critic；
4. intervention action 标为高质量候选，但用 SiLRI/Max-Q 式校验避免盲信次优人类动作；
5. 对 action flow loss 使用 clipped advantage weight；
6. 混入原始 SO101 成功数据，防止动作分布漂移；
7. 接触前后分别统计 critic calibration；
8. 每轮从稳定 SFT checkpoint 重新微调并保留旧模型对照，模仿 RECAP 的抗漂移方式。

这一版不需要外挂 actor 参与部署，但需要外挂 critic 参与训练。critic 可以在部署后移除。

### 9.2 第二版加入安全在线探索

用 residual/RLT actor 进行小幅真实探索：

\[
a_{exec}=a_{AHA}+g_{contact}\operatorname{clip}(\Delta a,-b,b).
\]

只在 near-contact phase 开启，接管立即取消旧 chunk。收集到比 AHA 更高价值的轨迹后：

- 继续保留 residual 部署；或
- 用 PLD/RLDG 风格将轨迹蒸馏回 ActionDiT；或
- 将 AHA、human、residual 三种候选交给 Q-filter，训练 ActionDiT 模仿较优者。

### 9.3 VideoDiT 只在有证据时加入

若 probe 显示 world representation 是瓶颈：

1. 用真实 rollout 视频做 standard video flow SFT；
2. 同时混入原始成功数据；
3. 对 frozen base latent 做 L2/KL/feature distillation；
4. 对 intervention、failure、success 添加来源/质量条件，或分别采样；
5. VideoDiT 学习率显著低于 ActionDiT；
6. 每次视频更新后重新抽取 critic training feature；
7. 先更新最后若干层的 LoRA/adapter，再考虑 full VideoDiT；
8. 不让稀疏成功 reward 无约束反传所有视频参数。

---

## 10. 文献对当前 AHA 项目的价值排序

| 论文 | 它真正更新什么 | 对 AHA 的直接价值 | 证据限制 |
|---|---|---:|---|
| [π*0.6 / RECAP](https://arxiv.org/abs/2511.14759) | value + advantage-conditioned 整体 VLA/action expert | **最高**：直接启发 ActionDiT RL 与 HG-DAgger 数据整合 | 大规模闭源数据/算力；非插入专项 |
| [π0.7](https://www.pi.website/download/pi07.pdf) | 质量/策略条件的统一模型，吸收 RL 数据 | **高**：解释如何把 specialist/失败/纠正蒸馏回 generalist | 不是可直接复现的 RL recipe |
| [πRL](https://arxiv.org/abs/2510.25889) | 冻 VLM，更新 flow action expert | **高**：证明 action-expert-only RL 可行 | 8×H100、并行仿真；真机证据不足 |
| [Self-Improving Embodied Foundation Models](https://arxiv.org/abs/2509.15155) | steps-to-go 稠密奖励 + 整体策略 REINFORCE | **高**：ALOHA 插入；直接启发视觉进度奖励 | 原策略可算 log-prob；AHA flow head 不能照搬更新式 |
| [ConRFT](https://arxiv.org/abs/2502.05450) | 冻 backbone，更新 consistency action head + Q | **高**：少 demo、HIL、contact-rich 真机 | 替换了 action head；多数任务较短 |
| [GR-RL](https://arxiv.org/abs/2512.01801) | frozen VLA 上的 latent noise policy | **高**：毫米级长时程精密，避免全量更新 | 单一高资源系统，新预印本 |
| [PLD](https://arxiv.org/abs/2511.00091) | residual RL 采数，再 SFT generalist | **高**：解决“外挂有用但想回到 AHA 本体” | 训练流程多阶段 |
| [RLT](https://arxiv.org/abs/2604.23073) | RL token、小 actor、双 Q | **高**：插入任务、精密真机、接口清楚 | 仍是外挂 actor；高性能平台 |
| [WAM-RL](https://arxiv.org/abs/2606.17906) | video SFT + actor RL | **高但后期**：直接回答 action-only vs joint | 仿真、8×A800、新论文 |
| [VLAW](https://arxiv.org/abs/2602.12063) | action-conditioned WM + 成功加权 flow VLA | **中高**：失败视频、world-policy co-improvement | AHA 当前视频不 action-conditioned |
| [RL²-VLA](https://arxiv.org/abs/2607.26991) | latent flow composition + failure gate | **中高**：只在需要时 steer，保留 base | 2026-07 新预印本 |
| [FPO](https://arxiv.org/abs/2510.09976) | flow policy PPO-style update | **中**：原生 flow RL 数学接口 | 主要仿真，直接真机风险高 |
| [Flow Policy Gradients](https://arxiv.org/abs/2602.02481) | flow policy gradient | **中**：长期直接优化路线 | 并非 AHA/WAM 专项 |
| [RLDT](https://arxiv.org/abs/2606.08602) | density transport fine-tune flow policy | **中**：不牺牲多模态 action distribution | 很新，工程复杂 |
| [CO-RFT](https://arxiv.org/abs/2508.02219) | full SFT 后 chunked offline RL | **中**：chunk critic 有价值 | 当前 policy 是 deterministic，不是 ActionDiT |

---

## 11. 对旧结论的正式修订

旧表述容易让人理解为：

> “AHA-WAM 太大，不能直接动参数，所以只能外挂 actor–critic。”

修订后的结论是：

1. **可以直接动参数。** π*0.6、πRL、ConRFT、FPO、CO-RFT 等证明了 full/action-head 级 VLA RL 的不同实现；
2. **AHA 最值得先动的是 ActionDiT + 视频—动作接口，不是 5B VideoDiT。** 这仍保留每层 video K/V 条件；
3. **critic 可以外挂、actor 不必外挂。** 外置 critic + ActionDiT advantage flow 是自然组合；
4. **外挂 residual 仍有价值。** 它是安全的在线探索器和 RL 数据生成器，之后可以蒸馏回 AHA；
5. **VideoDiT 不是永远冻结。** 当实验证明预测/表示是瓶颈时，应以 online video SFT + latent anchor 慢更新；
6. **不要用稀疏 reward 直接要求视频“更清晰”。** 任务 reward、视频预测误差和控制表示不是同一个目标；
7. **精密阶段先缩 execution horizon，而不是必然缩 video horizon。** 多速率异步仍然有意义；
8. **全 WAM 联合优化是后续研究问题，不是不可行问题。** 最合理形式是 WAM-RL/Knowledge-Insulation 式分层梯度，而非一个 critic 无约束更新全部模块。
9. **插入奖励值得增加视觉进度头。** steps-to-go 可将卡边、滑脱和重新对准变成稠密信号，但应交给 critic/advantage-flow 使用，而不是把 REINFORCE 公式机械套到 ActionDiT。

---

## 12. 最终建议

如果目标是在真实 SO101 上尽快得到可信结论，同时保留论文创新空间，建议主线定为：

```text
HG-DAgger 收集 learner-induced failures / recovery
                    │
                    ├──► conservative chunk value / advantage
                    │
                    ├──► ActionDiT advantage-conditioned flow fine-tuning
                    │
                    └──► bounded residual RL（安全在线探索对照）
                                      │
                                      ▼
                         高价值轨迹蒸馏回 ActionDiT
```

VideoDiT 走一条有准入条件的支线：

```text
只有 frozen-latent / future task-metric 证明视觉表示是瓶颈
                    │
                    ▼
rollout + intervention + failure video SFT
+ base-latent anchor
+ source/quality conditioning
                    │
                    ▼
与 ActionDiT RL 做分层联合消融
```

这样既没有把 AHA 降级成“只提供一个粗动作的可替换前端”，也没有在证据不足时让一个稀疏成功奖励无约束地改写整套 5B 视频生成器。它能直接回答一个有研究价值的问题：

> 对异步 WAM 的精密插入，性能瓶颈主要来自世界表示、动作翻译、chunk 内闭环，还是三者的耦合？不同数据规模下，ActionDiT-only、residual RL、RL-to-distillation 与分层 joint WAM 更新的边界在哪里？
