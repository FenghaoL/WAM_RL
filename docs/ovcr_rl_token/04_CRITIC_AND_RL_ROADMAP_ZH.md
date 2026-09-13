# 是否需要 critic，怎样升级为真正的 RL

## 1. 直接回答

**为了验证 OVCR adapter，不需要 critic；为了从成功／失败奖励学到超出人类示范的策略，需要明确的 RL 目标，但 critic 不是定义上的必需品。** REINFORCE／策略梯度可以不用学习 Q；在你这种真实交互昂贵、失败样本少、需要反复使用 replay 的场景，off-policy value/critic 通常是更实用的选择。

| 做法 | 是不是 RL |
| --- | --- |
| HUMAN action flow MSE | 不是；交互式模仿 |
| 用 takeover 标签训练 gate | 不是；风险监督分类 |
| 成功图片分类器 | 单独不是；可作为 reward 来源 |
| token reconstruction | 不是；表征学习 |
| 只训练 Q，但 actor 仍只做原 BC | 有价值学习，尚未构成基于 Q 的策略改进 |
| return/advantage 真正改变 actor 的更新，并迭代 rollout | 可以构成 RL／离线 RL，需说明估计与目标 |
| TD critic＋reward-driven actor＋真实新经验 | 标准 actor–critic RL 路线 |

不要为了名字“更像 RL”先添一个没进入策略更新的 critic。先验证：现有表征能否分辨偏差、HUMAN label 是否足够、adapter 是否可控制最终动作。这三点无效时，Q 的误差通常只会放大问题。

## 2. 推荐的两条 RL 扩展

### R1：保留 Action DiT，用价值加权的 flow 更新内部 adapter

这是最贴近本次主方法的扩展。actor 仍是 AHA 的生成式动作模型，只有 φ 更新。单独用 rollout transition 学 V/Q，再给真实动作样本赋权，避免每次 actor 更新都对 10-step sampler 完整反传。

可以借鉴 [IQL 的离线 value learning 与 advantage-weighted extraction](https://arxiv.org/abs/2110.06169)，但换成 AHA flow actor 后应标注为 IQL-inspired/AW-flow，而不是声称原 IQL 原样复现。一个可实现的候选为：

$$
\mathcal L_V=\mathbb E[\rho_\tau(\min_iQ_i(h,A)-V(h))],
\quad \rho_\tau(u)=|\tau-\mathbf 1[u<0]|u^2,
$$

$$
y=R^{(k)}+\gamma^k b\,V_{target}(h'),\qquad
\mathcal L_Q=\sum_{i=1}^{2}\mathbb E[(Q_i(h,A)-y)^2],
$$

$$
\hat A=\min_iQ_i(h,A)-V(h),\quad
w=\operatorname{clip}(\exp(\hat A/\beta),w_{min},w_{max}),
$$

$$
\mathcal L_\phi=\mathbb E[\operatorname{sg}(w)\,\mathcal L_{flow}(A;h,\phi)]
+\lambda_a\mathcal L_{anchor}.
$$

第一轮可用 expectile τ=0.7、wmax=10、β 按 validation advantage 的尺度选取；这些是起始假设，不是已经适配 SO101 的最佳值。权重须先 stop-gradient，并监控 effective sample size、极端权重比例和 positive/negative 数据占比。

这条路线可以利用 POLICY 实际动作作为 RL 数据，与“所有 POLICY 都不能训练”并不矛盾：不能把失败 POLICY 当 **专家标签**；经过价值建模后以明确权重训练它们，是不同目标。若所有权重都差不多，方法实际上退化为混合行为克隆，应在日志中揭示。

**关键工程条件：** h 必须与 actor 条件一致或足够解释它。只把当时的 z 存进 replay 能训练小 critic，却不自动足以重建完整 Action DiT 所需的每层 K/V。因此内部 AW-flow 还需要在线 KV 重建协议／有序 raw replay，或分轮固定 cache protocol 的离线再训练。若只能用 reset 后的离线图像重算 h，应把它叫近似 offline adaptation，不能宣称精确 on-policy latent 重放。

本轮尚未实现这套 value/weight/replay 训练；已经实现的 masked flow＋P0 anchor 是它的 actor 端起点。

### R2：RLT-inspired 外部 chunk actor＋twin Q

这条路线更接近 RLT 的功能分工。先固定任务 P0 和 readout E，然后训练小 actor。为减少与本体结构耦合，它可以直接读导出的稳定 z、当前 proprio 和 reference chunk。

我们针对 AHA 提出的状态扩展为：

$$
h_t=[z_t,s_t,\operatorname{age}(M_t),\operatorname{phase}_t,
\operatorname{committed\_actions}_t,\operatorname{valid\_queue\_mask}_t].
$$

actor 接口：\(A_t\sim\pi_\eta(h_t,A_t^{ref})\)。reference 必须是同一状态、同一基座、同一 cache generation 对应的 AHA chunk，不能从另一次 reset 后的策略随便补一个。

可采用小 MLP、固定方差 Gaussian、两个独立 Q、target critic，训练目标以 reward maximization 加 reference penalty 为主；这与 [RLT 方法](https://arxiv.org/html/2604.23073v1#S4) 的接口一致。若改成 \(A=A^{ref}+\delta A\) 且 δ 受限幅，就明确叫 bounded residual 对照，不应把这种参数化误称为 RLT 原文。

适合先尝试的规模：z=256 或512，3层宽256 MLP，action C×6。C 的初始值必须服从延迟预算；理论上更短更反应快，但若远程 Action DiT 还要为每个 C=2 步重算 reference，其请求延迟可能超过执行时长。外部 actor 若要每 tick 更新，必须有本地快速 observation encoder 或可及时获得的新 query；只重用旧 z 不能称作最新视觉闭环。

先用 demo/HUMAN 做 actor warm-start；Q 在同时包含成功、失败和恢复的数据上 warm-start；确认 Q 有区分能力后，再逐步降低 BC/reference 权重、增加真实自主数据。β、exploration std 与 action 坐标的关系需用旧 stats 和实际关节限制标定。

优势是更新 actor/Q 不必穿过大 Action DiT，replay 可以小得多。代价是部分动作由外部 actor 生成，研究问题更接近“强 WAM 表征是否是好的 RL state”，而不是“WAM 内部耦合能否被纠正”。两者都合理，应作为不同主张比较。

## 3. 为什么不第一版直接让 Q 反传 10-step Action DiT

若 actor 是内部 adapter，直接最小化 \(-Q(h,A_\phi)\) 的梯度是：

$$
\nabla_\phi Q=
\frac{\partial Q}{\partial A}
\frac{\partial A}{\partial v_1,\ldots,v_J}
\frac{\partial v_1,\ldots,v_J}{\partial \phi}.
$$

动作 A 来自 J 次 flow solver 更新，不能把某个时刻的 flow velocity 当作最终 action 输入 Q。现有 `infer_action_chunk` 等推理接口有 `@torch.no_grad()`，直接调用再写 `loss=-Q(...)` 不能训练内部 adapter。需要新增可微 sampler，并测试 solver、截断梯度、checkpointing、噪声固定和安全投影的梯度语义。

它不是不可行，而是比 R1 的单 timestep flow 学习更重，且 Q 容易诱导连续多步采样漂移。若后面完成 Flash student，可把短 sampler 当作独立研究条件，但不能默认 student 与原 10-step actor 的优化等价。

## 4. 第三条后续路线：把编辑 latent 本身当 RL action

受 [DSRL](https://openreview.net/pdf?id=jU7AbGq3se) 的 latent-action 思路启发，可先通过监督训练得到一个冻结的 KV writer，随后让小 RL actor 输出低维控制变量 u，由 writer 把 u 解码为 ΔKV；Action DiT 保持冻结。

这时 RL MDP 的 action 是 u，critic 必须用 \(Q(h,u)\)；环境包括冻结 writer＋Action DiT＋真实机器人。rollout 必须保存实际 u，不能只保存执行关节动作后“反推出唯一 u”，因为这个映射通常多对一。还要保存 sampler noise 或把它明确当作环境随机性。

此方案有机会同时保留原动作生成器和轻量 online updates，但本轮 T4 并未提供外部 u 输入，不能直接把它称为已完成的 latent-action RL。第一版暂不选择它，原因是需要新的 collector 和可控 latent 预训练，而当前 DH 没有对应 u 标签。

## 5. chunk TD 与时间语义

若在原始控制 tick 上定义 γ，某次决策实际经历 k 个 tick，则：

$$
R_t^{(k)}=\sum_{j=0}^{k-1}\gamma^j r_{t+j},\quad
y_t=R_t^{(k)}+b_t\gamma^k\min_i Q_i^{target}(h_{t+k},A').
$$

不能把以 chunk 为单位定义的 γ 再乘一次 chunk 长度，也不能只把奖励求和却错误地使用 γ 而非 γᵏ。γ=0.99/tick 时，8-step 折扣约0.923；γ=0.99/chunk 是另一个时间偏好。建议先按任务真实耗时定义折扣半衰期 T½：\(\gamma_{tick}=2^{-1/(15T_{1/2})}\)，让不同 C 的比较保持同一真实时间偏好。

成功终止：b=0。失败终止：b=0。时间截断：是否 bootstrap 取决于任务目标是有限时限还是无限期继续，必须固定定义，不能把缺少 next observation 的 reset 后首帧当下一状态。

`SYSTEM_HOLD` 没有专家动作，但真实时间仍在走。如果 hold 是部署动力学的一部分，RL return 的 k/Δt 必须包含这段时间，或者把 hold 单列为 transition。不能删帧后假装机器人连续无延迟执行。

## 6. 机械臂中途被接管时如何处理

首先明确 critic 的 action 定义，以下两种都可成立，但不能混用：

1. **执行目标定义：** Q 评价实际下发的关节目标；第一版只训练完整、同一控制源、固定长度的 C-step 块，不把人的后续动作拼成策略未执行尾部。对于不足 C 的成功终点，要单独实现正确的终止块表示／mask；不能为了固定 shape 把所有成功终点删掉。
2. **宏命令定义：** Q 评价整个已提交 planned chunk；执行器的安全投影、提前成功终止和约定的取消逻辑作为环境的一部分。此时正常终止前只执行前缀是合法的，但日志必须同时保留 planned chunk 和 executed prefix，且不能把人接管后的成功算作原命令的自主效果。

若想先验证最简单的 reward learner，可以使用 C=1 的完整 tick transitions，避开 variable chunk 表示；它的 credit assignment 更长，但数据语义最容易检查。真正切换到 C=8 前再做 fixed-chunk／terminal-prefix 的专项测试。

为了不奖励“等待人来救”，建议把 **自主无接管完成** 作为第一版目标。发生 takeover 的策略 transition 以取消／干预为 censoring 边界，不从后续 assisted-success bootstrap。HUMAN 恢复段可单独用于 demo BC 与有效行为 transition。

这会改变学习目标，使它偏向自主完成，**不是对原无干预环境 Q 的无偏修复**。如果要评估人机合作系统总收益，则允许恢复奖励影响干预前价值可能有其合理性，但 actor/state 需要显式建模人类 gate。两类目标不能只用一个“success=true”字段混在一起。

## 7. 最低 replay schema

| 字段 | 用途 |
| --- | --- |
| episode_id、request_id、decision_id、tick 范围 | 对齐 episode、请求和实际执行 |
| current/next observation timestamps | 检查 next state 与真实时间 |
| base checkpoint、adapter/readout version、stats hash | 防止表征和坐标版本混用 |
| planner generation、age、history-valid-length、chunk index/offset | 解释同一图像下不同内部条件 |
| z_t、z_next；必要时有序原始观测／cache 重放来源 | 小 critic 的输入与内部 actor 的条件重建 |
| proprio_t、proprio_next | 当前实际本体状态 |
| reference chunk、reference seed、reference condition ID | actor 的基座锚点 |
| planned chunk、executed targets、executed length、valid mask | 区分动作建议和真实执行 |
| committed queue、queue mask、interruption/cancel generation | 异步状态中已承诺的未来动作 |
| reward、reward timestamp、reward model version | 回报对齐与奖励版本 |
| terminated、truncated、intervened、bootstrap mask | 正确构造 TD target |
| control_source、safety_clamped、hold duration | 专家监督和执行协议 |
| autonomous_success、assisted_success | 避免干预信用混淆 |

这比“单图＋动作＋done”复杂，是 AHA 的有状态／异步机制决定的，不是为了让 RL 设计看起来复杂。[ARLI](https://arxiv.org/abs/2608.23831) 对 committed actions 与中间新观测的处理提供了直接相关的依据；这里扩展到 AHA cache metadata 是本项目推理。

## 8. 奖励、critic 验证与启动门槛

第一版 reward 可以来自人工标注的自主成功终止。若加入进度分类器或电流／跟踪误差辅助信号，应固定 reward version 并离线验证；双 RGB 的“看起来插进去了”可能与真实卡滞混淆，未校准舵机电流也不等价于力传感器。

训练 Q 后先检查：held-out 成功／失败可分性、相近状态正确恢复动作 vs 错误推进动作的排序、Q 对大幅越界动作是否异常高、twin-Q disagreement、bootstrap target 与经验回报尺度、不同 cache-age 下的误差。TD loss 小不等于 Q 好用。

只有以下条件同时满足，再进入真机 RL：P0 和 adapter 的 reset/推理协议固定；有真实成功和失败经验；reward 定义可执行；实际 action 与 next state 可追溯；人能及时接管；探索尺度已按 SO101 接口标定。这里复用现有执行限幅，不把 feature-space ΔKV 的范数误认为关节安全保证。

推荐顺序为：**V32/Q32 的监督纠正 → T4 对照 → 完整 replay 协议 → 离线 value/advantage 检查 → R1 或 R2 中选一条小规模真机 RL**。若主要目标是原生 WAM 方法，优先 R1；若主要目标是尽快测试奖励优化在硬件上能否带来增益，R2 通常更易调试。
