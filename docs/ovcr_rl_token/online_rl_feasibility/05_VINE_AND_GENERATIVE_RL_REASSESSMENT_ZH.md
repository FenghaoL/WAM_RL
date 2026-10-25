# VINE与生成策略RL：重新评估AHA-WAM路线

核查：2026-09-18。研究与实施建议，不代表已完成RL训练或真机验证。此前调研偏重冻结大模型的小模块路线，不能据此理解为不存在直接训练diffusion/flow策略的RL。

## 1. 决策

VINE值得做受控试验，但不宜仅凭论文的真机速度就优先全量更新AHA-WAM。目标若是最快跑通真机闭环，仍优先冻结WAM的小actor；目标若是研究奖励如何改进WAM自身动作生成，VINE与QAM值得重点比较。两种目标的排序不同。

VINE改变采样链，使critic对最终动作的梯度能穿过多步生成更新策略。它不是噪声空间的小actor，也不是免critic算法。其真机使用ResNet-10与三层MLP、单动作输出、固定插座和人工干预；大基础模型扩展仍是未来方向。OGBench总分领先不等于逐任务领先：cube-double为4，DSRL为74，scene-sparse为73，DSRL为99。核心证据见[论文§3–5](https://arxiv.org/html/2607.10369v1)。这些差异不证明DSRL在本项目更好，但足以否定“生成策略RL已经由VINE统一解决”。

作者[项目页](https://agibottech.github.io/vine/)报告真机20分钟、20/20成功、19.2%干预步；页面还展示25分钟运行视频，不能把几个时间口径当成本项目总墙钟承诺。[公开仓库](https://github.com/AgibotTech/vine)目前所见为项目网页，不是可直接复现训练的官方learner。原文不同baseline的骨干、动作粒度和初始化也不同，不能把系统差距全部归因于采样器。

## 2. 应补齐的方法地图

| 路线 | 如何使用奖励 | 大模型计算与迁移判断 |
|---|---|---|
| [Diffusion-QL](https://arxiv.org/abs/2208.06193) | diffusion行为约束与Q最大化结合 | 直接生成策略RL早已有之；主要是离线RL证据，不能等同真机在线后训练 |
| [DPPO](https://diffusion-ppo.github.io/) | 将去噪视为内部决策过程，用policy gradient/PPO优化 | 更新生成器；不等于沿最终动作Q梯度做完整链BPTT。对当前flow策略需适配概率路径；单臂在线采样成本应单独考虑 |
| [ReinFlow](https://reinflow.github.io/) | 给flow路径注入可学习噪声，构造可计算转移概率 | 直接面向flow后训练；有公开代码，主要基准不等同SO101实机证据 |
| [FPO](https://flowreinforce.github.io/) / [FPO++](https://hongsukchoi.github.io/fpo-control/) | 用flow-matching损失构造策略比率代理，做优势加权策略优化 | 无需把精确动作密度算出来；原始FPO与改进版稳定性不可混称。主要按on-policy预算评估，不默认最适合单臂 |
| [FQL](https://proceedings.mlr.press/v267/park25f.html) | 学flow行为模型，约束一个通过Q更新的一步策略 | 避免直接优化完整flow链；官方[实现](https://github.com/seohongpark/fql)可复用作小策略对照，但不是原样更新WAM生成器 |
| [QAM](https://arxiv.org/abs/2601.14234) | 将critic动作梯度转为逐步adjoint匹配目标 | 保留表达性，避免不稳定的完整链反传；仍有采样及adjoint计算，不能称零成本。对大DiT应与VINE实测显存和更新延迟 |
| [SAC Flow](https://sac-flow.github.io/) | 从序列计算角度重参数化速度网络，再用off-policy RL | 与VINE解决的问题接近；修改速度架构可能影响现有ckpt兼容性，不是仅换采样器 |
| [DSRL](https://diffusion-steering.github.io/) | 冻结策略，优化输入噪声 | SAC噪声Q路线可把learner与大DiT解耦；NA路线另有大量生成器前向。专家动作不是噪声标签 |
| [HIL-SERL](https://github.com/rail-berkeley/hil-serl) / [Policy Decorator](https://github.com/tongzhoumu/policy_decorator) | 人类辅助off-policy学习／冻结基础策略的残差RL | 有助于先跑通数据和奖励闭环；输出端更新不证明WAM内部得到改进 |

DPPO、ReinFlow与VINE都能改变生成策略，但梯度估计、数据复用和探索方式不同。不能将“没有完整去噪链BPTT”误解为“不训练生成模型”。也不能将所有“flow + RL”论文当成预训练大WAM实机后训练验证。

## 3. VINE迁移到本仓库的具体推导

以下为本项目适配推导，尚未实现验证。原文的时间从噪声走向动作；仓库[连续调度器](../../../../src/ahawam/models/wan22/schedulers/scheduler_continuous.py)定义：

$$x_\sigma=(1-\sigma)a+\sigma\epsilon,\qquad u_\theta\approx\epsilon-a.$$

因此把VINE的重构与端点估计改写为：

$$\widetilde x_k=(1-\sigma_k)\widehat a_k+\sigma_k\epsilon_k,$$
$$\widehat a_{k+1}=\widetilde x_k-\sigma_k u_\theta(\widetilde x_k,1000\sigma_k;c).$$

这里1000应实际读取scheduler的num_train_timesteps；不能硬编码。$c$包含当前观测、proprio、指令、视频上下文及必要的历史/offset。不能使用加号，也不能把scheduler返回的0–1000 timestep直接当0–1插值系数。

先保留相同去噪调用次数、固定checkpoint与同一批状态，比较原Euler和改造采样器。原始均匀时间网格与本仓库shift网格另作对照，不同时改步数、网格、动作范围和训练算法。

上述改造不保证保留原checkpoint的动作分布。端点预测是一个估计器，反复重构并不自动等价于原ODE的概率输运；理论上的最优速度场解释也不能保证有限数据、近似网络及学习中critic的全局稳定性。必须先测采样器替换后的基础行为，避免把采样变化当作RL收益。

actor的一般目标可以写成：

$$L_\pi=-\mathbb E Q_\phi(c,G_\theta(c,\epsilon_{0:K-1})) + \lambda_{\rm BC}L_{\rm BC}.$$

这里BC选择需要明确：数据动作端点MSE、flow-matching监督和基础策略蒸馏不是同一目标，混合使用属于本项目扩展，不能统称原VINE复现。尤其不要把失败自主动作当专家BC标签；这些transition仍用于critic。

## 4. 应改哪部分，计算代价在哪里

代码锚点：[ahawam.py](../../../../src/ahawam/models/wan22/ahawam.py)的`_predict_action_flow_with_video_state`、`rollout_action_prior_only`和`_prepare_distill_video_state`。

现有rollout辅助函数和视频条件准备有`no_grad`，且输出会detach，不能直接拿返回动作训练actor。需要新的可微采样入口，并检查所有attention/cache计算是否保留需要的输入梯度；不能全局删除部署端no_grad。现有蒸馏准备接口也不自动等价于异步部署缓存，必须核对video刷新时刻、历史索引和action offset。

| 更新范围 | 梯度路径 | 建议 |
|---|---|---|
| 外部actor，冻结WAM | Q→小actor | 最快闭环对照；基础WAM推理仍耗时 |
| Action DiT的LoRA/部分层，冻结视频 | Q→多步Action DiT→可训练权重 | VINE用于现有WAM的首选研究版本；视频条件可无梯度准备，但所有动作步仍需输入梯度 |
| OVCR参数，冻结Action DiT | Q→多步Action DiT→OVCR | 少优化器状态，不代表少反向激活。OVCR位于上游，仍需穿过动作网络 |
| 全量video+action | Q→动作链→视频/路由 | 参数、激活、缓存重建与表征漂移代价最高；不作为首个实机实验 |

SFT通常随机取一个噪声时刻训练；VINE actor要展开多个动作网络调用。相同步数的推理延迟可能接近，但learner不能据此视为同样便宜。记录更新/秒、峰值显存、采集时延、参数发布频率和每条新transition对应的实际更新数。LoRA主要节省参数和优化器状态，不自动消除完整链的反向成本。

## 5. WAM特有的问题，比采样器更先决定能否学到

**条件状态必须可重建。** 相同当前图像可能对应不同历史、视频缓存和未执行动作。critic只看单帧时会混淆这些情况。第一版冻结视频表征，保存原始时序、当前观测和必要的缓存元信息，采用与actor一致的有限历史状态或循环critic；之后更新视频时不可长期使用旧版本特征冒充当前特征。

**实际执行动作才决定transition。** 生成长chunk但只执行前$h$步，不能用完整预测chunk和前$h$步的结果形成语义不一致的Q样本。若按chunk建模，状态转移与折扣应匹配实际执行长度：

$$y=\sum_{j=0}^{h-1}\gamma^j r_{t+j}+\gamma^h(1-d)\min_i\overline Q_i(c_{t+h},A').$$

接管中断时不要把自主半段与人类半段打包成“策略生成的同一动作”。可在控制权切换处拆分transition；实际人工动作入动作critic，另保留BC监督。若中断与完成有关，不能简单假装为普通时间截断，需明确状态、终止和控制模式定义。

**旧DAgger短片段不是现成RL数据集。** 离线TD需要下一状态、奖励、终止与实际执行动作。已有成功rollout中的错误动作可得到后续人类救回的回报，但这不等于该自主动作本身可靠；需要保留控制来源，使用明确的混合行为数据语义，并用无干预部署验证。

**视频预测不能自动作奖励或模拟器。** 当前证据已显示PSNR上升不必然带来power adapter成功率上升。也尚无任意候选动作条件下准确预测接触结果的验证，因此不建议用“想象插入成功”取代真实成功奖励。

## 6. 推荐顺序与停止条件

### P0：先用一天内可完成的软件审计筛选路线

- 固定一个任务和当前经过复测的最好checkpoint；不同算法必须共用起点，不能一个用SFT、另一个用DAgger最好模型。
- 从真实状态重建条件，检查原采样器与VINE改造后的动作幅度、限位、chunk连续性、重复噪声下的变化。不要以和单条人类轨迹的MSE代替成功率。
- 用合成可微Q检查梯度连通和符号，再以真实replay做learner性能测试；合成Q仅验证实现，不是RL有效性证据。
- 测多步Action DiT更新能否在现有GPU预算内持续运行。不预设三卡能同时放下两个模型副本；根据峰值显存选择actor服务与learner共享、分卡或分时。

### P1：相同数据闭环的最小比较

| 实验 | 用途 |
|---|---|
| 固定ckpt+原采样器 | 起点 |
| 固定ckpt+VINE采样器，无RL | 分离采样器本身造成的改变 |
| 小reference-conditioned actor或bounded residual RL | 检查任务是否能通过较便宜的真实奖励学习改善 |
| 冻结视频、Action DiT部分参数+原采样器value-gradient RL | VINE最关键的同架构对照 |
| 同上+VINE采样器 | 隔离采样路径对稳定性和成功率的影响 |

稀疏真实成功奖励、人工成功确认和必要接管均可保留；随机性首先来自策略内部采样，不额外叠加未经标定的随机关节运动。所有方法使用相同安全边界和初始化集合。评估禁用接管，并分开统计自主成功、接管步比例、机器人交互分钟与总墙钟时间。

建议先做单任务局部插入验证，再做完整任务；这与原先论文完整任务成功率不可直接比较。若VINE采样替换已使原策略明显退化，或多步更新吞吐不可接受，优先修复/放弃该路径，而不是直接增加真机采集。

### P2：值得形成WAM研究的问题

1. **时序上下文是否帮助critic分清恢复状态？** 固定actor输入和主干，对比当前视觉critic与带真实历史critic；控制网络容量和数据预算，不能同时改变actor的历史。
2. **奖励应更新动作层还是预测表征？** 比较冻结视频、更新路由、更新视频末端；保持采样器、奖励、交互预算一致，检查成功率与遗忘，而非只报PSNR。
3. **异步预测上下文过期是否限制RL？** 记录缓存年龄与控制延迟，先做分组诊断，再控制刷新频率验证；不能把旧缓存下的失败统称VINE不稳定。

这些是待验证的研究假设。仅“将VINE接到WAM”不足以自动形成方法贡献；需要显示WAM特有的记忆、预测或异步结构改变了RL的困难，以及对应设计带来的收益。

## 7. 实施参数的证据边界

原文附录给出的是以OGBench为主的设置（例如MLP、batch512、学习率3e-4），不能直接作为大DiT真机学习率。在线replay比例、UTD、奖励定义等未从公开实现完整确认，不宣称有一份现成实机复现配置。

本项目先沿用已有小actor/critic路线的可审计实现作为闭环基线。Action DiT参数学习率、BC强度和actor更新频率需通过同起点稳定性小试确定；不能靠critic值持续上升选模型。至少检查有限梯度、实际自主成功、动作分布漂移与基础行为保持。在线有效性必须通过新rollout确认；完整有奖励的历史数据可用于离线预热，但不保证离线更新就能提升。
