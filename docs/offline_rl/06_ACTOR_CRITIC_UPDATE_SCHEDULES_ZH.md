# Actor、critic 怎样更新：在线/离线顺序、频率与代码审计

> 2026-10-24 本轮已选择 skip_hold＋自然完整尾部、SFT 006572 和直接 KV attention；当前实现以 [08](08_SYSTEM_IMPLEMENTATION_ZH.md) 为准，运行与资源见 [09](09_PARAMETERS_AND_RESOURCE_REPORT_ZH.md)。本文保留专项分析/早期原型记录。

核查日期：2026-10-24。补充用户关于具体梯度更新与训练频率的问题。本轮读取论文、作者配置及 learner/runner 循环，未改变本项目训练行为或启动机器人。

## 1. 先区分五个时钟

| 计数 | 含义 | 不能混淆成什么 |
| --- | --- | --- |
| control tick | 给机器人一个低层目标 | 不一定对应一次 RL 决策 |
| environment/RL decision | 提交一个动作块并得到结果 | C=8 不代表做了 8 次梯度更新 |
| replay record | 存入一条训练转移 | 滑窗可从一个执行块生成多条相关 record |
| optimizer step | 一次参数更新 | 双 Q 的两个头不自动计为两次 critic step；microbatch 累积也不等于多次 optimizer step |
| weight publish | learner 把参数发给执行策略 | 与 learner 每次更新、机器人每步控制都可以不同频 |

另一个独立数量是 flow/diffusion 的去噪步数：VINE 展开 10 次网络调用后反传一次，仍可以只做 **一次 actor optimizer step**，并非 actor 连续训练 10 步。

建议日志明确记录：新增原始 tick、非重叠决策、增广 replay record、Q/V/actor optimizer step、target EMA 次数和广播次数。不把所有数量都叫 step。

## 2. Critic 与 actor 每次究竟在学什么

以外部动作块 TD3 风格模型为例。状态 h 包含冻结视觉表征与当前本体状态，数据动作 A 是实际执行块，折扣 d 已含终止及实际持续时间：

$$
y=\operatorname{sg}\{R+d\min_iQ_{\bar\phi_i}(h',\pi_{\bar\theta}(h',A'_{ref})+\epsilon)\},
$$
$$
L_Q=\sum_i\mathbb E[(Q_{\phi_i}(h,A)-y)^2].
$$

一次 critic 更新对这项 loss 反传，更新 Q 的参数 phi；target、next-action 生成不因本次 critic loss 被训练。两个 Q 通常在同一 critic optimizer step 中共同更新。

actor 则产生自己的动作 A_pi，而非把数据动作直接拿来最大化：

$$
L_\pi=-\mathbb E[Q_\phi(h,A_\pi)]+\lambda L_{behavior},
\qquad A_\pi=\pi_\theta(h,A_{ref}).
$$

actor 更新时固定 Q 参数，但保留 `dQ/dA`，让梯度经过动作进入 actor。对 actor 输出 `detach()` 或把 Q 前向放在 `no_grad()` 中会切断这个价值梯度。Q 本身在这一步充当可微评价器，不会因 actor 想得高分就一起被更新为更高值。

behavior loss 的目标依算法不同：RLT 主要锚定 WAM reference，TD3+BC 锚定数据动作，SAC 还有 entropy/temperature；不能把它们统称为同一个 actor loss。

IQL 有额外的 V，actor 机制也不同：

$$
L_V=\mathbb E[\rho_{\tau}(\min_iQ_{target,i}(h,A)-V(h))],\quad
y=R+dV(h'),
$$
$$
w=\operatorname{sg}[\min(e^{\eta(\min_iQ_{target,i}(h,A)-V(h))},w_{max})],\quad
L_\pi=\mathbb E[w\,L_{BC}(\pi,A)].
$$

IQL 的 Q/V 更新不需要查询当前 actor 的数据外动作。actor 通过 detached advantage 权重学习实际动作，**不是对 -Q(h,pi(h)) 直接反传**。本项目固定方差均值 actor 使用 weighted MSE；原版实现使用优势加权 log-likelihood。

## 3. 在线训练的更新比率与数据复用比率

以下两个参数分别回答不同问题：

$$
G_Q=\frac{\text{critic optimizer updates}}{\text{新增指定单位的数据}},
\qquad
r=\frac{\text{critic updates}}{\text{actor updates}}.
$$

例如新增 10 条 RL transition，做 50 次 critic、25 次 actor，则 Q-UTD=5、actor-UTD=2.5、Q:A=2:1。每个 gradient batch 可以包含大量旧数据；UTD=5 不代表 batch size=5，也不代表 Q:A=5:1。

若每条 transition 是 8 tick 的非重叠块，5 次 Q 更新/块相当于 .625 次/tick；若又按 stride=2 保存重叠窗口，按 replay record 归一的数值还会变。比较论文中的 UTD 必须先核对分母。

典型离策略过程是：

```text
采集线程：使用执行端的 actor → 执行 → reward/next_obs → replay
训练线程：抽 B1 → Q 更新
          抽 B2 → Q 更新
          用 B2 或另抽 B3 → actor 更新
          按独立频率更新 target / 发布参数
```

因此“交替”可以是每小批交织、一次 Q-only 接一次 Q+A、组合 loss 同时更新，也可以按阶段冻结一方；不要求机械臂停下来等每一轮梯度完成。一次 actor 更新之后通常也不等待 Q 在新策略下完全收敛。

## 4. 各工作的实际训练循环

表中的比例统计对应已核查源码/配置，区别于论文中简写的一轮。Q 更新指整个 Q ensemble 一次更新，不按 Q 头数翻倍。

| 工作 | 一轮 learner 的顺序/频率 | 在线/离线与证据范围 |
| --- | --- | --- |
| RLT | 论文明确 Q:A=2:1，报告 UTD=5 | 在线异步采集与训练；未公开足够源码核定所有物理 tick/滑窗 record 的 UTD 计数细节 |
| HALO-WA online | 通常每子轮 Q 一次，每 2 子轮 A 一次；`run_training` 默认执行 5 个子轮 | `utd_ratio=5` 在异步代码里是每次 learner 调用的循环数，不是自动限制为每条新 transition 恰好 5 次 |
| HALO-WA offline helper | 每 outer step 做可配置的 k_Q 次 Q，再 k_A 次 A | 默认 k_A=1，k_Q 回落到 `critic_actor_ratio`，常见为 2；每次另抽 batch |
| IQL 原版 | V → actor → Q → target-Q，每轮各 1 次 | 离线随机 batch；在线继续训练也能复用其更新函数；无标准的 2:1 要求 |
| TD3+BC 原版 | 每调用 Q 一次，每 2 次调用 A 一次；target actor/Q 在延迟分支更新 | 固定数据，默认 `policy_freq=2` |
| FQL | 一次组合 loss 更新 Q、flow BC policy、一步 RL policy，随后 EMA target Q | 作者循环在线/离线都调用该 update；相应模块各一次，并非串行先 Q optimizer.step 再 A |
| QC-FQL | 与 FQL 类似；一个 update 的组合损失含 critic、flow BC、一步 policy | offline 每 batch 一轮；online 将 UTD 个 batch 用 scan 连续更新，默认 UTD=1 |
| VINE | Algorithm 1：一轮 Q，再一轮 actor；一次 actor 更新可展开 K=10 的采样链 | 论文算法是 1:1 示意；缺完整训练代码，不把它说成所有真机实验的精确调度 |
| ConRFT | 默认 `cta_ratio=2`：1 次 Q-only，再 1 次 Q+A | offline 和 online 均如此；offline 用 Cal-ConRFT，online 换 HIL-ConRFT loss |
| HIL-SERL | 默认 `cta_ratio=2`：1 次 Q-only，再 1 次 Q+A/temperature | 异步真机训练；本轮查的 RAM/USB 配置支持该默认，不能代替全部历史版本 |
| RLPD 原版 | 一个 update 内先 U 次 Q，再 1 次 A/temperature | 作者 README 的若干命令 U=20，因此是 20Q:1A；程序参数默认 U=1，不能忽略启动命令 |
| AWAC / RLkit | 默认每个 train batch 更新 Q ensemble、actor 各一次 | policy/q/target update period 默认都为 1，可配置；Q1/Q2 使用各自 optimizer，表中按 ensemble 更新一次计 |
| DSRL-SAC 示例 | 每个 vector-env step 后 `gradient_steps=20`；默认每步 Q 和 actor 各一次 | Q:A=1:1，但每次触发约 20:20，不是 20:1；按 vector env 与 chunk 定义解释 UTD |
| DSRL-NA 示例 | 同上 20 次动作 Q＋20 次噪声 actor，另做 10 次 noise-Q 蒸馏 | 动作 Q、噪声 Q 是两种功能，不是普通 twin-Q 两头的计数 |
| Cal-QL 作者默认 | 一次 `.train()` 更新 policy、Q1/Q2 和相应系数，再更新 target-Q | offline 默认 1000 epochs×1000 次；online 默认按新轨迹长度×UTD=1 批量训练 |
| V-GPS | 离线 trainer 每个 batch 调一次 agent.update；SAC 派生分支默认 A/Q/temperature 同轮 | 小 actor 是价值学习的训练配套；部署是用 learned Q 排序外部基础策略候选，不是在线更新这些大策略 |
| DPPO | 收集一批 on-policy rollout → GAE/return → 多 epoch/minibatch；暖机后每 minibatch A、V 各一步 | on-policy PPO 路线，不能直接套任意旧 replay 的高 UTD 配方 |

### 4.1 RLT：论文报告与算法示意要分开

[RLT §V](https://arxiv.org/html/2604.23073v1) 明确写两次 critic 对一次 actor，以及 UTD=5；伪代码把 update 压缩成每轮写一次 critic、一次 actor，不能仅按伪代码行数推翻正文频率。它在 warmup 后才开始学习，并允许 reference/action chunk 的中间滑窗增强；不据这些信息自行推断所有实现单位。

### 4.2 HALO：utd_ratio=5 和 weight_sync_interval=20 的实际分母

本地干净 checkout `611008193fd3438bf99cc8f8377858fbdfcb9555`：

- [bell 配置](https://github.com/YeanRoot/HALO-WA/blob/611008193fd3438bf99cc8f8377858fbdfcb9555/rlinf/examples/embodiment/config/online_rl_bell_ansyc.yaml) 是 `utd_ratio=5`、`critic_actor_ratio=2`、`target_update_freq=1`、`tau=.005`、`weight_sync_interval=20`。
- [`update_one_epoch`](https://github.com/YeanRoot/HALO-WA/blob/611008193fd3438bf99cc8f8377858fbdfcb9555/rlinf/rlinf/workers/actor/fsdp_gigawa_policy_worker.py#L2595) 每子轮抽一个 global batch，Q/A 可以复用该 batch；microbatch 只是 gradient accumulation。A 是否更新还受 warmup、冻结/训练阶段和 buffer readiness 控制。
- [`async runner`](https://github.com/YeanRoot/HALO-WA/blob/611008193fd3438bf99cc8f8377858fbdfcb9555/rlinf/rlinf/runners/async_embodied_runner.py#L168) 持续调用 learner，数据接收另有线程；它不是每拿到一个新块才允许执行一次该调用。
- 所以 20 个正常 learner 调用之间大约已有 100 次 Q、50 次 A，然后才广播一次。实际数据复用比仍取决于 rollout/learner 吞吐；冻结或未就绪时另算。
- [`offline helper`](https://github.com/YeanRoot/HALO-WA/blob/611008193fd3438bf99cc8f8377858fbdfcb9555/rlinf/rlinf/workers/actor/fsdp_gigawa_policy_worker.py#L1264) 可以设两个独立循环次数；它的 epoch 也不等于原视频播放一次。

这些是公开仿真代码的具体配置与行为，不宣称论文所有真机运行完全一致。

### 4.3 HIL-SERL / ConRFT：2:1 来自 n-1 次 Q-only，再 Q+A

[HIL-SERL learner](https://github.com/rail-berkeley/hil-serl/blob/main/examples/train_rlpd.py#L314) 和 [ConRFT learner](https://github.com/cccedric/conrft/blob/main/examples/train_conrft_octo.py#L322) 均按 `cta_ratio-1` 次 critic-only，最后再调用一次包含 critic/actor 的 update。因此 `cta_ratio=2` 总计是 **2Q+1A**，不是“2Q-only 再加 1Q+1A”。

默认值分别见 [HIL-SERL config](https://github.com/rail-berkeley/hil-serl/blob/main/examples/experiments/config.py)、[ConRFT config](https://github.com/cccedric/conrft/blob/main/examples/experiments/config.py)。两者还通过 `steps_per_update=50` 控制 learner 到执行端的参数广播；这不是每 50 次机器人动作才训练一次。

ConRFT 的 offline 外层默认 `pretrain_steps=2000` 不等于只做 2000 次 Q；每个外层还含上述子更新，确切次数也受代码 inclusive range 与恢复位置影响。online 每 batch 混合 demonstration 与 replay，loss 与 offline 阶段不同；更新比一样不代表算法一样。

### 4.4 IQL：没有必要把 TD3 的 actor delay 当成标准

[作者 `learner.py::_update_jit`](https://github.com/ikostrikov/implicit_q_learning/blob/master/learner.py#L25) 每次顺序是：

```text
target Q → 更新 V
target Q + 更新后的 V → 更新 actor
更新后的 V(next) → 更新 Q
EMA 更新 target Q
```

比例 V:Q:A=1:1:1。由于 Q/V 不依赖 actor 当前生成动作，也可以先学习 Q/V，再做优势加权 policy extraction；这在机制上可行，但不等于原作者每轮更新的训练过程完全相同。只有有限数据时，不能预设 Q/V 必须先训练到收敛，收敛的训练误差也不证明价值准确。

### 4.5 FQL / QC-FQL：同一步优化，不等于相互乱传梯度

[`FQL.total_loss/update`](https://github.com/seohongpark/fql/blob/main/agents/fql.py#L95) 和 [`QC-FQL`](https://github.com/ColinQiyangLi/qc/blob/main/agents/acfql.py#L110) 合并 Q、flow BC 与 distillation/Q actor loss 后调用一次 apply_loss_fn。这可以理解为各参数组同轮更新；actor 的 Q 梯度读取与 critic 参数优化仍有明确计算图边界，不能把一个总 loss 理解为让 critic 迎合 actor 抬高分数。

FQL 的 flow velocity 主要由 BC loss 学行为分布，一步 actor 由 Q 与 distillation 更新；这三个模块的更新不能只算成“一个普通 Gaussian actor”。QC 原始 best-of-N 分支的 policy improvement 还可体现为候选选择，不一定有一步 actor 的直接 Q 梯度。

### 4.6 DSRL：UTD=20 与另一个 noise critic

核查 [DSRL can 配置](https://github.com/ajwagen/dsrl/blob/main/cfg/robomimic/dsrl_can.yaml)、[入口](https://github.com/ajwagen/dsrl/blob/main/train_dsrl.py)、固定子模块 `10e5d31` 的 [SAC](https://github.com/ajwagen/stable-baselines3-dsrl/blob/10e5d31/stable_baselines3/sac/sac.py#L203) 与 [DSRL](https://github.com/ajwagen/stable-baselines3-dsrl/blob/10e5d31/stable_baselines3/dsrl/dsrl.py#L221)：

`train_freq=1`、`gradient_steps=20`，未覆写的 `actor_gradient_steps=-1` 表示 actor 也在全部 20 个 gradient iterations 中更新。DSRL-NA 再单独执行 `noise_critic_grad_steps=10` 次蒸馏；它不是把环境成功 reward 再算 10 次。vector env 数量和 wrapper 的 act_steps 影响“新增数据”的单位，不能直接把这个 20 套在 SO101 tick 上。

### 4.7 Cal-QL / V-GPS / DPPO：三个不同的外层循环

[Cal-QL `.train`](https://github.com/nakamotoo/Cal-QL/blob/main/JaxCQL/conservative_sac.py#L317) 对对应参数组各应用一次梯度，并更新 target Q；[main](https://github.com/nakamotoo/Cal-QL/blob/main/JaxCQL/conservative_sac_main.py#L199) 在线可以先收一条完整轨迹，再按其长度乘 online_utd_ratio 训练。在线并不要求每个环境 step 后立刻更新。

[V-GPS trainer](https://github.com/nakamotoo/V-GPS/blob/main/experiments/train.py#L231) 每 batch 调 agent.update；agent 可以由配置切换，不能把某个 agent 提供的 high-UTD helper 当成全部实验的实际调用。比如其 SAC helper 支持先多个 Q 后一个 actor，但默认 `.update()` 与 `.update_high_utd()` 是不同入口。

[DPPO learner](https://github.com/irom-princeton/dppo/blob/main/agent/finetune/train_ppo_diffusion_agent.py#L308) 对本轮 rollout 多次 epoch/minibatch 更新，暖机前仅执行 critic optimizer，之后每个 minibatch 通常 actor/critic 各一步，并可因 KL 提前停止。核查的 [can MLP 配置](https://github.com/irom-princeton/dppo/blob/main/cfg/robomimic/finetune/can/ft_ppo_diffusion_mlp.yaml) 为 `n_steps=300`、`update_epochs=10`、`n_critic_warmup_itr=2`。这里的 10 是对 rollout batch 的 epoch 数，不能理解为每个机器人动作更新 10 次 actor，更不是允许任意旧 DAgger replay 的离线方案。

### 4.8 RLPD / AWAC：高 UTD 不必同时多训 actor

[原版 RLPD `update`](https://github.com/ikostrikov/rlpd/blob/main/rlpd/agents/sac/sac_learner.py#L255) 将大 batch 分 U 份，连续做 U 次 critic，然后仅用最后一个 mini-batch 更新一次 actor/temperature。[作者 README](https://github.com/ikostrikov/rlpd) 的 locomotion/antmaze 等命令取 U=20，所以该例是 20Q+1A；而 DSRL 示例 `gradient_steps=20` 默认是 20Q+20A。**同样写 UTD=20，不保证 actor 频率相同。**

[AWAC 的 RLkit trainer](https://github.com/rail-berkeley/rlkit/blob/master/rlkit/torch/sac/awac_trainer.py#L717) 默认 `q_update_period=policy_update_period=target_update_period=1`，每 batch Q1/Q2 各一步、actor 一步，再按 period 更新 target。actor 是优势加权行为学习，其 Q backup 仍与策略相关，不能因为 actor loss 类似 weighted BC，就把它与 actor-independent 的 IQL Q/V backup 等同。实验可覆写 period 和外层训练/采集次数，这里只报告已核查的默认 trainer。

## 5. Target network 的更新是第三件事

target 一般不通过自己的 optimizer 学习，而做 EMA：

$$
\bar\phi\leftarrow(1-\alpha)\bar\phi+\alpha\phi.
$$

这里 alpha 是 EMA rate；不要与 IQL expectile 的符号 tau 混淆。EMA 频率与其系数共同决定跟随速度：每次 Q 后 .005 和每两次 Q 后 .005 不相同。

- [原 TD3+BC](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py#L136) 在延迟 actor 分支同时更新 target actor 和 target Q。
- 原 IQL 每次 Q 更新后更新 target Q，没有 target actor 参与 Q backup。
- HIL-SERL/ConRFT 在包含 critic 更新的调用后更新 target；HALO 有单独 target_update_freq。
- actor 广播到机器人不是 target network 更新；机器人通常执行 actor，target 网络主要用于 learner 内部构造稳定目标。

## 6. 在线与离线：不同的是数据流和约束，不是强制的更新顺序

离线：固定 D，反复采 batch，按设定 Q/V/A 计划更新；没有新的环境交互作为 UTD 分母。应报告 N_Q/N_V/N_A、batch size、数据集大小和终止样本覆盖。粗略样本重用量可用 `N_Q * B / |D|` 比较，但带 replacement、来源混采和重叠窗口时不等于精确 epoch。

在线离策略：D 持续扩展，采集与学习可同步或异步；可每步更新，也可每段/episode 后集中更新。建议报告实际 `ΔN_Q / ΔN_new_decisions`、`ΔN_A / ΔN_new_decisions` 和 actor staleness，而非仅记录配置里叫 utd 的变量。

离线更需控制数据外动作误差和过拟合；在线多做更新也可能让 actor 利用 Q 漏洞。增加 critic 更新数不是自动获得正确 critic；一个只预测 0 的 Q 在稀疏任务也可能有很低平均训练 loss。应联合看终止奖励覆盖、held-out 指标、Q 分歧/越界、actor 偏离、IQL 权重 ESS 及实际任务结果。

## 7. 我们当前代码到底怎样更新

当前 [learner.py](../../src/ahawam/offline_rl/learner.py) 的真实行为：

| 模式 | 每次 update | actor 频率 | target |
| --- | --- | --- | --- |
| IQL | 先 V，再 Q | 默认 actor_delay=2，每两次一次 | target-Q 每次更新 |
| TD3+BC | 先 Q | 默认每两次一次 | target-Q 每次、target-actor 每两次 |
| BC/actor_only | 只更新 actor | 每次 | 不作本次常规 RL target EMA |

两个应明确的适配差别：**当前 IQL 的 2:1 是本项目候选选择，不是 IQL 原版要求**；当前 TD3+BC target-Q 每次 EMA 也与作者的延迟分支更新不同。这些应作为可配置实验变量，不能称为逐行复现。

当前 CLI `bc_warmup=200` 是先进行 200 个 actor-only 数据 BC 更新，Q/V 没有在这 200 步一起预热。它不是“critic warmup”。恢复时延续全局 step 和该阶段计划。不要用这个参数实现先训 Q 再训 actor。

本轮在 CPU 给实际 optimizer 注册 post-step hook，并记录 target 调用，连续调用 4 次，得到：

| 配置 | A steps | Q steps | V steps | target-A EMA | target-Q EMA |
| --- | ---: | ---: | ---: | ---: | ---: |
| 当前 IQL delay=2 | 2 | 4 | 4 | 0 | 4 |
| IQL delay=1 对照 | 4 | 4 | 4 | 0 | 4 |
| 当前 TD3+BC delay=2 | 2 | 4 | 0 | 2 | 4 |

使用合成小 batch 验证调用，不是新的机器人训练结果。原有 16 项测试结果不被重新计为本轮新增实验。

## 8. 对本项目的具体建议

**离线 IQL 主线**：以 V/Q/A 各 1 次/learner batch 的原版频率建立清楚的基线，即 actor_delay=1；再比较 delay=2。保留可选的外部 actor BC 初始化，Q/V 预热应另定义阶段，不能与 BC warmup 混称。无需因为在线 RLT 是 2:1 就强制 IQL 也是 2:1。

**离线 TD3+BC 对照**：Q:A=2:1，明确 target-Q 是否也跟随延迟。先保证 BC 能表达已有行为，再小规模比较价值梯度；不要在不可靠的 Q 上做很多连续 actor-only 最大化更新。

**后续在线**：保留离线初始化的模型，先固定表示、动作块、hold 和 reward 合同。以明确的新 RL 决策计数预算 Q/A，比如候选 2Q+1A/新增决策，观测实际比率后再试更高 UTD。该数值是 SO101 的起始实验建议，不是论文已证实的最优值；参考 RLT 的 5 或 DSRL 的 20 时必须同时对齐单位和更新内容。

IQL 本身可以继续吸收在线数据。如果改切 TD3/SAC-style actor 更新，则 Q 的目标与 actor loss 已变化，应显式记录切换、验证价值尺度，并重新约定探索与 replay 比例；不能把它当成只修改频率的同一实验。

本轮不修改当前 learner 默认值，以保留已有候选实验的可比性；上述建议供最终配方讨论。

## 9. 本轮源码标识

RLT/VINE 频率结论来自公开论文描述，不宣称取得其完整真机 learner。其余核查文件 blob：

| 文件 | SHA |
| --- | --- |
| IQL learner.py | `e21ba9ac1258066fd8fc29ea36a561125b32d1c7` |
| TD3_BC.py | `36e3ffa0990d90d1e75b9066f6894f9c478f35ea` |
| FQL agents/fql.py | `efcef114aaf3e66151e6a40b294101af5c8bd7a3` |
| QC agents/acfql.py | `291acd830f4a652e8c2d566449e5eb42673412ae` |
| HIL-SERL train_rlpd.py | `6b33492097a3e2d2790f105028a9289b5a17ebdd` |
| HIL-SERL experiments/config.py | `efb6fcc1b4bfbae878290aa9df6cd76c9ace5e82` |
| ConRFT train_conrft_octo.py | `b1c63170e06878d741a70a7b39365cd7c27ad61e` |
| ConRFT experiments/config.py | `3c385bed55095d850a628132ebf90bd79db9fb69` |
| DSRL fixed-submodule dsrl.py | `77895cd3ecbaadd811cad4832c6cd63c9fcdb946` |
| Cal-QL conservative_sac.py | `281110589f3502fa59e2f996f5a8090da7c90349` |
| V-GPS experiments/train.py | `ce8feef474ab9e2948684a888cd49f2ed0380b31` |
| DPPO train_ppo_diffusion_agent.py | `998c638ee2107ce042a84f062a5fea38bfa9b010` |
| RLPD sac_learner.py | `90b8413506aa96368df840153bd6a69620e3635b` |
| AWAC RLkit awac_trainer.py | `56feeb4f21fb404e6fbb45da5fe1b364337fd1d5` |

这里标识本轮读取的版本，不将某个 default 配置外推为全部任务、所有作者后续版本的统一超参。
