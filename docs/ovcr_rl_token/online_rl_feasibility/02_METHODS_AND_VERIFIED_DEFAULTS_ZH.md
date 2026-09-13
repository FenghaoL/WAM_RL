# 相关方法与公开实现中的训练细节

## 1. 证据层级

“论文在另一台机器人上有效”“公开代码里有这套更新”“在本仓库实际跑通”是三个不同层级。本报告核对前两层，AHA-WAM 在线 RL 尚无第三层证据。

以下数值按具体配置文件记录，不能拼成一个所谓“统一最优超参数”。仿真的大量并行环境、Franka 的笛卡尔控制、SO101 的关节控制和不同控制频率，会改变探索、折扣率及实际 update-to-data ratio。

## 2. RLT：读出状态，冻结大模型，更新外部动作策略

RLT 先用 encoder/decoder bottleneck 重建 VLA 内部表征，获得供 RL 使用的读出。在线阶段冻结 VLA 和读出，以表征、proprio 及参考动作块为条件训练小策略；参考动作约束和 dropout 用于保留基础行为又允许改进。人工干预是可选项，不是 RL 定义上的必要条件。[^1]

论文具体报告：动作块 C=10，50 Hz 控制，Gaussian actor 使用固定小方差；小任务 MLP 为两层 256，螺丝任务为三层 512；双 Q 采用 TD3 风格 target，训练参考动作 dropout 为 50%；成功给 +1，chunk TD 使用块内折扣与 γ^C。[^2]

这些机制与“接在输出端”的路线直接相关，与目前 T4 写入 video K/V 的机制不同。对 AHA 的 readout，可以读取 OVCR 当前观测/路由特征，**作为 actor 的状态输入**；不必让 RL 梯度从 actor 反向进入 OVCR。

RLT 项目报告的小策略更新速度不能视作 AHA 推理吞吐。它的 Ethernet 例子中，15 分钟机器人经验对应约两小时总训练墙钟，复位等开销占了相当部分。比较本项目时必须同时记录机器人实际运行时间和总人时。[^1]

### 已有公开复现，不能再笼统地说“没有实现可参考”

截至本次检索，RLinf 有 RLT Stage 2 代码和实机配置。它是 **RLinf 团队的公开实现**，不是据此确认的原作者全部实验代码。原论文没有明确给出的学习率、batch、σ 数值，不能从复现配置反向声称是原论文原值。[^3]

固定版本：
`RLinf/RLinf@c94c67c12c27547ad6244d3957eb473955c7d46d`。

文件 `examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml`：

| 字段 | 配置值 | 解释 |
|---|---:|---|
| micro/global batch | 256 / 256 | 小策略 batch，不是 VLA SFT batch |
| actor / critic LR | 3e-4 / 3e-4 | FP32 小网络 |
| gradient clip | 10 | actor、critic |
| fixed_std | 0.002 | 所用动作坐标中的标准差；不是 SO101 角度 |
| Q / BC 权重 | 0.1 / 5 | actor = -0.1 Q + 5 BC |
| reference dropout | 0.5 | 作用于 actor 输入 |
| γ / τ | 0.96 / 0.005 | 代码对 chunk 奖励和 bootstrap 使用时间折扣 |
| critic_actor_ratio | 4 | learner 中每四次更新机会更新一次 actor |
| train_actor_steps | 2 | replay readiness 门槛，**不是每步做两次 actor 更新** |
| update_epoch | 8 | 不是按真实新 transition 定义的 UTD |
| reference / actor horizon | 20 / 10 | 基础参考与 RL 执行块长度不同 |
| target entropy / α | 固定 α=0 | **不是标准最大熵 SAC** |
| replay cache_size/min_buffer_size | 200 / 2 | 框架 replay 单位；不可当成 200/2 条 SO101 transition |

实际 `fsdp_rlt_ac_policy_worker.py` 的 BC target 在 intervention 位置换成人工动作，其他位置取基础 reference。target 使用双 Q 的 min；配置中的 actor 聚合为 q1。读到 config 名字里出现 SAC，并不意味着可以用标准 SAC 熵项解释这里所有 loss。[^4]

迁移建议：把它作为“外部 reference-conditioned 动作 actor”的可审计对照；第一版不把 RLinf 整套分布式框架搬进当前 server。可借鉴 transition 和 loss，先保留现有部署链路。

### 如何理解RLT的消融结果

RLT报告去掉BC约束、参考输入或读出后性能下降，但其单步对照同时更换了视觉encoder，不能把差距全部归因于chunk长度。DSRL对照也用了跨50步重复的32维噪声；HIL-SERL对照缺少原系统的动作边界，并在更高频率运行。因此这些结果不能证明SO101上的逐步有界residual或完整48维噪声控制一定较差。[^2]

本项目若因延迟改用逐步residual，要承认它会增加稀疏奖励信用分配的难度，并先缩短任务阶段；不能一边更换视觉反馈、一边把成功归因于actor算法。

## 3. HIL-SERL：实机数据闭环与混合 replay 的主要依据

HIL-SERL 通过示教、实时接管、奖励和 off-policy 更新做真实机器人学习。它对本项目最有用的部分是完整 transition、干预动作入库、demo/online 双 buffer、异步 actor/learner 和成功信号，而不是某个机器人专用控制器。[^5]

固定版本：
`rail-berkeley/hil-serl@c32939bccb65f3b8c43a9f9add3d322d4ab0264a`。

| 来源文件 | 已核对设置 |
|---|---|
| `examples/experiments/config.py` | batch 256；cta_ratio 2；training_starts 100；steps_per_update 50；discount 0.97 |
| `examples/train_rlpd.py` | 每次 batch：128 demo + 128 online；先一次 critic-only，再一次 critic+actor+temperature |
| `serl_launcher/.../utils/launcher.py::make_sac_pixel_agent` | 两层256、tanh、LayerNorm；双 critic；初始温度0.01；backup_entropy=False |
| `serl_launcher/.../agents/continuous/sac.py` | 默认 actor/critic/temperature LR均3e-4；target τ=0.005；target entropy默认 -动作维数/2 |

这里的 `steps_per_update=50` 是 learner 发布网络参数的间隔，不是 50 个实机动作后才优化。异步 learner 可以在同一批环境数据上更新很多次；CTA=2 仅指 critic 与 actor 的更新比例，**不等于 UTD=2**。[^6]

还有一个易忽略的实现差异：上述 HIL-SERL builder 关闭 target entropy backup；actor 使用 critic ensemble 的均值，target 使用保守聚合。下文给出的标准 SAC 双 Q min 公式是另一明确选择，不能宣称与 HIL-SERL 每一行完全相同。[^7]

本项目可以先采用相同的 50/50 replay 结构，但前提是 demo 内有真实 reward/next-state。仅有 HUMAN 标签的短截段，不足以自动成为 HIL-SERL 所需的完整 RL transition。

## 4. Policy Decorator：有界 residual 与逐步启用探索

Policy Decorator 冻结基础策略，训练 residual，以动作幅度限制和渐进启用来稳定在线优化。它提供“少量参数、少量偏移、保留基础行为”的直接依据；算法并不要求修改基础 diffusion 权重。[^8]

固定版本：
`tongzhoumu/policy_decorator@92ba9ba442587ae5989c286355ace2200a0537fb`。

具体是状态输入的 ManiSkill diffusion 示例 `online/pi_dec_diffusion_maniskill2.py`，不是所有视觉/实机实验的统一配置：

| 参数 | 值 |
|---|---:|
| actor / Q LR | 1e-4 / 1e-4 |
| batch | 1024 |
| γ / τ | 0.97 / 0.01 |
| initial α | 0.2，自动调节 |
| learning_starts | 8000 |
| UTD / train frequency | 0.25 / 64 |
| 并行训练环境 | 16 |
| residual scale | 0.1 |
| progressive exploration horizon | 100,000 |
| action horizon / DDIM steps | 4 / 4 |
| critic input / actor input | sum / obs |
| grad norm clip | 50 |

默认 actor 并非一定输入基础动作；该脚本提供 `obs_base_action` 选项。将参考动作加入 AHA residual actor 是本项目的明确设计，而非误读这个默认值。这里也不是“128 demo +128 online”的实现；这个 batch 结构来自 HIL-SERL/RLPD。[^9]

本项目不照搬 8000 次随机探索或 100,000 步渐进期。单臂实机交互和并行仿真预算不同，第一阶段应从零 residual 的可靠基础行为采集，而非随机关节运动。

## 5. DSRL-SAC 和 DSRL-NA：必须分开评估

DSRL 把冻结 diffusion 的初始噪声当作 RL action。噪声 actor 与 critic 可用常规 off-policy 更新；决定本项目计算成本的，是使用噪声 Q 还是同时学习动作 Q。[^10]

### DSRL-SAC

经验为 (h,w,r,h')，Q_w 直接评价 w。learner 不调用 diffusion；每次真实动作生成时才调用基础策略。旧 demo 没有 w，因此不能直接放进同一个噪声 critic 的 action 字段。

### DSRL-NA

先以关节动作经验训练 Q_a，再用冻结生成器将噪声映射为动作，蒸馏出 Q_w。actor 仍通过 Q_w 更新；**没有反传 diffusion，但有大量 diffusion 前向**。

固定版本的 `DSRL.train` 顺序是：

1. 从动作 replay 采样。
2. 采样下一状态噪声，用冻结 diffusion 得到下一动作，构造 Q_a 的 TD target。
3. 更新 Q_a；按配置通过 Q_w 更新 actor；更新 target Q_a。
4. 完成这组 gradient_steps 后，另做 noise_critic_grad_steps 次 Q_w 蒸馏。
5. 蒸馏噪声来自标准 Gaussian `randn`，并非只来自当前 actor；decoder 在 no-grad 中运行。

因此“每个 TD 更新之后立刻一次噪声蒸馏”也不符合这个版本代码。actor 使用的噪声 Q 可能在该训练调用后半段才更新。[^11]

| `cfg/robomimic/dsrl_can.yaml` | 已核对值 |
|---|---:|
| 默认算法 | dsrl_na |
| actor / critic 学习率 | 3e-4，共用 SB3 learning_rate |
| batch / τ / γ | 256 / 0.005 / 0.99 |
| 三层 MLP宽度 / LayerNorm | 2048 / 开启 |
| critic数 / target聚合 | 2 / min |
| noise box | [-1.5,1.5] |
| 自动温度 / target entropy | 开启 / 0.0 |
| train_freq / gradient_steps | 1 / 20 |
| noise critic更新数 | 10 |
| n_envs | 4 |
| 预填充 | 1501 次 vector rollout step |
| load_offline_data | False |
| 基础 denoise | DDIM 8步 |

这个 `utd:20` 被传给每次调用的 gradient_steps；每个 vector step 来自4个环境，因此按“每条新 transition 的梯度更新数”计算约为5，而不是20。NA 的10次噪声蒸馏还应单独记账。初始化虽然构造器 `learning_starts=1`，外层先预填充1501步；不能说算法仅收集一条数据就启动。[^12]

代码中还有需要重新适配的环境细节：`ActionChunkWrapper` 的 chunk 内循环及超时 done 语义不适合作为真实机器人规范照抄。真实 SO101 必须能中断、记录实际执行长度，并区分环境终止与时间截断。

结论：**AHA 首先试 DSRL-SAC，不能因为 NA 常被推荐更省交互就忽略其巨大的 learner 前向成本。** 固定视频条件下仅存小 feature，足够训练 Q_w，不足以恢复任意 h 上的完整 AHA decoder；NA 还需要可重放的条件缓存或有序历史重建。

## 6. FlowDAgger、UniSteer、FRS：人工动作可以成为噪声监督

### FlowDAgger：有 WAM 证据，但公开代码覆盖较窄

FlowDAgger 使用逆向积分与局部固定点迭代，把人工动作转换成噪声标签，再监督训练小噪声策略。论文报告 π0.5、GR00T 和 Cosmos-Policy/WAM 等实验；WAM 部分处理联合 world-action latent。**AHA 有单独的动作采样器，因此不能把 Cosmos 的 joint-latent 反演代码原样套上。**[^13]

公开仓库是 π0.5 + MetaWorld assembly 的最小参考实现，不等于公开了全部 WAM/实机实现。固定 commit：
`microsoft/FlowDAgger@d2e60b5bf5d39538783f1e73d460394b21eb0f28`。[^14]

| `flowdagger_pi05/train_flowdagger.py` 默认 | 值 |
|---|---:|
| 小策略 hidden dims | 256×3 |
| BC LR / batch | 1e-4 / 256 |
| BC更新 | 每采集一条 episode后100步 |
| inversion_method | perstep_fp |
| fp_per_step | 5 |
| inversion MSE门槛 | 0.001，原示例坐标 |
| seed expert episodes | 10 |
| noise_basis_k | 10 |
| filter_failures | 1；代码提示真实 HUMAN 应改0 |
| dual_buffer / auto_frac | 默认关闭 / 启用时0.5 |

此参考实现是 BC，不更新 critic。成功轨迹筛选在这种脚本化专家示例中有用途，**不能移植成在线 RL 丢弃所有失败 transition**。其 MSE门槛必须根据 SO101 实际单位和单关节最大误差重设。[^15]

### UniSteer：与“人工噪声监督 + 真正在线 RL”最贴近

UniSteer 将近似反演得到的 HUMAN 噪声加入监督和 RL 学习，论文给出了先监督后 RL 的更新顺序及相关消融。它对本项目有价值的依据是：人工纠正可以降低稀疏奖励下的噪声探索难度；前提是反演动作确实接近真正执行动作。[^16]

附录 D 报告：

| 参数 | 值 |
|---|---:|
| base | π0，10步denoise，16×7动作 |
| actor / critic | 1024×3 |
| batch / replay容量 | 256 / 33333 |
| actor RL LR / critic LR | 1e-4 / 3e-4 |
| actor SFT LR / temperature LR | 6e-5 / 3e-4 |
| γ / τ | 0.99 / 0.005 |
| initial α / target entropy | 1.0 / 0.0 |
| noise representation | 50维 |
| 固定点迭代 | 文中选 M=16 |

论文给出四个实机任务、单平台的验证；没有核对到可对应这些全部设置的公开训练仓库，因此把它标成“论文参数”，不声称完成逐行代码复现。它报告的单样本反演耗时也不能搬到 AHA DiT。

“50维 noise representation”尤其不能机械改成本模型40×6或8×6；需要匹配具体噪声展开函数。本项目先采用完整当前块48维，减少新增不可识别的投影。

### FRS：支持先用引导经验启动噪声 RL

FRS 用 flow reversal 将粗动作转为可解码噪声，还展示了噪声 BC 及用引导成功经验启动 DSRL。项目页当次显示代码仍为 Coming soon，因此这里仅作为机制与实验依据，不作为可直接复用的 learner。[^17]

这些工作修正了过于绝对的判断：人工动作**没有天然、已记录的噪声标签**，但并非永远不能利用。近似反演本身需要验证，不能把映射误差隐藏在 replay 里。

## 7. 全模型 RL 与模型内部 RL 的定位

DPPO 对 diffusion 策略本体做策略优化，有正式实验和代码。它适合说明 diffusion 并非不能直接 RL，但对现有 AHA flow 分块缓存、真机单环境和三张卡，增加大模型更新并不是第一轮最省资源的验证。[^18]

OVCR 上的动态 latent actor 是值得研究的独立课题，但目前没有与本仓库 V32/Q32/T4 完全一致的成功复现。先做输出/噪声两条有清晰动作语义的路径，有助于把失败归因于表征、探索或控制精度；一开始就在未知 latent 上训练 Q，会同时引入太多不确定项。

## 来源

[^1]: Physical Intelligence，[Precise Manipulation with Efficient Online RL](https://www.pi.website/research/rlt)，2026-03-19。
[^2]: Xu 等，[RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://arxiv.org/html/2604.23073v2)，2026，§IV、附录 B。
[^3]: RLinf，[RLT 实现文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html)，访问于2026-09-10。
[^4]: RLinf，[固定版本 Stage 2 配置](https://github.com/RLinf/RLinf/blob/c94c67c12c27547ad6244d3957eb473955c7d46d/examples/embodiment/config/realworld_rlt_stage2_ac_mlp.yaml)、[actor/critic loss](https://github.com/RLinf/RLinf/blob/c94c67c12c27547ad6244d3957eb473955c7d46d/rlinf/workers/actor/fsdp_rlt_ac_policy_worker.py)、[更新调度](https://github.com/RLinf/RLinf/blob/c94c67c12c27547ad6244d3957eb473955c7d46d/rlinf/workers/actor/fsdp_sac_policy_worker.py)。
[^5]: Luo 等，[HIL-SERL](https://arxiv.org/html/2410.21845v2)，2024/2025。
[^6]: HIL-SERL，[配置](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/examples/experiments/config.py)、[双buffer与异步learner](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/examples/train_rlpd.py)。
[^7]: HIL-SERL，[agent builder](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/serl_launcher/serl_launcher/utils/launcher.py)、[SAC losses/defaults](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/serl_launcher/serl_launcher/agents/continuous/sac.py)。
[^8]: Yuan 等，[Policy Decorator](https://arxiv.org/html/2412.13630)，2024/ICLR 2025。
[^9]: Policy Decorator，[固定版本示例代码](https://github.com/tongzhoumu/policy_decorator/blob/92ba9ba442587ae5989c286355ace2200a0537fb/online/pi_dec_diffusion_maniskill2.py)。
[^10]: Wagenmaker 等，[DSRL](https://arxiv.org/html/2506.15799v2)，2025。
[^11]: DSRL，[NA learner](https://github.com/ajwagen/stable-baselines3-dsrl/blob/10e5d311b5c42fe571ff3eeb72f622ca616a149d/stable_baselines3/dsrl/dsrl.py)。
[^12]: DSRL，[任务配置](https://github.com/ajwagen/dsrl/blob/8d21b9cf55459f022c0e769bca1343edd9d30e5a/cfg/robomimic/dsrl_can.yaml)、[训练入口](https://github.com/ajwagen/dsrl/blob/8d21b9cf55459f022c0e769bca1343edd9d30e5a/train_dsrl.py)、[环境wrapper](https://github.com/ajwagen/dsrl/blob/8d21b9cf55459f022c0e769bca1343edd9d30e5a/env_utils.py)。
[^13]: Murray 等，[FlowDAgger](https://arxiv.org/html/2607.08877v1)，2026-07-09。
[^14]: Microsoft，[FlowDAgger公开范围](https://github.com/microsoft/FlowDAgger/blob/d2e60b5bf5d39538783f1e73d460394b21eb0f28/README.md)。
[^15]: FlowDAgger，[参数入口](https://github.com/microsoft/FlowDAgger/blob/d2e60b5bf5d39538783f1e73d460394b21eb0f28/flowdagger_pi05/train_flowdagger.py)、[逐步反演](https://github.com/microsoft/FlowDAgger/blob/d2e60b5bf5d39538783f1e73d460394b21eb0f28/flowdagger_pi05/flow_matching_inverter.py)、[反演误差筛选](https://github.com/microsoft/FlowDAgger/blob/d2e60b5bf5d39538783f1e73d460394b21eb0f28/flowdagger_pi05/train_utils.py)。
[^16]: Lu 等，[UniSteer](https://arxiv.org/html/2605.10821v1)，2026-05-11，§5、附录 C/D。
[^17]: Tang 等，[Flow Reversal Steering 官方项目](https://flow-reversal-steering.github.io/)，2026。
[^18]: Ren 等，[DPPO项目与论文](https://diffusion-ppo.github.io/)，ICLR 2025。

所有引用的完整可复用范围和代码版本汇总见 [来源清单](SOURCES_ZH.md)。
