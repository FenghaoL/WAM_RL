# 方法来源、代码审计与后续路线

日期：2026-10-23。区分“本轮读过关键公式/源码”“关联阅读”“已经复现实验”。本轮没有复现任何论文的真机结果，也不把相关工作的分钟数或成功率外推到 SO101。

此前项目已有 [HALO 迁移长文](../research/HALO_WA_RL_AND_AHAWAM_MIGRATION_GUIDE_ZH.md) 和 [在线生成策略 RL 比较](../ovcr_rl_token/online_rl_feasibility/05_VINE_AND_GENERATIVE_RL_REASSESSMENT_ZH.md)。本文针对**已有完整 HIL 日志＋人工 success frame 的离线阶段**补充决策，不重写既有 DAgger 结论。

## 1. 核心工作的可迁移部分

| 工作与一手资料 | 本轮实际核查内容 | 对本项目的结论 |
| --- | --- | --- |
| [RLT 论文](https://arxiv.org/html/2604.23073v1)、[作者项目](https://www.pi.website/research/rlt) | reader 训练、chunk TD、reference 条件和 penalty、接管 reference 替换、附录小网络与稀疏奖励 | 借用“先适配表征，再冻结并训练小 actor-critic”的分工；不能把在线 loss 当完整离线方案 |
| [HALO-WA](https://arxiv.org/html/2607.04265v1)、[公开代码](https://github.com/YeanRoot/HALO-WA) | 本地完整 wrapper 的 VAE latent 提取、attention actor/critic、worker 的 target/BC 路径及离线脚本 | 借用多 token 读取和 WA reference；重写 AHA 接口、时序和离线更新，不引入整套 RLinf |
| [IQL](https://arxiv.org/html/2110.06169)、[作者代码](https://github.com/ikostrikov/implicit_q_learning) | `critic.py`、`learner.py`、`value_net.py` 的 expectile、数据动作双 Q、V bootstrap、target Q 更新 | 首版离线价值/策略提取；本项目加入 chunk、terminal mask、参考动作条件 |
| [TD3+BC 论文](https://arxiv.org/abs/2106.06860)、[作者源码](https://github.com/sfujim/TD3_BC/blob/main/TD3_BC.py) | 两个 Q、target noise、actor delay、`alpha / abs(Q)`、数据 BC | 保留直接 Q 梯度的简单对照；BC 对齐数据，不等于只锚定 WAM reference |
| [Q-chunking](https://arxiv.org/html/2507.07969v4)、[作者项目](https://colinqiyangli.github.io/qc/index.html)、[代码](https://github.com/ColinQiyangLi/qc) | chunk MDP 的推导、behavior constraint；`agents/acfql.py` 的 flatten、`gamma^h`、flow BC 和 distillation | 支持评价整个 chunk，但要求奖励和执行区间一致；它不自动修复 hold、接管取消或隐藏缓存 |
| [ConRFT](https://arxiv.org/html/2502.05450)、[代码](https://github.com/cccedric/conrft) | 离线/在线两阶段，`conrft_single_octo_cp.py` 中 CQL 随机/策略动作采样、MC 校准下界、consistency actor | 支持小数据离线初始化再接在线；完整 Cal-ConRFT 比首版双 Q IQL 更复杂，不是“小 critic＋普通 BC”即可复现 |
| [HIL-SERL](https://arxiv.org/abs/2410.21845)、[网络工厂](https://github.com/rail-berkeley/hil-serl/blob/main/serl_launcher/serl_launcher/agents/continuous/sac.py) | `create_pixels` 的 frozen pretrained ResNet-10、小 actor/critic、双 Q 默认与可配置 ensemble | 不必先训大 VLM critic；它提供在线系统经验，并不是当前固定数据的离线算法证明 |
| [VINE](https://arxiv.org/html/2607.10369v1)、[项目](https://agibottech.github.io/vine/) | 新插值采样路径、端到端价值梯度、OGBench 与附录 actor/critic、公开 GitHub 目录 | 保留为内部 Action DiT 更新路线；与外部 actor、IQL policy extraction 是不同优化机制 |

没有将社区转述作为关键机制的证据。公开 RLT 补充实现只作为接口参考：本轮阅读了 [yknxh/rlt-openpi 的 reader](https://github.com/yknxh/rlt-openpi/blob/main/src/rlt_openpi/models/rl_token.py)，它是独立实现，不是 PI 官方完整真机训练代码。其 decoder/mask 细节也不能免审计复制。

## 2. HALO-WA 本地审计：为什么不直接搬

本地 checkout：`/home/fenghao/workspace/HALO-WA`，commit `611008193fd3438bf99cc8f8377858fbdfcb9555`。这是本轮实际读取的版本，不声称等于远端最新代码。

### 表征实际来自哪里

`giga_world_policy.py::_extract_visual_latent_from_ref` 对 reference image 预处理后调用 pipeline 的 `prepare_latents`，得到 `latent_condition`；`encode_visual` 选时间首帧或平均、展平成空间 tokens，再经过 compressor 和位置编码。这里不是“从 Video DiT 中拿最后层给 Action DiT 的 K/V”。

因此本项目的 post-OVCR V 是一个 **需要验证的新输入选择**。它的合理性在于与当前 action branch 使用的条件一致；局限是受 planner/cache 过期、层选择、token 压缩和视频任务表征偏差影响。

### Critic 的顺序问题

`adapter/gigawa_actor_critic.py::CrossAttentionCritic` 将 state/ref/candidate 投影为 query，对 visual tokens 独立 cross-attention，经过逐 token FFN 后平均池化；没有候选动作位置编码或候选 token 间 self-attention。

对这个模块而言，置换 candidate token 顺序只会置换这些 query 的输出，平均值保持不变。因此它无法区分同样的目标集合先向下再抬起，还是先抬起再向下。用实际类、随机特征、C=8、D=6 做倒序测试，最大 Q 差 `1.862645149230957e-09`，与浮点误差一致。

这是模块级的结构分析和实测，不是断言整篇工作的效果不成立。公开代码中 `TwinQHead` 的两个大 MLP 还共享前面的 attention 表征；本项目首版采用更小、编码器也分离的双 Q，便于控制相关误差和模型容量。大头 `[2048,1024,512]` 并不因为叫轻量 adapter 就适合 7 条 episode。

### Loss 和时间不能照抄

当前 worker 对 chunk 内 reward 求和，target 使用 `self.discount`；其配置/初始化层还会处理块长折扣。不能仅看 helper 的 `discount` 变量名便推断是 tick 还是 chunk。AHA 的稀疏奖励要明确在块内哪一步出现，并用实际 m tick 的 `gamma^m`。

worker actor 的 BC 走 `ref_action`，而本项目离线 IQL/TD3+BC 的数据 BC 走实际执行目标；两者不是同一种行为约束。HALO 的 Q 上界 clamp/penalty 也不能消除离线分布外高估；超过上界后硬 clamp actor Q 还可能失去梯度。本实现先监控越界，避免用 clamp 让日志显得稳定。

公开离线训练脚本依赖 Giga/RLinf replay 和指定配置；即使名称是 offline RL，也不是能直接消费本项目 Parquet/success JSON 的开箱入口。

## 3. 为什么暂不选其他路线为首版

**CQL / Cal-QL**：对数据外动作施加保守约束；Cal-QL 还用回报校准以便后续在线学习。本轮在 ConRFT 源码中核查了该路径。48 维 chunk 的随机采样、保守系数和回报下界都需要额外校准；旧日志包含接管和时序缺口时，MC return 的语义也要先处理。适合作为下一阶段比较，而非一开始把所有 regularizer 叠加。[Cal-QL 作者实现](https://github.com/nakamotoo/Cal-QL)，[论文](https://arxiv.org/abs/2303.05479)

**FQL / QC-FQL**：flow 拟合行为分布，小的一步策略通过 Q 与 distillation 更新。它保留多模态行为约束，并避免对完整生成链做价值反传；当 Gaussian/MLP actor 出现左右纠正取均值的问题时，值得优先考虑。冻结 AHA-WAM 的单个 reference 并不等于已经实现 FQL 的行为分布约束。[FQL 作者项目](https://seohong.me/projects/fql/)，[代码](https://github.com/seohongpark/fql)，[QC-FQL 实现](https://github.com/ColinQiyangLi/qc/blob/main/agents/acfql.py)

**V-GPS / best-of-N**：可以只训练价值，给同状态的多个基础策略候选排序，避免直接修改 actor。它是验证 critic 是否有用的辅助路线；但当前大 WAM 每个候选要多次采样，固定任务的延迟和计算成本需实测。只在原动作旁加噪打分，并不能证明 Q 排序正确。[作者项目](https://nakamotoo.github.io/V-GPS/)，[论文](https://arxiv.org/abs/2410.13816)，[源码](https://github.com/nakamotoo/V-GPS)

**DSRL / latent noise actor**：冻结生成策略，优化输入噪声。旧 DAgger 只保存了执行关节目标，通常不存在可直接监督的唯一噪声动作；不能将实际关节动作当 latent action 塞进 noise Q。若重建噪声/重新采样另建流程，才可作为独立路线。[作者工作](https://arxiv.org/abs/2506.15799)

以上作为路线比较，不声称本轮逐行审计了 Cal-QL、V-GPS、DSRL 的全部实现和所有实验。选中的 IQL、TD3+BC、HALO 接口与数据链路进行了更深入核查；扩展实现前仍需对选定版本做专项验证。

## 4. 预训练与数据覆盖的几个易忽略问题

1. **预训练的视觉表示不等于预训练的数值 Q。** 先复用 DAgger WAM/reader；Q/V 从当前 0/1 回报学习，不搬其他 embodiment 的 Q 权重。
2. **Video→Action 特征也可能不足。** 精细插入需要腕部局部偏差、接触/运动历史；视觉不可观测的力不能靠更大 Q 自动恢复。比较当前图像与有历史特征才知道瓶颈在哪里。
3. **全层 KV 不是免费的数据增强。** 它可能是每层大量空间 token，缓存和 I/O 比 learner 计算更重。先选一层/少量层；层、pooling、时间来源都写入 feature_spec。
4. **冻结 reader 后缓存才稳定。** 在线更新 reader/OVCR/Video 分支会改变旧 cache 语义；需版本化重算或冻结一份 critic 表征，不可混用。
5. **离线 reference 是重新生成的条件。** 旧 `policy_action` 来自旧 checkpoint/旧缓存，不可用作新 DAgger reference；同状态多个参考噪声和实际数据动作间的模式一致性也要检查。
6. **少量数据的 actor 可能平均不同恢复模式。** 先看优势权重 ESS、参考条件敏感性和 BC；若加权 MSE 无法表达多峰，改小 flow actor 比先扩 Q 更有针对性。
7. **整条失败不表示每个动作都坏。** 全零终局回报通过 TD/泛化评价其动作；不能把失败 episode 的全部目标当负动作分类标签。
8. **没有成功覆盖，离线 RL 无法创造反馈。** 若只取最后精细阶段但没有其成功转移、或在过滤 hold 时丢掉成功，调更大 critic 也救不了目标。

## 5. 内部 Action DiT 路线如何保留

用户已要求保留，但外部小 actor 优先。建议下一步先比较以下两类，不立即实现/训练：

| 内部更新方法 | 梯度如何到动作模型 | 本项目需要额外完成 |
| --- | --- | --- |
| IQL-inspired advantage-weighted flow | 对真实 action chunk 的 flow loss 乘 detached advantage weight | 正确重建内部 conditioning/KV；按数据 chunk 合同构造 flow 目标；冻结 critic 特征源 |
| VINE 价值梯度 | critic 对最终动作的梯度穿过多步重新插值采样 | 可微 sampler、时间/速度符号适配、显存与吞吐、参考/BC 约束、采样器独立对照 |

VINE 的关键是改变生成路径来改善多步价值梯度，不是一个 reader。其离线 OGBench 附录用双 residual value network，不能把这里的 ResNet 理解成“用 ImageNet ResNet 图像 critic”。本轮公开 GitHub 目录仍是项目网页/视频，没有找到完整训练源码；不据此承诺可直接复制真机训练配方。[VINE 论文附录](https://arxiv.org/html/2607.10369v1)，[当前公开目录](https://github.com/AgibotTech/vine)

内部路径仍需当前状态、历史和计划缓存，紧凑 frozen z 不足以复现整个 Action DiT 条件。LoRA/仅 OVCR 更新能省可训练参数和 optimizer，不能省掉梯度穿过 Action DiT 的计算。

至少保留“原 checkpoint＋原 sampler”“原 checkpoint＋新 sampler、不训练”“同可训练参数的普通 Q 梯度”“VINE Q 梯度”对照。外部 actor 和内部更新最终应比较同一初始策略下的无接管成功、机器人交互量、训练吞吐和旧任务保持。

## 6. 源码版本记录

| 来源 | 审计标识 |
| --- | --- |
| HALO-WA 本地 checkout | `611008193fd3438bf99cc8f8377858fbdfcb9555` |
| IQL 作者 master，本轮 API 查询 commit | `09d700248117881a75cb21f0adb95c6c8a694cb2` |
| IQL `critic.py` blob | `a49d2b1ed3dcc12e15b726cf1d4f592a955725f7` |
| TD3+BC `TD3_BC.py` blob | `36e3ffa0990d90d1e75b9066f6894f9c478f35ea` |
| QC main，本轮 API 查询 commit | `48283b4f662bd1c127ec9fa80647b49759f10653` |
| QC `agents/acfql.py` blob | `291acd830f4a652e8c2d566449e5eb42673412ae` |
| ConRFT main，本轮 API 查询 commit | `a779fde7fa5db5a469960a8490c100f35b41b49e` |
| rlt-openpi `rl_token.py` blob | `54a6c06e7e07b0f1aca40d9c074dbc3e974dddea` |

远端文件读取和 commit 查询不是原子事务，因此 blob SHA 才是读过的具体文件证据；没有把代码复制进本仓库或增加这些仓库的运行依赖。
