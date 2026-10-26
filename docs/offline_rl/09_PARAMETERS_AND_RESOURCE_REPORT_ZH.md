# 参数选择、缓存规模与资源实测

日期：2026-10-24。配套 [实现与命令地图](08_SYSTEM_IMPLEMENTATION_ZH.md)。这里区分论文参数、当前工程默认和本机测量；500-step pilot 是链路与资源检查，不是最终超参数定论。

## 1. 文献与代码对照

| 工作 | 实际输入与数据/阶段 | 可核对的参数 | 对本项目的含义 |
| --- | --- | --- | --- |
| [IQL 原文附录 B/C](https://arxiv.org/html/2110.06169) | D4RL 状态离线 benchmark；部分实验再 online | 2×256 MLP，Adam 3e-4，target tau=.005；常见 expectile=.7、beta=3，AntMaze .9/10；大多 1M 更新，灵巧操作预训练有 25k 设置 | 学习率与网络大小可作起点，不能把原任务 reward scale、beta 或 1M steps 当作 103 轨迹的定律 |
| [TD3+BC 原文表 4](https://arxiv.org/html/2106.06860) | 固定 D4RL MuJoCo 状态数据；每条动作低维、无 KV attention | batch256、2×256 MLP、lr3e-4、gamma .99、tau .005、actor每2Q更新；alpha2.5，noise .2/clip .5 | 本项目 8×6 chunk、绝对关节 z-score 与稀疏单次成功不同；alpha/noise 不能直接认为同一尺度 |
| [RLT 附录 B](https://arxiv.org/html/2604.23073v1) | 冻结的 learned RL token；4 项真机任务、50Hz、约5小时在线数据 | 多数任务 2×256 MLP，螺丝任务3×512；chunk10、stride2、reference dropout50%；表征先训2k–10k | 支持“小 actor 读冻结表征”的方向，但它先压成 token；本项目直接 KV decoder 计算与参数多一些 |
| [HALO-WA 审计版本](https://github.com/YeanRoot/HALO-WA/tree/611008193fd3438bf99cc8f8377858fbdfcb9555) | 模拟/在线世界动作模型表征；实际公开 actor-critic 输入不等于本项目 post-OVCR KV | `online_rl_bell_ansyc.yaml` global batch64、micro batch4；actor lr2.5e-5、critic lr3e-5；适配器 self/cross attention；Q 两个独立 MLP head [2048,1024,512] | 参考其 token 融合和共享双 Q 结构。我们用投影256、head256，不能拿其大head与全系统GPU预算直接估计本项目 |
| [QC-FQL/QC 原文表3](https://arxiv.org/html/2507.07969v4) | OGBench 长时域 sparse 任务及 robomimic；多任务不同规模，含大数据设置 | batch256、lr3e-4、512宽4层、UTD1、通常1M offline更新，2 Q（FQL系列） | 支持 chunk Q 的动机，但其数据规模、flow actor 与终止/奖励约定不同，不能照搬训练总量 |
| [VINE 附录 A.4](https://arxiv.org/html/2607.10369v1) | 状态型 OGBench 离线实验，与本文的 KV 外部 actor 不同 | batch512、1M steps、lr3e-4；actor4×512、双ResNet Q宽512深4/2；gamma.995、tau.005、flow10步 | 是内部生成策略 RL 的后续参考，不是当前 deterministic actor 的必要复杂度 |
| [V-GPS 作者仓库](https://github.com/nakamotoo/V-GPS) | Bridge+Fractal 跨机器人/多任务视觉数据离线训练 Q，部署时重排多个基础策略动作 | 已发布 Q checkpoint 使用 batch256、500k更新 | 数据与机器人不同，直接拿其 Q 权重不等于适配 SO101；无需为首版额外依赖这个预训练 critic |

HALO 原文件：[actor/critic 结构](https://github.com/YeanRoot/HALO-WA/blob/611008193fd3438bf99cc8f8377858fbdfcb9555/rlinf/rlinf/models/embodiment/adapter/gigawa_actor_critic.py)、[bell 配置](https://github.com/YeanRoot/HALO-WA/blob/611008193fd3438bf99cc8f8377858fbdfcb9555/rlinf/examples/embodiment/config/online_rl_bell_ansyc.yaml)。这些是具体实验配置，不代表作者所有任务统一参数。

## 2. 当前参数与选择理由

| 项 | 首轮值 | 理由与后续比较 |
| --- | --- | --- |
| checkpoint / 模型精度 | 指定 SFT 006572 / BF16 | 与采集策略一致；A100 支持 BF16，动态范围更适合该模型；不做 int8/FP16 量化漂移 |
| 缓存并行 | 4 GPU，各1副本，batch1 | 重型 WAM 推理按 episode 并行；不是分布式训练或四卡模型切分 |
| 仅导出层 | 最后一层 post-OVCR K/V、routed、query | 保留未池化信息，约76GiB可接受；不存30层或每个transition的两份复制 |
| RL 网络 | width256，depth2，4heads | 冻结表征后只学小策略；先验证链路。512宽/独立双Q作为后续消融 |
| batch size | 64 | 本轮先测32/64/128；不必因显存富余就用512/4096。滑动窗口强相关，64窗口不等于64独立轨迹 |
| LR | actor/Q/V统一1e-4、Adam | 低于IQL/TD3状态基准3e-4，直接训练视觉读出且数据小，先保守；后续比较3e-5/3e-4 |
| IQL | expectile .7，beta3，max-weight20，V/Q/actor每次各1 | 原IQL常见量级；权重截断比原100更保守，监测ESS是否接近纯BC |
| TD3+BC | 2Q:1actor；alpha.1；noise .1、clip.2 | 对归一化chunk的保守对照，不声称等价于原论文alpha2.5/noise.2；alpha后续扫描 .1/.5/2.5 |
| target EMA | tau .005 | IQL每步，TD3随延迟actor步更新，匹配两种算法各自调度 |
| actor 初始化 | reference＋零初始化残差；reference dropout .1 | 初始保留采集策略行为；较RLT .5更少屏蔽参考，以免小数据首轮难学 |
| gamma | .997 / 有效动作 | 对8步chunk折扣约.97625；HOLD去除后不按墙钟折扣。与旧tick gamma数值相同但含义不同 |
| BC warmup | 200 actor-only updates | 验证小 actor 能拟合实际动作；不把它称为Q/V预热 |
| 首轮总更新 | 每算法500 | 200 BC＋300 RL，验证真实链路、资源、异常。初步建议正式候选5k，再决定是否延长到20k |
| 采样 / 验证 | 训练窗口均匀有放回；`_val`整episode隔离 | 不让相邻窗口随机分到训练与验证；首轮不靠人为正例重采样改变分布 |

以当前 16,554 窗口中 16,016 训练窗口计算，batch64：500 次抽样约相当于2.0次窗口覆盖，5k约20.0次，20k约79.9次；这不是独立样本 epoch，重叠窗口仍来自少量真实 episode。1M更新会达到约3996次重复覆盖，缺乏必要性，可能只把少量轨迹记住。验证 losses、Q分歧与ESS用于发现问题，最终优劣需真机或可靠独立评估。

本轮保留 `_val` 下已就绪的3条episode、538窗口作验证；其余16,016窗口训练。9条缺标注 episode 的列表在 [数据审计](audit_20261024.json)。已标记成功81条，其中79条能产生覆盖成功的完整动作块；535个正奖励窗口不是535次成功。16个成功事件按用户规则前移1–3帧。

## 3. 存储估算

103条共43,680原始观察。384×320相机拼图，VAE/patch网格120 token，planner宽3072：

| 内容 | 每观察 | 全量 |
| --- | ---: | ---: |
| 最后一层 K+V BF16 | 1.40625MiB | 59.99GiB |
| routed＋projected_query BF16 | 0.375MiB | 16.00GiB |
| state＋reference FP32 | 216字节 | 9.0MiB |
| 合计纯tensor | 约1.78146MiB | 约76.0GiB / 81.6GB |

另有几十MiB级的JSON/索引开销，建议预留85–90GiB。每128观察分片约228MiB。源视频和parquet本身约1.6GB，transition索引只有MiB级。4张卡不使缓存扩大4倍，每张处理不同episode；offset每帧只一个版本也不乘8。最初的约60GB估算只含K/V，不含这次追加的routed/query。

## 4. 资源测量与运行产物

机器为4张 A100-SXM4-40GB；启动前均空闲。模型首次加载需读取约14GB SFT checkpoint，并完成SHA256校验，启动I/O时间不能混入每观察推理吞吐。实际峰值与速度由下列文件更新：

- `runs/offline_rl/cylinder103_skip_hold/probe_cache/`：少量真实帧，验证形状/有限值/落盘。
- `cache/progress_rankN.json` / `done_rankN.json`：各卡model加载时间、每观察平均秒数、allocated/reserved/peak显存、分片与视频hash。
- `resource_profile.json`：同一架构、真实缓存，batch32/64/128的IQL/TD3逐步耗时和显存。小子集有放回重复只用于资源测量。
- `pilot_iql/profile.json`、`pilot_td3_bc/profile.json`：全体已就绪轨迹的单卡真实训练资源；以及各自metrics/config/provenance/checkpoint。

四卡导出耗时估算用 `max(每rank帧数 × 该rank平均秒/观察) + 首次加载/校验`，再考虑写盘/共享I/O余量；不是简单用总帧数除以4后套一个无历史单帧模型时间。导出会重放最多6个历史planner观察，但同一个8帧anchor只prefill一次；计时包含这些开销。长段/接管gap不同，最终均速可能变化。

缓存完成后IQL/TD3训练都不再运行冻结WAM。只按需读取KV、当前/后继state和reference、实际动作、奖励与discount；TD3的target actor用后继reference，IQL的V备份不需要后继生成动作。训练主要计算是小decoder及其Q/V/target前向/反向。重型特征与参考动作推理通常是准备阶段的大头，实测读盘也可能成为小网络训练瓶颈；profile会分别记录batch加载与更新时间。后续一般一张40GB卡即可，四卡更适合特征提取或多seed并行，而不是为小头做DDP。

### 本轮已取得的实测

真实 WAM 探针导出16个连续观察，K/V 为 `[120,3072]`、routed/query 为 `[32,3072]`，分片约28.5MiB。峰值 allocated **18,202.8MiB（17.78GiB）**、reserved **18,360MiB（17.93GiB）**。首次模型加载＋来源校验559.9秒；16帧推理均值0.734秒/观察，末帧0.504秒。该段没有长历史，不能当作全目录的最终均速；正式worker将CPU线程限制为2，探针首次运行未作这一限制。

真实缓存资源基准，每个设置预热3步再量20步：

| 算法 | Batch | 秒/更新，含加载 | 秒/加载 | 峰值allocated MiB | 峰值reserved MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| IQL | 32 | .0815 | .0194 | 757 | 774 |
| IQL | 64 | .1276 | .0396 | 1244 | 1324 |
| IQL | 128 | .2572 | .1206 | 2240 | 2390 |
| TD3+BC | 32 | .0626 | .0195 | 709 | 756 |
| TD3+BC | 64 | .1180 | .0537 | 1201 | 1318 |
| TD3+BC | 128 | .2811 | .1511 | 2197 | 2390 |

基准只覆盖了探针中**1个完整真实transition**，有放回重复，用于验证真实BF16接口、梯度链路与显存。它不是多轨迹训练结果，热缓存吞吐不能预测全目录随机读盘；完整测量JSON为 `runs/offline_rl/cylinder103_skip_hold/resource_profile_probe.json`。batch128显存仍充足，但加载与拷贝开销更大，目前batch64很合理，单卡足够。

若保持该热缓存速度，单算法5k更新约10–11分钟，加上验证/保存/冷读盘；这只是小网络计算的参考下界，全数据pilot会记录实际值。对全量43,680帧，0.734秒/观察按四卡均分约2.23小时，再加模型加载与更长历史/共享I/O；当前先预留2–4小时量级，后续用各rank进度修正。不是承诺完成时间。

四卡后台队列已在当地时间2026-10-24 23:35启动第二次尝试。第一次启动暴露新脚本名与标准库profile冲突，已更名benchmark.py，错误日志单独保留。当前任务覆盖全部103条原始录像；缓存完成并验证后，依次单卡跑IQL500步、TD3+BC500步。实时PID与命令在 `sequence_launch.json` / `cache/launch.json`，状态以新的worker日志、complete/sequence_done标记为准。尚未完成的阶段不能当作实测结果。


### 多观察补测与四卡启动后的进度

又在已落盘分片上覆盖197个真实transition进行每设置10步补测（与后台WAM导出并行，数值包含资源竞争，不作算法质量评估）：

| 算法 | Batch32 秒/步 | Batch64 秒/步 | Batch128 秒/步 | Batch64 峰值allocated |
| --- | ---: | ---: | ---: | ---: |
| IQL | .177 | .277 | .534 | 1244MiB |
| TD3+BC | .123 | .235 | .327 | 1201MiB |

结果在 `resource_profile_partial.json`。真实多观察加载使batch64耗时高于单transition热缓存，但显存与前测相同。按该补测，每算法5k步约20–23分钟（仍不含完整数据冷读盘、验证与保存；完整pilot将更新实际数据）。

第二次四卡启动的加载＋校验已降为135.6–143.4秒：复用经文件大小/mtime核对的checkpoint摘要，避免各卡重复扫描14GB。四卡均写出首个128观察分片，已实际经过48/96帧历史条件。该快照均速分别为.551/.518/.543/.501秒每观察，峰值allocated最大18,287MiB，reserved18,484MiB。四卡分工帧数10,871 / 11,349 / 11,402 / 10,058，按此初步均速估算全量约1.76小时加后续I/O/长历史变化，先保留2–4小时预算。该状态不是缓存完成标记。


真实端到端CLI也已验证：在4条episode、667个已有缓存transition构成的隔离小子集上，IQL跑6步并从checkpoint恢复至8步，TD3+BC跑6步；两者均包含2步BC后切换RL、整episode验证隔离、保存优化器/target/RNG。已加载实际保存的actor，通过KV推理API得到有限的 `[1,8,6]` 归一化/绝对动作。路径为 `integration_iql/`、`integration_td3_bc/`。这是完整链路测试，不代表正式收敛或真机改进。21项CPU测试、Python编译、文档本地链接与diff空白检查通过。正式四卡导出仍持续进行，后台500步pilot尚未开始。


补充审计：没有成功窗口的2条分别为 `20260911_225809/episode_000017` 与 `20260912_010747_val/episode_000002`。两条成功都标在HUMAN→POLICY切换处，成功图像的incoming相邻记录 tick 差分别15和14，存在未记录时段，并非自然尾部不足8动作。当前不推断这些缺失tick全部是HOLD，因此不跨该缺口构造奖励transition；缓存与原始标注仍完整保留。之后若能从独立控制日志恢复实际执行/等待，可再纳入这两个事件。
