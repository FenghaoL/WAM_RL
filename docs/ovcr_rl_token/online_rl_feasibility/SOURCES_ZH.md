# 来源与版本清单

## 1. 论文与项目

| 编号 | 作者/发布方；标题 | 时间 | 用于本报告的内容 |
|---|---|---|---|
| 1 | Physical Intelligence；[Precise Manipulation with Efficient Online RL](https://www.pi.website/research/rlt) | 2026-03-19 | RL token读出、外部小策略、局部阶段、真实交互与总耗时区别 |
| 2 | Charles Xu等；[RL Token: Bootstrapping Online RL with Vision-Language-Action Models](https://arxiv.org/html/2604.23073v2) | 2026-04-30，v2 | 参考动作、Gaussian actor、chunk TD、网络大小与dropout |
| 3 | Jianlan Luo等；[Precise and Dexterous Robotic Manipulation via Human-in-the-Loop Reinforcement Learning](https://arxiv.org/html/2410.21845v2) | 2024，v2 | HIL-SERL实机训练系统；[项目](https://hil-serl.github.io/) |
| 4 | Xiu Yuan等；[Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model](https://arxiv.org/html/2412.13630) | 2024，ICLR 2025 | bounded residual与progressive exploration |
| 5 | Andrew Wagenmaker等；[Steering Your Diffusion Policy with Latent Space Reinforcement Learning](https://arxiv.org/html/2506.15799v2) | 2025-06，v2 | DSRL-SAC/NA、冻结decoder与噪声MDP |
| 6 | Michael Murray等；[FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space](https://arxiv.org/html/2607.08877v1) | 2026-07-09 | 动作反演、噪声BC、WAM迁移边界；[项目](https://microsoft.github.io/FlowDAgger/) |
| 7 | Junjie Lu等；[Unified Noise Steering for Efficient Human-Guided VLA Adaptation](https://arxiv.org/html/2605.10821v1) | 2026-05-11 | UniSteer固定点反演、监督+RL、附录超参数、单平台局限 |
| 8 | Andy Tang等；[Improving Robotic Generalist Policies via Flow Reversal Steering](https://flow-reversal-steering.github.io/) | 2026 | FRS引导噪声、噪声BC、DSRL初始化；项目当次标注代码Coming soon |
| 9 | Allen Z. Ren等；[Diffusion Policy Policy Optimization](https://diffusion-ppo.github.io/) | ICLR 2025 | diffusion本体策略优化，与小actor路线比较 |
| 10 | RLinf团队；[RLT实现文档](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/rlt.html) | 访问2026-09-10 | 可参考的公开RLT实现，不能冒充原论文未披露设置 |

时间为论文/项目标注或版本信息，不将搜索结果抓取时间当作发表时间。关于参数的细节优先以固定commit源码为准。

## 2. 已直接阅读的公开代码

| 仓库 | 固定commit | 重点文件 | 可复用范围 |
|---|---|---|---|
| [ajwagen/dsrl](https://github.com/ajwagen/dsrl/tree/8d21b9cf55459f022c0e769bca1343edd9d30e5a) | `8d21b9cf55459f022c0e769bca1343edd9d30e5a` | `train_dsrl.py`、`env_utils.py`、`cfg/robomimic/dsrl_can.yaml`、`.gitmodules` | 官方训练调用、示例参数、噪声/动作环境包装 |
| [ajwagen/stable-baselines3-dsrl](https://github.com/ajwagen/stable-baselines3-dsrl/tree/10e5d311b5c42fe571ff3eeb72f622ca616a149d) | `10e5d311b5c42fe571ff3eeb72f622ca616a149d` | `stable_baselines3/dsrl/dsrl.py`、`sac/sac.py` | DSRL主仓固定的submodule；NA更新顺序、冻结前向、SAC |
| [rail-berkeley/hil-serl](https://github.com/rail-berkeley/hil-serl/tree/c32939bccb65f3b8c43a9f9add3d322d4ab0264a) | `c32939bccb65f3b8c43a9f9add3d322d4ab0264a` | `examples/train_rlpd.py`、`experiments/config.py`、agent builder、SAC | 实机双buffer、异步learner、网络与loss；机器人动作需改 |
| [tongzhoumu/policy_decorator](https://github.com/tongzhoumu/policy_decorator/tree/92ba9ba442587ae5989c286355ace2200a0537fb) | `92ba9ba442587ae5989c286355ace2200a0537fb` | `online/pi_dec_diffusion_maniskill2.py` | 状态输入ManiSkill示例，不代表所有实机默认值 |
| [RLinf/RLinf](https://github.com/RLinf/RLinf/tree/c94c67c12c27547ad6244d3957eb473955c7d46d) | `c94c67c12c27547ad6244d3957eb473955c7d46d` | `realworld_rlt_stage2_ac_mlp.yaml`、`fsdp_rlt_ac_policy_worker.py`、`fsdp_sac_policy_worker.py`、`rlt_mlp_policy.py` | 公开RLT复现的实际配置/loss/调度，非原作者完整实验包 |
| [microsoft/FlowDAgger](https://github.com/microsoft/FlowDAgger/tree/d2e60b5bf5d39538783f1e73d460394b21eb0f28) | `d2e60b5bf5d39538783f1e73d460394b21eb0f28` | `flowdagger_pi05/train_flowdagger.py`、`flow_matching_inverter.py`、`train_utils.py`、`dagger_loop.py` | π0.5/MetaWorld最小参考实现；未提供本仓AHA适配 |

这里的“直接阅读”指源代码审计，未安装这些外部系统的全部环境依赖，未运行其原论文实验。公开代码存在不意味着本机真机端已经具备对应机器人控制接口。

## 3. 本地代码与数据证据

| 路径 | 证据 |
|---|---|
| [ahawam.py](../../../src/ahawam/models/wan22/ahawam.py) | prefill噪声入口、关闭action history、OVCR当前观测路径 |
| [ahawam_chunk_base.py](../../../src/ahawam/models/wan22/ahawam_chunk_base.py) | 当前块噪声切片、去噪、state推进与父类cache写入 |
| [base_wam.py](../../../src/ahawam/models/wan22/base_wam.py) | 显式start_latents和seed生成 |
| [scheduler_continuous.py](../../../src/ahawam/models/wan22/schedulers/scheduler_continuous.py) | 非均匀σ网格、负delta Euler积分 |
| [rl_token_readout.py](../../../src/ahawam/models/wan22/rl_token_readout.py) | 独立重建读出原型，不是已接入的RL系统 |
| [ahawam_policy.py](../../../deploy/server/ahawam_policy.py) | action服务参数、normalization及异步snapshot接口 |
| [run_action_server.sh](../../../scripts/so101_pipeline/online_deploy/run_action_server.sh) | 已支持从部署profile读取serving；不重复旧入口错误推断 |
| [hil_human_only_dataset.py](../../../scripts/so101_pipeline/hil_human_only_dataset.py) | HUMAN来源及有效性/连续性约束 |
| [SYSTEM_HOLD约定](../../hil/SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md) | control_source=2 |
| [v2实验文档](../v2_experiments/README_ZH.md) | 既有监督实验、任务数据和训练策略 |
| [audit_snapshot.json](audit_snapshot.json) | 全部70个历史HIL parquet及manifest/request ledger的聚合统计 |

数据目录：`data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v2`。快照读取所有parquet的control_source、clamp/drop、loop_dt和action_age，并聚合所有 `episodes_manifest.jsonl`、`policy_chunks.jsonl`。分位数使用NumPy percentile，排除非有限延迟值；schema为一个样例parquet的列名。聚合outcome按manifest标签计数，不重新人工判定成功。

用户提供的前置信息：附件 `pasted-text.txt`，含HALO-WA、插入RL及既有迁移研究的历史说明；用于理解任务范围，技术结论仍以当前源码与一手方法证据为准。

## 4. 仍未被证实的事项

- 输出residual、RLT读出、DSRL-SAC中哪条在这台SO101上更好。
- 当前部署模型的实时推理延迟；旧HIL日志不是当前测速。
- 当前OVCR表征是否足以识别插入接触状态。
- AHA动作反演的误差、接受率、计算速度与动作范围。
- 小actor本地15Hz执行是否满足本机算力和相机延迟条件。
- UniSteer全部原始训练细节的公开代码对应关系。
- RLT原作者未在已阅读材料中明确给出的全部数值设置。

这些不确定项在实验计划中各有独立诊断；没有用“算法曾在别处有效”替代本地验证。

