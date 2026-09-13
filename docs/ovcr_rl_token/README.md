# AHA-WAM：OVCR 小适配器与 RL Token 专项

在线 RL 后续研究（2026-09-10）：[框架可行性、官方实现参数、噪声反演与实机实验协议](online_rl_feasibility/README_ZH.md)。包含历史 HIL 时延/来源审计；补充 RLT 公开复现、DSRL-SAC/NA 的计算差异，以及 FlowDAgger/UniSteer 对人工噪声监督的依据。

审计日期：2026-09-05。范围：本地 AHA-WAM 代码、SO101 双 RGB 插入数据、RLT 原论文及相关一手资料。

**建议先做 Value-only 与 Query correction 的配对实验，再试多 token K/V writer。第一轮使用 demo + HUMAN 的监督 flow loss，不需要 critic。** 如果路由纠正有效，再决定用奖励加权 flow 更新内部 adapter，还是增加独立 RLT actor–critic。两者都是可研究的方向，但成本和科学问题不同。

本次已经实现三种原生 OVCR adapter、训练入口、同噪声 P0 anchor、checkpoint 元数据及加载、独立 readout 原型，并完成 CPU 回归验证。**没有完成大模型 GPU 训练、真机评测或在线 RL 闭环**；本次环境 `torch.cuda.is_available()=False`，`nvidia-smi` 无法连接驱动。代码可运行与任务成功率提升是两个不同验证层级。

| 阅读顺序 | 内容 |
| --- | --- |
| [01：RLT 与相关工作](01_RLT_AND_RELATED_WORK_ZH.md) | 原机制、迁移边界、文献依据、已有文档需要修正的判断 |
| [02：代码与方法设计](02_CODE_AND_METHODS_ZH.md) | 注入位置、张量、公式、梯度、三方案差异与实现契约 |
| [03：训练和实验](03_TRAINING_AND_EXPERIMENTS_ZH.md) | 从哪个 ckpt 出发、用什么数据、命令、参数、消融、验收 |
| [04：critic 与真正的 RL](04_CRITIC_AND_RL_ROADMAP_ZH.md) | BC/RL 区别、两条 RL 扩展、chunk TD、干预与异步 replay |
| [05：验证记录](05_IMPLEMENTATION_AND_VALIDATION_ZH.md) | 文件清单、已执行检查、当前限制、后续工程缺口 |

## 实验轮次

| 轮次 | 目录 | 内容 |
| --- | --- | --- |
| power-adapter v2（2026-09-05 起） | [v2_experiments](v2_experiments/README_ZH.md) | V32/Q32/T4、1:1 全量纠正、从预训练重训四轮，以及机制复核与部署审计 |
| cylinder v1（2026-09-08 起） | [cylinder_experiments](cylinder_experiments/README_ZH.md) | 同一套协议搬到 cylinder 任务，新增 wandb 在线记录与更大的固定验证子集 |

## 已实现方案

| ID | 配置 | 改写位置 | 默认可训练参数 | 推荐角色 |
| --- | --- | --- | ---: | --- |
| Q32 | [query.yaml](../../scripts/so101_pipeline/ovcr_adapter/query.yaml) | 原 OVCR query 输入 | 2,744,640 | 检验“现有 planner 内容足够，只需重新路由” |
| V32 | [value.yaml](../../scripts/so101_pipeline/ovcr_adapter/value.yaml) | 原 OVCR 输出的 V | 2,058,560 | 首选：不直接改变本层注意力寻址的纠正 |
| T4 | [token_kv.yaml](../../scripts/so101_pipeline/ovcr_adapter/token_kv.yaml) | 少量 token 解码出的 K/V 增量 | 5,511,360 | 检验压缩读写接口，容量更大 |

参数是按本地维度实例化后计数，默认只改第 20–29 层（从 0 编号）、rank=32；不是等参数量对照。T4 的 token writer **不是 RLT 原文的只读 RL token**。原始所有权重被冻结，最终动作仍由原 Action DiT 的 flow 采样生成。

训练入口：[train.py](../../scripts/so101_pipeline/ovcr_adapter/train.py)。可选的只读重建模块：[rl_token_readout.py](../../src/ahawam/models/wan22/rl_token_readout.py)，不自动进入部署热路径。

## 与已有项目材料的关系

本目录承接 [DAgger 与 WAM 专项设计](../research/AHAWAM_DAGGER_METHODS_AND_WAM_SPECIFIC_DESIGN_ZH.md)、[HUMAN-only 训练实现](../hil/AHAWAM_HIL_HUMAN_ONLY_TRAINING_IMPLEMENTATION_ZH.md)、[IG-OVCR 采集路线](../hil/SO101_AHAWAM_IG_OVCR_RICH_V4_COLLECTION_AND_REALTIME_EXECUTION_HANDOFF_ZH.md) 和 [架构逐式说明](../architecture/AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md)。本轮保留已有采集／部署改动；没有替换正在使用的服务器配置或启动机器人。

这里将“论文事实”“代码事实”“待实验假设”“已实现／仅设计”分别说明。不存在已经证明 OVCR adapter 优于 Action LoRA、或者冻结 Video 永远更好的结论。
