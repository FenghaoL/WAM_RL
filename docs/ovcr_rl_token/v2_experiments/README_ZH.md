# SO101 power-adapter v2：OVCR 纠正训练实验

本目录对应 2026-09-05 的数据审计和可执行实验。旧目录中的 v1 数据示例不适用于本轮。这里的 adapter 实验是**带 replay 的纠正模仿学习**，不是已经实现的 RLT 在线强化学习。

- [研究依据与初始化选择](01_RESEARCH_AND_DECISIONS_ZH.md)
- [数据、采样、训练预算与消融](02_PROTOCOL_ZH.md)
- [运行、验证与当前状态](03_EXECUTION_ZH.md)

先从已有任务 SFT **step_005328.pt** 出发，冻结 P0，仅训练零初始化 OVCR residual adapter。首轮比较 V32 / Q32 / T4，均使用 demo:HUMAN=1:1 和 demo P0 anchor。随后用 V32 比较自然混合、3:1、取消 anchor。HUMAN-only 为低优先级压力测试。

配置、数据 manifest、模型和日志统一保存在 `runs/ovcr_adapter_padapter_v2/`；可复现脚本在 `scripts/so101_pipeline/ovcr_adapter/v2_experiments/`。不会把旧训练数据加入本轮配置。

2026-09-06 新增：[Demo + HUMAN 全量微调与条件视频监督](04_FULL_FINETUNE_ZH.md)，作为 V/Q/T 真机效果一般后的直接模型更新对照。

2026-09-07：[V32/Q32/T4机制复核、真正RL路线、四轮SFT与部署审计](05_MECHANISMS_RL_AND_DEPLOYMENT_REVIEW_ZH.md)。包含启动脚本未指向 `settings_dagger.yaml` 的核查结果。

2026-09-07 已按用户要求启动：[B方案：AHA预训练出发，自然混合四轮全量SFT](06_B_PRETRAINED_NATURAL_4EPOCH_ZH.md)。
