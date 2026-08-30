# AHA-WAM 项目文档导航

> 更新日期：2026-08-30
> 说明：这里按“当前实验决策—架构—研究综述—工程训练”组织入口。为避免破坏已有引用，没有移动或重命名原文档。

## 1. 当前最应该先读

### [02+04 Demo 与 HUMAN Correction 两组 5-Epoch 训练实现](./AHAWAM_HIL_HUMAN_ONLY_TRAINING_IMPLEMENTATION_ZH.md)

当前可直接运行的训练方案。说明 v1/v2/v3 HUMAN-only logical episode、短段 action-only/长段 video+action、episode 尾部 padding、固定 stats、八卡 A800 ZeRO-2 no-offload，以及 natural concat 和严格 2:2 两个无参数入口。

### [HIL 数据采完后的首轮训练与 RECAP 决策](./AHAWAM_HIL_AFTER_COLLECTION_TRAINING_AND_RECAP_DECISION_ZH.md)

对应当前已经采完的 35 条 HIL rollout。审计 73-demo task-SFT、97 段 correction、SYSTEM_HOLD 与 checkpoint provenance，明确第一轮应从采集 policy continuation 还是从官方 AHA base 重训、怎样做 75:25/50:50 分源采样、为什么现有全成功数据还不能独立支撑 RECAP value，以及 HIL-SERL/RLDG/WAM-RL 的后续优先级。

### [SYSTEM_HOLD 训练端数据契约](./SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md)

定义 rich-v3 的 SYSTEM_HOLD、未来 40 步 action supervision mask、旧 rich-v1/v2 时间缺口、correction-only 导出兼容性和训练端验收测试。实现本轮 HG/IWR loader 前必须阅读。

### [SO101 + AHA-WAM 在线接管数据采集实现交接](./SO101_AHAWAM_HIL_DATA_COLLECTION_IMPLEMENTATION_HANDOFF_ZH.md)

面向机械臂部署电脑的可直接执行说明。比较 HG-DAgger、IWR、HIL-SERL 与 RLDG 的数据语义，判断 LeRobot v0.6 的复用边界，并规定 AHA action chunk 清队列、在途 reply 失效、leader/follower 平滑交接、富标注 schema、逻辑 episode、数据导出和分阶段安全测试。

### [从 π*0.6 / RECAP 到 SO101 插头插入：实验方案](./PI06_RECAP_TO_SO101_PLUG_EXPERIMENT_PLAN_ZH.md)

对应当前“抓起插头并插入插线板”实验。它回答：

- π*0.6 / RECAP 每一轮从哪个 checkpoint 开始、是否累计数据、各实验用了几轮和多少数据；
- HG-DAgger 怎样实时接管，接管前后数据怎样保存和训练；
- 当前 73 条训练 episode、2/4 epoch checkpoint 和部署配置中有哪些混杂因素；
- 今天应该先补局部插入 demo，还是直接采 HG；
- SFT-only、HG-DAgger、IWR 与 RECAP-like 的公平消融和论文叙事。

## 2. 理解 AHA-WAM 本体

### [AHA-WAM 架构：从数学到代码](./AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md)

从前置数学、张量形状和注意力机制出发，结合仓库代码解释 VideoDiT、ActionDiT、异步 video/action 推理、chunk-KV editor、训练目标及部署数据流。

### [HALO-WA RL 架构与迁移指南](./HALO_WA_RL_AND_AHAWAM_MIGRATION_GUIDE_ZH.md)

解释 HALO-WA 的完整流程、强化学习模块与 AHA-WAM 的结构差异，并区分可以迁移的算法思想和不能直接复制的实现假设。

## 3. 插入、RL 与数据闭环研究

### [AHA-WAM 插入任务 RL 文献综述](./AHAWAM_INSERTION_RL_LITERATURE_REVIEW_ZH.md)

覆盖 WAM + RL、WAM + 插入、VLA/扩散动作 + RL、接触丰富操作和精密装配，重点讨论各路线迁移到 SO101/AHA-WAM 的条件与风险。

### [AHA-WAM RL 范围与 video/action 联合优化重审](./AHAWAM_RL_SCOPE_AND_VIDEO_ACTION_COTRAINING_REVIEW_ZH.md)

讨论外挂 actor–critic、ActionDiT 直接优化、VideoDiT 适配及全模型联合训练的边界；补充标题中不直接写 “RL” 的 VLA 经验学习工作。

## 4. 训练工程

### [Hugging Face 训练资产目录与自动恢复](./AHAWAM_HF_TRAINING_ASSETS_ZH.md)

给出 02、04、HIL、预训练权重、固定 stats 与文本缓存的远端目录一一映射、本地上传命令，以及两个无参数训练入口如何从 public Dataset 匿名恢复缺失资产到默认路径。

### [AHA-WAM ZeRO-2 微调说明](./AHAWAM_ZERO2_TUNING_ZH.md)

记录多卡全量微调、ZeRO-2/offload、显存与吞吐、checkpoint 及常见训练问题。

## 5. 推荐阅读顺序

如果目标是**马上推进当前 SO101 插头实验**：

1. [当前 HUMAN-only 两组训练实现](./AHAWAM_HIL_HUMAN_ONLY_TRAINING_IMPLEMENTATION_ZH.md)
2. [HIL 数据采完后的首轮训练与 RECAP 决策](./AHAWAM_HIL_AFTER_COLLECTION_TRAINING_AND_RECAP_DECISION_ZH.md)
3. [SYSTEM_HOLD 训练端数据契约](./SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md)
4. [SO101 + AHA-WAM 在线接管采集交接](./SO101_AHAWAM_HIL_DATA_COLLECTION_IMPLEMENTATION_HANDOFF_ZH.md)
5. [π*0.6 / RECAP 到 SO101 实验方案](./PI06_RECAP_TO_SO101_PLUG_EXPERIMENT_PLAN_ZH.md)
6. [AHA-WAM 架构全解](./AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md)
7. [插入任务 RL 文献综述](./AHAWAM_INSERTION_RL_LITERATURE_REVIEW_ZH.md)
8. [RL 范围与联合优化重审](./AHAWAM_RL_SCOPE_AND_VIDEO_ACTION_COTRAINING_REVIEW_ZH.md)

如果目标是**实现完整的 RL/HG 数据管线**：

1. [HIL 数据采完后的首轮训练与 RECAP 决策](./AHAWAM_HIL_AFTER_COLLECTION_TRAINING_AND_RECAP_DECISION_ZH.md)
2. [SYSTEM_HOLD 训练端数据契约](./SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md)
3. [SO101 + AHA-WAM 在线接管采集交接](./SO101_AHAWAM_HIL_DATA_COLLECTION_IMPLEMENTATION_HANDOFF_ZH.md)
4. [AHA-WAM 架构全解](./AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md)
5. [HALO-WA 迁移指南](./HALO_WA_RL_AND_AHAWAM_MIGRATION_GUIDE_ZH.md)
6. [π*0.6 / RECAP 到 SO101 实验方案](./PI06_RECAP_TO_SO101_PLUG_EXPERIMENT_PLAN_ZH.md)
7. [ZeRO-2 微调说明](./AHAWAM_ZERO2_TUNING_ZH.md)
