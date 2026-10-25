# AHA-WAM 项目文档导航

> 更新日期：2026-09-07
> 说明：文档已按 `architecture/`、`research/`、`hil/`、`training/`、`project/` 和专项 `ovcr_rl_token/` 分类；本页是统一导航入口。

### [项目中英文简历表述与岗位关键词](./project/AHAWAM_RESUME_PROJECT_DESCRIPTION_ZH_EN.md)

提供一页简历、研究型 CV 和 LinkedIn 可直接使用的中英文项目介绍，并按机器人学习、操作算法、ML Systems 和部署岗位整理关键词；严格区分已实现的 HIL/HG-DAgger式监督训练与尚未完成的完整 RL/成功率结果。

## 1. 当前最应该先读

### [已有 DAgger 数据上的离线 RL：RLT/HALO-WA 接口、时序选择与可运行候选](./offline_rl/README_ZH.md)

针对 cylinder HIL v2 的成功帧标注，比较连续 tick 与原 chunk＋hold 宏动作，提供双 Q 的 IQL、TD3+BC、特征读出与训练入口；记录真实指定批次审计、CPU 验证以及尚待决定的部署时序和基座。

### [OVCR 小适配器与 RL Token：研究、实现和实验方案](./ovcr_rl_token/README.md)

核查 RLT 原文及 AHA 代码，提供 Query、Value-only 和多 token K/V writer 三种可训练方案，说明任务 checkpoint、HUMAN 数据、P0 保持损失与实验命令；单独讨论 critic、异步 replay 和真正 RL 的扩展边界，并记录已完成的 CPU 验证。

已完成的实验轮次：[power-adapter v2](./ovcr_rl_token/v2_experiments/README_ZH.md)；进行中：[cylinder v1（含 wandb 记录）](./ovcr_rl_token/cylinder_experiments/README_ZH.md)。

### [02+04 Demo 与 HUMAN Correction 两组 5-Epoch 训练实现](./hil/AHAWAM_HIL_HUMAN_ONLY_TRAINING_IMPLEMENTATION_ZH.md)

当前可直接运行的训练方案。说明 v1/v2/v3 HUMAN-only logical episode、短段 action-only/长段 video+action、episode 尾部 padding、固定 stats、八卡 A800 ZeRO-2 no-offload，以及 natural concat 和严格 2:2 两个无参数入口。

### [HIL 数据采完后的首轮训练与 RECAP 决策](./hil/AHAWAM_HIL_AFTER_COLLECTION_TRAINING_AND_RECAP_DECISION_ZH.md)

对应当前已经采完的 35 条 HIL rollout。审计 73-demo task-SFT、97 段 correction、SYSTEM_HOLD 与 checkpoint provenance，明确第一轮应从采集 policy continuation 还是从官方 AHA base 重训、怎样做 75:25/50:50 分源采样、为什么现有全成功数据还不能独立支撑 RECAP value，以及 HIL-SERL/RLDG/WAM-RL 的后续优先级。

### [SYSTEM_HOLD 训练端数据契约](./hil/SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md)

定义 rich-v3 的 SYSTEM_HOLD、未来 40 步 action supervision mask、旧 rich-v1/v2 时间缺口、correction-only 导出兼容性和训练端验收测试。实现本轮 HG/IWR loader 前必须阅读。

### [SO101 + AHA-WAM 在线接管数据采集实现交接](./hil/SO101_AHAWAM_HIL_DATA_COLLECTION_IMPLEMENTATION_HANDOFF_ZH.md)

面向机械臂部署电脑的可直接执行说明。比较 HG-DAgger、IWR、HIL-SERL 与 RLDG 的数据语义，判断 LeRobot v0.6 的复用边界，并规定 AHA action chunk 清队列、在途 reply 失效、leader/follower 平滑交接、富标注 schema、逻辑 episode、数据导出和分阶段安全测试。

### [IG-OVCR 截稿期最小采集方案与 rich-v4 路线图](./hil/SO101_AHAWAM_IG_OVCR_RICH_V4_COLLECTION_AND_REALTIME_EXECUTION_HANDOFF_ZH.md)

基于最新版 rich-v3 的逐字段和时序审计，给出不改服务端/执行器的 deadline-minimal v3.1：保存完整 raw，但训练使用 demo anchor + HUMAN correction，并只补 checkpoint hash、takeover-frame 映射、assisted outcome 和 exporter/validator。文档后半保留 full rich-v4、异步预取、RTC 与 paired residual 作为截稿后路线图。

### [从 π*0.6 / RECAP 到 SO101 插头插入：实验方案](./research/PI06_RECAP_TO_SO101_PLUG_EXPERIMENT_PLAN_ZH.md)

对应当前“抓起插头并插入插线板”实验。它回答：

- π*0.6 / RECAP 每一轮从哪个 checkpoint 开始、是否累计数据、各实验用了几轮和多少数据；
- HG-DAgger 怎样实时接管，接管前后数据怎样保存和训练；
- 当前 73 条训练 episode、2/4 epoch checkpoint 和部署配置中有哪些混杂因素；
- 今天应该先补局部插入 demo，还是直接采 HG；
- SFT-only、HG-DAgger、IWR 与 RECAP-like 的公平消融和论文叙事。

## 2. 理解 AHA-WAM 本体

### [AHA-WAM 架构：从数学到代码](./architecture/AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md)

从前置数学、张量形状和注意力机制出发，结合仓库代码解释 VideoDiT、ActionDiT、异步 video/action 推理、chunk-KV editor、训练目标及部署数据流。

### [AHA-WAM 异步推理、RTC、两步去噪与 SO101 部署审计](./architecture/AHAWAM_ASYNC_RTC_FLASH_AND_SO101_DEPLOYMENT_AUDIT_ZH.md)

区分 AHA 的慢 planner/快 executor 异步、RTC 的旧块执行与新块 inpainting，以及 AHA-WAM-Flash 的 2-step ODE 蒸馏；逐行审计当前服务端/客户端，说明为何不能只给 `run_action_server.sh` 加一个开关，以及 SO101 控制端必须补的双连接协议、相机拼接、reset 与 HIL generation fencing。

### [AHA-WAM-Flash ODE 蒸馏：从直觉、数学到本项目代码](./architecture/AHAWAM_FLASH_ODE_DISTILLATION_FROM_FIRST_PRINCIPLES_ZH.md)

解释当前 Flash 训练蒸馏的是动作 ODE 计算路径而非新任务能力，展开 teacher trajectory、per-chunk anchor 和两步采样数学，并比较 SO101 四卡配置与论文官方数据/batch 规模。

### [HALO-WA RL 架构与迁移指南](./research/HALO_WA_RL_AND_AHAWAM_MIGRATION_GUIDE_ZH.md)

解释 HALO-WA 的完整流程、强化学习模块与 AHA-WAM 的结构差异，并区分可以迁移的算法思想和不能直接复制的实现假设。

## 3. 插入、RL 与数据闭环研究

### [插入实验中底座位姿是否变化：逐项核查](./research/INSERTION_RECEPTACLE_POSE_RANDOMIZATION_AUDIT_ZH.md)

逐项区分 HALO-WA、RLT、HIL-SERL、RLDG、ViSk、EDRIAD、SHIELD、ACT 等工作究竟是
移动母口、只随机公头/末端，还是论文没有说明；同时抽样审计本地 cylinder 数据，并给出
反事实位置对、held-out 位姿和视觉消融的最低成本实验设计。

### [从 DAgger 到 WAM-specific Interactive Correction：文献、设计与迁移判断](./research/AHAWAM_DAGGER_METHODS_AND_WAM_SPECIFIC_DESIGN_ZH.md)

针对当前最重要的方法选择：系统比较 IWR、SIRIUS、APO、Diff-DAgger、CR-DAgger、FlowCorrect、world-model gating 与 HALO-WA，分别给出影响力、证据可靠性、AHA 适配度和实现成本评分；提出 intervention-gated OVCR correction，并详细回答 HUMAN correction 应更新 Video DiT、Action DiT、LoRA 还是 OVCR/KV editor。

### [AHA-WAM DAgger 快速实验清单](./research/AHAWAM_DAGGER_FAST_TRACK_EXPERIMENT_PLAN_ZH.md)

上述综述的短版执行入口：数据五类标签、最低主表、flow uncertainty 离线检查、Action-LoRA→OVCR adapter 实现顺序、rich-v4 shadow-policy 日志与 Go/No-Go 标准。

### [AHA-WAM 插入任务 RL 文献综述](./research/AHAWAM_INSERTION_RL_LITERATURE_REVIEW_ZH.md)

覆盖 WAM + RL、WAM + 插入、VLA/扩散动作 + RL、接触丰富操作和精密装配，重点讨论各路线迁移到 SO101/AHA-WAM 的条件与风险。

### [AHA-WAM RL 范围与 video/action 联合优化重审](./research/AHAWAM_RL_SCOPE_AND_VIDEO_ACTION_COTRAINING_REVIEW_ZH.md)

讨论外挂 actor–critic、ActionDiT 直接优化、VideoDiT 适配及全模型联合训练的边界；补充标题中不直接写 “RL” 的 VLA 经验学习工作。

## 4. 训练工程

### [SO101 AHA-WAM-Flash 蒸馏运行目录](../scripts/so101_pipeline/flash_distillation/README.md)

从 power-adapter-v2 `step_005328.pt` 启动仓库原生 ODE 蒸馏：严格继承 6D SO101 模型/数据形状，使用 16-step teacher、2-step student、ZeRO-2 optimizer offload，并提供两卡或四卡启动、状态和停止入口。

### [Flash + 异步部署与机械臂端交接](../scripts/so101_pipeline/flash_async_deploy/ROBOT_CLIENT_HANDOFF_ZH.md)

说明蒸馏 student checkpoint 自动解析、双 TCP 连接、显式 reset、双相机拼接、15 Hz
chunk 执行、HIL generation fencing、时间戳和上机门槛；同时明确异步
planner/executor 不是 RTC action inpainting。

### [SO101 同步 rich-v3 HUMAN-only DAgger 采集交接](./hil/SO101_AHAWAM_SYNC_V3_HUMAN_ONLY_COLLECTION_HANDOFF_ZH.md)

暂停 Flash/异步路线时，使用四轮 SFT step_005328.pt 按原同步 infer 进行 rollout；说明
为何 v3 足够构造连续 HUMAN correction 训练 view、推理 HOLD 应如何记录，以及本轮只需
补齐 checkpoint provenance 而不必升级完整 rich-v4。

### [SO101 同步 v3/data-lite 多用途 DAgger 采集交接](./hil/SO101_AHAWAM_SYNC_V3_MULTIPURPOSE_DAGGER_COLLECTION_HANDOFF_ZH.md)

当后续可能同时使用 HUMAN correction、带推理 HOLD 的自主数据、world/video loss、
IWR 或粗粒度 value 时使用；给出 v3 的能力边界和不阻塞采集的 request/chunk ledger
最小扩展。

### [Hugging Face 训练资产目录与自动恢复](./training/AHAWAM_HF_TRAINING_ASSETS_ZH.md)

给出 02、04、HIL、预训练权重、固定 stats 与文本缓存的远端目录一一映射、本地上传命令，以及两个无参数训练入口如何从 public Dataset 匿名恢复缺失资产到默认路径。

### [AHA-WAM ZeRO-2 微调说明](./training/AHAWAM_ZERO2_TUNING_ZH.md)

记录多卡全量微调、ZeRO-2/offload、显存与吞吐、checkpoint 及常见训练问题。

## 5. 推荐阅读顺序

如果目标是**马上推进当前 SO101 插头实验**：

1. [当前 HUMAN-only 两组训练实现](./hil/AHAWAM_HIL_HUMAN_ONLY_TRAINING_IMPLEMENTATION_ZH.md)
2. [HIL 数据采完后的首轮训练与 RECAP 决策](./hil/AHAWAM_HIL_AFTER_COLLECTION_TRAINING_AND_RECAP_DECISION_ZH.md)
3. [SYSTEM_HOLD 训练端数据契约](./hil/SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md)
4. [SO101 + AHA-WAM 在线接管采集交接](./hil/SO101_AHAWAM_HIL_DATA_COLLECTION_IMPLEMENTATION_HANDOFF_ZH.md)
5. [截稿期最小采集方案与 rich-v4 路线图](./hil/SO101_AHAWAM_IG_OVCR_RICH_V4_COLLECTION_AND_REALTIME_EXECUTION_HANDOFF_ZH.md)
6. [π*0.6 / RECAP 到 SO101 实验方案](./research/PI06_RECAP_TO_SO101_PLUG_EXPERIMENT_PLAN_ZH.md)
7. [AHA-WAM 架构全解](./architecture/AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md)
8. [插入任务 RL 文献综述](./research/AHAWAM_INSERTION_RL_LITERATURE_REVIEW_ZH.md)
9. [RL 范围与联合优化重审](./research/AHAWAM_RL_SCOPE_AND_VIDEO_ACTION_COTRAINING_REVIEW_ZH.md)

如果目标是**实现完整的 RL/HG 数据管线**：

1. [DAgger/WAM-specific 方法综述](./research/AHAWAM_DAGGER_METHODS_AND_WAM_SPECIFIC_DESIGN_ZH.md)
2. [DAgger 快速实验清单](./research/AHAWAM_DAGGER_FAST_TRACK_EXPERIMENT_PLAN_ZH.md)
3. [HIL 数据采完后的首轮训练与 RECAP 决策](./hil/AHAWAM_HIL_AFTER_COLLECTION_TRAINING_AND_RECAP_DECISION_ZH.md)
4. [SYSTEM_HOLD 训练端数据契约](./hil/SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md)
5. [SO101 + AHA-WAM 在线接管采集交接](./hil/SO101_AHAWAM_HIL_DATA_COLLECTION_IMPLEMENTATION_HANDOFF_ZH.md)
6. [截稿期最小采集方案与 rich-v4 路线图](./hil/SO101_AHAWAM_IG_OVCR_RICH_V4_COLLECTION_AND_REALTIME_EXECUTION_HANDOFF_ZH.md)
7. [AHA-WAM 架构全解](./architecture/AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md)
8. [HALO-WA 迁移指南](./research/HALO_WA_RL_AND_AHAWAM_MIGRATION_GUIDE_ZH.md)
9. [π*0.6 / RECAP 到 SO101 实验方案](./research/PI06_RECAP_TO_SO101_PLUG_EXPERIMENT_PLAN_ZH.md)
10. [ZeRO-2 微调说明](./training/AHAWAM_ZERO2_TUNING_ZH.md)
