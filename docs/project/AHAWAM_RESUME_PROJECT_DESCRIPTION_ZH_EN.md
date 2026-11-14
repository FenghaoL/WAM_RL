# AHA-WAM × SO-101 精密操作项目：中英文简历表述与岗位关键词

2026-10-26 补充：包含 HG-DAgger、末层 planner KV 小 actor、IQL/TD3+BC 及计划内在线 RL 的新版材料见 [项目完整总结与简历稿](AHAWAM_HGDAGGER_OFFLINE_ONLINE_RL_RESUME_ZH.md)。本文件保留为 2026-08-30 的早期快照。

> 整理日期：2026-08-30  
> 适用方向：机器人学习、具身智能、Robot Learning Research Engineer、Robotic Manipulation ML Engineer、VLA/WAM Research、强化学习与模仿学习工程。  
> 当前状态：真机 SFT、在线人工接管数据闭环和 HUMAN-only 联合 video/action 训练管线已经实现；最终真机成功率、完整 value/advantage RL 和跨任务结果仍在实验中。

## 1. 推荐项目名称

### 中文

**基于异步世界—动作模型与人类纠正学习的 SO-101 精密插接系统**

更短版本：

**AHA-WAM 精密操作与人机协同后训练**

### English

**Precision Insertion on SO-101 via an Asynchronous World–Action Model and Human-Correction Learning**

Short version:

**Human-in-the-Loop Post-Training for AHA-WAM Robotic Manipulation**

“AHA-WAM”是已有模型，因此不要把项目标题写成“提出/发明 AHA-WAM”。准确的贡献边界是：将预训练 AHA-WAM 适配到 SO-101，构建真机部署、人工接管、数据治理和后训练系统，并研究其精密操作能力。

## 2. 一句话介绍

### 中文

将由 5B Video DiT 与 1.02B Action DiT 构成的异步 AHA-WAM 部署到双相机 SO-101，并围绕插头抓取—对准—插入任务构建了采用 15 Hz 控制协议的真机控制、在线人工接管、可追溯纠正数据和多卡联合 video/action 后训练闭环。

### English

Adapted a 6.02B-parameter asynchronous AHA-WAM, comprising a 5B Video DiT and a 1.02B Action DiT, to a dual-camera SO-101 and built a real-robot pipeline under a 15 Hz control protocol for plug grasping, alignment, and insertion with online human intervention, traceable correction data, and distributed video–action post-training.

## 3. 一页中文简历：推荐直接使用版

**AHA-WAM 精密操作与人机协同后训练｜机器人学习项目｜2026.08–至今**  
`Python` · `PyTorch` · `AHA-WAM/Wan2.2` · `Flow Matching` · `LeRobot` · `HG-DAgger` · `DeepSpeed ZeRO-2` · `Hugging Face`

- 将官方预训练 AHA-WAM 适配至低成本 SO-101，打通固定相机、腕部相机、6维关节状态/动作和远程模型服务，按 15 Hz 控制协议执行 8-step action chunks；对端到端时序进行 profiling，定位动作去噪与有状态异步推理中的主要延迟和历史缓存一致性问题。
- 构建面向精密插接的 Human-in-the-Loop/HG-DAgger 式采集闭环，设计 `POLICY/HUMAN/SYSTEM_HOLD` 控制状态、旧 action queue/in-flight reply 失效和 human→policy hard reset 语义；采集并审计 35 条真机 rollout、22,314 个控制帧及 97 段人工纠正（9,158 帧）。
- 实现 HUMAN-only 时序数据加载与损失掩码：按控制帧连续性切分逻辑 episode，阻止自主动作跨入 40-step expert target，并针对短纠正采用 action-only、长纠正采用 video+action 监督；验证所有训练 anchor 均不包含自主控制帧。
- 搭建 6B 级模型的 bf16/gradient-checkpointing/DeepSpeed ZeRO-2 多卡全量微调与预检流程，支持 demo–correction 自然拼接和严格 2:2 分源采样，为 SFT、HG-DAgger/IWR 及后续 RECAP-like RL 建立可复现对照。

如果一页简历空间紧张，优先保留前三条；如果目标岗位强调大模型训练，再保留第四条并压缩第一条。

## 4. One-page English résumé: recommended version

**Human-in-the-Loop Post-Training for AHA-WAM Precision Manipulation | Robot Learning Project | Aug. 2026–Present**  
`Python` · `PyTorch` · `AHA-WAM/Wan2.2` · `Flow Matching` · `LeRobot` · `HG-DAgger` · `DeepSpeed ZeRO-2` · `Hugging Face`

- Adapted a pretrained asynchronous World–Action Model to a low-cost SO-101, integrating fixed and wrist RGB cameras, 6-DoF joint observations/actions, and remote inference with 8-step action chunks under a 15 Hz control protocol; profiled end-to-end timing and diagnosed denoising latency and stateful-cache consistency issues.
- Built a Human-in-the-Loop, HG-DAgger-style data-collection loop with explicit `POLICY`, `HUMAN`, and `SYSTEM_HOLD` control states, stale action/reply invalidation, and hard policy resets after interventions; collected and audited 35 real-robot rollouts containing 22,314 control frames and 97 human-correction segments (9,158 frames).
- Implemented a temporally safe HUMAN-only dataset and masked training objective that split logical episodes at control discontinuities, prevented autonomous actions from leaking into 40-step expert targets, and applied action-only supervision to short corrections and joint video–action supervision to long corrections; verified zero autonomous frames in the training anchors.
- Engineered reproducible full-parameter fine-tuning for a 6.02B-parameter dual-DiT model using bf16, gradient checkpointing, and DeepSpeed ZeRO-2, with natural demo–correction concatenation and strict 2:2 source-balanced sampling for SFT, HG-DAgger/IWR, and future RECAP-style comparisons.

## 5. 研究型 CV / 实验室申请详细版

### 中文

**项目目标。** 研究低成本、无腕部力/力矩传感器的 SO-101 能否利用异步视频世界模型、双相机 RGB 反馈和少量人类纠正完成抓取—搬运—毫米级对准—插入，并分析精密阶段的性能瓶颈来自视觉世界表征、动作生成，还是二者的异步耦合。

**主要工作。**

- 完成 AHA-WAM 训练与推理代码审计，梳理 Video DiT、Action DiT、OVCR/KV editor、flow-matching action generation、action chunking 和有状态异步缓存的数据流，并将其映射到 SO-101 的 6维绝对关节动作空间。
- 建立双 RGB 相机真机部署链路和 15 Hz 控制协议，支持模型 prefill、action chunk 请求、远程 reset、观测历史重建和推理延迟记录；通过实测分析发现当前硬件上的 action-chunk 推理约为 533.6 ms，并将主要耗时定位到多步动作去噪。
- 面向模型在抓偏、粗对准和孔口附近失败时产生的 on-policy 状态，设计可多次接管/交还的 HIL 数据采集协议；逐帧区分 policy proposal、human proposal 与实际 executed action，并保存接管原因、请求/chunk ID、相机与控制时间戳及安全裁剪信息。
- 审计 35 条 HIL rollout：原始数据共 22,314 帧，其中 9,158 帧为 HUMAN、对应 97 个连续纠正段；按 episode、frame/tick 连续性和 dropped tick 切分 logical episodes，消除网络等待和暂停阶段造成的伪连续时间轴。
- 修改 AHA-WAM dataset、sampler、processor、trainer 和 video/action loss，使 15 个不足49帧的短段仅提供 action supervision，82 个长段提供 video+action supervision；对 episode 尾部保留逐 token padding mask，并确保自主 POLICY/SYSTEM_HOLD 帧不进入本轮训练输入或损失。
- 构建从共同预训练权重重启的累计数据训练方案，比较原始 demonstrations、natural demo+correction、source-balanced correction training 和 correction-only continuation；为后续 success/failure value learning、advantage-conditioned Action DiT 和插入 specialist RL 保留统一数据契约。

**当前实验与下一步。** 当前正在进行真机 checkpoint 对比和插入成功率评测。计划在固定初始化集合上报告完全插入率、部分啮合率、完成时间、接管时间、失败类型和二项置信区间；在完成前不应将“性能提升”写成既成结论。

### English

**Objective.** Investigate whether a low-cost SO-101 without wrist force/torque sensing can use an asynchronous video world model, dual-RGB feedback, and limited human corrections to perform grasping, transport, millimeter-scale alignment, and insertion, while identifying whether the precision bottleneck lies in world representation, action generation, or their asynchronous coupling.

**Contributions.**

- Audited the AHA-WAM training and inference stack, tracing the dataflow across the Video DiT, Action DiT, OVCR/KV editor, flow-matching action generator, action chunking, and stateful asynchronous cache, and mapped the policy to the SO-101's 6-DoF absolute joint-action space.
- Developed a dual-RGB real-robot deployment pipeline under a 15 Hz control protocol with model prefill, chunked action requests, remote reset, observation-history reconstruction, and latency telemetry; measured approximately 533.6 ms per action-chunk request on the current hardware and localized the dominant cost to iterative action denoising.
- Designed an intervention protocol for learner-induced grasping, coarse-alignment, and near-contact failures, supporting multiple takeovers and releases per rollout while separately logging policy proposals, human proposals, executed actions, intervention reasons, request/chunk identifiers, camera/control timestamps, and safety clamps.
- Audited 35 HIL rollouts with 22,314 frames, including 9,158 HUMAN frames across 97 continuous correction segments; constructed logical episodes from episode/frame/tick continuity and dropped-tick metadata to eliminate false temporal adjacency caused by inference waits and operator pauses.
- Extended the AHA-WAM dataset, sampler, processor, trainer, and video/action objectives so that 15 correction segments shorter than 49 frames receive action-only supervision while 82 longer segments receive joint video–action supervision; preserved token-level end-of-episode padding and excluded all autonomous `POLICY/SYSTEM_HOLD` frames from the current training view.
- Established cumulative-data retraining from a shared pretrained initialization and reproducible comparisons among demonstration-only SFT, natural demo–correction mixing, source-balanced correction training, and correction-only continuation, leaving a compatible path toward success/failure value learning, advantage-conditioned Action DiT training, and insertion-specialist RL.

**Status.** Real-robot checkpoint evaluation is in progress. Final reporting will use a fixed initialization set and include full-insertion success, partial engagement, completion time, intervention time, failure categories, and binomial confidence intervals.

## 6. LinkedIn / 个人主页短段落

### 中文

我正在将异步视频—动作世界模型 AHA-WAM 迁移到双相机 SO-101，用于插头抓取、对准和插入等接触丰富操作。项目覆盖从真机控制和远程推理、在线人类接管，到带严格时间连续性和 supervision mask 的纠正数据训练。我已构建35条HIL rollout、97段人工纠正的数据闭环，并实现6B级模型的多卡 video/action 联合后训练；当前正在评测HG-DAgger式纠正学习，并准备进一步研究 advantage-conditioned action training 与插入阶段 specialist RL。

### English

I am adapting the asynchronous AHA-WAM world–action model to a dual-camera SO-101 for contact-rich plug grasping, alignment, and insertion. The project spans real-robot control and remote inference, online human intervention, temporally consistent correction datasets, and masked video–action post-training. I have built a data loop with 35 HIL rollouts and 97 human-correction segments and implemented distributed post-training for the 6B-scale model; current work evaluates HG-DAgger-style correction learning and explores advantage-conditioned action training and insertion-specialist RL.

## 7. 岗位关键词：中英文映射

以下关键词来自当前机器人操作岗位中反复出现的能力要求，并结合了本项目实际完成的内容。ATS关键词必须自然出现在项目 bullet 或 Skills 中，不能无依据堆叠。

| 中文关键词 | English keyword | 当前项目证据 |
| --- | --- | --- |
| 机器人学习 | Robot Learning | AHA-WAM 真机后训练与评测 |
| 机器人操作、精密操作 | Robotic Manipulation, Precision Manipulation | 抓取、对准、插入 |
| 接触丰富操作 | Contact-Rich Manipulation | 插头—插座接触与卡阻恢复 |
| 世界—动作模型、世界模型 | World–Action Model, World Model | Video DiT 与 Action DiT 异步耦合 |
| 视觉—语言—动作模型 | Vision–Language–Action Model, VLA | 可作为相邻方向关键词；正文优先准确写WAM |
| 模仿学习、行为克隆 | Imitation Learning, Behavior Cloning | demonstration SFT 与 correction SFT |
| 人在回路学习 | Human-in-the-Loop Learning, HIL | 在线接管、交还和纠正数据 |
| HG-DAgger、交互式模仿学习 | HG-DAgger, Interactive Imitation Learning | learner rollout中的人工纠正 |
| 强化学习后训练 | Reinforcement Learning Post-Training | 当前是后续方向，完整RL实现后再列为已掌握成果 |
| 流匹配 | Flow Matching | Video/Action DiT训练目标 |
| 扩散Transformer | Diffusion Transformer, DiT | 5B Video DiT + 1.02B Action DiT |
| 动作分块 | Action Chunking | 40-step horizon、8-step chunks |
| 多模态学习 | Multimodal Learning | 双RGB、语言、proprioception、action |
| 真机部署 | Real-Robot Deployment | SO-101远程推理和控制闭环 |
| 遥操作与数据采集 | Teleoperation, Robot Data Collection | SO-101 leader接管与富标注日志 |
| 策略评测、失效分析 | Policy Evaluation, Failure-Mode Analysis | 固定评测与时序/失败诊断 |
| 安全关键控制、恢复行为 | Safety-Critical Control, Recovery Behavior | stale reply失效、hard reset、安全裁剪 |
| 分布式训练 | Distributed Training | Accelerate + DeepSpeed ZeRO-2 |
| 数据管线与质量控制 | Data Pipeline, Data Quality, Quality Gates | logical episode、mask、preflight |
| 多相机视觉 | Multi-Camera Vision | fixed + wrist RGB |

### 按岗位选择关键词

**Robot Learning / Research Engineer**

`Robot Learning`, `World Models`, `VLA/WAM`, `Imitation Learning`, `Human-in-the-Loop Learning`, `Flow Matching`, `Diffusion Transformer`, `Policy Evaluation`, `Real-Robot Experiments`, `PyTorch`

**Robotic Manipulation ML Engineer**

`Robotic Manipulation`, `Contact-Rich Manipulation`, `Behavior Cloning`, `Teleoperation Data`, `Training Pipelines`, `Failure-Mode Analysis`, `Recovery Behaviors`, `Real-Robot Deployment`, `Distributed Training`

**ML Systems / Foundation Model Engineer for Robotics**

`Large-Scale Model Training`, `PyTorch`, `DeepSpeed ZeRO-2`, `Mixed Precision`, `Gradient Checkpointing`, `Data Loading`, `Checkpointing`, `Latency Profiling`, `Reproducible Experiments`

**Robotics Software / Deployment Engineer**

`Real-Time Control`, `Action Arbitration`, `State Machine`, `Multi-Camera`, `Remote Inference`, `Concurrency`, `Safety`, `Telemetry`, `Dataset Logging`

本项目当前没有直接使用 ROS、MuJoCo、Isaac Sim、TensorRT、ONNX、力/力矩或触觉传感器，因此即使岗位描述常出现这些词，也不应放进本项目的已用技术列表。

## 8. 获得真机结果后替换成结果导向版本

当前最缺的是公平评测结果。完成评测后，把中文简历第一或第四条替换为：

> 在 `[N]` 个固定、分层随机初始化的真机 trial 上，将完全插入成功率由 demonstration-only SFT 的 `[A%]` 提升至 `[B%]`，同时将平均人工接管时间降低 `[C%]`；使用相同初始状态集合和 Wilson 95% 置信区间进行比较。

English:

> Improved full-insertion success from `[A%]` with demonstration-only SFT to `[B%]` over `[N]` stratified real-robot trials while reducing mean human-intervention time by `[C%]`; compared policies on matched initializations and reported Wilson 95% confidence intervals.

若没有达到显著提升，也可以诚实写成诊断性结果：

> 通过分阶段评测发现 correction training 将末端对准成功率提高 `[X]`，但完整任务仍受抓取偏移限制；据此将后续数据采集预算从完整demo转向 learner-induced grasp recovery states。

这类结论依然体现研究判断，比只写“调参并训练模型”更有价值。

## 9. 不应提前写入简历的表述

- 不写“提出 AHA-WAM”——模型主体来自已有工作。
- 暂不写“实现了 RECAP/强化学习”——当前完成的是 HIL/HG-DAgger式纠正数据与监督训练；完整 value、advantage conditioning 和 reward experience 尚未落地。
- 不写“将成功率提升至……”“达到毫米级精度”——在固定协议的真机评测完成前没有证据。
- 不写“实时15 Hz推理”——当前控制协议为15 Hz，但实测 chunk 推理和等待行为需要准确区分；可写“15 Hz control protocol”或给出实测延迟。
- 不写“八卡A800训练完成”——训练脚本支持相应配置，但简历中的硬件数量应以实际完成日志为准。
- 不把训练 loss 降低直接等同于真机操作性能提升。

## 10. 为什么采用这种写法

技术简历的通用标准是“动作 + 具体任务/方法 + 可验证结果”，而不是只列职责。UC Davis建议使用 `Action Verb + Context = Results`，MIT和UF的技术简历指南同样强调动作、项目和结果，并要求在有意义时量化。[UC Davis Career Center](https://careercenter.ucdavis.edu/resumes-and-materials/resumes/accomplishment-statements)、[MIT CAPD](https://capd.mit.edu/resources/resumes-writing-about-your-skills/)、[University of Florida Technical Resume Guide](https://careerhub.ufl.edu/resources/technical-resume-guide/)

当前机器人学习岗位对该项目最匹配的要求包括：

- Diligent Robotics要求VLA、行为克隆、Transformer/扩散策略、遥操作日志数据管线、真机评测、安全和失效分析。[Diligent Robotics — ML Engineer, Manipulation](https://job-boards.greenhouse.io/diligentrobotics/jobs/7651459003)
- Omakase Robotics明确要求真实遥操作数据上的VLA/模仿学习、flow-matching action heads、多GPU训练、数据质量门和超越loss曲线的真机评测。[Omakase Robotics — Robotics Researcher, Manipulation](https://job-boards.greenhouse.io/omakaserobotics/jobs/6134913004)
- Maven Robotics强调机器人操作中的RL/IL、Python/PyTorch以及从研究算法到真机部署。[Maven Robotics — ML Engineer, Robot Manipulation](https://job-boards.greenhouse.io/mavenrobotics/jobs/4139600008)
- Skild AI强调大模型机器人后训练、真实部署数据、可靠性、安全性以及强化/模仿学习。[Skild AI — Research Engineer, Post-training & Deployment](https://job-boards.greenhouse.io/skildai-careers/jobs/4091494008)

因此推荐bullet有意突出“真机系统—数据—训练—评测”闭环，而没有把篇幅全部用于罗列论文缩写。
