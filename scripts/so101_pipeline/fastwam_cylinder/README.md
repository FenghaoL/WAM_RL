# FastWAM cylinder experiments

四项训练、公开权重与优化器说明见
[完整文档](../../../docs/project/FASTWAM_CYLINDER_SFT_AND_CORRECTION_20261105_ZH.md)。

当前队列：`runs/fastwam_cylinder_20261105/queue.json`。
顺序：Franka joint → cylinder demo SFT；RoboCOIN → 同样 demo SFT 对照；
再从 Franka SFT 分别启动 action-only / full-video correction 两个独立分叉。

Native FastWAM：无 history、无 offset/OVCR，49 原始帧 → 13 视频帧＋48 动作，
执行时取前 8 动作。所有阶段仅初始化权重，每阶段只保留最新一份权重，无 optimizer state。
