# ICRA 2027 写作参考与复现交接

**写作者先读[论文参考骨架](00_PAPER_SKELETON_ZH.md)。** 正文聚焦完整模型的纠正训练、跨接管历史与未来视觉监督；原生OVCR作为基础模型接口解释，未采用的小模块与其他探索不再列入正文。

最新用户确认：按相同13个初始化场景、每模型每场景第一次rollout，cylinder **SFT为1/13（7.7%），基本纠正full为5/13（38.5%），vidext为6/13（46.2%）**。额外重复测试与这一共同口径分开。具体测试step和逐场景成功/失败记录仍待核对，尚不能计算配对显著性；第一次rollout的筛选口径是本次回溯澄清，不声称预注册。

| 文档 | 用途 |
|---|---|
| [论文参考骨架](00_PAPER_SKELETON_ZH.md) | 自包含的背景、方法、损失与实验正文参考 |
| [复现交接表](01_WRITER_HANDOFF_ZH.md) | 有效数据、关键权重与代码入口 |
| [新结果后的实验取舍](02_CONTRIBUTION_AND_EXPERIMENT_PLAN_ZH.md) | 低成本平台的论证边界、vidext对照、明天补实验优先级与具体协议 |
| [脚本入口与验证](03_ENTRIES_AND_VALIDATION_ZH.md) | 已有可复用入口及power v3的当前准备限制 |
| [最新用户结果记录](user_report_results.json) | 用户汇总，未伪造逐trial或逐模型结果 |
| [历史工程快照](evidence_snapshot.json) | 当时配置/文件/日志证据，不是当前训练状态或实机成绩 |

三个重点：基本full已经有video loss；vidext仅扩大成功混合轨迹的视频目标，动作仍限定HUMAN；当前主效果属于整个纠正配方，尚不能单独归因于Video DiT或机械精度变化。

power新纠正目录为 `data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v3`；旧v1/v2因相机问题排除。新集合已存在，划分、训练和实机结果待补。建议先完成第二任务，再补同数据历史对照和无需训练的局部恢复范围评估。
