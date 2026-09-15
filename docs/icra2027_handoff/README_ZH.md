# ICRA 2027 写作参考与复现交接

**2026-09-15论文改稿入口：[英文稿](main.tex)、[完整中文稿](main_中文全文.md)、[已编译PDF](main.pdf)。** 已按当前实现替换初稿中的未实现RL/ego/加速内容，保留4张表及2个图位；当前7页含引用，待测项明确标注。图件素材、八页预算和修改依据见[本次修订说明](06_MANUSCRIPT_REVISION_AND_FIGURES_ZH.md)。

**写作者先读[三个贡献简版](04_FINAL_CONTRIBUTIONS_ONE_PAGE_ZH.md)，再读[最终实验表格与配图](05_FINAL_TABLES_AND_FIGURES_ZH.md)。** 这两份是根据两个任务、八个纠正checkpoint收敛后的写作入口；[论文参考骨架](00_PAPER_SKELETON_ZH.md)提供背景与方法展开。正文聚焦完整模型的纠正训练、跨接管历史与未来视觉监督；原生OVCR作为基础模型接口解释，未采用的小模块与其他探索不再列入正文。

最新用户确认：按相同13个初始化场景、每模型每场景第一次rollout，cylinder **SFT为1/13（7.7%），基本纠正full为5/13（38.5%），vidext为6/13（46.2%）**。额外重复测试与这一共同口径分开。具体测试step和逐场景成功/失败记录仍待核对，尚不能计算配对显著性；第一次rollout的筛选口径是本次回溯澄清，不声称预注册。

| 文档 | 用途 |
|---|---|
| [三个贡献简版](04_FINAL_CONTRIBUTIONS_ONE_PAGE_ZH.md) | 可直接转发共同作者的工作定位、三项贡献与证据边界 |
| [最终实验表格与配图](05_FINAL_TABLES_AND_FIGURES_ZH.md) | 八个纠正模型加两个SFT的矩阵，全部主表/附表、恢复指标、视频诊断及执行优先级 |
| [论文参考骨架](00_PAPER_SKELETON_ZH.md) | 自包含的背景、方法、损失与实验正文参考 |
| [复现交接表](01_WRITER_HANDOFF_ZH.md) | 有效数据、关键权重与代码入口 |
| [新结果后的实验取舍](02_CONTRIBUTION_AND_EXPERIMENT_PLAN_ZH.md) | 低成本平台的论证边界、vidext对照、明天补实验优先级与具体协议 |
| [WAM贡献论证与相关工作研究](contribution_research/README_ZH.md) | 自监督与人类纠正的准确表述、数学形式、配比分析及免重训优化可行性 |
| [脚本入口与验证](03_ENTRIES_AND_VALIDATION_ZH.md) | 已有可复用入口及power v3的当前准备限制 |
| [最新用户结果记录](user_report_results.json) | 用户汇总，未伪造逐trial或逐模型结果 |
| [历史工程快照](evidence_snapshot.json) | 当时配置/文件/日志证据，不是当前训练状态或实机成绩 |

三个重点：基本full已经有video loss；vidext仅扩大成功混合轨迹的视频目标，动作仍限定HUMAN；当前主效果属于整个纠正配方，尚不能单独归因于Video DiT或机械精度变化。

power新纠正目录为 `data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v3`；旧v1/v2因相机问题排除。当前版本已提供轨迹划分与实验配置，见[HIL v3记录](../ovcr_rl_token/padapter_hil_v3/README_ZH.md)；旧文档的“尚未划分”属于历史快照。各模型的训练完成状态和实机结果仍需独立登记。优先完成第二任务，再做共享偏差起点下的H/HV恢复对照。
