# ICRA 2027 写作与复现入口

当前稿：[英文PDF（8页，含参考文献）](latex/main.pdf)、[英文TeX](latex/main.tex)、[中文PDF（7页）](latex/main_中文全文.pdf)、[中文TeX](latex/main_中文全文.tex)、[中文Markdown](main_中文全文.md)。编译方法见[LaTeX工程说明](latex/README_ZH.md)。

当前标题：**Beyond the Takeover: Human-Corrected Predictive Learning for World–Action Models**。

## 当前论文主线

1. 一种面向WAM的交互学习框架。
2. 人类校正的预测式策略自改进。
3. 理解视频预测如何支持错误恢复。

先读[贡献简版](04_FINAL_CONTRIBUTIONS_ONE_PAGE_ZH.md)和[本次结果与机制分析](11_LAYOUT_CITATIONS_AND_FINAL_RESULTS_ZH.md)。第三项采用每任务三个checkpoint的验证视频质量与独立实机成功率比较，不是逐次rollout的配对成功预测，也不是新增推理时视频规划。

| 模型 | Cylinder成功率 | Power adapter成功率 |
|---|---:|---:|
| SFT：SFT | 7.7% | 50.0% |
| Segment-only：segdemo | 30.8% | 75.0% |
| History：跨接管历史 | 38.5% | 65.0% |
| History+Future：历史+视频延伸 | 46.2% | 55.0% |

两任务视频预测结果均已入文。圆柱呈现同向排序，适配器呈现预测改善未转化为进一步控制收益的差异。受控偏差恢复范围实验及2:1混合比例实验已从本次论文删除。预测质量与任务表现的关联不等于因果机制；遮挡解释只作为讨论中的假设。

## 证据和图件

- [逐模型成功率记录](analysis/task_success_results.json)：用户汇报的整数及适配器History+Future名称纠正，不伪造逐试验日志。
- [适配器视频指标审计](analysis/padapter_video_results_audit.json)。
- [视频指标审计](analysis/video_results_audit.json)、[重算脚本](analysis/audit_video_results.py)：online与none的胜出次数不同，以原始metrics.json为准。
- [系统流程](latex/figures/interactive_learning_framework.jpg)。
- [方法总览](latex/figures/method_overview.pdf)、[正文rollout拼图](latex/figures/rollout_comparison.pdf)、[完整33帧拼图](latex/figures/rollout_comparison_all_frames.pdf)。
- [编译验证](manuscript_validation.json)。
- [训练数据、权重与脚本交接](01_WRITER_HANDOFF_ZH.md)：工程复现参考；本次实测状态以主稿和11号记录为准。

## 历史资料

[原始骨架](00_PAPER_SKELETON_ZH.md)、[先前实验规划](05_FINAL_TABLES_AND_FIGURES_ZH.md)、[图件改版记录](09_FIGURES_AND_CONTRIBUTIONS_ZH.md)保留作设计过程参考，其中未执行实验不属于当前投稿结果。原始TeX在 `latex/old/`，原始照片仍在 `figures/rollouts/`。当前主稿不包含早期OVCR adapter负结果、未实现RL或推理加速贡献。
