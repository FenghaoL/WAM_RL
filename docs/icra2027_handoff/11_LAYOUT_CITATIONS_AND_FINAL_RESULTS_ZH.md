# 两任务视频结果、引用与版式更新（2026-09-15）

本文件取代10号记录中“适配器视频待补”的状态。当前主稿为 `latex/main.tex` 和 `latex/main_中文全文.tex`，中文阅读稿同步到 `main_中文全文.md`。

## 文章中的主要发现

纠正后训练在两任务上都提高了实机完成率和验证视频质量，但视频延伸的控制收益具有任务依赖性。圆柱的SFT/History/History+Future在预测与成功率上排序一致；适配器History+Future的视频预测进一步改善，成功率却从History的65%降至55%。这支持将第三贡献写为“理解视频预测如何支持错误恢复”，同时研究预测与执行之间的差异，而不是宣称视频更准必然使控制更好。

| 任务 | 模型 | PSNR (dB) | 固定视角 | 腕部视角 | MSE | 实机成功率 |
|---|---|---:|---:|---:|---:|---:|
| Cylinder | SFT | 21.45 | 24.07 | 15.71 | 501.3 | 7.7% |
| Cylinder | History | 22.74 | 25.07 | 17.19 | 362.8 | 38.5% |
| Cylinder | History+Future | 23.56 | 26.02 | 18.09 | 303.9 | 46.2% |
| Power adapter | SFT | 19.63 | 23.25 | 13.36 | 759.3 | 50.0% |
| Power adapter | History | 21.08 | 24.18 | 15.05 | 524.9 | 65.0% |
| Power adapter | History+Future | 21.61 | 24.87 | 15.52 | 469.4 | 55.0% |

Segment-only的成功率仍保留于主结果表：圆柱30.8%，适配器75%。未测Segment-only的视频预测，不能补造数值。

## 完整评估口径（正文不展开工程模式）

主表沿用两任务原始记录的online PSNR；逐窗口配对增益也来自同一模式。fixed、wrist、MSE为该次评估的相应指标。不是将none列与online配对增益混用。不同模式原始值均保留在审计JSON。

- 圆柱：7窗口、4个源episode；History/History+Future分别7/7优于SFT，平均增益1.30/2.11 dB。
- 适配器：12窗口、7个源episode；History/History+Future分别11/12优于SFT，平均增益1.44/1.97 dB。History+Future对History也为11/12，平均增益0.53 dB。
- 适配器验证集：`data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v3/20260915_211458_val`。
- 原始结果：`runs/ovcr_adapter_padapter_hil_v3/video_preview/val/` 下三个checkpoint的summary与逐窗口metrics。
- 可复算：`python3 docs/icra2027_handoff/analysis/audit_video_results.py --task padapter`；圆柱使用 `--task cylinder`。
- 审计输出：[圆柱](analysis/video_results_audit.json)、[适配器](analysis/padapter_video_results_audit.json)。包含准确权重路径、窗口数、重复尾帧与逐窗口差值。

所有窗口的预测范围固定；按接管段长度分组不等于改变预测时域。适配器长段组只有2段，圆柱只有1段，不据此宣称长时外推已得到验证。两个适配器窗口分别重复4帧与2帧末端参考图像，所有模型使用相同参考。预测验证与实机试验是独立数据，仅作checkpoint级比较。

运动幅度比值没有进入正文，也不作为PSNR改善的已知原因。腕部PSNR改善更大只说明视角差异，尚不能定位为接触阶段收益。遮挡解释保留为有文献动机、尚未验证的假设。

## 图件和作者

1. 图1：用户提供的系统流程JPEG，呈现示范、人类干预、后训练的整体闭环。图注说明当前只评估一轮纠正。
2. 图2：可编辑矢量图，保留时间上下文、动作与视觉监督边界及联合更新机制；去掉与图1重复的上方流程。
3. 图3：横向rollout，时间从左到右；每个任务SFT在上、DAgger在下，两任务上下排列。适配器SFT最后一帧改为RPC634；完整33帧来源图仍保留原图636。图片没有生成或修改场景内容。

作者顺序：Fenghao Liu（University of Michigan）、Yunpeng Zhu（Westlake University）、Lixi Fan（Zhejiang University）、Zhangyuan Wang（Westlake University）。前三位共同贡献，最后一位通讯作者；未添加未知邮箱。

## 引用如何服务论文主线

- [DAgger](https://proceedings.mlr.press/v15/ross11a.html)、[HG-DAgger](https://arxiv.org/abs/1810.02890)：分布偏移与人类门控交互。
- [IWR](https://arxiv.org/abs/2012.06733)、[Sirius](https://www.roboticsproceedings.org/rss19/p005.pdf)：纠正经验、混合回放和部署后学习。均衡回放本身不作为原创贡献。
- [CR-DAgger](https://compliant-residual-dagger.github.io/files/CR_DAgger_extended.pdf)：接触丰富任务的人类纠正；区别在于我们研究预测式策略的时间上下文，而非柔顺残差控制。
- [AHA-WAM](https://serene-sivy.github.io/aha-wam/)：继承骨干、记忆与上下文路由。明确没有新建骨干。
- [VideoVLA](https://arxiv.org/abs/2512.06963)：联合视觉与动作预测，以及想象质量与操作效果之间的联系，支撑第三贡献的问题设置。
- [Temporal Skip Connections](https://arxiv.org/abs/1710.05268)：自监督视觉规划与时间信息对遮挡处理的作用；不用于证明本任务遮挡的原因。
- [Model-Based Runtime Monitoring](https://arxiv.org/abs/2310.17552)、[Hi-WM](https://arxiv.org/abs/2604.21741)：区分预测触发干预、模型内纠正与本文真机纠正。
- [WorldGym](https://openreview.net/forum?id=hidBHy1CAw)：动作条件模拟中的策略评估，与本文独立视频验证和实机成功率的比较不同。
- [Models, Pixels, and Rewards](https://arxiv.org/abs/2012.04603)：预测与行为评价之间的关系并非新问题，应明确我们的增量是纠正后训练下的实证分析。
- [Vision and Touch](https://arxiv.org/abs/2410.23860)：插入时视觉与接触信息的互补性。

正文已删减重复动机、模式名与工程实现展开。保留监督边界、掩码目标、采样比例、训练预算和评价限制等影响方法理解或实验解释的要素。

## 展示名称

论文统一使用SFT、Segment-only、History、History+Future。分别对应审计文件中的S、C、H、HV；保留机器可读键和checkpoint路径，避免影响结果复算。History+Future指延伸未来视频目标，History并非彻底删除所有视频损失。

## 标题与八页扩写

标题采用“Beyond the Takeover: Human-Corrected Predictive Learning for World–Action Models”。摘要按作者提供文本同步改写；将“WAM无法从交互改进”限定为示范训练尚未利用闭环经验，保留初步实机发现与预测—行为分析目标。

借鉴旧稿的三类动机结构，新增内容集中于交互后训练、时间上下文和预测学习的行为锚定。正文移除GPU数量、分布式训练描述、具体帧索引等实现细节；保留监督边界与主要训练对照条件。没有添加未执行的RL、DPO、加速或额外实机结果。英文为8页（含参考文献），中文阅读版7页。
