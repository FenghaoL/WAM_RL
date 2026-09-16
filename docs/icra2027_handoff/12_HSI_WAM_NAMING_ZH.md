# HSI-WAM命名、行文与匿名版本（2026-09-16）

当前标题：**HSI-WAM: Human-Corrected Predictive Self-Improvement for World–Action Models**。

## 修改范围

摘要只在提出框架处加入HSI-WAM，并将核心机制术语统一为human-corrected predictive self-improvement；保留作者原有结构、动机和结果表述。引言、贡献、方法节、图注、讨论与结论统一名称。HSI-WAM指交互后训练框架，继承AHA-WAM骨干，不表述为新预训练模型。

SFT、Segment-only、History、History+Future继续作为实验名称；说明History+Future使用框架的完整时间上下文，避免与方法名混淆。自改进仍定义为在人类指导下从自身部署经验后训练，不追加自主RL或多轮迭代的实验主张。

## 参考的呈现方式

[VideoVLA](https://arxiv.org/abs/2512.06963)、[Hi-WM](https://arxiv.org/abs/2604.21741)和[WAM-TTT](https://arxiv.org/abs/2607.06988)均在提出方法时引入名称，再说明对应机制。本次采用这种统一命名方式，没有照搬其贡献或结果。

补充了WAM-TTT引用与方法比较：它从人类视频预测适配轻量记忆并冻结WAM；本文从真机人类纠正与真实上下文联合更新现有视频—动作策略。新增一段阐明世界模型在本文中是被纠正策略的一部分，而非采集纠正的独立模拟环境。这些比较用于定位贡献，同时充实匿名版篇幅。

## 匿名与验证

当前中英文TeX和PDF均无作者署名与致谢，中文Markdown同步匿名。历史草稿和内部交接资料保留历史信息；匿名检查针对当前稿，不能将整个历史目录直接视为匿名投稿材料。

英文8页（含15条参考文献），中文7页。未调整字号或页边距以压缩篇幅。验证记录见[manuscript_validation.json](manuscript_validation.json)。
