# 2026-09-15：论文改稿、八页预算与最终图件

> 后续融合修订：`main.tex`已恢复，初稿引言/骨干/总览已融合，PDF当前为8页，英文插座脚注已同步。下文关于7页和main.tex暂缺的描述保留为当时记录，最新取舍见[融合说明](07_OLD_DRAFT_INTEGRATION_ZH.md)。

### 插座夹具补充说明

用户补充：SO101无法提供原装插座所需的插入力，实际使用移除内部组件的插座外壳。已在中文稿V-A任务描述后加一句，限定为几何对准与插入评估。该事实影响任务定义，放在实验设置或首次描述任务的脚注即可，不必在摘要或贡献中反复强调。

参考放置方式：[CR-DAgger §4.1](https://arxiv.org/html/2506.16685#S4.SS1)将齿轮材质、轴和底板抬高设置放在任务描述；[RSS 2021插入研究](https://www.roboticsproceedings.org/rss17/p088.pdf)在任务/实验叙述中交代连接器、夹具和测试条件。这是对信息位置的参考，未查到可核实的“同样拆空插座并仅用脚注说明”的直接先例，不据此声称别人也采用相同简化。

随后`main.tex`已恢复，英文脚注现已直接插入第V-A节（Robot, Tasks, and Data）首次介绍power-adapter任务的句末，源文件第145行；PDF已重新编译，仍为7页。可复用文本保存在[socket_fixture_note.tex](socket_fixture_note.tex)：

> To accommodate the SO101's force limits, we use a socket housing with its internal components removed, evaluating geometric alignment and insertion rather than electrical mating or the insertion-force requirements of an intact socket.

## 1. 现在读哪份

- [main.tex](latex/main.tex)：英文论文源稿，按现有实现整体重写。
- [main_中文全文.md](main_中文全文.md)：对应的完整中文阅读稿，公式统一使用`$`和`$$`。
- [main.pdf](latex/main.pdf)：实际编译的双栏工作稿，当前**7页，包含参考文献**。为新结果、真实组合图与最终复现参数保留约1页；不应为凑满8页加入未做方法。
- [references.bib](latex/references.bib)：本次核验并实际引用的8个条目，替代不存在的`new-refs.bib`依赖。
- [figures/method_overview.tex](figures/method_overview.tex)：可编辑TikZ方法矢量图，已经进入PDF。

当前PDF用于协作写作：有`pending`和实验图占位，不是可以直接投稿的完整结果论文。填结果时应同步摘要、结果分析、结论和中文版；没有完成的实验需要删去相应承诺，不能仅把将来时改为过去时。

## 2. 官方格式：本年度与旧习惯的区别

[ICRA 2027官方投稿页](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)明确：全文最多8页，包含图、表、致谢和参考文献；采用双栏及双匿名。FAQ还说明补充文字材料也要放在8页以内，不能默认另有不计页数的supplement PDF。

因此本稿使用官方`ieeeconf`，移除初稿中含占位作者A/B的作者栏，改为匿名。没有缩小正文、修改页边距或用负间距挤版面。官方class来自[PaperCept模板入口](https://ras.papercept.net/conferences/support/tex.php)，同目录`ieeeconf.cls`保持下载原样。

当前已确认的编译结果是7页，而不是已经排满8页的终稿。预留空间主要用来填两任务结果、实际偏移/超时/部署参数和扩大真实实验图。替换图或增加段落后必须重新编译检查，不能依赖本次页数。

## 3. 参考了哪些论文的写法

| 论文/一手来源 | 参考的写作组织 | 对本稿的具体影响 |
|---|---|---|
| [HG-DAgger，ICRA 2019](https://arxiv.org/pdf/1810.02890) | 定义交互方式，再给方法与机器人验证 | 将human-gated清楚定义为人类接管；不把采集循环本身包装为新RL |
| [Model-Based Runtime Monitoring，ICRA 2024](https://arxiv.org/html/2310.17552) | 明确实验问题，将系统表现与组件测试分开 | 完整任务成功率与固定偏差恢复分开；不以局部指标代替系统成功 |
| [Sirius，RSS 2023](https://www.roboticsproceedings.org/rss19/p005.pdf) | 人机数据来源、学习目标、整体性能与数据分析形成闭环 | 解释不同控制来源的数据为什么承担不同角色；不声称均衡回放首创 |
| [CR-DAgger](https://arxiv.org/html/2506.16685) | 接触任务中由具体失败问题引出方法，对采集和更新分别说明 | 以短纠正缺失前后文为问题，不虚构柔顺/力控、残差头贡献 |
| [AHA-WAM](https://arxiv.org/abs/2606.09811) | 骨干结构与已有目标的直接来源 | 明确Video DiT、Action DiT、OVCR和flow目标来自骨干；本文改训练样本构造 |

后两篇不是被当作“已接受ICRA论文”的依据。本文的结构选择是参考这些论文后的编辑判断，不是ICRA强制要求每篇必须有几张图或几个公式。

## 4. 最终推荐：两张图，其中只有一张方法结构图

### 图1：双栏方法总图，已有可编译版本

上半部分：demo→任务SFT→真机人类门控交互→demo与纠正混合的后训练。下半部分：同一条轨迹的policy/human/hold，并列标注过去上下文、人类动作标签、真实未来视频目标。

这张图应让读者一眼明白“控制权边界不等于视觉输入/监督边界”。无需再独立画一张AHA-WAM全层架构图；如果美化时补双相机图标和骨干框图，应标明是继承组件，不能让读者误以为OVCR是本工作的新增设计。图1是方法示意，没有虚构实机照片或数据。

### 图2：双栏实验组合图，需要真实素材

- (a) 两任务真实装置：相机位置、抓持对象和插入目标。固定相机/腕部视角可作为小窗。
- (b) 局部恢复条件：中心＋四方向各两档偏移，给实际毫米值或经标定的坐标；每点分别标H/HV成功次数/3。只有9个离散条件，不绘制未经测量的连续恢复边界。
- (c) 一组匹配偏差起点下的纠正过程，标真实时间戳。应配一个失败/无收益例子，或者在文字明确案例选择规则。

将最终图保存为`figures/experiment_panel.pdf`，main.tex自动替换占位。预留高度约5cm；真实三面板通常需要6–7cm，优先保证两栏缩印后的文字和孔口仍可辨认。用实际出图后页数决定是否拆成两行。

如果真实材料太多，可以将图2拆成“任务设置单栏图＋恢复结果双栏图”，总共3张图，但仍只要1张方法图。默认两张总图最紧凑。

### 视频预测图是否要独立放

默认不单独占一张正文图。当前主证据是动作恢复，不是视频画质。只有固定输入H/HV对照有可靠发现时，才把生成/真实对照加入图2或用剩余页数增加图3。生成视频明确标生成，不称为候选动作的反事实模拟。纯定性材料可进入合规的附带视频，不能承担正文唯一证据。

## 5. 四张表，各自只回答一个问题

| 表 | 目的 | 当前内容 | 最终需补 |
|---|---|---|---|
| I 配置矩阵 | 哪些因素不同 | S/C/H/HV/H2，两任务共8纠正＋2SFT | 核对最终训练配置不再改变 |
| II 数据与监督覆盖 | 为什么短接管需要特殊处理 | 两任务人类帧、片段、轨迹数 | 同一loader口径的段长、零历史率、视频有效率 |
| III 完整任务 | 方法是否改善实际任务完成 | cylinder 1/13、5/13、6/13及描述性区间 | power、C/H2、最终评估step和实际分母 |
| IV 局部恢复 | 视频延伸是否改变同错误状态的恢复 | H/HV、两任务，所有格待测 | 偏差成功率、中心参考、截断耗时、重试 |

不要再加一张重复表III的成功率柱状图。回放敏感性直接读H/H2行，历史对照读C/H，延伸读H/HV。H2当前没有延伸，HV对H2不是单变量。Power C还改了history stride及HOLD过滤，正文已明确这是历史构造整体对照。

表II若来不及审计，删除未统计行，保留已有规模数字或将它们并入实验设置文字。不能从不同offset或抽样口径拼一列“有效率”。表IV若无法完成，则删表、删图中对应结果面板，并弱化第三贡献为评估视角；保留全是待测格的表不能构成投稿证据。

## 6. 为什么只有四个编号公式

1. 历史与动作条件：说明memory是什么，以及历史编码的stop-gradient边界。
2. 三种时间范围：本文方法的核心定义；控制来源不统一决定历史和视频边界。
3. Flow插值/速度目标：让读者知道“action/video loss”实际在预测什么，不假装是新目标。
4. mask后的联合目标：动作保持人类标签、视频只平均有效样本；说明延伸改变了哪些监督。

demo比例用行内$q_r=r\mathcal D+(1-r)\mathcal H$足够；超时计分用文字足够。删除原稿DPO、Gaussian actor、critic/RL公式，也不展开原生OVCR每层attention。没有假设验证的Lipschitz上界不放正文；没有新增理论保证就不凑定理。中文版使用美元数学分隔；LaTeX保留标准`equation`环境以自动编号和交叉引用。

## 7. 工程细节如何取舍

正文保留会改变解释或复现的细节：完整模型更新范围、两阶段起点、3卡/global batch 12、学习率与更新预算、6帧历史、同rollout拆分、HOLD不进入延伸执行时间线、完整视频窗口和动作mask规则、同部署条件测试。

正文省略：服务器/隧道路径、yaml文件名、队列管理、恢复训练日志、历史失败脚本、adapter探索、TensorRT部署尝试，以及长checkpoint路径。这些读[复现交接](01_WRITER_HANDOFF_ZH.md)和[最终实验映射](05_FINAL_TABLES_AND_FIGURES_ZH.md)即可。最终推理步数、执行chunk、相机尺寸/频率等需要从**实际测试配置**填写一小段，不能从一个默认yaml猜成所有实测都相同。

初稿未完成的VLM标签、DPO、冻结骨干小actor、ego预训练、加速及“首个WAM+RL”均已删去。硬件漂移和Video DiT导致恢复的机制仍未被隔离验证，不写成结论。

## 8. 八页预算与补结果顺序

建议正文与图表约6.8–7.2页，引用与必要声明约0.6–0.8页，总数不超过8。当前编译为7页，有余量而没有人为撑页。

1. 优先填power完整任务和两任务最终checkpoint绑定，保证两任务主线有证据。
2. 填H/HV局部恢复及真实图2；这最贴近第二贡献。
3. 补C与H2结果、表II覆盖审计。
4. 将第VI节“如果……则……”分析准则替换为实际发现，删去已不需要的计划语句。
5. 根据最终证据重写摘要和三条贡献；只有实际完成的项目使用完成时态。

当前结果不支持“HV显著更优”“机械漂移已解决”“两任务都改善”。原有13场景的汇总仍是初步、回溯明确口径的记录。新结果需要真实计数，不从额外重复中挑选最有利的分母。

## 9. 编译与验证

使用同目录[build_paper.sh](latex/build_paper.sh)。优先latexmk，其次本机tectonic，再次pdflatex＋bibtex。例：

```bash
cd docs/icra2027_handoff
./build_paper.sh
```

本次临时编译器在`/tmp/icra_tex_tool/tectonic`，不作为项目依赖发布；若仍存在，可执行：

```bash
TECTONIC_BIN=/tmp/icra_tex_tool/tectonic XDG_CACHE_HOME=/tmp/icra_tex_cache ./build_paper.sh
```

本次已实际编译并检查：7页US Letter双栏、引用/公式引用可解析、无overfull盒子、字体嵌入且无Type 3；图1和结果表页面已视觉检查。仍有少量underfull排版提示，不是越界。最终真实图和新增结果会改变分页，需重跑；这不替代PaperPlaza最终PDF检查。

官方要求对AI生成内容作具体披露，本次涉及方法/实验文字重写和结构图，因此加入了事实性致谢声明，而非仅将工作称为语法润色。作者应核对最终保留范围。另据同一[官方页面](https://2027.ieee-icra.org/contribute/call-for-icra-2027-papers-now-accepting-submissions/)，附带视频第二上传窗口为9月17–22日，最多180秒/20MB；它不是延期补写正文实验结果的窗口。
