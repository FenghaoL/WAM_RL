# 价值函数的全量视频检查

日期：2026-10-25。输出目录：`runs/offline_rl/value_videos_20261025`；入口是目录中的 **`index.html`**。

## 评价了哪些模型、哪些动作

IQL 和 TD3+BC 的外部 actor 实验都已完成 10,000 步，Action DiT 的加权 LoRA 实验也已完成 2,000 步。本次固定两项外部实验最终 checkpoint，使用 **当前 critic 的两路 Q**，以及 **IQL 的 V**；没有使用慢更新的 target Q。文件路径、SHA256、cache 和训练数据来源记录在输出的 `provenance.json`。

IQL 有训练过的 Q1/Q2/V；TD3+BC 训练 Q1/Q2，没有独立的已训练 V。统一 learner 对象里即使有 value 模块，也不能把 TD3 下那个未优化模块当成训练好的 V。

V 的输入是冻结 planner/OVCR KV＋当前关节状态，KV 中含历史条件，所以不是只给单张 RGB 图评分。Q 还接收完整 **8×6 数据动作块**：从当前观察之后的命令开始，只去掉 SYSTEM_HOLD，保留 POLICY/HUMAN，自然往后取 8 个实际执行命令。不是把一个动作复制八次，不是 actor 输出，也不是 reference。数值的含义是“此状态下执行这个动作块的估计价值”，不能单独解释为图片的进度。

诊断构造的每一个原训练/验证窗口都会和训练文件的 `action_frames`、实际动作逐值比较；动作归一化使用和训练 loader 相同的 FP32 乘、加、clamp。原始 parquet、双相机视频也会与导出 KV 时的哈希核对。两路 Q 共享前干，各有独立输出 head；差异小不能证明它们的共同估计正确。

## 输出覆盖与命名

全量覆盖原 103 episodes、43,680 个原始观察，不只抽几条漂亮结果：

| 目录 / 索引筛选 | 数量 | 是否参与 Q/V 拟合 |
| --- | ---: | --- |
| `train/success` | 78 | 是，使用成功前有效窗口 |
| `train/failure` | 13 | 是，使用有效窗口 |
| `val/success` | 3 | 否，原验证划分 |
| `excluded_early_stop/train_source` | 8 | 否，原训练来源目录中的提前停止录像 |
| `excluded_early_stop/val_source` | 1 | 否，原验证来源目录中的提前停止录像 |

现有 `_val` 目录没有失败 episode，不能为了凑齐类别而把训练过的失败 episode 改称 held-out。9 条提前结束录像是用户报告实验最终成功、但录制时未记录到完整成功的轨迹，不是失败。文件名包含 `excluded_early_stop`、`success_early_stop_no_success_frame`，保留其真实来源分区。

原视频完整播放，包括 HOLD 和成功后的部分；训练集 episode 不代表它的所有帧都参与了拟合。画面顶部 `scope` 逐帧标明：

- `TRAIN_WINDOW`：原训练窗口起点。
- `VAL_WINDOW`：原验证窗口起点。
- `EXCLUDED_EPISODE_PROBE`：9 条排除 episode 的探测评分。
- `POST_SUCCESS_PROBE`：有效成功帧及之后；没有当作普通训练起点。
- `NONTRAINING_OBSERVATION`：其余观察，如 HOLD、丢帧附近、末尾。

**V 在全部原始帧上计算；HOLD、成功后等区域要按外推结果看。** Q 仅在当前帧为非 HOLD、能自然取满 8 个有后继观察的命令、且区间内没有缺 tick / dropped row 时计算。否则显示 N/A，曲线断开，不填零、重复最后动作、延用上一帧评分或跨 gap 偷接。

## 画面和文件

MP4 为 H.264、1280×960，保留原数据 FPS 与全部帧。固定相机和腕部相机左右并排，保留宽高比。中间显示 IQL V、IQL Q1/Q2、TD3 Q1/Q2、REC return；下方显示最近 300 原始帧，数值未平滑、不做每 episode 的归一化。

- 所有视频统一纵轴，由全体原始评分的范围确定，并包含 0、1；实际预测不裁剪。
- 灰色背景表示 HOLD；绿色竖线是奖励对齐后的有效成功观察，橙色竖线是原成功标注观察，两者可重合。
- Q 两个 head 接近时曲线会重叠，数值保留四位小数，CSV 保存更高精度，不能通过看一条粗线就断言它们完全相等。
- `index.html` 可按分组与 run / episode 名筛选，视频按需加载；每条提供 MP4、CSV 和 JSON 链接。JPG 是接近成功或录像后段的预览。
- `scores/*.csv` 包含帧号、原控制 tick、真实经过秒数、控制来源、五个分数、Q min、REC return、原窗口 reward / discount、IQL 一步 Bellman target、动作帧列表、缺失原因与 scope。
- `summary.json` 按分组汇总原有效窗口的值域、均值、Q 差异、成功/失败 terminal 拟合误差、IQL Bellman 残差，并保留每 episode 统计。主要分组统计不会把成功后探测帧混进原训练窗口。
- `complete.json` 记录最终视频数量、总帧数、文件体积、耗时与评分显存；每个视频在发布前核对编码帧数。

图上的时间轴以原始帧号为准；`elapsed` 使用录制 monotonic 时间戳，保留控制缺帧造成的时间跳变。视频没有额外插帧去填补丢失控制 tick。

## 分数基准与正确读法

本任务成功只发一次 +1，其余即时奖励 0，保留动作时间的 gamma=0.997。理想折扣回报范围是 **[0,1]**；越晚得到成功奖励，折扣越多。但这些网络是无约束标量回归，不是 sigmoid 分类器，预测可以略小于 0 或大于 1，**不能将 0.8 直接称为“成功概率 80%”**。IQL V 还是 expectile 价值，不是普通分类概率。

两种算法使用同一奖励尺度，但后续策略和估计方式不同；同一帧上 TD3 的 Q 比 IQL 高，不等于 TD3 的估计更准确。V 与动作条件 Q 也不是必须逐点相同。是否准确应结合各自的监督目标与独立评估，而不能只比较谁的数值大。

绿色 `REC return` 按已经记录的行为轨迹，计算从当前动作到有效成功动作的 `gamma^(剩余有效动作数−1)`；失败录像的记录行为回报为 0。它只在有效 Q 起点且后续奖励路径被完整观察时画出；跨 missing tick 的正回报、成功后区域与提前停止的 9 条都留空。

这条参考线有助于看趋势和尺度，但**不是当前 actor 的 Q 真值**。Q 的后续行为来自 TD3 actor 或 IQL 的隐式策略近似，而录像后续可能包括 HUMAN 接管；一条最终失败的行为轨迹中，前段状态仍可能被另一策略救回。不能仅凭前段 Q>0 就断言 critic 错误。

更直接的拟合检查是：

1. 包含成功的训练窗口没有 bootstrap，Q 的目标就是块内折扣奖励；这些窗口的误差是直接的 reward 拟合误差。
2. 原真实失败终止窗口的 target 为 0；该窗口估值高说明没有拟合好这部分监督。
3. IQL 其他原窗口可比较 `Q(s,A)` 与 `r + discount*V(next_obs)`，但小 Bellman 残差不保证 Q 的绝对准确性。
4. 对训练与 held-out 分开比较；本数据的 held-out 样本只有 3 个完整成功事件，且相邻帧强相关，不能把几百帧当成几百次独立试验。
5. 9 条排除录像适合观察泛化趋势，但由于未来奖励未记录，无法由它们计算最终成功回报误差或证明评分校准。

成功后的 V 不一定变成 0：训练在成功窗口停止 bootstrap，也没有直接把所有成功后观察的 V 都监督为 0。画面保留这些数值，是展示外推行为，不是给它们补上训练时不存在的零标签。

## 实现和重跑

| 文件 | 作用 |
| --- | --- |
| [value_videos.py](../../scripts/so101_pipeline/offline_rl/value_videos.py) | checkpoint / 数据一致性、GPU 评分、CSV/JSON、统计、HTML、并发编码调度 |
| [value_video_render.py](../../scripts/so101_pipeline/offline_rl/value_video_render.py) | PIL 绘制、精确原始视频解码、H.264 编码和帧数核对 |
| [value_diagnostics.py](../../src/ahawam/offline_rl/value_diagnostics.py) | 不依赖 reward 标签的诊断动作块、记录行为回报、episode 分组 |
| [test_value_diagnostics.py](../../tests/test_value_diagnostics.py) | HOLD/尾部/gap、奖励时序、未录到成功不能变成失败的回归测试 |

```bash
PYTHONPATH=src:. CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 \
  /home/fenghao/miniconda3/envs/ahawam/bin/python \
  scripts/so101_pipeline/offline_rl/value_videos.py \
  --output runs/offline_rl/REPLACE_WITH_NEW_VIDEO_OUTPUT --workers 4
```

默认 checkpoint、cache 和数据路径都指向本轮最终实验。`--iql`、`--td3` 可指定其他匹配的 checkpoint；`--limit 2` 用于小样例验证；`--device cpu` 可仅用 CPU 评分。输出目录来源不一致会报错，应换新目录；评分的 episode JSON 已落盘时可以复用，渲染会重做，确保全视频统一纵轴。不会修改训练数据、标注或模型权重。

## 本轮检查结果

新增三个时序/分组测试已通过；先生成两条真实成功/失败样例，核对画面、原始分数和 H.264 帧数后，再生成全量。

**103 个视频全部完成，共 43,680 帧，MP4 合计 496,376,068 bytes（约 473 MiB），总耗时约 12 分钟。** 评分在单张 GPU 上运行，显存峰值约 602 MiB；渲染用 4 个 CPU 进程，无需重跑大模型。Q 评分有效起点共 18,357 个，其中原训练窗口 16,016、原验证窗口 538；其余包括排除 episode 和成功后探测。V 覆盖全部原始帧。

建议先打开输出中的 `WATCH_FIRST.md`，内含 7 条成功、失败、验证、提前停止轨迹的直接视频链接和具体异常帧。`review_report.json` 保留以下检查及验证集最大残差案例，`examples/` 有对应画面截图。

| 直接 terminal 目标拟合 | 窗口数 | IQL Qmin 的 MAE | TD3 Qmin 的 MAE |
| --- | ---: | ---: | ---: |
| 训练成功窗口 | 522 | 0.00801 | 0.00765 |
| 验证成功窗口 | 13 | 0.00574 | 0.01558 |
| 训练失败终止窗口 | 13 | 0.04809 | 0.04677 |

这些成功窗口没有 bootstrap，目标是已记录块内奖励，所以结果支持“成功附近的直接奖励拟合较好”。但验证集的 13 个窗口只对应两个相互独立的成功事件，不是 13 次试验；第三条完整验证成功轨迹因为成功前 tick gap，没有合法成功奖励窗口。没有验证失败 episode，无法检查 held-out 失败终点拟合。

IQL 在原训练窗口的平均 `|Qmin−(r+discount*Vnext)|` 为 **0.01217**，原验证窗口为 **0.03849**。这属于内部一致性检查，不是真实 Q 误差。验证集 `20260912_010747_val / ep000002` **第111帧（视频约7.40秒）**出现明显局部不一致：

| IQL V | IQL Q1 | IQL Q2 | TD3 Q1 | TD3 Q2 | IQL Bellman target |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.65289 | 0.10861 | 0.11237 | 0.68205 | 0.68334 | 0.68197 |

这里两路 IQL Q 很接近，却共同偏离自身一步目标约 0.57。只能据此判断 IQL 局部 Bellman 不一致，不能拿 V 或 TD3 的值冒充真实 Q 标签，也不能把双 Q 接近当成正确性证明。另有训练失败 `20260911_225809 / ep000026` 第146帧：真实终止目标为0，TD3 Qmin仍约0.253；同 run 的 ep000005 第459帧 IQL Qmin约0.187，也值得观看。

9 条提前停止录像的有效 Q 评分不为零，也未被当成失败：训练来源的8条上 IQL/TD3 Qmin均值约0.751/0.738，验证来源的1条约0.747/0.736。因为后续奖励没有记录，不能给这些分数算成功回报准确率，更不能用高分宣称当前画面已经插到底。

这些视频能检验对所记录状态与动作的分数变化，但没有比较**同一状态的多个候选动作**，因此也不能单凭它们证明 critic 已能可靠指导 actor 改进。当前结论是：直接成功奖励已有较好拟合，同时存在局部一致性与失败终点误差，需要结合异常视频继续分析。
