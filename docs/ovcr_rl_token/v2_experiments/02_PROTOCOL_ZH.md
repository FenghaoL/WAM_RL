# 数据、采样与实验协议

## 数据范围与划分

只使用以下来源，normalization 完全复用 P0 的 `runs/so101_ahawam_padapter_v2/assets/dataset_stats.json`，不在纠正数据上重新拟合。

| 集合 | 原始 episode 数 | 连续 HUMAN 段 | 帧数 |
|---|---:|---:|---:|
| demo train：v2 原训练三目录 | 88 | — | 15,984 |
| demo val：v2 原 `20260831_225142_val` | 10 | — | 1,735 |
| HIL 全量 | 70 | 130 | 26,658 原始帧，5,498 HUMAN 帧 |
| HUMAN train | 54 | 104 | 4,437 |
| HUMAN val | 15 | 26 | 1,061 |

70 次 rollout 中有 69 次含有效 HUMAN。HUMAN 段长 min/Q1/median/Q3/max 为 1/28.25/42/49.75/143 帧，93 段短于 49 帧。约 366.5 秒 HUMAN 动作，不是 70 条完整演示。

五个 HIL session 按原始 episode 划分，以固定哈希种子 20260905 在每个 session 留约 20% episode。一次 rollout 的全部片段属于同一集合；训练随机种子不改变 split。数据清单保存每个 parquet 的 SHA256、episode、段长、HUMAN 帧数和 safety clamp 数。runner 验证 manifest、stats 和实际加载长度；不在每次启动重新哈希全部视频。若原始数据被修改，应创建新实验版本。

HUMAN 选择沿用 `control_source/intervention/human_action_valid/dropped_tick/action_supervision_mask` 的边界检查，监督实际执行的 `action`。history、future video、action chunk 均在段内截断/pad，不跨到自主动作。不是按单帧随机拆分 train/val。由于同一操作员、同一 session 仍同时出现在 train/val，验证只衡量 episode 内分布的新 rollout 泛化，不声称跨场景泛化。

片段短会使 48 帧窗口大量 padding；采样覆盖按“窗口起始帧”计，不能等同独立动作标签数量。沿用 P0 的 offset 0–7，尾部可能在 offset 后没有有效监督，action loss 用 mask 处理；没有为增加有效比例而偷偷跨 HUMAN 边界。未来可单独消融有效 offset/起点过滤，但必须所有比较组一致重建清单。

## 采样机制

使用打乱后的索引流，而不是每步独立有放回抽样。来源内部一个完整排列用完后才重新打乱；一个全局 epoch 按较长来源所需步数取整，较小 HUMAN 来源允许重复多个循环。每 GPU batch4；Accelerate 分发完整本地 batch，所有 rank 对应同一全局索引流。

| 模式 | 每 GPU demo/HUMAN | 每全局 step demo/HUMAN | 2664 步的起点曝光 |
|---|---:|---:|---|
| 1:1 | 2/2 | 6/6 | demo 15,984；HUMAN 15,984，约 3.60 遍 |
| 3:1 | 3/1 | 9/3 | demo 23,976；HUMAN 7,992，约 1.80 遍 |
| natural | 随排列变化 | 非精确固定 | 总 31,968，期望 HUMAN 约 6,946；实际由排列/epoch 尾部决定 |
| HUMAN-only | 0/4 | 0/12 | HUMAN 31,968，约 7.20 遍；无 demo |

1:1 的 2664 步刚好让全部 demo 起点看一遍，全部 HUMAN 至少看三遍。自然比例是训练集合中的 `4437/(15984+4437)=21.73%`；若没留 HUMAN 验证集则约 25.59%，两者不可混用。

“不放回”和“少数类过采样”不冲突：可在每个来源的循环内不放回，跨循环重复。严格全训练过程都不允许 HUMAN 重复，则无法在数据量不等时既做到 1:1 又覆盖全部 demo。按 epoch 重新打乱会丢弃上一 epoch 的未完成少数来源循环尾部，但每个完整 epoch 已覆盖该来源所有起点；不保证跨 epoch 永不重复。

## 第一轮：固定算力预算的筛选

所有组从 P0 独立初始化，固定 seed42、3 GPU（默认 0/1/2）、每 GPU batch4、accumulation1、global batch12、max_steps2664、bf16、ZeRO2 + CPU optimizer offload、AdamW lr1e-4 / betas(0.9,0.95) / weight_decay0.01、grad clip1、cosine、5% warmup（133步）。P0 full SFT 的 lr 是 1e-5；小模块采用 1e-4 是待验证的参数选择。

所有组冻结 backbone、VAE、原 OVCR、观测编码器，仅训练指定 adapter；video loss 权重0。每444步评估；每888步保存（888/1776/2664），验证文件还记录 step0 P0 基线。固定优化步数和 batch 比固定 epoch 更适合比较不同混合比；anchor 需要额外 teacher forward，因此相同步数不等于相同 GPU 小时。

| 优先级 / 配置名 | 变量 | anchor λ | 可学习参数 | 对照关系 |
|---|---|---:|---:|---|
| P1 `p1_v32_balanced` | value rank32，1:1 | 1 | 2,058,560 | 默认候选 |
| P1 `p1_q32_balanced` | query rank32，1:1 | 1 | 2,744,640 | 改插入位置 |
| P1 `p1_t4_balanced` | token_kv，4 tokens / rank32，1:1 | 1 | 5,511,360 | 改信息聚合方式，但参数量也增加 |
| P2 `p2_v32_natural` | value，自然混合 | 1 | 2,058,560 | 与 P1 V 比采样 |
| P2 `p2_v32_3to1` | value，3:1 | 1 | 2,058,560 | 与 P1 V 比采样 |
| P2 `p2_v32_noanchor` | value，1:1 | 0 | 2,058,560 | 隔离 anchor 效果 |
| P3 `p3_v32_human_only` | value，HUMAN-only | 0 | 2,058,560 | 必须与 noanchor 比较才能隔离 replay 效果 |

参数量不同意味着前三组是“方案比较”，不是严格等容量的结构归因。如果 T4 胜出，后续要加入参数量接近的宽 rank V/Q 以及固定4 token改变rank的对照。本轮也不把 full finetuning 的旧失败结果当作正式对照：旧实验的数据、预算与验证协议未匹配。

必要的后续对照：P0 不训练；P0 上 full / action-only mixed BC（1:1，较低 lr1e-5，独立显存测试）；从 AHA 预训练重新用 v2 demo+HUMAN 训练。后两项成本更高，先列为待实现的大预算扩展，不能宣称当前 adapter 优于它们。

## 模型选择和正式验证

训练时每个来源固定每 rank 4 个样本，共12个 loss 样本；其中每 rank1个做10步动作推理，共3个动作样本。固定样本和 torch/NumPy RNG，保持评估前后的训练 RNG；这是低成本趋势检查，样本量不足以排序或宣称性能提升。

筛选后需扩大固定验证子集（建议每 rank128个 loss 样本），记录每个来源的 action flow loss、action L1/L2 和有效 mask。demo 与 HUMAN 分开报告，不以混合总体 loss 掩盖退化。重复 seed42/142/242，仍使用同一 episode split；报告均值与各 seed 结果，不只挑最优 seed。初始化和样本量是数据冻结后的预定条件。

真机验证在本地 SO101 执行：P0、最佳 V、最佳另一结构使用相同初始物体位置列表、相同动作分块和执行频率，交错测试顺序。先少量调试，正式每模型至少30次完整任务；以成功率、完成时间、接管次数/时长、阶段失败类型、控制限幅次数为指标。成功定义、超时和紧急停止计分规则需在测试前固定。对同一摆放条件报告配对结果，置信区间以 rollout 为单位；不把连续帧当独立重复。

先锁定 checkpoint 再测最终测试集；不要在同一组测试摆放上反复选 checkpoint 后仍报告该组最高值。对“原有能力保持”和“纠正后的恢复能力”分别安排正常起点/困难起点。机器人部署设置保持实验间一致，否则延迟/分块变化会混入模型效果。
