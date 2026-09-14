# 抓取对准后"不合爪、一直停顿"：诊断与纠正窗口改造

> **2026-09-13 更新**：第 4、6 节的 A/B 变体与起点过滤方案已被取代，实际执行见 [10_FULL_VARIANTS_ZH.md](10_FULL_VARIANTS_ZH.md)。诊断部分（第 1–2 节）仍然有效。

2026-09-12。现象：训练出的 checkpoint 在抓取阶段进入矫正位置后容易一直停着、不合爪；插入阶段较少出现。
DAgger 人工干预大部分发生在"合爪前对准夹爪"，对准后人没有合爪，直接交还给模型。

## 1. 数据证据（hil_v2 训练集）

夹爪是第 5 个关节（demo 中 90/94 条先开后合，开/合分界约 19.7）。121 段人类接管按接管开始时夹爪开合分类：

| | 抓取对准（夹爪开） | 持物插入（夹爪合） |
|---|---:|---:|
| 段数 / 帧数 | 62 / 1,607 | 59 / 1,853 |
| 段末静止帧（中位 / p90） | 6 / 11 | 1 / 9 |
| 起点落在段末静止段 | 30.1% | 11.3% |
| **整个被监督动作窗口静止（所有关节变化 <1°）** | **31.9%** | **13.1%** |
| **段内人合过爪** | **0 / 62** | — |
| 交还后模型合爪（成功 rollout，执行帧） | 中位 7、p90 14 | — |

对照 demo：人在合爪前也会停约 8 帧（中位），但随后一定合爪。DAgger 片段恰好截到停顿、没截到合爪。

**结论**：抓取对准的接管样本里约三分之一在教"张着爪保持不动"，而且因为动作窗口截断在接管段末尾，模型在这个位姿上
永远看不到"接着合爪"。1:1 采样下每个接管帧被重复约 5.7 次，这个信号被放大。插入阶段静止比例低得多，与观察一致。

## 2. 排除的因素

- **动作末尾 padding 不会让模型学成静止**：padding 的动作不进 loss（按有效数平均）；动作分支注意力是按 chunk 因果的
  （`chunk_ids[query] >= chunk_ids[key]`），后面被 padding 的 chunk 影响不到前面有效 chunk 的预测。上面 32% 的"静止目标"
  来自真实录制的静止帧，不是 padding。
- **anchor**：`p1_full_balanced_video` 系列是全量微调，本来就没有 anchor；adapter 实验的 anchor 只作用在 demo 样本上，
  不直接影响接管样本教出的"停住"。
- **部署配置**：`cylinder_v2_p1_full_balanced_video.yaml` 在 09-12 19:34 之前用了 power-adapter 的 stats 和指令。
  那段时间观察到的行为不能用于判断训练问题。

## 3. 起点过滤规则的选择

最初按"每帧所有关节变化 <0.5° 且连续 ≥4 帧"定义停顿，结果排除了 79% 的起点——人类精细对准本身就很慢（每帧约 0.05°），
会被误判为停顿。对比多种规则：

| 规则 | 抓取 | 插入 | 合计 |
|---|---:|---:|---:|
| 逐帧速度 <0.5°、连续 ≥4 帧 | 88.1% | 71.3% | 79.1% |
| 位置在 0.5° 内保持 4 帧 | 44.9% | 29.1% | 36.5% |
| **被监督动作窗口整体静止 <1°（变体 A 时间线）** | **31.9%** | **13.1%** | **21.8%** |
| **被监督动作窗口整体静止 <1°（变体 B 时间线）** | **1.5%** | **3.0%** | **2.3%** |

采用"被监督动作窗口整体静止"：它直接对应"样本只教保持不动"这一机制，并且沿训练时实际使用的动作时间线判断。

## 4. 两个变体（均基于 `p1_full_balanced_video_hilv2`，其余设置不变）

共同改动（仅作用于接管纠正样本）：

1. **未来视频延伸**：从起点 t 起的时间线 = 接管段剩余帧 → 该 rollout 之后的执行帧（跳过 SYSTEM_HOLD），只在结果为
   success 的 rollout 上延伸（86/99 条）；视频损失仍要求 13 张采样图全部有效。
2. **静止起点排除**：被监督动作窗口整体静止（<1°）的起点不作为 t；其他起点的窗口照常包含这些帧。

| 变体 | 动作/状态标签 | 保留起点 | 排除 |
|---|---|---:|---:|
| **A**（`_winA`） | 仍只在接管段内（纯人类标签） | 2,704 | 756（21.8%） |
| **B**（`_winB`） | 与视频同一时间线延伸，含成功 rollout 里交还后模型的合爪动作 | 3,380 | 80（2.3%） |

A 能停止"教停住"，但仍无法教"合爪"（接管段里从来没有合爪）；B 把合爪纳入动作监督，是对症方案。
B 的风险是动作标签里混入了模型自主动作——用"仅成功 rollout"和"跳过 SYSTEM_HOLD"限制；如需更保守，可以把延伸限制到合爪完成为止。

HUMAN 验证集与基线完全相同（不过滤、不延伸），`human_*` 曲线与 `p1_full_balanced_video_hilv2` 直接可比。

## 5. 实现与验证

- `lerobot_dataset.py::_get_query_indices` 增加可选的查询下标重映射挂点：行、时间戳、视频帧都跟随同一组下标，模态天然对齐；未设置时行为不变。
- `hil_human_only_dataset.py`：`_FutureWindowRemapper`（时间线）、`static_start_mask`（静止起点）、成功标签读取
  （`recording_context/episodes_manifest.jsonl`），配置项 `correction_future_extension`、`correction_extension_success_only`、
  `correction_exclude_static_starts`、`correction_static_range_deg`。
- `run.py` 支持 `experiment.manifest_file`，两个变体各一份 manifest。
- `check_future_window.py`：从原始 parquet 控制列和结果标签**独立重算**时间线，与训练数据集逐帧比较；两变体各 12 个样本、
  60 次图像与 54/56 次动作逐帧一致，所有段的保留/排除起点与独立重算一致，扩展时间线内无 SYSTEM_HOLD，失败 rollout 不延伸。
- 单元测试 18 项通过（含时间线、静止判定在 A/B 下的差异、失败 rollout 不延伸）。

## 6. 排队

两个变体排在当前 hil_v2 队列（raw48、skip72 对照）之后，GPU 0/1/2，**先 B 后 A**：

```bash
PREV=runs/ovcr_adapter_cylinder_hil_v2
RUN_ROOT=runs/ovcr_adapter_cylinder_hil_v2_window SESSION=ovcr_cylinder_hilv2_window GPU_IDS=0,1,2 \
NAMES="p1_full_balanced_video_hilv2_winB p1_full_balanced_video_hilv2_winA" \
PRE_COMMAND="GPU_WAIT_SECONDS=43200 bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/wait_for_queue.sh $PREV p1_v32_balanced_hilv2_skip72 $PREV/p1_v32_balanced_hilv2_skip72/model/checkpoints/weights/step_003285.pt" \
bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_suite.sh
```

等待脚本要求 skip72 以退出码 0 完成且 GPU 0/1/2 各有 ≥36GB 空闲（最长等 12 小时）才启动。若手动停掉前一个队列，
等待脚本不会放行——此时去掉 `PRE_COMMAND` 直接运行即可。每个全量实验约 8.5 小时。

部署这些权重时：`num_history_frames: 6`、cylinder stats、原任务指令。
