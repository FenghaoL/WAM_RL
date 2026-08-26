# 后续怎么用这套代码

**只有一个入口，直接跑，不用传参数：**

```bash
bash scripts/so101_pipeline/baseline_video_experiments/run_experiment.sh
```

要改什么，只改 `run_experiment.sh` 顶部【配置区】里的变量。
它会依次做三件事，中间产物都落在 `OUTPUT_ROOT` 下：

```
1. export_reference_samples.py  挑评测窗口 + 导出真值  -> $OUTPUT_ROOT/samples/
2. wan22_base/run_video_branch_preview.py  每个 ckpt 生成视频 -> $OUTPUT_ROOT/<名字>/
3. compare_runs.py              合并成对比表 + 并排视频 -> $OUTPUT_ROOT/comparison/
```

第 1 步只在窗口还不存在时跑。**同一个 `OUTPUT_ROOT` 下所有 run 打的是同一批窗口**，
所以数字可以直接横向比；第 3 步会把这个目录里跑过的所有 run 都合进表里。

```
scripts/so101_pipeline/baseline_video_experiments/
├── run_experiment.sh                  ← 唯一入口，改这个
├── export_reference_samples.py        第 1 步
├── wan22_base/run_video_branch_preview.py   第 2 步
├── compare_runs.py                    第 3 步
├── common/
│   ├── sample_selection.py            挑窗口的逻辑
│   └── reporting.py                   指标 / 对比图 / summary
├── minimax_h3/                        MiniMax-H3 基线（独立 conda 环境，见第 3 节）
├── verify_video_path_equivalence.py   工具：验证 preview == 训练视频链路
├── PATH_AUDIT_ZH.md                   链路一致性审计
├── HISTORY_EXPLAINED_ZH.md            历史 K/V 是什么、怎么输入、怎么开关
├── WHY_SLOW_ZH.md                     int4/MiniMax 为什么慢
├── README.md                          实验结果 + 指标缩写对照 + 模型大小/显存
└── HOW_TO_USE_ZH.md                   本文
```

## 配置区里最常改的几项

| 变量 | 作用 |
| --- | --- |
| `RUN_DIR` | 训练 run 的 `model` 目录（要有 `config.yaml` 和 `checkpoints/weights/`） |
| `CHECKPOINTS` | 要评测哪些。写 step 名自动到 `$RUN_DIR/checkpoints/weights/` 找；也可以写完整 `.pt` 路径；写 `wan22-base` 就是未训练的原始 Wan2.2 对照 |
| `OUTPUT_ROOT` | 结果放哪。**同一个 root 共用窗口、结果自动汇总** |
| `NAME_PREFIX` | 输出目录前缀，避免不同 run 的同名 step 撞车（如 `ahawambase_`） |
| `GPUS` | 用哪些卡，一张卡跑一个 checkpoint，多的排队 |
| `DATASET_DIRS` | 留空 = 用训练配置里的目录；填了 = 覆盖（比如打验证集）。归一化统计永远沿用训练 run 的，不会重算 |
| `NUM_SAMPLES` | 挑几个窗口，按任务平均分 |
| `STEPS` / `SEED` | 去噪步数 / 随机种子。同一 seed 下各 checkpoint 拿到同一份初始噪声 |
| `HISTORY_MODES` | `online` / `training` / `none` / `wan_native`，逗号分隔；第一个用于并排视频 |
| `FORCE_REEXPORT` | `1` = 强制重挑窗口（会让旧结果不可比） |

脚本顶部还写了三个常用配方（多 checkpoint / 加 Wan2.2 对照 / 打验证集），照着改就行。

---

## 最常见的几件事

### 1. 我训了个新 checkpoint / 换数据 / 换卡

全都在 `run_experiment.sh` 的配置区改，然后 `bash` 一下。见上面那张表。

想绕过入口脚本、手动跑单个 python 脚本也可以，三个脚本的参数
和配置区变量是一一对应的，`--help` 能看到全部开关。

### 2. 关于评测窗口

挑选逻辑在 `common/sample_selection.py`：只取不含 padding 的窗口，
按 `task → 历史深度 → episode` 三层轮转，所以 24 个样本会是每个任务 12 个、
覆盖不同 episode、不同录制 session、不同历史长度。想改口味就改 `_order_within_task`。

数据集目录写了多个（多个录制 session）时直接用，不用改任何东西。
样本目录名里的 `s0/s1/s2` 就是第几个 session
（`episode_index` 在每个 session 里都从 0 重新开始，光看 episode 号会重名）。

**换了窗口，所有模型都要重跑**，否则数字不可比。旧窗口目录别删，它是那批结果的凭据。

### 3. 我想跑 MiniMax-H3

一次性准备（已经做完了，重装环境时才需要）：

```bash
bash scripts/so101_pipeline/baseline_video_experiments/minimax_h3/setup_env.sh

huggingface-cli download MiniMaxAI/MiniMax-H3 \
  --local-dir checkpoints/MiniMaxAI/MiniMax-H3 --max-workers 16 \
  --include "modular_model_index.json" "model_index.json" \
            "transformer/*" "text_encoder/*" "vae/*" "audio_vae/*" \
            "processor/*" "tokenizer/*" "scheduler/*" "audio_scheduler/*"
```

跑：

```bash
bash scripts/so101_pipeline/baseline_video_experiments/minimax_h3/run_minimax_h3_preview.sh
```

环境变量可以覆盖：

```bash
GPU=1 LIMIT=16 HEIGHT=768 WIDTH=640 STEPS=40 \
  bash scripts/so101_pipeline/baseline_video_experiments/minimax_h3/run_minimax_h3_preview.sh
```

| 变量 | 默认 | 说明 |
| --- | --- | --- |
| `GPU` | `1` | 物理卡号 |
| `LIMIT` | `6` | 跑多少个窗口。H3 很慢，先小跑再放开 |
| `HEIGHT`/`WIDTH` | `576`/`480` | 必须是 32 的倍数，且保持 384:320 = 1.2 的比例，否则画面会被拉伸；生成完会缩回 384x320 再算指标 |
| `STEPS` | `32` | 去噪步数；`0` = 用 pipeline 自带默认值（50 个 sigma 点 = 49 次前向） |

还有一个不在环境变量里、但很关键的开关：`--memory-reserve-margin`（默认 `3GB`）。
int8 后的 transformer 大约 31 GB，A100 只有 40 GB；
如果这个 margin 留太大（比如默认过的 `10GB`），`ComponentsManager` 会算出
"放不下"，于是**每一个去噪步都把 transformer 换出再换入**，
单步从几秒变成 90 秒。日志里出现
`no combination of models to offload to cpu is found, offloading all models`
就是这个症状。

要注意的硬约束（都是 MiniMax-H3 自己的限制，不是我们加的）：

- 固定 24 fps，时长必须在 5~15 秒，帧数会被上取整到 `17n+5`。
  我们一个窗口只有 3.2 秒，所以脚本会请求最短合法长度（124 帧 = 5.17 秒），
  再按 GT 的 13 个时间戳重采样回来打分；完整的 5.17 秒片子存成
  `prediction_full_24fps.mp4`，想看长一点的想象就看它。
- 量化选 int8 还是 int4 见下一节。要跑全精度就传 `--quantization none`，
  但两个大件加起来 124 GiB，靠 `ComponentsManager` 的 auto CPU offload 硬扛，会非常慢。
- 模型会同时生成音轨，我们只取 `videos`，音频直接丢掉。

### 3b. int8 还是 int4？

`QUANT` 环境变量切换，**两者都不需要额外下载**——torchao 直接从磁盘上已有的
bf16 权重现场量化：

```bash
QUANT=int8 bash .../run_minimax_h3_preview.sh                      # 默认，快
QUANT=int4 LIMIT=16 bash .../run_minimax_h3_preview.sh \
  --memory-reserve-margin 20GB                                     # 省显存
```

输出目录会自动带上后缀（`minimax_h3_int8` / `minimax_h3_int4`）。

| | 常驻权重 | 实测峰值 | 单样本耗时 | 画质 |
| --- | --- | --- | --- | --- |
| int8 | ~31.6 GiB | 33.89 GiB | 232 s | 同左 |
| int4 + `--memory-reserve-margin 20GB` | 19.46 GiB | **21.31 GiB** | 988 s | 同右 |

- **int4 在 A100 上比 int8 慢 4.3 倍。** 一句话：视频模型是算力瓶颈，
  而 weight-only 量化只省带宽、还额外增加反量化开销；再叠加 sm_80 上唯一可用的
  int4 内核 tinygemm `_weight_int4pack_mm` 是给 LLM 单 token 解码写的，
  实测在大矩阵上比 bf16 慢 17 倍。完整解释和实测数据见
  [WHY_SLOW_ZH.md](WHY_SLOW_ZH.md) 第 1 节。
- int4 的打包算子 `aten::_convert_weight_to_int4pack` **只有 CUDA 实现**，
  所以脚本对 int4 会用 `device_map=cuda` 在卡上量化，量化完立刻 park 回 CPU
  再去装下一个组件（两个加起来 39 GiB，同时放会挤掉 VAE）。int8 在 CPU 上就能打包，
  走的是原来的路径。
- 用 `--memory-reserve-margin` 控制同时有几个组件常驻：预算 = 卡显存 − margin。
  20GB 的 margin 让预算只有 20 GB，装得下 int4 transformer（19.46 GiB）但装不下
  transformer + conditioner，于是文本编码完就被换出去。
- 社区的 `-FP8` / `-nvfp4` 量化包在 A100 上**用不了**（sm_80 没有对应张量核），
  `-int4` 那些是 ComfyUI 单文件格式，diffusers 读不了。

### 4. 我想把所有结果并排看

```bash
python scripts/so101_pipeline/baseline_video_experiments/compare_runs.py \
  --runs runs/so101_g05_square_v1/baseline_experiments/*/summary.json \
  --output runs/so101_g05_square_v1/baseline_experiments/comparison
```

产出：

- `comparison.md` / `comparison.csv`：各 run 的均值表 + 每个窗口的 PSNR 表；
- `side_by_side/<窗口>.mp4`：GT 和每个模型的预测横向拼在一起；
- `contact_sheet_last.png`：所有窗口末帧的总览图。

`--no-video` 可以只出表格（快很多）。
这个脚本只用 numpy/PIL/imageio，`ahawam` 和 `minimax_h3` 两个环境里都能跑。

### 5. 在留出验证集上评（而不是训练集）

在 `run_experiment.sh` 里改两处：

```bash
OUTPUT_ROOT="runs/<数据集>/baseline_experiments_val"   # 换个新的，别和训练集窗口混
DATASET_DIRS=(
  data/<数据集>/<任务A>/<session>_val
  data/<数据集>/<任务B>/<session>_val
)
```

脚本会自动带上 `--val-set-proportion 0`（配置里默认 0.05，会再切走 5% 的 episode；
留出目录本身就是验证集，不需要再切）。

**千万不要在验证集上重算归一化统计。** 那等于偷偷改了评分标准，训练/验证的数字就不可比。
脚本开头会把实际生效的 `pretrained_norm_stats` 打印出来，跑之前扫一眼确认它指向训练集那份。

两边的 `comparison/comparison.md` 并排看，就是泛化差距。

---

## 输出目录长什么样

```
runs/so101_g05_square_v1/baseline_experiments/
├── samples/                        # 固定的评测窗口（第 1 步产物）
│   ├── manifest.json
│   └── <窗口>/{first_frame.png, ground_truth.mp4, frames/gt_###.png,
│               history_strip.png, meta.json}
├── wan22_base/                     # 每个 run 一个目录
│   ├── summary.json  summary.csv  README.md
│   ├── contact_sheet_middle.png  contact_sheet_last.png
│   └── <窗口>/{ground_truth.mp4, prediction_online.mp4,
│               prediction_training.mp4, prediction_none.mp4,
│               comparison.mp4, comparison_middle.png,
│               comparison_last.png, metrics.json}
├── ahawam_pretrained/
├── so101_finetuned/
├── minimax_h3_int8/                # 多一个 prediction_full_24fps.mp4
├── minimax_h3_int4/
├── comparison/
└── path_equivalence.json
```

指标含义：

| 字段 | 含义 |
| --- | --- |
| `future_rgb_psnr` | 整帧 PSNR，**不含第 0 帧**（那是白送的条件帧，含进去所有模型都会虚高） |
| `fixed_rgb_psnr` | 只看上方 256 行的固定相机区域 |
| `wrist_rgb_psnr` | 只看左下角 128x160 的腕部相机区域 |
| `pred_temporal_abs_diff` | 预测视频的相邻帧平均绝对差 = 画面动得多剧烈 |
| `target_temporal_abs_diff` | GT 的同一个量。**这两个数要一起看**：预测远大于 GT = 模型在乱抖；远小于 = 模型在出静止画面 |

PSNR 本身对"物体位置对不对"不敏感，一个原地不动的预测也能拿到不低的 PSNR。
所以**永远要配合 `*_temporal_abs_diff` 和 `comparison.mp4` 一起判断**，别只看一个数。

---

## 验证链路一致性

换了模型结构、改了 mask 模式、动了 history 逻辑之后，可以再跑一遍：

```bash
python scripts/so101_pipeline/baseline_video_experiments/verify_video_path_equivalence.py \
  --config runs/so101_g05_square_v1/full_zero3/<run>/model/config.yaml \
  --weights runs/.../weights/step_004896.pt \
  --sample-index 18014 \
  --device cuda:0 \
  --output runs/so101_g05_square_v1/baseline_experiments/path_equivalence.json
```

它会检查三件事：`pre_dit` 忽略 action、MoT 联合前向不改变视频输出、
一次性 prefill 和逐帧滚动 prefill 得到同一份 K/V。
详细解释见 [PATH_AUDIT_ZH.md](PATH_AUDIT_ZH.md)。

如果哪天 `action_conditioned` 被改成 `true`，**这套 preview 就不再等价了**，
到时候必须先跑动作分支再跑视频分支。脚本里有断言会直接报错拦住你。

---

## 环境备忘

| 环境 | 用途 | 关键版本 |
| --- | --- | --- |
| `ahawam` | 第 1 步、AHA-WAM 视频分支、验证脚本、`compare_runs.py` | python 3.10, torch 2.7.1+cu128, transformers 4.49.0 |
| `minimax_h3` | 只给 MiniMax-H3 用 | python 3.12, torch 2.11+cu128, diffusers main, transformers 5.x, torchao |

两个环境是**故意分开**的：MiniMax-H3 需要 diffusers `main`（集成比任何 release 都新）
和带 Qwen3-VL 的 transformers，而 `ahawam` 钉死 `transformers==4.49.0`，
就地升级会把训练环境搞坏。

跑任何 GPU 任务前都记得：

```bash
source /home/fenghao/use-nvidia-580159.sh
```
