# 后续怎么用这套代码

这个目录里的东西是**可重复跑的对照实验框架**，不是一次性脚本。
核心思路只有一句：**先把评测窗口固定下来落盘，之后所有模型都打这同一批窗口**，
这样不同 checkpoint、甚至不同架构的模型，数字才可以直接比。

```
scripts/so101_pipeline/baseline_video_experiments/
├── export_reference_samples.py        # 第 1 步：挑窗口 + 导出 GT（只跑一次）
├── common/
│   ├── sample_selection.py            # 挑窗口的逻辑
│   └── reporting.py                   # 指标 / 对比图 / summary（不依赖 ahawam 和 torch）
├── wan22_base/
│   ├── run_video_branch_preview.py    # 第 2 步 A：AHA-WAM 视频分支，权重可换
│   └── run_video_branch_preview.sh    # 一键跑多个 checkpoint
├── minimax_h3/
│   ├── setup_env.sh                   # 建 minimax_h3 conda 环境（只跑一次）
│   ├── run_minimax_h3_preview.py      # 第 2 步 B：MiniMax-H3 图生视频（int8 / int4）
│   └── run_minimax_h3_preview.sh
├── compare_runs.py                    # 第 3 步：合并所有 run 出对比表 + 并排视频
├── verify_video_path_equivalence.py   # 独立工具：验证 preview == 训练视频链路
├── PATH_AUDIT_ZH.md                   # 链路一致性审计（先读这个）
├── WHY_SLOW_ZH.md                     # int4/MiniMax 为什么慢，能不能进 WAM
├── README.md                          # 实验结果 + 指标缩写对照 + 模型大小/显存
└── HOW_TO_USE_ZH.md                   # 本文
```

---

## 最常见的几件事

### 1. 我训了个新 checkpoint，想看它的视频分支怎么样

编辑 `wan22_base/run_video_branch_preview.sh` 里的 `CONDITIONS` 数组，加一行：

```bash
CONDITIONS=(
  "wan22_base|wan22-base"
  "ahawam_pretrained|checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt"
  "so101_finetuned|runs/.../weights/step_004896.pt"
  "my_new_run|runs/.../weights/step_009792.pt"      # <- 加这行
)
```

然后：

```bash
bash scripts/so101_pipeline/baseline_video_experiments/wan22_base/run_video_branch_preview.sh
```

格式是 `<输出目录名>|<权重>`。`<权重>` 有两种写法：

- 字面量 `wan22-base` → 不加载任何 AHA-WAM checkpoint，
  视频专家直接用 `Wan-AI/Wan2.2-TI2V-5B` 官方权重（也就是"没训练过"的起点）；
- 一个 `.pt` 路径 → 正常加载那个 checkpoint。

已经跑过的 condition 会被覆盖重跑。如果只想跑新加的那个，
把别的行注释掉，最后 `compare_runs.py` 那步仍然会把全部 `summary.json` 合起来。

单独跑一个也行：

```bash
python scripts/so101_pipeline/baseline_video_experiments/wan22_base/run_video_branch_preview.py \
  --config runs/so101_g05_square_v1/full_zero3/<run>/model/config.yaml \
  --samples-dir runs/so101_g05_square_v1/baseline_experiments/samples \
  --output runs/so101_g05_square_v1/baseline_experiments/my_new_run \
  --weights runs/.../weights/step_009792.pt \
  --run-name my_new_run \
  --history-modes online,training,none \
  --steps 20 --seed 42 --device cuda:0
```

常用开关：

| 开关 | 作用 |
| --- | --- |
| `--history-modes` | 逗号分隔，可选 `none` / `training`（RoPE 间隔 12）/ `online`（RoPE 间隔 10）/ `wan_native`；第一个是"主"模式，会被 `compare_runs.py` 拿去做并排视频 |
| `--steps` | 去噪步数，默认 20 |
| `--seed` | 每个样本用 `seed + 序号`，所以同一个 seed 下不同 checkpoint 拿到的是同一份初始噪声 |
| `--limit N` | 只跑 manifest 里前 N 个窗口，调试用 |

`wan_native` 是个特殊模式：它把视频专家临时切回 Wan2.2 自己的推理设置
（`video_attention_mask_mode="bidirectional"`、从 0 开始的连续 RoPE、不给历史），
用来回答"原生 Wan2.2 在这批数据上能做到什么"。
只对 `--weights wan22-base` 有意义，对训练过的 checkpoint 用它等于故意错配。
它和别的模式**不要写在同一个 `--history-modes` 里**，单独跑一次
（`run_video_branch_preview.sh` 里的 `RUN_WAN22_NATIVE=1` 就是干这个的）。

### 2. 我想换一批评测窗口（换数据集 / 换数量）

```bash
python scripts/so101_pipeline/baseline_video_experiments/export_reference_samples.py \
  --config runs/so101_g05_square_v1/full_zero3/<run>/model/config.yaml \
  --output runs/so101_g05_square_v1/baseline_experiments/samples_v2 \
  --num-samples 32
```

然后把两个 run 脚本里的 `SAMPLES_DIR` 指过去。
**换了窗口以后，所有模型都要重跑**，否则数字不可比。
旧窗口目录别删，它就是那批结果的凭据。

挑选逻辑在 `common/sample_selection.py`：
只取不含 padding 的窗口，按 `task → 历史深度 → episode` 三层轮转，
所以 16 个样本会是 4 个 task 各 4 个、覆盖不同 episode 和不同历史长度。
想改口味就改 `_order_within_task`。

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

### 5. 我想一次比较同一个 run 的多个 checkpoint（多卡并行）

单张卡跑一个 checkpoint 大约 10 分钟（模型加载占大头），所以多个 checkpoint
直接一卡一个并行最省事。**先导出窗口，再并行**——窗口必须是同一批：

```bash
R=runs/<你的 run>/model
OUT=runs/<你的数据集>/baseline_experiments

# 只跑一次
python scripts/so101_pipeline/baseline_video_experiments/export_reference_samples.py \
  --config "$R/config.yaml" --output "$OUT/samples" --num-samples 24

# 一个 checkpoint 一张卡
i=1
for step in 001720 002580 003440; do
  CUDA_VISIBLE_DEVICES=$i python scripts/so101_pipeline/baseline_video_experiments/wan22_base/run_video_branch_preview.py \
    --config "$R/config.yaml" --samples-dir "$OUT/samples" \
    --output "$OUT/step_$step" --weights "$R/checkpoints/weights/step_$step.pt" \
    --run-name "step_$step" --device cuda:0 --steps 20 --seed 42 \
    --history-modes online,training,none &
  i=$((i+1))
done
wait

python scripts/so101_pipeline/baseline_video_experiments/compare_runs.py \
  --runs "$OUT"/step_*/summary.json --output "$OUT/comparison"
```

注意 `--device cuda:0` 配合 `CUDA_VISIBLE_DEVICES=$i`：每个进程只看得见一张卡，
所以它自己的 `cuda:0` 就是物理第 `$i` 张。峰值显存约 19 GiB，一张 40GB 卡放一个进程绰绰有余。

**先确认目标卡是空的**（`nvidia-smi`）——在线部署的 action server 也会占卡。

### 6. 数据集是多个目录（多个录制 session）怎么办

直接用，不用改任何东西。`dataset_dirs` 列多少个目录，窗口挑选都会跨它们做轮转，
样本目录名里的 `s0/s1/s2...` 就是第几个 session
（`episode_index` 在每个 session 里都从 0 重新开始，所以光看 episode 号会重名）。

> 这一点在 2026-08-14 之前是个 bug：旧代码只在第一个子数据集里查 task 名，
> 多目录数据集下会把后面 session 的样本全部标错任务。现在已修。

### 7. 我想在留出验证集上评（而不是训练集）

不用改配置文件，两个开关就够：

```bash
VAL1=data/<数据集>/<任务A>/<session>_val
VAL2=data/<数据集>/<任务B>/<session>_val

python scripts/so101_pipeline/baseline_video_experiments/export_reference_samples.py \
  --config "$R/config.yaml" \
  --dataset-dirs "$VAL1" "$VAL2" --val-set-proportion 0 \
  --output "$OUT_VAL/samples" --num-samples 24

# 每个 checkpoint 也要带上同样的两个开关
python scripts/so101_pipeline/baseline_video_experiments/wan22_base/run_video_branch_preview.py \
  --config "$R/config.yaml" \
  --dataset-dirs "$VAL1" "$VAL2" --val-set-proportion 0 \
  --samples-dir "$OUT_VAL/samples" --output "$OUT_VAL/step_003440" \
  --weights "$R/checkpoints/weights/step_003440.pt" --run-name step_003440 \
  --device cuda:0 --steps 20 --seed 42 --history-modes online,training,none
```

| 开关 | 为什么必须 |
| --- | --- |
| `--dataset-dirs` | 换成留出目录，但**其余一切照搬训练配置**——分辨率、帧率、历史长度、以及最关键的 `pretrained_norm_stats` |
| `--val-set-proportion 0` | 配置里默认 `0.05`，会再从每个目录切走 5% 的 episode。留出目录本身就是验证集，不需要再切，传 0 才是用满 |

**千万不要在验证集上重算归一化统计。** 那等于偷偷改了评分标准，
训练/验证的数字就不可比了。脚本会在开头把实际生效的
`pretrained_norm_stats` 打印出来，跑之前扫一眼日志确认它指向训练集的那份。

输出目录建议和训练集的分开（`baseline_experiments/` vs `baseline_experiments_val/`），
两边的 `comparison/comparison.md` 并排看就是泛化差距。

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
