# SO101 视频分支对照实验

这个目录做三件事：

1. **实验 A**：把**没训练过的 Wan2.2-TI2V-5B**（也就是 AHA-WAM 的初始化权重）
   放进和你训练时完全一样的 VideoDiT 链路里，跑同一批 SO101 窗口，
   和 AHA-WAM-Pretrained、你的 SO101 微调 checkpoint 横向对比。
2. **实验 B**：把最近开源的 **MiniMax-H3**（33B 全模态视频+音频模型，int8 / int4 量化）
   当作图生视频基线，在同一批窗口、同一个首帧、同一句任务指令下生成，看它在你的数据上是什么效果。
3. **链路审计**：回答"原来那套 preview 到底能不能代表训练/推理走的 DiT 链路"。

配套文档：

- [PATH_AUDIT_ZH.md](PATH_AUDIT_ZH.md) —— 链路一致性审计，**建议先读这个**
- [HISTORY_EXPLAINED_ZH.md](HISTORY_EXPLAINED_ZH.md) —— "历史"到底是什么形态、怎么输入、部署里怎么开关
- [WHY_SLOW_ZH.md](WHY_SLOW_ZH.md) —— 为什么 int4 更慢、为什么 MiniMax 比 Wan 慢 200 倍、这类模型能不能进 WAM
- [HOW_TO_USE_ZH.md](HOW_TO_USE_ZH.md) —— 后续怎么继续用这套代码

结果都在 `runs/so101_g05_square_v1/baseline_experiments/`。


## 一句话结论

| | |
| --- | --- |
| 那套 preview 能代表训练链路吗？ | **能，而且是逐位相同**（数值验证见 [PATH_AUDIT_ZH.md](PATH_AUDIT_ZH.md) 第 4 节）。但要知道两件事：原脚本用的是**部署**的历史间隔（RoPE 10）而不是**训练**的（RoPE 12）；而且在线动作推理**根本不生成未来视频**，这条链路是旁路诊断 |
| 未训练的 Wan2.2 什么水平？ | 放进 AHA-WAM 的推理配置里是**纯噪声**（8.23 dB）；换回 Wan2.2 自己的配置能到 11.16 dB |
| MiniMax-H3 什么水平？ | 零样本 **13.11 dB**（int8，16 窗口），画面质量很干净，但运动是 GT 的 2 倍且和真实轨迹无关；单样本 232 秒 |
| 你的微调 checkpoint 呢？ | **15.89 dB**，帧间运动 7.40 ≈ GT 的 6.65，**明显优于所有基线**，单样本 1.2 秒 |

---

## 评测设置

16 个窗口，覆盖 4 个 task × 不同 episode × 不同历史深度，一次性导出后固定下来
（`samples/manifest.json`），**所有模型打的是同一批窗口**，所以数字可以直接横向比。
这一点和你原来的脚本不同：原脚本每次运行都重新挑窗口，跨 checkpoint 的数字其实不可比。

一个窗口 = 13 个视频帧 @ 3.75 Hz = 3.2 秒真实时长，分辨率 384x320
（上 256 行 fixed 相机 + 左下 128x160 wrist 相机 + 右下黑色占位）。
指标一律**排除第 0 帧**，因为那是所有模型都白送的干净条件帧。

---

## 实验 A：同一条视频分支，四种权重/配置

命令：`bash scripts/so101_pipeline/baseline_video_experiments/wan22_base/run_video_branch_preview.sh`

四个 run 跑的是**同一份代码**，区别只在视频专家的权重从哪来、以及用哪套推理配置：

| run | 视频专家权重 | 推理配置 | 说明 |
| --- | --- | --- | --- |
| `wan22_base` | `Wan-AI/Wan2.2-TI2V-5B` 官方权重 | AHA-WAM 的（per_frame_causal + 历史 K/V + 稀疏 RoPE） | 完全没经过 AHA-WAM 训练的起点 |
| `wan22_native` | 同上 | **Wan2.2 自己的**（双向注意力 + 连续 RoPE + 无历史） | 公平版："原生 Wan2.2 在你数据上能做到什么" |
| `ahawam_pretrained` | `AHA-WAM-pretrained.pt` | AHA-WAM 的 | 作者在 RoboCOIN/RoboTwin 上训的 |
| `so101_finetuned` | `step_004896.pt` | AHA-WAM 的 | 你在 SO101 上微调 4 epoch 的结果 |

### 结果（16 个窗口均值，`--history-mode online`）

| run | 整帧 PSNR ↑ | fixed 相机 PSNR ↑ | wrist 相机 PSNR ↑ | MSE ↓ | 预测帧间运动 | GT 帧间运动 |
| --- | --- | --- | --- | --- | --- | --- |
| `wan22_base` | **8.23** | 8.45 | 7.87 | 10398 | **30.82** | 6.65 |
| `wan22_native` | **11.16** | 10.90 | 11.06 | 5259 | **15.74** | 6.65 |
| `ahawam_pretrained` | **12.63** | 12.93 | 10.68 | 4101 | **13.78** | 6.65 |
| `so101_finetuned` | **15.89** | 16.07 | 13.06 | 1792 | **7.40** | 6.65 |

逐窗口的表在 `comparison/comparison.md`，并排视频在 `comparison/side_by_side/`，
每个 run 目录下还有 `contact_sheet_last.png`（所有窗口末帧的总览）。

### 怎么读这几行

**"预测帧间运动 vs GT 帧间运动"这一列比 PSNR 更能说明问题。**
GT 是 6.65，也就是这批数据里 3.2 秒内画面本来就动得不多。

- **未训练的 Wan2.2 放进 AHA-WAM 的推理配置里，输出是纯噪声**（8.23 dB，运动 30.8，
  接近 GT 的 5 倍）。看 `wan22_base/contact_sheet_last.png` 就是一片彩色雪花。
  这不是"Wan2.2 视频质量差"，而是 AHA-WAM 的 `per_frame_causal` 注意力掩码 +
  历史 K/V 前缀 + 不连续的时间 RoPE 位置，这一整套推理配置 Wan2.2 原生从来没见过。
- **所以我又加了 `wan22_native`**：同一份权重，但换回 Wan2.2 自己的配置
  （双向注意力、从 0 开始的连续 RoPE、不给历史）。分数立刻从 8.23 涨到 **11.16**，
  运动从 30.8 降到 15.7，画面里能看出物体轮廓了（虽然颜色还是会崩）。
  这证实了上面的判断：**8.23 那个数主要是"配置不匹配"的惩罚，不是权重本身的水平。**
  > 注意：`wan22_native` 仍然偏低，因为这套 harness 不做 CFG（`text_cfg_scale=1`）、
  > 只跑 20 步、画布只有 384x320（远小于 Wan2.2 的训练分辨率）。
  > 它是"同一条流水线下的可比参照"，不是 Wan2.2 的能力上限。
- **AHA-WAM-Pretrained 是 12.63，运动 13.8，仍是 GT 的 2 倍**：在 RoboCOIN/RoboTwin
  上学到的"机械臂视频"先验明显起作用了（相对 `wan22_native` +1.5 dB），
  但它没见过你的桌面、你的相机标定、你的物体，所以还是动得过头。
- **你的微调 checkpoint 是 15.89，运动 7.40，几乎正好压到 GT 的 6.65**：
  模型已经学会了"这个场景在 3.2 秒里应该动多少"。
  这是这批结果里最有说服力的一条——微调不只是把画面拟合得更像，
  而是把**运动幅度的先验**校准到了你的数据上。

另外，训练过的两个 checkpoint 都是 fixed 相机明显好于 wrist 相机
（微调后 16.07 vs 13.06，pretrained 12.93 vs 10.68）——腕部相机视角变化剧烈、
运动幅度大，本来就更难预测。未训练的 `wan22_native` 反而两边差不多（10.90 / 11.06），
因为它两边都没预测对，这个差距本身就是"学到了固定视角"的信号。

### 历史 K/V 的作用（同一批窗口，三种 history mode）

| run | online (RoPE 间隔 10) | training (RoPE 间隔 12) | none (不给历史) |
| --- | --- | --- | --- |
| `wan22_base` | 8.231 | 8.106 | 7.021 |
| `ahawam_pretrained` | 12.634 | 12.676 | 10.763 |
| `so101_finetuned` | 15.891 | 15.750 | 15.818 |

（`wan22_native` 只有一种模式，按定义就不带历史。）

两个观察：

1. **训练间隔（12）和部署间隔（10）几乎没差别**（±0.15 dB）。
   这两个数不一样是 AHA-WAM 里真实存在的一处 train/deploy mismatch
   （详见 [PATH_AUDIT_ZH.md](PATH_AUDIT_ZH.md) 第 5 节），
   但从这批数据看，**它目前不是主要矛盾**，不用急着去改。
2. **你的微调 checkpoint 对历史几乎不敏感**（给不给历史都是 15.8~15.9），
   而 AHA-WAM-Pretrained 差 1.9 dB。
   合理的解释是：你这批数据 3.2 秒内场景变化小、任务只有 4 个、
   首帧已经包含了几乎全部所需信息，所以模型学会了"主要看当前帧"。
   这本身不是坏事，但意味着**如果你想让历史 K/V 真正起作用，
   需要更长时程、更依赖上下文的任务**，或者更长的历史窗口。

---

## 实验 B：MiniMax-H3 图生视频基线

命令：`bash scripts/so101_pipeline/baseline_video_experiments/minimax_h3/run_minimax_h3_preview.sh`

MiniMax-H3 是 2026-08-03 开源的全模态视频+音频生成模型
（`transformer` 分区 ~62 GiB bf16，Qwen3-VL 条件塔另有 ~62 GiB）。
它和 AHA-WAM **唯一共享的条件是同一个首帧和同一句任务指令**——
没有历史 K/V、没有 per-frame-causal 掩码、没有动作分支、也没有 3.75 Hz 控制时钟的概念。

### 怎么跑起来的

| 项 | 值 |
| --- | --- |
| 权重 | `MiniMaxAI/MiniMax-H3`，diffusers 格式，只下 `fl2va` 需要的分区（144 GB） |
| 量化 | torchao **int8 weight-only**。A100 是 sm_80，**没有 FP8/NVFP4 张量核**，社区那些 `-FP8`、`-nvfp4` 包在这台机器上跑不了 |
| 画布 | 576x480（保持 384:320 = 1.2 的比例，生成后缩回 384x320 算指标） |
| 时长 | 请求 124 帧 @24fps = 5.17 秒（模型最短就是 5 秒），再按 GT 的 13 个时间戳重采样回 3.2 秒打分 |
| 步数 | 32 |
| 显存 / 耗时 | 峰值 33.9 GiB，加载 248 秒，**每个样本约 232 秒** |

完整的 5.17 秒生成结果保存在每个样本目录的 `prediction_full_24fps.mp4`，
用来打分的 13 帧版本是 `prediction_minimax_h3.mp4`。

> 踩过的坑：`--memory-reserve-margin` 默认给 10GB 时，int8 后 31 GB 的 transformer
> 在 40 GB 卡上放不下，`ComponentsManager` 会**每个去噪步都换出再换入**，
> 单步 92 秒。调到 3GB 之后单步 3.3 秒，快了 28 倍。

### 结果（同样 16 个窗口）

| run | 整帧 PSNR ↑ | fixed ↑ | wrist ↑ | MSE ↓ | 预测帧间运动 | GT 帧间运动 | 单样本耗时 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `wan22_base` | 8.23 | 8.45 | 7.87 | 10398 | 30.82 | 6.65 | 1.2 s |
| `wan22_native` | 11.16 | 10.90 | 11.06 | 5259 | 15.74 | 6.65 | 1.2 s |
| `ahawam_pretrained` | 12.63 | 12.93 | 10.68 | 4101 | 13.78 | 6.65 | 1.2 s |
| **`minimax_h3_int8`** | **13.11** | **13.11** | **10.97** | **3380** | **13.60** | 6.65 | **232 s** |
| `so101_finetuned` | **15.89** | 16.07 | 13.06 | 1792 | **7.40** | 6.65 | 1.2 s |

MiniMax-H3 逐窗口 PSNR 区间 10.74 ~ 15.96，中位数 13.25。

### 结论

1. **MiniMax-H3 零样本就打平了 AHA-WAM-Pretrained**（13.11 vs 12.63）。
   它从没见过机器人数据、没见过你的场景，只拿到首帧和一句
   "A video recorded from a robot's point of view executing the following instruction: ..."。
   看 `minimax_h3_int8/*/comparison_last.png`：**画面质量明显更干净**——
   桌面颜色、纸卷和白块的位置、甚至那个"上面固定相机 + 左下腕部相机 + 右下黑块"的
   拼图版式，它都完整保住了。它是当作一张普通图片去续写的，却续得很像。
2. **但它不知道机械臂该怎么动**：帧间运动 13.60，是 GT 的 2 倍，
   而且它编出来的手臂轨迹和真实轨迹无关（视频里能明显看到）。
   这正好说明 PSNR 13 附近这一档的含义是"画面对、运动错"。
3. **微调后的 SO101 checkpoint 仍然明显领先**（15.89，运动 7.40 ≈ GT 6.65）。
   一个 5B 的模型在本地数据上微调 4 epoch，
   在"预测这个具体场景的未来"这件事上，超过了一个 33B 的通用视频模型。
   这是符合预期的，也说明你的微调确实学到了东西。
4. **代价差 200 倍**：AHA-WAM 视频分支 1.2 秒一个样本，MiniMax-H3 要 232 秒。
   MiniMax-H3 在这里只适合当离线参照，不可能进控制回路。

一句话：**MiniMax-H3 可以当作"视觉先验上限"的参照物，但它不是 world model 的替代品**——
它缺的恰好是 AHA-WAM 训练要教的那部分（这个机器人在这个场景下接下来会怎么动）。

---

## 指标缩写对照

所有 `summary.json` / `summary.csv` / `metrics.json` 里的字段：

| 字段 | 全称 / 公式 | 含义 | 方向 |
| --- | --- | --- | --- |
| `future_rgb_mse` | Mean Squared Error，`mean((pred - gt)^2)`，像素 0~255 尺度 | 逐像素平方误差 | 越小越好 |
| `future_rgb_psnr` | Peak Signal-to-Noise Ratio，`10 * log10(255^2 / MSE)`，单位 dB | 和 MSE 一一对应，只是换成对数刻度。**每 +3 dB 表示 MSE 减半** | 越大越好 |
| `fixed_rgb_psnr` | 同上，只算画面上方 256 行 | 固定相机区域 | 越大越好 |
| `wrist_rgb_psnr` | 同上，只算左下角 128x160 | 腕部相机区域 | 越大越好 |
| `pred_temporal_abs_diff` | `mean(abs(pred[t+1] - pred[t]))`，0~255 尺度 | 预测视频"动得多剧烈" | **应当接近下一行** |
| `target_temporal_abs_diff` | 同上但对真值 | 真值本身动得多剧烈。同一批窗口下是常数（这里 6.65） | 参照基准 |
| `generation_seconds` | | 单个窗口的生成耗时，**不含模型加载** | |
| `peak_allocated_gib` | `torch.cuda.max_memory_allocated()` | 生成过程中 PyTorch 实际分配的显存峰值。比 `nvidia-smi` 小，因为不含 caching allocator 缓存下来的空闲块 | |
| `history_frames` | | 这个窗口实际用上了几帧历史。episode 开头不够 6 帧时会更少 | |
| `history_rope_indices` | | 每帧历史所在的时间 RoPE 位置，前面的 0 是被 mask 掉的 padding | |

所有 PSNR/MSE 都**排除第 0 帧**，因为那是所有模型都白送的干净条件帧，
算进去会让每个模型都虚高同样的量，反而掩盖真正的差距。

### 两个必须一起看的数

**PSNR 对"物体位置对不对"并不敏感。** 一个完全静止、把首帧复制 13 遍的预测，
在这批数据上也能拿到不低的 PSNR（因为 3.2 秒内背景本来就几乎不动）。
所以一定要**同时看** `pred_temporal_abs_diff` 和 `target_temporal_abs_diff`：

| 情况 | 含义 | 本次例子 |
| --- | --- | --- |
| pred ≫ target | 模型在乱动 | `wan22_base` 30.8 vs 6.65 |
| pred ≈ 2x target | 画面对但运动编错了 | `minimax_h3` 13.6、`ahawam_pretrained` 13.8 |
| pred ≪ target | 输出接近静止画面 | 本次没出现 |
| pred ≈ target 且 PSNR 高 | 真的预测对了 | `so101_finetuned` 7.40 vs 6.65 |

### PSNR 分档（针对这批 384x320 机器人画面）

| 区间 | 大致意味着 |
| --- | --- |
| < 10 dB | 和真值基本无关，通常是彩色噪声 |
| 10 ~ 13 dB | 能看出场景轮廓，但内容和运动都不对 |
| 13 ~ 15 dB | 画面正确、运动错 |
| > 15 dB | 画面和运动都大致对 |

这个分档只在这批数据、这个分辨率下成立，换数据集不要直接套。
**最终还是要打开 `comparison.mp4` 看**，指标只用来排序和发现异常。

---

## 模型大小与显存

参数量和磁盘占用是固定的；**显存占用取决于量化和 offload 策略**，下面两张表分开列。

### AHA-WAM（视频分支 preview，bf16）

| 组件 | 参数量 | 显存 (bf16) |
| --- | --- | --- |
| `video_expert`（Wan2.2-TI2V-5B DiT） | 5.00 B | 9.31 GiB |
| `action_expert`（ActionDiT） | 1.02 B | 1.90 GiB |
| `chunk_kv_cache_editor` | 1.23 B | 4.57 GiB |
| `vae`（Wan2.2 VAE） | 0.71 B | 1.31 GiB |
| **常驻合计** | **8.05 B** | **17.29 GiB** |
| **实测生成峰值** (`peak_allocated_gib`) | | **18.87 GiB** |

- 磁盘：checkpoint `.pt` **13.69 GiB**；Wan2.2 官方发布权重 18.63 GiB（fp32 存储，加载成 bf16 是 9.31 GiB）。
- **T5 文本编码器没有加载**（配置里 `load_text_encoder: false`），走的是预先算好的
  embedding cache，所以省掉了 11.89 GiB。如果哪天要在线编码文本，要把这部分加回去。
- 峰值 18.87 减去常驻 17.29 ≈ **1.6 GiB 是激活值**（13 帧 384x320、4 个 latent 帧）。

### MiniMax-H3

| 组件 | 参数量 | bf16（磁盘） | int8（估算） | int4 group 128（实测） |
| --- | --- | --- | --- | --- |
| `transformer` (FL2VA) | 33.12 B | 61.70 GiB | ~31.6 GiB | **19.46 GiB** |
| `text_encoder` (Qwen3-VL) | 33.36 B | 62.13 GiB | ~33.1 GiB | **19.45 GiB** |
| `vae`（视频） | 2.60 B | 9.70 GiB | 9.70 GiB | 9.70 GiB |
| `audio_vae` | 0.15 B | 0.56 GiB | 0.56 GiB | 0.56 GiB |
| **合计** | **69.24 B** | **134.13 GiB** | ~75 GiB | **49.17 GiB** |

两个 VAE 不量化（体积小、且量化后画质损失明显）。
transformer 里 0.85 B、conditioner 里 2.15 B 参数在 `modules_to_not_convert` 名单上，
保持 bf16——上表已经把它们算进去了。

> int4 那一列是脚本启动时打印的实测值（`weight_footprint_gib` 也会写进 `summary.json`）。
> 注意 19.46 GiB 高于"33 B x 0.5 字节 = 16.5 GiB"的朴素估算：tinygemm 的
> `tile_packed_to_4d` 布局要额外存每组的 scale/zero，还有 tile 对齐的 padding。
> 另外**不能用 `numel * element_size` 去量 torchao 张量**——量化后的张量仍然对外
> 报告 bf16 的逻辑形状，那样量出来还是 61.7 GiB。要顺着 `__tensor_flatten__`
> 找到里面真正的打包张量（`component_footprint()` 就是这么做的）。

**这四个组件不会同时待在卡上。** `ComponentsManager` 的 auto CPU offload
按需搬运，所以真正的峰值显存由"同一时刻最多有谁常驻 + 激活值"决定，
而这由 `--memory-reserve-margin` 控制（预算 = 卡的显存 − margin）：

| 配置 | 预算 | 谁能同时常驻 | 实测峰值 | 单样本耗时 |
| --- | --- | --- | --- | --- |
| int8, margin 10GB（初版） | 30 GB | 谁都放不下 → **每步换出换入** | — | **~92 s/步** |
| int8, margin 3GB | 37 GB | transformer 一个（两个大件加起来 65 GiB，放不下） | **33.89 GiB** | 232 s |
| int4, margin 3GB | 37 GB | transformer + VAE 等多个一起赖着不走 | **34.15 GiB** | 881 s |
| **int4, margin 20GB** | 20 GB | 只有 transformer | **21.31 GiB** | ~988 s |

### 关于"能不能压到 AHA-WAM 那个级别"

能压到**同一个量级，但压不到同一个数**。对比：

| | 常驻权重 | 实测峰值 | 单样本 |
| --- | --- | --- | --- |
| AHA-WAM 视频分支 (bf16) | 17.29 GiB | **18.87 GiB** | 1.2 s |
| MiniMax-H3 int8 | ~31.6 GiB | 33.89 GiB | 232 s |
| MiniMax-H3 int4 + margin 20GB | **19.46 GiB** | **21.31 GiB** | 988 s |

int4 把峰值从 33.9 压到 **21.3 GiB**，已经能和 AHA-WAM 塞进同一张卡的同一个预算档，
但比 18.87 还高 13%。**19.46 GiB 是硬下限**——那是 transformer 自己 int4 后的重量，
再怎么调 offload 预算也降不下去。想再往下只有两条路：

1. `--height 384 --width 320`（回到原生分辨率）能砍掉大部分激活值，
   峰值大约能到 20 GiB 出头，但画布远小于 MiniMax-H3 的训练分辨率，画质会掉；
2. 给 transformer 开 `enable_group_offload(offload_type="block_level")`，
   把 transformer 的 block 逐个从 CPU 流进来。常驻能降到几 GiB，
   但每步都要搬运权重，会比现在的 988 秒再慢好几倍。

**代价要说清楚**：int4 在 A100 上比 int8 慢 **4.3 倍**（988 s vs 232 s）。
原因是 sm_80 上唯一可用的 int4 内核是 tinygemm `_weight_int4pack_mm`，
它是为 LLM 单 token 解码那种瘦长 GEMM 优化的；视频 DiT 这种上万 token 的
大 GEMM 走它反而比 int8 慢。所以：

- **想省显存 → int4**；
- **想快 → int8**；
- 画质上两者在这批数据里看不出差别（同一个窗口 int4 15.14 dB vs int8 14.67 dB，
  单样本差异在采样噪声范围内）。

顺带一提：**int4 不需要额外下载**。torchao 直接从磁盘上已有的 bf16 权重现场量化，
社区那些 `-int4`、`-nvfp4` 包是 ComfyUI 单文件格式，既要转换又大多需要
Ada/Hopper/Blackwell 的内核，在 A100 上用不了。

---

## 目录结构

```
scripts/so101_pipeline/baseline_video_experiments/
├── export_reference_samples.py        # 第 1 步：挑窗口 + 导出 GT（ahawam 环境）
├── common/
│   ├── sample_selection.py            # 窗口挑选逻辑
│   └── reporting.py                   # 指标 / 对比图 / mp4 / summary（无 ahawam、无 torch 依赖）
├── wan22_base/
│   ├── run_video_branch_preview.py    # 实验 A
│   └── run_video_branch_preview.sh
├── minimax_h3/
│   ├── setup_env.sh                   # 建 minimax_h3 conda 环境
│   ├── run_minimax_h3_preview.py      # 实验 B（QUANT=int8 / int4）
│   └── run_minimax_h3_preview.sh
├── compare_runs.py                    # 合并所有 run：对比表 + 并排视频
├── verify_video_path_equivalence.py   # 链路等价性数值验证
├── PATH_AUDIT_ZH.md
├── WHY_SLOW_ZH.md
├── HOW_TO_USE_ZH.md
└── logs/
```

去噪本身仍然调用 `deploy/server/ahawam_policy.py:denoise_future_video_latents()`，
历史 K/V 仍然复用 `archive/video_path_experiments/exact_video_path.py` 里的
`_prepare_history_cache` / `_prepare_online_history_cache`，
所以新脚本和你原来的 `video_preview/run_preview_batch.sh` 走的是同一份代码。
