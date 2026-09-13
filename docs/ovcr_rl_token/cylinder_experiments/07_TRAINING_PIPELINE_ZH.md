# 训练管线逐层讲解：一个 sample 是什么、边界怎么处理、一个 step 发生了什么

面向"没有前置知识"的读者，按"磁盘 → sample → 模型 → 损失 → 一次参数更新"的顺序展开，
所有数字都来自本仓库代码和对 cylinder 数据的实测。

## 0. 前置概念

| 概念 | 解释 |
|---|---|
| episode | 一次连续录制。本项目 15 fps，一帧 = 1/15 秒 ≈ 67 ms |
| 动作 | 每帧一条 6 维关节目标（SO101 六个关节） |
| action chunk | 模型一次生成的一小段连续动作。本项目 `action_chunk_size=8`，即 8 帧 ≈ 0.53 秒 |
| flow/扩散训练 | 给真值动作/视频加噪声，让模型预测"去噪方向"。每个样本随机抽一个噪声强度（时间步） |
| token | Transformer 的最小处理单位。图像被切成小块后每块是一个 token |
| KV cache | 注意力里算过的 Key/Value 缓存，可被后续 token 复用，避免重算 |
| VAE latent | 图像先被压缩成小得多的隐变量再喂给模型。本项目时间压缩 4 倍、空间压缩 16 倍 |

## 1. 一个 sample 的构成（实测）

一个训练样本 = 以某一帧 t 为"当前帧"截取的一个时间窗口。实测（`train[0]`）：

| 张量 | 形状 | dtype | MB |
|---|---|---|---:|
| `video` | [3, 13, 384, 320] | float32 | 19.17 |
| `chunk_obs_images` | [5, 3, 384, 320] | float32 | 7.37 |
| `chunk_obs_images_no_offset` | [5, 3, 384, 320] | float32 | 7.37 |
| `video_history` | [3, 6, 384, 320] | float32 | 8.85 |
| `context`（文本 embedding，预计算缓存） | [128, 4096] | bfloat16 | 1.05 |
| `action` / `proprio` | [48, 6] / [48, 6] | float32 | 0.002 |
| `action_is_pad` / `image_is_pad` / `proprio_is_pad` | [48] / [13] / [49] | bool | ~0 |
| **合计** | | | **43.81** |

窗口定义在 `robot_video_dataset.py:117-118`：`obs_size=num_frames=49`、`action_size=48`，
`base_lerobot_dataset.py:74-97` 把它翻译成"相对当前帧 t 的时间偏移查询"：状态取 t…t+48（49 个），
动作取 t…t+47（48 个），图像取 `image_sample_indices`。视频只取其中 13 帧（0,4,8,…,48，
即 `action_video_freq_ratio=4`），两个相机拼成一张 384×320 图。

`chunk_obs_images` 是 5 张"每个 chunk 起点的观测图"（40 个动作 = 5 chunk × 8）。
`action_offset` 是每次取样用 `np.random.randint(0..7)` 抽的随机起点（`robot_video_dataset.py:388-391`），
用来增强；chunk 观测图按这个偏移挑帧（`:393-408`）。**比较两个数据集的样本时必须固定随机种子，否则这一项必然不同。**

## 2. 边界怎么处理

**结尾不够**：查询超出 episode/片段末尾时，LeRobot 把查询夹到范围内并置 `*_is_pad=True`。
实测一个 recover 样本：`action_is_pad` 48 个里有 7 个为真、`image_is_pad` 13 个里有 2 个为真。
片段最后一帧的样本则是 47/48 个动作、12/13 张图都是 padding。

**片段边界**：DAgger 人类接管片段用 `hil_human_only_dataset.py` 的
`_clamp_sequence_to_human_boundary` 把跨越接管边界的查询截断并标 pad；新录的短片段用
`logical_view`（同文件）把每条 episode 的前 N 帧裁掉，逻辑起点后移，其余机制不变。
两者都不改磁盘数据，只改"从哪儿开始取、到哪儿为止有效"。

**开头没有历史**：历史帧按 `idx - 48k` 取（`robot_video_dataset.py:487-527`），
不够就补零并记 `video_history_valid_len`，模型再按这个长度做 mask。所以"开头没有历史"是
自动处理的，不需要特殊配置。

## 3. 历史（memory）的真实开销

sample 里装的是**历史图像**，不是 KV。KV 在训练时现算：`ahawam.py:758-800` 先取
`effective_history_frames = batch 内 valid_history_len 的最大值`，**为 0 就整段跳过**；
不为 0 时在 `torch.no_grad()` 下编码并 `detach()`，因此历史不参与反向传播、不产生梯度。

一帧历史需要当前起点距片段开头 ≥48 帧，再多一帧又要 48 帧。实测：

| 数据 | 平均历史帧 | 完全无历史 | 最大 |
|---|---:|---:|---:|
| demo | 1.763 | 22.9% | 6 |
| DAgger 人类接管 | 0.148 | 88.7% | 3 |
| 09-11 纠正片段 | 0.070 | 93.5% | 2 |
| recover 片段（裁剪后） | 0.000 | 100% | 0 |

**结论：对短片段，带不带历史没有区别**（输入里根本没有历史帧）。差别只出现在 batch 里
另一半 demo 样本，以及在 demo 上计算的 P0 anchor。注意混合 batch 里只要有一个 demo 样本有历史，
历史分支就会为整个 batch 执行一次（其余样本被 mask 掉），这部分算力省不掉，但没有梯度。

## 4. 片段短到什么程度会有问题

模型每次固定预测 40 个动作（5 个 chunk）。一个起点实际能监督多少个动作 =
`clamp(片段剩余长度 - action_offset, 0, 40)`，其余位置被 `action_is_pad` 排除在损失外
（`ahawam_chunk_base.py:975-985`，按有效数归一，`valid_sum.clamp(min=1)`）。

| 数据 | 片段数 | 长度中位数 | 平均被监督动作 | 完全没有有效动作 | 不足一个 chunk(8) |
|---|---:|---:|---:|---:|---:|
| demo episode | 94 | 210 | 35.6/40 | 1.7% | 5.0% |
| DAgger 人类接管 | 77 | 24 | 16.4/40 | 11.3% | 33.7% |
| 09-11 纠正片段 | 101 | 45 | 20.4/40 | 7.5% | 22.5% |
| recover（裁剪前） | 30 | 37 | 17.2/40 | 9.0% | 27.1% |
| recover（裁掉 8 帧后） | 30 | 29 | 13.6/40 | 11.4% | 34.1% |

要点：

- **不会报错，也不会"凑不满就丢弃"**：不足的部分是 padding，只是不进损失。
- 片段长度 < 9 帧时，任何起点都填不满一个 chunk（但仍可训练）。
- `valid=0` 的样本（片段末尾 + 较大随机偏移）对动作损失的贡献精确为 0，却照样消耗一次完整前向反向。裁掉开头会让这个比例上升（9.0% → 11.4%），这是裁剪的代价。
- 视频损失另有要求：当前起点的 13 张采样帧必须全部有效（≥49 帧剩余）。recover 数据裁剪后最长 47 帧，**永远不会有视频损失**；adapter 实验本来就 `lambda_video=0`，所以无影响。

## 5. 模型前向与损失

1. `video`（13 帧）过 VAE → 4 个 latent 帧（13 = 1 + 3×4），第一帧被替换成干净的当前帧
   latent（`latents[:, :, 0:1] = first_frame_latents`），后 3 个 latent 步是预测目标；
2. 动作侧从 48 个动作里按 `action_offset` 切出 40 个（`_training_loss_action_offset`），
   加噪；`action_train_timestep_mode: per_chunk` 表示 5 个 chunk 各抽一个噪声强度；
3. video DiT 与 action DiT 在 MoT 里联合前向；OVCR chunk-KV editor 在这一步把 32 个观测 query
   读到的信息写回 video KV 的当前帧槽位——**adapter 就挂在这里**；
4. 动作损失：MSE，按 `~action_is_pad` 加权平均；
   视频损失：先把 12 个尾帧按 4 帧一组映射到 3 个 latent 步（`base_wam.py:687-732`，
   一组 4 帧全是 padding 才算该 latent 步无效），再按 `video_supervision_mask` 决定该样本是否计入
   （`ahawam.py:1226-1258`）；
5. adapter 实验额外有 P0 anchor：同噪声、同时间步下对比冻结原模型的输出，只作用在 demo 样本上。

## 6. 一个 step 是什么

`trainer.py:1131-1160`：取一个 batch → `training_loss` → `backward` → 梯度裁剪 →
`optimizer.step()` → `global_step += 1`。本项目每卡 batch 4、3 张卡、累积 1，
所以**一个 step = 12 个样本**；采样器保证每卡的 4 个样本是 2 个 demo + 2 个纠正
（`samplers.py` 的 `BalancedTwoSourceEpochSampler`）。3285 步 = demo 起点正好过一遍（19708/6）。
实测约 3.3 s/step，其中数据加载约 0.01 s，其余是前向/反向。

每步从 CPU 送到 GPU 的数据约 12 × 43.8 MB ≈ 526 MB，这也是 `video`/`chunk_obs_images` 用
float32 存放的代价；真正占显存大头的是模型和激活，不是样本本身。
