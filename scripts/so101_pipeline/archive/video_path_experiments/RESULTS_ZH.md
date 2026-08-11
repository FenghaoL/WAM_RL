# AHA-WAM 真实视频链路对照实验

更新时间：2026-08-09

## 1. 先回答两个直接问题

### 旧 preview 有没有使用任务文字？

有。旧 `generate_future_video()` 会把当前任务填入：

```text
A video recorded from a robot's point of view executing the following instruction: {task}
```

然后读取这个完整字符串对应的 UMT5 embedding，传给 VideoDiT 的 `pre_dit()`。VideoDiT 每层通过 text cross-attention 使用这些 token。所以旧 preview 的问题不是漏掉任务文字。

### 旧 preview 有没有使用历史视觉 K/V？

没有。它只把当前相机画面编码成第一个干净 latent，后面的 latent 从随机噪声开始生成。

这会有影响。历史 K/V 不是一串原始 RGB，也不是历史动作；它是历史观察经过 VideoDiT 每层后留下的 key/value memory。当前视频 token 可以查询它，从而知道较早时刻的场景布局和变化趋势。

需要注意：AHA-WAM 的历史观察很稀疏。它每隔一个完整视频 horizon 取一个历史锚点，不保存控制频率下的每一帧，因此能提供“跨 horizon 的场景演化”，但不能像光流或完整录像那样无损恢复机械臂刚才每一小步怎么运动。

## 2. 为什么本实验是正确的视频推理链路

新脚本位于 `exact_video_path.py`，每个去噪步依次执行：

1. 当前第一帧经 Wan VAE 编成干净 latent，未来 latent 使用固定随机种子的高斯噪声。
2. 历史 RGB 分别经同一个 Wan VAE 编码。
3. 历史 latent 使用时间步 0、真实全局 RoPE 帧号和任务文字，调用 `video_expert.pre_dit()`。
4. 调用训练代码同一个 `mot.prefill_video_cache()`，得到 30 层历史 K/V。
5. 当前 noisy latent 使用真实全局 RoPE 帧号和相同任务文字，调用 `video_expert.pre_dit()`。
6. 逐层执行训练联合 forward 中相同的视频 Q/K/V、per-frame causal self-attention、历史前缀注意力、text cross-attention 和 MLP。
7. 调用相同 `video_expert.post_dit()` 得到 flow，再由相同 inference scheduler 积分 20 步。
8. 每一步都把第一个 latent 固定为真实当前观察，最后使用相同 Wan VAE 解码 RGB。

本实验没有运行 ActionDiT，但不会改变视频输出。当前配置明确为 `video_dit_config.action_conditioned: false`；联合 MoT 中每层也是先计算视频，再让动作分支读取视频 K/V，没有 ActionDiT 输出反向写入 VideoDiT。因此，这里省掉的只是与视频结果无关的动作计算。

这比“调用 training_loss 看一段视频”更准确。训练时不会自由生成完整 MP4：它把真实未来 latent 加到随机噪声等级，只预测一次 flow 并计算损失。这里测试的是部署时真正关心的、更困难的自由生成：从纯未来噪声连续积分 20 步。

## 3. 实验变量

两路预测严格共享：

- 同一个 checkpoint
- 同一个真实样本和任务文字
- 同一个第一帧
- 同一个随机种子 42 和初始噪声
- 同样 20 个 scheduler 步
- 同一个 VAE 和 VideoDiT

唯一变量是是否给当前视频分支添加训练同款历史 K/V。因此两路差异可以归因于历史视觉记忆。

输出中的 `ground_truth.mp4` 是数据集真实未来，`prediction_no_history.mp4` 是旧 preview 的核心条件，`prediction_with_history.mp4` 是修正后的条件，`comparison.mp4` 将三者横向排列。

一键脚本按物理采样率保存：SO101 为 3.75 FPS，RoboCOIN 为 6.25 FPS。首轮实验产物另保留了快速检查版本，并额外导出 `comparison_physical_fps.mp4` 供按真实时间观看。

## 4. SO101 结果

使用：

- checkpoint：4 epoch SO101 step 6524
- 样本索引：8696
- 任务：`Put the white block on the paper roll.`
- 数据：15 Hz 原始控制/相机时间轴
- 当前 clip：连续 49 个原始时刻，每 4 个取一张，共 13 张视频帧
- 真实 horizon：48 / 15 = 3.2 秒
- 历史：4 个真实锚点
- 所有未来视频点和动作点均无 padding

| 指标 | 无历史 K/V | 训练式历史 K/V |
|---|---:|---:|
| Future RGB MSE | 7790.642 | 5294.007 |
| Future RGB PSNR | 9.215 dB | 10.893 dB |
| 预测帧间绝对变化 | 18.631 | 14.335 |
| 真实帧间绝对变化 | 20.284 | 20.284 |
| 去噪加解码耗时 | 1.325 s | 1.202 s |
| 峰值 allocated 显存 | 18.726 GiB | 18.891 GiB |

历史 K/V 使 MSE 降低约 32%，PSNR 提高 1.68 dB。固定相机的桌面和物体布局明显更稳定，说明旧 preview 漏历史确实是一个真实问题。

但修正后仍然不能正确生成机械臂的大幅运动，腕部区域也严重失真。历史输入缺失不是 SO101 视频崩坏的唯一原因。

结果目录：`runs/so101_g05_square_v1/video_path_experiment/`

## 5. RoboCOIN 原生域结果

下载的数据：`RoboCOIN/Cobot_Magic_move_the_bread`，979 MB，99 episodes，三相机，50 Hz。三路相机正好对应 checkpoint 的 head、left wrist、right wrist 布局。

使用：

- checkpoint：`AHA-WAM-pretrained.pt`
- episode：0，共 528 帧
- 当前 clip：第 384 到 448 帧，每 8 帧取一张，共 9 张视频帧
- 真实 horizon：64 / 50 = 1.28 秒
- 历史锚点：第 0、64、128、192、256、320 帧
- 任务文字逐字读取自数据集 `meta/tasks.jsonl`

| 指标 | 无历史 K/V | 训练式历史 K/V |
|---|---:|---:|
| Future RGB MSE | 1233.727 | 610.216 |
| Future RGB PSNR | 17.219 dB | 20.276 dB |
| 预测帧间绝对变化 | 6.904 | 4.113 |
| 真实帧间绝对变化 | 6.738 | 6.738 |
| 去噪加解码耗时 | 1.259 s | 1.176 s |
| 峰值 allocated 显存 | 18.708 GiB | 18.956 GiB |

历史 K/V 使 MSE 降低约 50.5%，PSNR 提高 3.06 dB。固定相机中的绿色桌面、盘子、苹果和背景在整个预测中保持正确，末帧没有跳到其他场景。腕部小视图仍有形变和少量手部伪影，所以模型的未来 RGB 不是精确录像复现。

加载 14D RoboCOIN checkpoint 到 6D SO101 外壳时，只跳过 action encoder weight、action head weight 和 action head bias。这三个参数没有在本实验中执行；VideoDiT、文本 cross-attention、scheduler 和 VAE 均来自 RoboCOIN checkpoint，因此不影响视频结论。

结果目录：`runs/robocoin/video_path_experiment/`

## 6. 可以得出的结论

1. 旧 preview 使用了任务文字，但遗漏了历史视觉 K/V。
2. 历史 K/V 在 SO101 和 RoboCOIN 上都明显改善生成，应该接回在线 preview。
3. 同一视频生成链路在 RoboCOIN 原生相机、机器人和场景分布上能生成结构正确的视频，说明 scheduler、VAE、文本输入和 VideoDiT 调用顺序没有根本性接线错误。
4. SO101 在补齐历史后仍明显更差，主要证据指向 domain/embodiment/camera-layout shift，而不是单纯的 preview 输入错误。
5. 当前 SO101 训练冻结了整个 VideoDiT，而且核对过其 825 个 VideoDiT tensor 与 RoboCOIN 初始化完全相同。因此 4 epoch SO101 训练没有让视频主干学习单臂、双相机和黑桌面分布。
6. 视频差不等于动作一定差。动作分支可通过 action loss 学习，并读取干净当前观察的 VideoDiT K/V；动作服务也不需要先解码未来 RGB。但“机械臂完全不动”加上视频域偏移，说明后续仍应分别检查动作数值、归一化、动作执行协议和动作模型训练，而不能只怪视频头。

## 7. 在线 preview 的修正

`deploy/server/ahawam_policy.py` 现在会：

- 把任务文字继续传给 VideoDiT。
- 使用模型已积累的滚动历史 K/V。
- 通过图像哈希识别“当前帧已经做过 planner prefill”的情况，避免把当前帧同时作为 first frame 和历史前缀重复两次。
- 给当前和未来 latent 使用全局 RoPE 时间坐标，不再每次从 0 开始。
- 在响应中返回 `history_frames_used` 和 `video_current_frame_index`。

第一帧请求没有历史时，`history_frames_used=0` 是正常的。经过多个 planner refresh 后，该值会逐渐增加，最多为配置中的 `num_history_frames`。

已用两张真实 SO101 观察做过 policy 级 smoke test，实际经过 observation update、planner prefill 和 `generate_future_video()`：`history_frames_used=1`、`video_current_frame_index=10`、输出 13 帧，端到端 preview 为 2.234 秒。该在线时间包含历史编码、输入准备和文件输出，因此高于表中只统计去噪加解码的约 1.2 秒。

独立视频进程也做了连续两次纯 `video_preview` smoke：第 1 次返回 `history_frames_used=0, video_current_frame_index=0`，生成后自动缓存该观察；第 2 次返回 `history_frames_used=1, video_current_frame_index=12`。因此双 GPU 架构中的视频进程可以积累自己的 preview history，不需要共享动作进程状态。

## 8. 一键命令

SO101：

```bash
bash scripts/so101_pipeline/video_path_experiments/run_so101.sh
```

RoboCOIN 下载、文字 embedding 和实验：

```bash
bash scripts/so101_pipeline/video_path_experiments/download_robocoin.sh
bash scripts/so101_pipeline/video_path_experiments/prepare_robocoin_text.sh
bash scripts/so101_pipeline/video_path_experiments/run_robocoin.sh
```

常用简单参数：

```bash
GPU=0 STEPS=20 SEED=42 bash scripts/so101_pipeline/video_path_experiments/run_so101.sh
EPISODE=0 START_FRAME=384 bash scripts/so101_pipeline/video_path_experiments/run_robocoin.sh
```

## 9. 资料来源

- [AHA-WAM paper](https://arxiv.org/abs/2606.09811)
- [AHA-WAM RoboCOIN pretrained checkpoint card](https://huggingface.co/SereneC/AHA-WAM-Pretrained)
- [RoboCOIN official repository](https://github.com/FlagOpen/RoboCOIN)
- [Cobot Magic move the bread dataset](https://huggingface.co/datasets/RoboCOIN/Cobot_Magic_move_the_bread)
