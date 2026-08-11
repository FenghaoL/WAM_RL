# 批量历史视频预测对照实验

这组脚本检验：使用模型在联合训练/推理时同一条 VideoDiT 网络路径，加入真实
历史 K/V 后，SO101 和 RoboCOIN 本域数据分别能生成怎样的未来视频。

## 1. 实验路径

每个样本执行：

1. 从真实 episode 取历史观察、当前帧和真实未来帧；
2. 用训练相同的 VAE encoder 得到 latent；
3. 历史帧逐层 prefill 成 rolling Video K/V；
4. 当前帧作为固定 clean prefix，未来 latent 从高斯噪声开始；
5. 使用 Video expert 的 `pre_dit -> 30 blocks -> post_dit` 和同一 flow scheduler
   做 20 步采样；
6. VAE decode 为 RGB，并与真实未来视频逐帧并排输出。

该 checkpoint 的 `video_dit_config.action_conditioned=false`，所以 VideoDiT 接收
历史视觉、当前视觉、全局 RoPE 时间位置和任务文字，但不接收 ActionDiT 本次生成
的动作。这是模型当前设计的自由未来预测，不是 action-conditioned simulator。

## 2. 命令和产物

```bash
bash scripts/so101_pipeline/video_path_experiments/batch/run_so101_batch.sh
bash scripts/so101_pipeline/video_path_experiments/batch/run_robocoin_batch.sh
```

模型每次只加载一次，然后连续处理 8 个样本。每个样本保存：

- `history_strip.png`：送入 K/V memory 的历史帧；
- `ground_truth.mp4`：真实当前帧和未来；
- `prediction_with_history.mp4`：预测当前帧和未来；
- `comparison_gt_vs_prediction.mp4`：左右同步对比；
- `comparison_middle.png`、`comparison_last.png`：中间和末帧静态对比；
- `metrics.json`：MSE、PSNR、时间变化量、历史长度与生成时间。

根目录另有 contact sheet、`summary.csv`、`summary.json` 和自动生成的 `README.md`。

## 3. 本机结果

| 数据 | 样本 | MSE | PSNR | 预测帧间变化 | 真实帧间变化 | 生成时间 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SO101 | 8 | 4749.192 | 11.921 dB | 13.716 | 9.455 | 1.478 s | 18.891 GiB |
| RoboCOIN | 8 | 602.709 | 20.390 dB | 4.664 | 6.795 | 1.278 s | 18.956 GiB |

输出目录：

```text
runs/so101_g05_square_v1/video_path_batch_history/
runs/robocoin/video_path_batch_history/
```

视觉结论：

- SO101 加历史后场景比单帧 preview 稳定，但机械臂形态和运动轨迹仍明显错误；
- RoboCOIN 本域的背景、桌面和头部相机更稳定，PSNR 高约 8.5 dB；
- RoboCOIN 的预测动态小于真实动态，说明它也常把运动“抹平”，并非可靠物理轨迹；
- 因此 SO101 的坏视频不只是 preview 输入链路错误，域差异和未微调 VideoDiT 都是
  更有力的解释，这也是本次尝试全量 WAM 微调的依据。

重要：上面 SO101 的结果生成于旧的“腕部拉满 `128x320` 底栏”布局。现在训练已
改为“腕部左下 `128x160` + 右下黑槽”。旧 checkpoint 也学习了旧布局，因此不能
只换输入布局后公平重跑；应在新的全量 checkpoint 训练到一个 epoch 后，再用相同
脚本生成新基线。

参考：

- [AHA-WAM 官方仓库](https://github.com/serene-sivy/AHA-WAM)
- [AHA-WAM 论文](https://serene-sivy.github.io/aha-wam/static/aha-wam.pdf)
- [FastWAM 官方仓库](https://github.com/yuantianyuan01/FastWAM)
