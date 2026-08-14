# SO101 在线同路径批量视频 Preview

> 想跨 checkpoint 比较（含未训练的 Wan2.2 初始化、MiniMax-H3），用
> [`../baseline_video_experiments/`](../baseline_video_experiments/README.md)：
> 它把评测窗口固定下来让所有模型打同一批，数字才可比。本目录每次运行都会重新挑窗口。
> 这条 preview 链路与训练/部署链路的逐项审计见
> [`../baseline_video_experiments/PATH_AUDIT_ZH.md`](../baseline_video_experiments/PATH_AUDIT_ZH.md)。

这个目录提供一个入口：

```bash
bash scripts/so101_pipeline/video_preview/run_preview_batch.sh
```

需要调整 GPU、样本数、checkpoint 或输出目录时，直接修改脚本开头的变量即可。

## 为什么它和在线推理一致

在线服务和这个离线批处理现在共同调用
`deploy/server/ahawam_policy.py` 中的 `denoise_future_video_latents()`：

1. 当前真实图像经同一个 VAE encoder，作为固定 clean first frame；
2. 从 episode 按在线 planner 的 40 个控制帧间隔取历史图像；
3. 历史图像按 RoPE 间隔 10 逐帧 prefill，使用与在线服务相同的 rolling K/V；
4. 使用相同的任务文字 embedding 和当前/未来 temporal RoPE 位置；
5. 未来 latent 从相同 seed 的高斯噪声开始；
6. 每一步都执行同一个 `VideoDiT pre_dit -> video-only MoT -> post_dit`；
7. 使用同一个 flow scheduler 和 VAE decoder。

离线历史来自录制数据，在线历史来自机械臂运行期间的真实观察，所以图像内容来源不同；
观察间隔、RoPE 时间位置、历史编码、rolling K/V 和未来视频去噪实现相同。训练 dataset
原本使用 48 控制帧、RoPE 12 的历史窗口；这里特意不用那个间隔，因为在线动作服务是
每 40 个控制帧刷新一次 planner。

当前模型配置为 `video_dit_config.action_conditioned=false`。VideoDiT 不读取动作 tensor，
MoT 中的信息方向是 VideoDiT 生成视频 K/V，再供 ActionDiT 使用；ActionDiT 不反向修改
VideoDiT。因此这个 preview 就是联合推理中的视频分支，不需要先生成动作。

## 输出

默认输出到：

```text
runs/so101_g05_square_v1/full_zero3/video_preview_online_history/
```

每个样本目录包含：

- `ground_truth.mp4`：数据中的真实未来；
- `prediction_with_history.mp4`：在线同路径预测；
- `comparison_gt_vs_prediction.mp4`：真实与预测并排播放；
- `history_strip.png`：送入 rolling K/V 的历史帧；
- `comparison_middle.png`、`comparison_last.png`：中间帧和末帧；
- `metrics.json`：任务文字、历史长度、生成速度、显存和画面误差。

根目录还会生成 `summary.csv`、`summary.json` 和两张 contact sheet。
