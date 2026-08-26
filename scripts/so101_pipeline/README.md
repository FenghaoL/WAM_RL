# SO101 使用入口

这里保留几套实际使用的代码和独立复现实验。第一次使用时先选一个模块，再阅读
该目录内唯一的 `README.md`。

```text
scripts/so101_pipeline/
├── online_deploy/            在线控制与未来视频服务
├── fastwam_deploy/           FastWAM 原生权重审计与 14-D 服务验证
├── video_preview/            在线同路径的离线批量未来视频对照
├── baseline_video_experiments/  跨 checkpoint / 跨模型的视频分支对照实验
├── partial_finetune_zero2/   冻结 VideoDiT 的 ZeRO-2 微调
├── full_finetune_zero2_offload/  FastWAM 初始化的四卡全量 ZeRO-2 offload 微调
├── full_finetune_zero3/      VideoDiT + ActionDiT 全量 ZeRO-3 微调
├── zero2_repro/              官方 ZeRO-2 四卡全量 OOM 复现实验
├── benchmark/                两个训练脚本共用的监控工具，不直接运行
├── config_utils.py           两套训练共用的 Hydra 配置加载器
├── so101_dataset.py          两套训练共用的 SO101 数据集适配器
└── archive/                  历史实验，仅供追溯
```

## 应该使用哪套训练

| 目标 | 入口 | 特点 |
| --- | --- | --- |
| 快速适配 SO101 动作 | `partial_finetune_zero2/` | 冻结 VideoDiT，3 卡，速度较快 |
| 新 SO101 数据全量训练 | `full_finetune_zero2_offload/` | 4 卡，optimizer CPU offload，当前推荐 |
| 同时适配场景和未来视频 | `full_finetune_zero3/` | 全量训练，4 x A100 40GB，速度较慢 |

需要向别人复现“官方 ZeRO-2、四卡全量训练为何 OOM”时，使用 `zero2_repro/`。它是
1-step 诊断，不是正式训练入口。

想搞清楚"训练和部署各自在算什么、多久算一次、和论文是否一致"，读
`PIPELINE_TIMING_ZH.md`。它有实测耗时，并指出了两处需要注意的地方：
当前 `num_inference_steps=10` 让动作生成比实时慢 2.7%，以及
`run_action_server.sh` 并没有开启仓库里那套异步双流。

想横向比较不同 checkpoint（含未训练的 Wan2.2 初始化）或别的视频模型（MiniMax-H3）
在同一批 SO101 窗口上的视频生成效果，用 `baseline_video_experiments/`。它先把评测
窗口固定落盘，再让每个模型打同一批窗口，所以数字可以直接对比；
`video_preview/` 每次运行会重新挑窗口，跨 checkpoint 的数字并不可比。
该目录的 `PATH_AUDIT_ZH.md` 还记录了“这条 preview 链路是否等于训练/部署链路”的
逐项审计和数值验证。

新数据 `data/so101_ahawam_square_v1` 使用
`full_finetune_zero2_offload/`。`full_finetune_zero3/` 保留为显存更紧张时的备选和
历史结果。数据、checkpoint 和 `runs/` 都不在本目录整理范围内，原样保留。

## 配置位置

- ZeRO-2 训练：`partial_finetune_zero2/settings.yaml`
- ZeRO-2 offload 全量训练：`full_finetune_zero2_offload/settings.yaml`
- ZeRO-3 训练：`full_finetune_zero3/settings.yaml`
- 在线部署：`online_deploy/settings.yaml`
- FastWAM 原生验证：`fastwam_deploy/settings_native_robotwin.yaml`

通常只需要修改这三份 YAML 和相应 `.sh` 文件开头的 GPU、端口、任务文字等简单
默认值。`archive/` 中脚本的路径可能已经过期，不应作为正式入口。
