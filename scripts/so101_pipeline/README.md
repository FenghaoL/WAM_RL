# SO101 使用入口

这里保留三套实际使用的代码和一套独立复现实验。第一次使用时先选一个模块，再阅读
该目录内唯一的 `README.md`。

```text
scripts/so101_pipeline/
├── online_deploy/            在线控制与未来视频服务
├── partial_finetune_zero2/   冻结 VideoDiT 的 ZeRO-2 微调
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
| 同时适配场景和未来视频 | `full_finetune_zero3/` | 全量训练，4 x A100 40GB，速度较慢 |

需要向别人复现“官方 ZeRO-2、四卡全量训练为何 OOM”时，使用 `zero2_repro/`。它是
1-step 诊断，不是正式训练入口。

当前正在运行的是第二种。数据、checkpoint 和 `runs/` 都不在本目录整理范围内，原样
保留。

## 配置位置

- ZeRO-2 训练：`partial_finetune_zero2/settings.yaml`
- ZeRO-3 训练：`full_finetune_zero3/settings.yaml`
- 在线部署：`online_deploy/settings.yaml`

通常只需要修改这三份 YAML 和相应 `.sh` 文件开头的 GPU、端口、任务文字等简单
默认值。`archive/` 中脚本的路径可能已经过期，不应作为正式入口。
