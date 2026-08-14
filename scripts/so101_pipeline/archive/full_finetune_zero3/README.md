# SO101 ZeRO-3 全量微调

这套训练令 `freeze_video_dit: false`，同时训练 VideoDiT、ActionDiT、MoT/OVCR 和
SO101 6D 适配层。UMT5 文本编码器和 Wan VAE 仍冻结。它是当前 4 x A100 40GB 上
用于适配动作与场景视频的正式方案。

## 文件

| 文件 | 作用 |
| --- | --- |
| `settings.yaml` | 数据、batch、epoch、学习率、冻结策略和初始化 checkpoint |
| `accelerate_zero3_hybrid.yaml` | Accelerate 四进程入口 |
| `ds_zero3_hybrid.json` | ZeRO-3，35% optimizer state 放 CPU |
| `aligned_sampler.py` | 保证四个 rank 使用相同 history 分支 |
| `zero3_trainer.py` | ZeRO-3 权重聚合和便携 checkpoint 保存 |
| `train_entry.py` | ZeRO-3 外部参数协调、schedule 和 Trainer 入口 |
| `self_test.py` | 不加载大模型的 sampler/配置自检 |
| `smoke_test.sh` | 默认一 step 四卡训练短测 |
| `benchmark_batch2.sh` | 正式任务结束后测试 micro batch 2 |
| `train_4epochs.sh` | 前台正式训练 |
| `start_training.sh` | 在 tmux 中后台启动 |
| `status_training.sh` | 查看进度 |
| `stop_training.sh` | 停止正式任务 |

`benchmark/gpu_monitor.py` 和 `benchmark/summarize.py` 是两套训练共用的内部工具，
不需要直接执行。

## 训练配置

日常只编辑 `settings.yaml`：

```yaml
batch_size: 1                    # 每卡 micro batch
gradient_accumulation_steps: 4
num_epochs: 4
learning_rate: 1.0e-5
freeze_video_dit: false
init_checkpoint: checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt
```

当前 effective batch 是 `1 x 4 GPU x 累积 4 = 16`。相机布局、动作维度、history、
数据路径、stats 和文本缓存都在同一个 YAML 里。DeepSpeed 细节只在
`ds_zero3_hybrid.json`，一般不要修改。

四张 40GB A100 无法用官方 ZeRO-1 容纳全量 7.25B AdamW 状态，因此这里固定使用
已经压力测试通过的 hybrid ZeRO-3。pure-GPU 配置只有约 200MiB 余量，CPU 全
offload 又明显更慢，均已从正式目录移除。

## 运行

启动前先检查是否已有正式任务；若仍在运行，不要重复启动：

```bash
bash scripts/so101_pipeline/full_finetune_zero3/status_training.sh
```

未来新开训练时，先自检和短测：

```bash
PYTHONPATH="$PWD" python scripts/so101_pipeline/full_finetune_zero3/self_test.py
bash scripts/so101_pipeline/full_finetune_zero3/smoke_test.sh
```

再后台启动：

```bash
bash scripts/so101_pipeline/full_finetune_zero3/start_training.sh
```

输出位于 `runs/so101_g05_square_v1/full_zero3/formal-*/`。每个 epoch 保存一个约
14GiB、可直接部署的 `model/checkpoints/weights/step_*.pt`。默认不保存巨大的 Adam
状态，所以 checkpoint 是 weight-only 续训，不是精确恢复 optimizer。

## Batch 2 测试

当前正式任务结束后运行：

```bash
bash scripts/so101_pipeline/full_finetune_zero3/benchmark_batch2.sh
```

它在最吃显存的 `history_len=5` 上测试 `batch_size=2、gradient accumulation=2`，
effective batch 仍为 16。脚本检测到正式训练仍活跃时会拒绝启动。测试通过后，再在
下一次训练中同步修改 `settings.yaml` 的这两个值。
