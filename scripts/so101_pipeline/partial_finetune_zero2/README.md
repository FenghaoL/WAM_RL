# SO101 ZeRO-2 部分微调

这套训练冻结约 5B 的 VideoDiT，训练 ActionDiT、MoT/交互模块和 SO101 6D
输入输出层。它适合先验证动作学习，显存和时间都明显少于全量训练。

## 文件

| 文件 | 作用 |
| --- | --- |
| `settings.yaml` | 数据路径、batch、学习率、epoch、冻结策略和初始化 checkpoint |
| `prepare_data.sh` | 一次完成格式转换、校验、stats 和文本 embedding |
| `convert_v3_to_v21.py` | 将当前 LeRobot v3 数据转换为本 repo 可读的 v2.1 |
| `validate_dataset.py` | 检查转换后的 episode、图像、state 和 action |
| `make_stats.py` | 计算动作和 state 的归一化统计量 |
| `train_entry.py` | 计算每个 epoch 的准确 step 数并调用 repo Trainer |
| `smoke_test.sh` | 8-step 三卡短测，不写 W&B 正式 run |
| `train_4epochs.sh` | 前台正式训练 |
| `start_training.sh` | 后台启动正式训练 |
| `status_training.sh` | 查看进度 |
| `stop_training.sh` | 停止后台任务 |

## 训练配置

唯一配置文件是 `settings.yaml`。当前关键值：

```yaml
batch_size: 1                    # 每张卡的 micro batch
gradient_accumulation_steps: 4
num_epochs: 4
learning_rate: 5.0e-5
freeze_video_dit: true
init_checkpoint: checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt
```

数据路径、stats 和文本缓存也都在同一文件的 `data.train` 下。训练脚本默认使用 GPU
`0,1,2`；需要调整时修改 `train_4epochs.sh` 和 `smoke_test.sh` 开头的
`NUM_GPUS/GPU_IDS`。

## 运行顺序

首次使用新数据：

```bash
bash scripts/so101_pipeline/partial_finetune_zero2/prepare_data.sh
```

先短测：

```bash
bash scripts/so101_pipeline/partial_finetune_zero2/smoke_test.sh
```

再后台正式训练：

```bash
bash scripts/so101_pipeline/partial_finetune_zero2/start_training.sh
bash scripts/so101_pipeline/partial_finetune_zero2/status_training.sh
```

每个 epoch 保存一次 checkpoint，输出位于
`runs/so101_g05_square_v1/formal-3gpu-zero2-4ep-*/model/checkpoints/`。完成后将所需
`step_*.pt` 写入 `../online_deploy/settings.yaml`。
