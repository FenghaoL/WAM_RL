# cylinder-v1：关闭视频记忆（num_history_frames=0）的 SFT 与纠正训练

2026-09-10 按用户要求新增两项，顺序执行、使用 3 张 GPU（0/1/2），GPU3 留给部署：

1. `sft_nohist_pretrained_4ep`：用原 cylinder 任务 SFT 的配方重新训练，只把视频记忆关掉；
2. `p1_full_balanced_video_nohist`：在 (1) 的最终权重上跑与 `p1_full_balanced_video` 完全相同的 1:1 demo/HUMAN 全量纠正训练，同样不用记忆。

(2) 只有在 (1) 正常结束（退出码 0）后才会启动，并加载 (1) 的 `step_006572.pt`。

> **2026-09-11 更新**：按用户要求，(2) `p1_full_balanced_video_nohist` 已取消（配置移至 `runs/ovcr_adapter_cylinder_v1/configs_cancelled/`），SFT 继续运行；其后改为四个加入 09-11 纠正片段的实验，见 [04_NOHIST_CLIPS_ZH.md](04_NOHIST_CLIPS_ZH.md)。

## 1. "记忆步=0" 在代码里到底改了什么

AHA-WAM 的记忆就是 `model.num_history_frames`（原配置为 6）：当前 clip 之前的若干历史帧先编码成 video KV 前缀，再让当前 clip 注意到它们。设为 0 后：

- 数据侧：`build_datasets` 按 `model.num_history_frames` 配置所有数据集，样本里不再生成 `video_history`、`video_history_valid_len`、`video_history_frame_indices`（已用真实数据确认）；
- 模型侧：训练（`training_loss`）与推理（prefill）都经过 `num_history_frames > 0` 的同一分支判断，历史编码和 KV 前缀整段跳过；当前 clip 的时间位置也不再使用数据给出的绝对 `video_temporal_position_ids`，而是退回模型的默认相对位置。这是 AHA 支持的无记忆模式，训练和推理一致；
- 采样器：原 SFT 开启的 `history_aware_batching` 在全部历史长度为 0 时只有一个桶，等价于普通打乱，不影响训练，所以保留原设置不改。

**部署时必须同步改成 `num_history_frames: 0`。** 现有 `online_deploy/settings*.yaml` 都是 6；如果不改，服务端会给一个从没见过历史输入的模型喂 6 帧历史。

## 2. 配置与原实验的逐项差异

`prepare_nohist.py` 直接复制原 SFT 运行时解析后的 `config.yaml`，并逐键校验模型部分只允许 `num_history_frames` 变化（`lambda_video=1.0` 是原配置依赖的同一个默认值，这里显式写出）。实测差异：

| 键 | 原 SFT | 本轮 SFT |
|---|---|---|
| `model.num_history_frames` | 6 | **0** |
| `eval_num_val_batches` | 8 | 16（与 P1 协议一致，便于直接比较数值） |
| 验证集 | 仅 demo val | demo val + 同一组 HUMAN val（只评估，不训练） |
| wandb | project `ahawam-so101-cylinder-v1` | project `ahawam-so101-cylinder-hil-v1`，group `cylinder-v1-nohist` |

其余全部相同：AHA 预训练初始化 + 14→6 shape 适配、同样 4 个 demo session（19,708 帧，训练中不含任何 HUMAN 数据）、4 epoch × 1643 = 6572 步、每卡 batch 4 × 3 卡、lr 1e-5、cosine 5% warmup、AdamW wd 0.01、grad clip 1、bf16、ZeRO2 + CPU offload、每 1643 步存一次、每 410 步验证、seed 42、num_workers 4。

`p1_full_balanced_video_nohist` 与 `p1_full_balanced_video` 的差异只有 `init_checkpoint`（换成本轮 SFT 的最终权重）和 `num_history_frames: 0`；数据切分、1:1 采样、3285 步、lr、视频监督规则（HUMAN 只在完整 49 帧窗口上算视频损失）、验证子集全部相同。

## 3. 训练入口的改动

原 SFT 用的 `full_finetune_zero2_offload/` 已被移到 `scripts/so101_pipeline/archive/`（archive 说明其内容不再维护、内部路径可能过期），OVCR 队列原先直接引用它，移动后已不能启动。本轮没有恢复旧目录，也没有修改 archive 里的文件，而是：

- 把 `zero2_trainer.py`（与 HEAD 逐字节一致）、`ds_zero2_offload.json` 和指向它的 `accelerate_zero2_offload.yaml` 放进 `ovcr_adapter/v2_experiments/`，`run.py` 与 `launch.sh` 改为引用这里；
- `run.py` 增加 `experiment.training_data: demo_only`，使同一套 runner 能跑纯 demo 的任务 SFT；训练器仍是原 SFT 用的 `Zero2WeightOnlyTrainer` + 全量 DiT 训练模式。

仍引用旧路径、目前无法直接启动、本轮未改动的脚本：`hil_human_only_training/run_common.sh`、`val_eval/run_all.sh`、`ovcr_adapter/train.py`。

## 4. 命令

```bash
# 生成两个配置（已执行；拒绝覆盖已存在的配置）
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/prepare_nohist.py

# 启动两步队列（tmux 会话 ovcr_cylinder_nohist）
bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_nohist_chain.sh

# 查看
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/ovcr_adapter_cylinder_v1
tmux attach -t ovcr_cylinder_nohist
```

启动脚本会先把上一轮（history-6 五项）的 `suite_status.json` 另存为 `suite_status_before_nohist_<时间>.json`。

## 5. 结果怎么读

- 同一 demo/HUMAN 固定子集上，把 `sft_nohist_pretrained_4ep` 的最终验证与原 SFT 的 P0 基线（demo action loss 0.1694 / L1 2.298，HUMAN 0.0795 / L1 1.556）对比，回答"去掉记忆后任务 SFT 本身变好还是变差"；
- 把 `p1_full_balanced_video_nohist` 相对其 step 0 的变化，与 `p1_full_balanced_video` 相对其 step 0 的变化对比，回答"纠正训练的收益是否依赖记忆"；
- 仍只比 `eval/*_loss_action` 和 `eval/*_action_l1`，不比总损失；
- 验证损失不能替代真机：记忆主要影响闭环里的时序一致性，这恰恰是离线 teacher-forcing 验证最看不出来的部分，最终要做同起始位姿的配对真机测试。

## 6. 启动前验证与启动记录（2026-09-10）

| 检查 | 结果 |
|---|---|
| 配置逐键对比 | SFT 模型部分只有 `num_history_frames` 6→0（加显式的默认 `lambda_video=1`）；p1 只有 init 与 history 两处不同 |
| 数据 | SFT `DATA_LENGTHS [19708, 0, 661, 674]`，p1 `[19708, 2393, 661, 674]`，与冻结 manifest 一致；样本中已无任何 `video_history*` 字段 |
| SFT 三卡两步 | 退出码 0；全量 7,349,379,396 参数；预训练 14→6 适配；训练与验证 `history_len=0`；grad norm 9.47 / 11.54；成功保存权重 |
| p1 三卡两步（替身初始化） | 真正的初始化要等 SFT 训完才有，所以用原 history-6 SFT 权重作替身跑工程检查：严格加载进无记忆模型成功，`history_len=0`，grad norm 1.42 / 1.54，成功保存；替身 smoke 不进入任何比较 |
| 最终权重是否会保存 | trainer 先按 `save_every` 保存再检查 `max_steps`，6572 是 1643 的整数倍，所以 `save_at_end: false` 只是省掉重复的第二次保存，p1 依赖的 `step_006572.pt` 一定会写出 |
| 单元测试 | 12 项全部通过 |

队列于 20:04 在 tmux 会话 `ovcr_cylinder_nohist` 启动（GPU 0/1/2，GPU3 空闲）。SFT wandb run：[2qzbssc7](https://wandb.ai/fenghaoliu8-university-of-michigan/ahawam-so101-cylinder-hil-v1/runs/2qzbssc7)。

实测 6.59 s/step，比 history-6 的全量实验（约 7.0–7.6 s/step）快，因为不再编码历史帧。预估 SFT 约 12.3 小时（含 17 次验证），随后 p1 约 6.2 小时，整条队列约 18.5 小时。每卡约 31GB 显存；CPU optimizer offload 常驻内存，运行期间不要在本机跑其他吃内存的任务。

SFT 的 step 0 是未适配 SO101 动作头的预训练权重（demo action loss 1.705、HUMAN 1.464，48 样本/来源），只作为曲线起点，不与 0.1694 的 P0 基线比较；可比的是 SFT 训练结束时的数值。
