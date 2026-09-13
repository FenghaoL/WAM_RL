# Demo + HUMAN 全量微调对照（2026-09-06）

根据真机反馈，V32/Q32/T4 效果一般，因此增加直接更新原模型的对照。这里的“全量”沿用原任务 SFT 的定义：action DiT、video DiT、原 OVCR、观测 query 编码器和 proprio/visual 投影全部可训练；VAE 保持冻结，文本使用预计算缓存。没有新增 adapter，也没有 P0 anchor。

此项可用于判断小模块是否限制了纠正学习，但不能视为当前数据的性能上限：优化器、预算、超参数、采集质量和控制时序都会限制结果。

## 初始化与数据

- 从原始 v2 任务 SFT `step_005328.pt` 出发，不从 V/Q/T adapter checkpoint 出发，也不从随机模型开始。
- 仅用 v2 原 demo train 15,984 帧和 HUMAN train 4,437 帧。
- 保留相同独立 demo val 1,735 帧、HUMAN val 1,061 帧及原 manifest 的 episode 划分，未把验证数据加入训练。
- 保留 P0 的 normalization statistics、两相机布局、40步动作 horizon、8步 chunk、0–7 action offset、6帧 history。
- demo:HUMAN=1:1，每个来源循环内打乱且不放回；每张 GPU batch4，全局 batch12。

## Video loss 的实际判定

`loss = action_flow_loss + video_flow_loss`，两项系数均1；action 仍按原 `action_is_pad` 排除片段边界以后的动作。

HUMAN video loss 同时要求：

1. 连续 HUMAN 片段长度至少49帧；
2. **当前采样起点**对应的13张视频采样帧（原始时间0,4,…,48）全部有效，`image_is_pad` 没有 True。

因此长片段尾部不足49帧的窗口也只计算 action loss。旧脚本只有“片段总长≥49”条件，加上通用视频 padding mask；通用 mask 对一个 latent 内部分 padding 仍可能保留监督。为了避免把补齐出来的尾部当作视频目标，本次增加了可选 `video_supervision_require_full_window: true`，只对本实验的 HUMAN dataset 开启，不改变旧 adapter 实验的配置。

HUMAN 训练集有31段≥49帧，共**538个完整视频窗口起点**；其余3,899个起点仅监督动作。HUMAN 验证集有6个长段，共79个完整视频窗口起点。这里的538是窗口数，不是独立 episode 数。

demo 继续使用原始视频监督和 padding 规则。HUMAN 的观测历史、动作和视频都仍被限制在连续接管边界内，不引入自主运行帧作为标签。video loss 在每个本地 batch 内按满足条件的样本数归一化，全部不满足时得到可微分的零；该归一化方式沿用原实现。

“只计算 action loss”不等于冻结 video DiT：原模型中动作依赖 video/OVCR 特征，因此 action 梯度仍可能更新 video DiT。被关掉的是该窗口的视频重建监督。

## 训练安排

| 参数 | 本次设置 |
|---|---|
| 配置名 | `p1_full_balanced_video` |
| GPU | 0、1、2；不占用 GPU3 |
| 模式 | bf16，DeepSpeed ZeRO2，CPU optimizer offload |
| batch / accumulation | 每卡4 / 1，全局12 |
| 学习率 | 1e-5，沿用原 full SFT 量级 |
| 优化器 | AdamW，betas=(0.9,0.95)，weight_decay=0.01 |
| 日程 | cosine，5% warmup（133步），grad clip1 |
| 步数 | 2664；恰好一遍 demo 起点、约3.60遍 HUMAN 起点 |
| checkpoint | 888、1776、2664步，完整模型权重 |
| 验证 | step0、每444步及训练结束；demo/HUMAN 分开 |

先做与 adapter 相同步数的 full finetuning 对照，保留早期 checkpoint，以免只看最终模型漏掉过拟合前的较好状态。相同步数不等于相同 GPU 小时；本项还启用了 video loss，因此与旧 adapter 的差异包含参数范围和训练目标两方面，不能将差异完全归因为参数量。若结果值得继续，再单独做 full action-only 或延长预算。

验证日志中的总 loss 现在包含 video，**不能与 adapter 的总 loss 直接比较**。比较两组时看 `demo_loss_action`、`human_loss_action`、动作推理误差及同条件真机表现。当前固定小子集仅作趋势检查；HUMAN 视频合格窗口稀少，小子集可能没有合格窗口，出现 human_loss_video=0 并不意味着训练未启用视频监督。

## 复用代码与入口

复用 `Zero2WeightOnlyTrainer`、`Wan22Trainer._apply_dit_only_train_mode`、原 HUMAN dataset 和视频 loss reduction。扩展已有 v2 runner 的 `FullTrainer` 分支，以保留原有 manifest 校验和双验证集。

没有直接调用旧 `train_entry.py`：它会根据旧的计数逻辑重新推导 max_steps，而且旧配置指向 plug/v1 数据，不能直接用来代表本轮固定 split。模型、优化器、ZeRO2 和保存机制仍复用旧实现。

```bash
# 此配置已生成；生成器拒绝覆盖，复现到新工作目录时使用
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/v2_experiments/prepare_full.py

# 真实 HUMAN 窗口边界检查
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/v2_experiments/check_full_data.py

# 两步三卡检查，结果写入独立 _smoke 目录
scripts/so101_pipeline/ovcr_adapter/v2_experiments/launch.sh \
  --config runs/ovcr_adapter_padapter_v2/configs/p1_full_balanced_video.yaml --smoke-steps 2

# 正式训练；不要与正在运行的同名任务重复启动
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/suite.py --names p1_full_balanced_video
```

配置：`runs/ovcr_adapter_padapter_v2/configs/p1_full_balanced_video.yaml`。
正式日志：`runs/ovcr_adapter_padapter_v2/logs/p1_full_balanced_video.log`。
输出：`runs/ovcr_adapter_padapter_v2/p1_full_balanced_video/model/`。
最终权重：`checkpoints/weights/step_002664.pt`。

正式权重可通过现有 action server 加载，使用 power-adapter-v2 stats 和对应任务指令；无需 adapter 配置。权重保存不含 optimizer state，因此不能把 weight-only restart 称作精确恢复训练。

## 本次验证与启动

两步真实三卡训练通过并保存完整权重，训练参数数目为7,349,379,396。首步 `loss_action=0.0392`、`loss_video=0.1519`、`video_supervised_fraction=0.6667`、`grad_norm=1.1904`；第二步视频监督比例0.5833，梯度有限。真实边界读取检查确认剩余49帧可监督、48帧不可监督，短片段和尾部保留动作样本。

另发现并修复验证转换 `_to_batched_eval_sample` 丢失 `action_is_pad`、`image_is_pad`、`proprio_is_pad`、`video_supervision_mask` 和 `data_source` 的问题。训练 DataLoader 原本保留这些字段，训练不受此问题影响。新增回归测试通过。正式训练使用修复后的验证；此前 adapter 和本次 smoke 的旧验证结果需要按修复后的统一流程重新评估，不能直接与新的验证数字比较。

正式任务使用独立 tmux 会话 `ovcr_v2_full`，最新状态见 `suite_status.json`；之前完成的 T4 状态归档为 `suite_status_before_full_finetune.json`。正式效果尚待训练和真机验证。
