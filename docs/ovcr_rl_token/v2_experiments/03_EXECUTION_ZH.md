# 执行与验证

## 当前已完成的验证

- 实际读取四个集合，长度分别为 15,984 / 4,437 / 1,735 / 1,061；视频、动作、历史、文本缓存可读取。
- `test_ovcr_adapter.py` 和 `test_ovcr_v2_sampling.py` 共 11 项测试通过，覆盖零初始化恒等、native action 路径梯度、anchor、checkpoint，以及三卡来源比例、循环覆盖、确定性和采样位置恢复。
- GPU 0/1/2 上 V32 两步真实 ZeRO2/bf16 训练退出码0，grad norm 为 0.0114 / 0.0230；成功保存 `p1_v32_balanced_smoke/model/checkpoints/weights/step_000002.pt`。
- 检查保存权重：60 个 adapter tensor 均有限，10/10 层 up weight 非零，checkpoint 含 adapter 配置。真实两步未测 Q/T；其梯度路径已通过小模型测试。
- 逐张量核对 P0 与 smoke checkpoint：1,976 个冻结 tensor 在统一到运行 dtype 后完全相等，只有新增 adapter 更新；记录在 `frozen_weight_check.json`。
- smoke 的固定3样本/来源验证：demo action loss 0.428751→0.428676；HUMAN 0.144413→0.143402。**这些数字只证明训练和评估链路可运行，不是有效性结论。**

正式队列按 V32 → Q32 → T4 执行，每项2664步，一项失败即停止。动态状态以 `runs/ovcr_adapter_padapter_v2/suite_status.json` 和各 run 的日志为准。正式运行尚未完成；尚无真机成功率结果。

## 脚本入口

以下从仓库根目录执行，先激活 `ahawam` 环境并令 `PYTHONPATH=src:.`。

```bash
# 生成数据清单和七组配置；suite 运行时禁止重新生成
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/v2_experiments/prepare.py

# 只检查实际数据（不需要 GPU）
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/v2_experiments/run.py \
  --config runs/ovcr_adapter_padapter_v2/configs/p1_v32_balanced.yaml --data-only

# 查看当前队列与最新验证值
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py

# 当前已启动此队列，勿重复启动
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/suite.py

# P1 完成后，运行 P2 单因素对照
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/suite.py --names \
  p2_v32_natural p2_v32_3to1 p2_v32_noanchor

# 低优先级：HUMAN-only 压力测试
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/suite.py --names p3_v32_human_only
```

`launch.sh` 会激活 conda、加载服务器已有的进程级 NVIDIA 库兼容设置，要求恰好三张 GPU 且每张至少36GB空闲，默认 `GPU_IDS=0,1,2`；W&B 关闭。不会修改系统驱动或占用第四张 GPU。

queue 有进程锁、PID、状态文件、每项独立日志，拒绝覆盖已有 output。停止队列可向 `suite.pid` 中的 PID 发送 SIGTERM；它只终止自己启动的训练进程组。权重按888步保存，不包含 optimizer state，因此停止后不能声称精确恢复 optimizer；需要精确恢复时应另开启 trainer state 保存，并记录新的配置。

单独测试其他模式可用 `launch.sh --config ... --smoke-steps 2`，输出到独立 `_smoke` 目录。smoke 缩短日程、关闭动作推理，仅用于工程检查；其 checkpoint 不进入正式比较。

## 扩大离线验证

等训练队列释放 GPU 后执行以下命令。新输出目录不得已有验证文件；此命令只评估指定 checkpoint，不执行更新。

```bash
ENTRYPOINT=scripts/so101_pipeline/ovcr_adapter/v2_experiments/evaluate.py \
  scripts/so101_pipeline/ovcr_adapter/v2_experiments/launch.sh \
  --config runs/ovcr_adapter_padapter_v2/configs/p1_v32_balanced.yaml \
  --checkpoint runs/ovcr_adapter_padapter_v2/p1_v32_balanced/model/checkpoints/weights/step_002664.pt \
  --output-dir runs/ovcr_adapter_padapter_v2/evaluations/v32_step2664 \
  --loss-samples-per-rank 128 --action-samples-per-rank 16
```

该扩展入口已编写，尚未在正式 checkpoint 上运行。共384个固定 loss 样本/来源、48个动作推理样本/来源，默认10步 diffusion/flow 推理，继承 P0 参数。它仍然不能替代真机评估。实际来源写入 `evaluation_source.json`；evaluation 记录中的 step0 是新评估进程计数，不是 checkpoint 的训练步数。

## 结果位置与阅读顺序

- `data_manifest.json`：数据边界、划分、数量和来源校验。
- `configs/*.yaml`：每个实验的完整配置；`model/experiment_config.yaml` 是启动时快照。
- `logs/<experiment>.log`：逐步 loss、anchor、梯度、耗时和异常。
- `<experiment>/model/validation.jsonl`：demo/HUMAN 分别的固定验证；损失不含 anchor 正则，避免 λ 变化污染评估定义。
- `<experiment>/model/checkpoints/weights/step_*.pt`：可部署的完整 AHA 权重，含 OVCR adapter 元数据；单个约14GB，并非仅几MB的 adapter 文件。

先看是否正常收敛，再看 demo 是否退化和 HUMAN 是否改善，最后看真机。若训练后 HUMAN loss 降低但成功率变差，应检查片段边界、offset、推理历史与采集控制节奏，而不能仅靠增大训练步数。

旧版 online deployment 配置保留用户当前选择；训练队列不会自动切换到未评估 checkpoint。完整模型加载已支持 adapter 元数据，但本轮未进行机器人远程部署或运动测试。

## 重复随机种子

初筛后可用以下命令生成独立配置；它不修改正在运行的配置，也不重新划分数据。然后将新配置名传给 `suite.py --names`。

```bash
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/clone_seed.py \
  --config runs/ovcr_adapter_padapter_v2/configs/p1_v32_balanced.yaml --seeds 142 242
```

## 本次启动记录（2026-09-05）

已确认正式 V32 完成 step0 的双来源验证与动作推理，并进入2664步训练，首步 loss=0.0391、grad_norm=0.0114。GPU 0/1/2 各占约18.8GB，GPU3原有进程未变动。Q32、T4 等待前一项成功结束后由队列自动启动。正式结果需继续查看动态日志，不能由 smoke 数字推断。
