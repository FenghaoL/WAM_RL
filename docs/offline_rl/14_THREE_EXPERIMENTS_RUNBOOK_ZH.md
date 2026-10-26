# IQL、TD3+BC 与 IQL 加权 Action DiT 实验

日期：2026-10-25。运行根目录：`runs/offline_rl/cylinder103_experiments_20261025`。

## 本次安排

| 实验 | 初始化与训练目标 | GPU | 预算 |
| --- | --- | --- | --- |
| IQL＋外部 actor | 冻结采集 SFT 表征；独立训练小 actor、双 Q、V | 0 | 10,000 optimizer iterations，batch 64 |
| TD3+BC＋外部 actor | 相同初始化种子、划分和网络；独立从头训练 | 1 | 10,000 iterations，batch 64 |
| IQL＋Action DiT | 固定第一项最终 Q/V，导出数据动作的 advantage 权重；从采集 SFT 初始化原生 Action DiT 的 LoRA | 2 | 2,000 actor optimizer updates，microbatch 1，累积 8 |

前两项同时运行，均先做 200 次纯 BC 更新。之后 IQL 每步更新 V、Q、actor；TD3+BC 每步更新 Q、每两步更新 actor 和 target networks。两项不是相互接续的 checkpoint。第三项依赖**本轮第一项正常退出**，并依赖新版 Action DiT 的真实数据验证通过；不会使用以前的 500 步 pilot 作为正式微调的 critic。

这是首轮单种子实验预算，不代表经过超参数搜索或已经收敛。前两项约消费 40 个训练集等量的 minibatch 样本（有放回抽样）；Action DiT 消费 16,000 个窗口，约一个训练集等量，也有放回抽样，不保证每条数据被看见一次。

## 沿用的数据与模型合同

- 缓存已完成：103 episodes、43,680 个观察、81,594,083,608 bytes（约 75.99 GiB）。训练缓存来自 `cylinder103_skip_hold/cache`，不重新导出全量 KV。
- 执行训练仍为 94 episodes、16,554 个窗口；其中训练 16,016、验证 538。9 条提前结束且未录到完整成功的轨迹继续排除，未改成终止失败。
- 只去掉 SYSTEM_HOLD 动作，保留 POLICY＋HUMAN。自然完整 8 动作块可跨过成功；成功窗口不自举。奖励和 HOLD 前移规则不变。
- 冻结基座为采集这些数据的 SFT `step_006572.pt`。保持原归一化、因果 history、planner anchor 与当前观察 offset。Action DiT 从该 SFT 初始化，和外部小 actor 不共享可训练参数。

## 新代码在哪里

| 文件 | 职责 |
| --- | --- |
| [experiments.py](../../scripts/so101_pipeline/offline_rl/experiments.py) | 持久 supervisor：并行启动前两项、检查退出码、按依赖启动第三项；写 `queue.json` |
| [system.py](../../scripts/so101_pipeline/offline_rl/system.py) | 已有外部 actor 训练入口，本轮继续使用 |
| [action_dit.py](../../scripts/so101_pipeline/offline_rl/action_dit.py) | IQL 权重导出、原生单 8 动作 FM 训练、验证、显存、checkpoint、恢复 |
| [validate_action_dit.py](../../scripts/so101_pipeline/offline_rl/validate_action_dit.py) | CPU 回归与真实视频 GPU 验证，成功后生成队列放行凭据 |
| [flow_actor.py](../../src/ahawam/offline_rl/flow_actor.py) | FP32 LoRA、per-chunk 加权 FM、冻结条件重算、adapter 加载 |
| [test_offline_flow_actor.py](../../tests/test_offline_flow_actor.py) | 权重 detach、padding、loss 归约、LoRA 零初始化、冻结与恢复测试 |

没有把最后一层 KV 复制到全部 Action DiT 层。`FrozenConditionProvider` 从原始视频重新播放相同历史和 anchor，重算**所有层** post-OVCR KV；仅保留当前 episode 的视频 reader 与当前 anchor，不额外生成约 1.76 TiB 的全层磁盘缓存。Q/V 权重继续使用现有最后一层缓存，因此需要验证重算条件与现有缓存匹配。

原生 Action DiT 的可训练入口是 `flow_actor.predict_native_flow`：保留原服务的 `pre_dit`、branch embedding、attention mask、`forward_action_with_video_cache` 与 `post_dit`，去掉服务外层的 `no_grad` 包装。每次抽噪声级、预测 flow，保留 scheduler 时间权重后乘 detached IQL 权重。没有在训练时展开十次采样，也没有让 Q 的动作梯度穿过去噪链。[数学与论文依据见文档 12](12_IQL_WEIGHTED_FLOW_ACTION_DIT_ZH.md)。

精度对齐需要单独处理：OVCR 的输出可能是 FP32，但 exporter 保存 BF16，服务 attention 也把 KV 转到 action query 的 dtype。因此验证先按 exporter 的 dtype 量化再做精确比较；训练时按服务方式转换 KV，而不是把 FP32 editor 输出直接传进 BF16 attention。新增回归测试覆盖该情况。

本轮训练与验证对齐缓存的独立 anchor 协议：每个样本是当前位置的一个 8 动作 prior chunk，chunk index 重置为 0。不是给原 40 动作/5 chunks 的联合 loss 乘一个权重。adapter 能装入原模型，但原服务连续使用 chunk index 0–4 的完整异步流程尚未在本轮离线实验中验证；实际部署时仍需检查条件刷新与 chunk position 是否匹配，不能仅凭 adapter 能加载就声称部署等价。

LoRA 覆盖所有 Action DiT blocks 的 self-attention q/k/v/o，rank 16、alpha 16，初始化 B=0。模型主体 BF16，adapter 参数与 AdamW 状态 FP32；学习率 1e-5、weight decay 0、梯度裁剪 1。planner、OVCR、观察编码器、本体编码器、原始 Action DiT 权重都冻结。这是原模型的受控微调首试；不是全参数微调。`eval()` 关闭 dropout，但训练 forward 保留 autograd。

外部网络仍为 width 256、depth 2、共享前干双 Q、学习率 1e-4、expectile 0.7、beta 3、最大权重 20；TD3 alpha 0.1。详见 [参数说明](09_PARAMETERS_AND_RESOURCE_REPORT_ZH.md)、[TD3 损失说明](13_TD3_BC_TARGETS_AND_REFERENCE_ZH.md)。

## 日志、结果和检查点

根目录的 `queue.json` 是任务当前状态；`iql.log`、`td3_bc.log`、`iql_action_dit.log` 是各自进程日志。supervisor 自身日志位于其同级的 `cylinder103_experiments_20261025.queue.log`。进程独立于当前工具会话运行。

每个外部实验目录包含 `config.json`、`provenance.json`、`metrics.jsonl`、`last.pt`。每 1,000 步原子更新 `last.pt`，正常结束后写 `profile.json`。每 100 步的验证诊断使用固定前 256 个验证窗口，未用于选 checkpoint；它不是完整 538 窗口评估，也不是实机成功率。

Action DiT 目录会包含：

- `weights.pt`：每个 transition 的 target-Q minimum、V、advantage、权重及窗口键、数据/cache/critic 哈希；约 MB 量级。
- `weight_report.json`：训练/验证分别的权重分位数、ESS、按 episode 统计；验证数据没有参与 critic 拟合或权重归一化。
- `equivalence.json`：重算 KV 与原缓存、可反传 flow 与原服务 flow 的一致性检查。
- `baseline_validation.json`：LoRA 零初始化时固定噪声的验证 FM 损失。
- `metrics.jsonl`：训练加权/未加权 FM、权重均值、梯度范数、耗时、显存；每 100 步再做固定 16 个验证窗口的 FM 检查。
- `step_000000.pt`、`last.pt`、每 100 步 `step_*.pt`：仅保存 adapter、优化器、RNG 和 provenance，无需重复存基座。正常结束后写 `profile.json`。

第三项保存的 adapter 不是完整 AHA-WAM 权重文件。先加载匹配的原 SFT，再调用 `flow_actor.load_action_lora(model, adapter_path, actual_base_sha256)` 即可装载；基座哈希或参数结构不一致会报错。本轮没有启动机器人服务或修改机器人执行配置。

## 运行、观察与恢复

当前队列已经启动，**不要重复执行启动命令**。查看日志可用：

```bash
cat runs/offline_rl/cylinder103_experiments_20261025/queue.json
tail -n 2 runs/offline_rl/cylinder103_experiments_20261025/iql/metrics.jsonl
tail -n 2 runs/offline_rl/cylinder103_experiments_20261025/td3_bc/metrics.jsonl
tail -n 20 runs/offline_rl/cylinder103_experiments_20261025/iql_action_dit.log
```

复现实验需使用新输出目录：

```bash
PYTHONPATH=src:. /home/fenghao/miniconda3/envs/ahawam/bin/python \
  scripts/so101_pipeline/offline_rl/experiments.py \
  --transitions runs/offline_rl/cylinder103_skip_hold/data/transitions.pt \
  --cache runs/offline_rl/cylinder103_skip_hold/cache \
  --output runs/offline_rl/REPLACE_WITH_NEW_OUTPUT --gpus 0 1 2
```

supervisor 必须看到输出目录的 `action_dit_validated.json` 才启动第三项，该文件记录真实验证结果与训练代码 SHA256；验证后代码改变会阻止启动。复现时应重做验证并写对应凭据，不应无检查复制放行文件。

放行验证的可执行入口如下，`--smoke-output` 必须是新目录；它先运行 CPU 测试，再在指定卡上用已有 pilot 验证三次真实 optimizer update（microbatch 1、累积 8），最后才原子写入放行文件：

```bash
PYTHONPATH=src:. /home/fenghao/miniconda3/envs/ahawam/bin/python \
  scripts/so101_pipeline/offline_rl/validate_action_dit.py \
  --queue-output runs/offline_rl/REPLACE_WITH_QUEUE_OUTPUT \
  --smoke-output runs/offline_rl/REPLACE_WITH_NEW_SMOKE_OUTPUT \
  --pilot-checkpoint runs/offline_rl/cylinder103_skip_hold/pilot_iql/last.pt \
  --transitions runs/offline_rl/cylinder103_skip_hold/data/transitions.pt \
  --cache runs/offline_rl/cylinder103_skip_hold/cache --gpu 3
```

Action DiT 的 `--resume .../last.pt --steps 2000` 表示恢复后继续到总计 2,000 步，数据、权重、LoRA、累积批量和代码 provenance 必须一致。外部 `system.py --resume` 的 `--steps` 则表示**额外**更新次数。恢复前须确认原任务已经退出，避免两个进程同时写同一目录。

`action_dit.py --uniform-weights` 可做同架构普通 FM-BC 消融，但本次没有偷偷加第四项任务。仅凭 weighted FM 或 Bellman loss 下降，无法断言插入成功率提高；固定验证、权重 ESS、原 SFT 对照及后续实机评估需要共同判断。

## 验证与资源实测

本轮先运行原有 21 项测试与新增 7 项测试，再用真实视频、KV、SFT 和 pilot critic 验证 Action DiT 前向、反向、冻结范围与 checkpoint。真实验证结果、参数量、显存和速度见运行目录的 `profile.json`、`smoke_passed.json` 与本节后续记录。pilot 仅用于验证程序，不参与正式第三项的权重。开发过程中较早的 smoke/validation 日志保留用于诊断；正式放行凭据指向通过验证的那一次运行。

最终通过的真实验证目录为 `runs/offline_rl/cylinder103_action_dit_validated_v4_20261025`，放行文件已写入队列根目录。检查包括 episode 起点，以及另一个具备 6 帧 history、offset 非零的观察：两个样本的 BF16 K/V 与原缓存逐值相同，两条 flow forward 逐值相同（最大误差均为 0）。连续 3 次真实更新后，240 个 adapter 张量均有变化；冻结参数没有梯度，checkpoint 恢复逐值相同。程序仍会在正式第三项开始时重复条件与 flow 一致性检查。

| 实测项目 | 结果 | 解读 |
| --- | --- | --- |
| 外部 IQL，batch 64 | PyTorch allocated 峰值 1,244 MiB，reserved 1,288 MiB；近期约 0.20–0.23 秒/更新 | 约 35–40 分钟/10,000 步，受 cache I/O 与并发影响 |
| 外部 TD3+BC，batch 64 | allocated 峰值 1,201 MiB，reserved 1,268 MiB；近期约 0.18–0.21 秒/更新 | 约 30–35 分钟/10,000 步 |
| 原生 Action DiT LoRA | 7,864,320 个可训练参数；allocated 峰值 18,514.5 MiB（18.08 GiB），reserved 19,052 MiB（18.61 GiB） | 单张 A100 40 GB 足够本配置；nvidia-smi 会额外计入 CUDA context |
| 原生单卡，microbatch 1×累积 8 | 三次更新分别约 10.62、10.64、10.64 秒；平均 10.63 秒 | 2,000 步纯更新约 5.9 小时，加加载、权重导出、验证和保存，先估 6–7 小时；短测不能覆盖全部 episode 的耗时变化 |
| 本次原生模型加载 | 约 233 秒 | 复用缓存时已保存的基座 SHA256，按 path/size/mtime 校验后避免再读一遍 14 GB 做 hash |

这说明外部小 actor 的主要成本已是 cache 读取；原生 Action DiT 路线仍需每样本重算冻结 planner/VAE/OVCR，并经过整个动作网络反传，计算量明显更大。当前最后一层 cache 不能消除第三项的这些计算。

两个正式外部实验的 6,000 步 checkpoint 已另外在 CPU 上重新加载，检查全部参数有限，并在真实缓存输入上得到有限的 `[1,8,6]` 动作；结果保存在队列根目录的 `checkpoint_audit.json`。这些属于运行链路验证，不是策略效果结论。
