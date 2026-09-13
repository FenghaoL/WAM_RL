# cylinder-v1：启动前验证、队列状态与限制

## 1. 启动前已经真实执行的检查

| 检查 | 结果 |
|---|---|
| HIL 数据 schema | 5 个 session 的 parquet 列与 power-adapter v2 完全一致（rich-v3：`control_source`/`intervention`/`human_action_valid`/`action_supervision_mask` 等 35 列），HUMAN 行动作全部有限 |
| 数据规模复核 | `DATA_LENGTHS [19708, 2393, 661, 674]`，与冻结 manifest 完全一致；四个数据集的样本都能解码出 13 帧视频、6 维动作、历史与文本缓存 |
| 任务文本 | demo 与 HIL 的 `tasks.jsonl` 都是 `Pick up the cylinder and insert it into the matching hole.`，命中已有 T5 缓存 |
| V32 三卡两步真实训练 | 退出码 0，`TRAINABLE_PARAMETERS 2058560`，grad_norm 0.0044 / 0.0116，保存 14.7GB 完整权重 |
| 冻结检查 | 逐张量对比 smoke 权重与 cylinder P0：**1,976 个原模型张量完全相同、0 个被改动**；新增 60 个 adapter 张量，两步后全部非零（零初始化的 up 层确实收到梯度）。记录在 `runs/ovcr_adapter_cylinder_v1/frozen_weight_check.json` |
| wandb 在线 | 使用 `~/.netrc` 凭据成功登录并同步 run（project `ahawam-so101-cylinder-hil-v1`），smoke 以 `<name>_smoke` 命名，不占用正式 run 名 |
| B 配置两步真实训练 | 全量参数 7,349,379,396；预训练加载按原任务 SFT 方式做 14→6 适配（`proprio_encoder: adapted weight input dim 14 -> 6 by adaptive average pooling`） |
| 单元测试 | `python -m unittest discover -s tests`：12 项全部通过（adapter 零初始化恒等、真实小模型反传、anchor、checkpoint 往返、采样比例与覆盖） |

step 0 的 P0 基线验证（V32 smoke，3 样本/来源，仅用于确认链路）：demo action loss 0.1898、HUMAN 0.0117。**样本太少，不能作为效果结论**；正式运行用 48 样本/来源。

## 2. 资源占用实测

- adapter 实验：每卡约 18.6GB 显存。
- 全量实验：每卡约 30.7GB 显存，且 ZeRO2 CPU offload 把 optimizer state 放在内存，实测占用约 443GB（本机 503GB）。**因此全量实验期间不要在同一台机器上跑别的吃内存任务，队列也必须串行。**
- 磁盘：每个 checkpoint 约 14.7GB。五个实验共 16 个 checkpoint，约 235GB；启动时 `/home` 剩余 2.1T。

## 3. 时间预估

按 padapter 轮实测速度（adapter 约 3.4 s/step，全量约 7.0–7.6 s/step）加上本轮更密的验证：

| 实验 | 步数 | 预估 |
|---|---:|---|
| `p1_v32_balanced` | 3285 | 约 3.5 小时 |
| `p1_q32_balanced` | 3285 | 约 3.5 小时 |
| `p1_t4_balanced` | 3285 | 约 3.5 小时 |
| `p1_full_balanced_video` | 3285 | 约 6.7 小时 |
| `b_full_pretrained_natural_4ep` | 7368 | 约 16 小时 |

整个队列约 33 小时，串行执行。这是**估计值**，实际以日志中的 `speed=` 和 `eta=` 为准。

## 4. 队列状态

见本文件末尾的"启动记录"和实时文件：

- 队列：`runs/ovcr_adapter_cylinder_v1/suite_status.json`
- 单项日志：`runs/ovcr_adapter_cylinder_v1/logs/<experiment>.log`
- 验证：`runs/ovcr_adapter_cylinder_v1/<experiment>/model/validation.jsonl`
- wandb：project `ahawam-so101-cylinder-hil-v1`，group `cylinder-v1-ovcr-correction`

```bash
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/ovcr_adapter_cylinder_v1
tmux attach -t ovcr_cylinder_v1        # 队列输出；Ctrl-b d 退出
kill -TERM "$(cat runs/ovcr_adapter_cylinder_v1/suite.pid)"   # 停止队列
```

## 5. 现在还不能说的结论

- 没有任何真机结果。验证损失下降 ≠ 闭环成功率提高。
- 五个实验之间预算不同（P1 是 3285 步 1:1，B 是 7368 步自然混合），**不是等算力对照**；只能分别对比它们各自的 step 0 基线和 demo/HUMAN 两条曲线。
- 1:1 采样让 HUMAN 起点重复约 8.2 遍，本轮尤其要看 HUMAN 验证损失是否先降后升，以及 demo 验证是否退化；如果出现，应选更早的 checkpoint，而不是继续加步数。
- 本轮没有 critic、reward 或在线 rollout，属于纠正模仿学习，不是 RL。

## 6. 启动记录（2026-09-08 03:59 EDT）

队列已在 tmux 会话 `ovcr_cylinder_v1` 中启动，使用 GPU 0/1/2，GPU3 保持空闲以便随时部署测试。顺序为

```
p1_v32_balanced → p1_q32_balanced → p1_t4_balanced → p1_full_balanced_video → b_full_pretrained_natural_4ep
```

一项失败即整队停止；每项完成后自动启动下一项。要改顺序或只跑其中几项，用 `NAMES="..." bash .../launch_suite.sh`（需先等当前队列结束或停止它）。

`p1_v32_balanced` 已进入正式训练：wandb run [fm4lgvhm](https://wandb.ai/fenghaoliu8-university-of-michigan/ahawam-so101-cylinder-hil-v1/runs/fm4lgvhm)，实测约 3.6 s/step，每卡约 19GB 显存。

训练前 step 0 的 P0 基线（48 loss 样本 + 12 动作推理样本/来源，本轮所有实验的共同参照点）：

| 来源 | action loss | action L1 | action L2 |
|---|---:|---:|---:|
| demo（`20260904_024009_val`） | 0.1694 | 2.2979 | 17.9992 |
| HUMAN（held-out 13 条 rollout） | 0.0795 | 1.5563 | 10.7795 |

`b_full_pretrained_natural_4ep` 从预训练出发，其 step 0 基线必然远高（smoke 实测 demo action loss 1.95、HUMAN 1.44），这是未适配 SO101 动作头的预期表现，不能与上表并列比较。

smoke 产物保留在 `p1_v32_balanced_smoke/` 和 `b_full_pretrained_natural_4ep_smoke/`，只用于工程验证，不进入正式比较。
