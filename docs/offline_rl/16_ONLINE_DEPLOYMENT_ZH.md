# 三项离线 RL policy 的同步部署

日期：2026-10-25。复用 `scripts/so101_pipeline/online_deploy/run_action_server.sh`、现有 TCP server 和机器人客户端。

## 配置和启动

| 方案 | 部署 YAML | GPU | 端口 | RL checkpoint |
| --- | --- | ---: | ---: | --- |
| IQL 外部 actor | [cylinder_offline_iql.yaml](../../scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_iql.yaml) | 0 | 12130 | `cylinder103_experiments_20261025/iql/last.pt`，10,000步 |
| TD3+BC 外部 actor | [cylinder_offline_td3_bc.yaml](../../scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_td3_bc.yaml) | 1 | 12131 | 同轮 `td3_bc/last.pt`，10,000步 |
| IQL＋Action DiT | [cylinder_offline_iql_action_dit.yaml](../../scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_iql_action_dit.yaml) | 2 | 12132 | 同轮 `iql_action_dit/last.pt`，LoRA 2,000步 |

YAML 的 `checkpoint_path` 全部是采集用 SFT `step_006572.pt`，`offline_rl.checkpoint_path` 才是本次小 actor / LoRA。配置记录 sidecar SHA256、基座 SHA256 与预期训练步数，防止 `last.pt` 被其他实验覆盖后静默切换模型。

分别在三个终端运行：

```bash
bash scripts/so101_pipeline/online_deploy/run_action_server.sh scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_iql.yaml
bash scripts/so101_pipeline/online_deploy/run_action_server.sh scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_td3_bc.yaml
bash scripts/so101_pipeline/online_deploy/run_action_server.sh scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_iql_action_dit.yaml
```

也可沿用在脚本注释区选择 `DEPLOY_CONFIG` 的方法。传入第一个参数优先级最高，其次是 `AHAWAM_DEPLOY_CONFIG` 环境变量，最后是脚本中的选择。不传参数时按已有选择运行。

同一根目录内其他模型的 GPU/端口由各自 YAML 决定，不能仅根据文件名假定都在某张卡。新三项各占一张卡，可同时加载；原 SFT/DAgger 也要同开时，需自行分配剩余卡或停止占用冲突的服务。

## 实际生成动作的链路

- **外部 IQL / TD3+BC**：冻结采集 SFT → planner 与 OVCR → 原 Action DiT 生成 reference → 最后一层 post-OVCR KV 转 BF16＋当前归一化 state＋clamped reference → 本次外部 actor → 反归一化绝对目标。
- **IQL Action DiT**：采集 SFT 加载本次 LoRA，planner/OVCR 保持冻结 → 使用 LoRA 后的原 Action DiT 进行10步采样 → 反归一化绝对目标。没有额外小 actor 或 Q 候选动作选择。

外部服务只加载 actor 参数，不构建 Q/V/优化器，也不恢复训练 RNG。IQL 与 TD3+BC 的 critic 不参与在线执行动作的选择；它们的作用已经通过离线训练进入 actor。Action DiT 的 LoRA 保留 FP32 adapter、BF16 主体，与训练相同。

复用原相机拼接、state normalization、动作反归一化。输出仍为 `(8,6)` 的 G05/model 绝对关节目标，机器人端继续使用 `model_to_arm`，不能再把绝对目标当残差叠加或重复反归一化。修改的是模型如何生成这8步，不是 TCP 数据格式。

## 条件和 history：实际匹配与在线近似

离线缓存和 Action DiT 提取均使用独立 anchor 协议，且每次动作推理的 chunk index 为0。因此本实现每个请求都独立刷新 planner 条件，使用当前请求观察作为 anchor（offset=0），只执行原40动作布局中的第一个8动作 chunk。没有把原服务的 chunk index 1–4 当成已经用 RL 训练过的位置。

离线表征还使用最多6个历史观察、原始录制帧间隔48；现有机器人接口只在请求动作时发送一个当前观察，未发送逐控制 tick 图像、HOLD 时长或实际执行数量。所以无法从服务器凭空还原训练时的原始帧历史。本轮声明采用 **请求执行动作时钟**：

1. 第0个动作请求的虚拟命令位置为0，第n个请求为 `8*n`；假设上一个请求完整执行8个命令。
2. 取 `t−288, t−240, …, t−48` 对应的历史请求观察，存在才使用；再加入当前观察。
3. 每次从这些历史观察开始独立重放冻结 planner，并设置对应 RoPE `command_step//4`。不把此前已经积累过另一组上下文的 KV 冒充新 replay 的 KV。
4. 只在动作推理成功之后推进请求计数，内存只保留最近37个请求观察；`reset` 清空模型与请求历史。

在无缺 tick、无 HOLD、每请求正好跨8个原始帧的录制回放中，该过程可以与离线缓存逐点对照；真实机器人含推理等待、HOLD、人工接管时，**48个执行动作与48个原始控制 tick 不等价**。offset 也只覆盖训练偏移分布中的0，不是在线自动随机偏移。这些是当前接口的实际限制，不能宣称完整复现缓存的所有异步时序。

机器人新 episode、接管后恢复、动作块没有执行满或被丢弃时，客户端应 `reset` 后重建 history。原先 SFT/DAgger 服务通常每5个 chunk 刷新 planner，和本三项的刷新周期不同；跨旧服务比较的结果会包含调度差异。三个新增 RL 方案之间采用相同的单 chunk 调度。

`health` 返回 `offline_rl_mode`、checkpoint、训练步数、基座 SHA、history 时钟和 `last_conditioning`，可确认实际连接到哪个策略。此部署只支持同步 action server；未来视频诊断和 async/dual-GPU backend 不能绕开这个适配直接复用。

## 本轮代码

| 文件 | 改动 |
| --- | --- |
| [offline_rl_policy.py](../../deploy/server/offline_rl_policy.py) | 新增三方案适配；独立 replay、单 chunk、actor/LoRA 加载、normalizer 与基座核对、reset 和 metadata |
| [inference.py](../../src/ahawam/offline_rl/inference.py) | 外部 actor 改为仅加载 actor，避免服务携带 Q/V/优化器或恢复训练 RNG |
| [run_action_server.sh](../../scripts/so101_pipeline/online_deploy/run_action_server.sh) | 添加三项配置注释；接受 YAML 参数/环境变量；从 YAML 读取 policy module/class |
| [deployment_preflight.py](../../scripts/so101_pipeline/online_deploy/deployment_preflight.py) | 原完整 WAM 检查之外，校验小 checkpoint 的类型、算法、步数、SHA 与 normalization/history |
| [smoke_offline_rl.py](../../scripts/so101_pipeline/online_deploy/smoke_offline_rl.py) | 临时启动真实 TCP server，用已录制数据测37个请求、history、reset 和缓存对照，结束后关闭服务 |
| [test_offline_rl_deploy.py](../../tests/test_offline_rl_deploy.py) | history 边界、内存上限、失败请求不前进、reset、BF16 KV、坐标反归一化、GPU/端口配置 |

没有修改原 `AHAWAMPolicy` 或 planner/OVCR 计算，训练缓存的模型代码来源校验依然有效。原 YAML 未指定新 policy class 时，启动器仍选择原 AHAWAMPolicy。

## 验证

相关 CPU 测试共34项通过，三个 YAML 的 preflight 与 shell 语法检查通过。真实 TCP 测试输出在 `runs/offline_rl/deploy_smoke_20261025/{iql,td3_bc,iql_action_dit}`，包含 `server.log`、`progress.json`、成功后的 `passed.json`。本测试只使用录制图像与关节数据，不连接或控制机械臂。

三个真实 TCP 测试均已通过：37个连续录制观察请求，覆盖完整6帧 history；再 reset 后重放第0帧，动作可重现。health 确认模式、步数、offset=0 和 chunk index=0。外部 actor 在原始帧0、48、288与缓存直接推理对比，IQL 最大差 `7.63e-6` 度、TD3 最大差 `1.15e-5` 度。测试人为将第n个请求的输入选为录制第8n帧，前缀没有缺 tick，因此虚拟时钟和缓存的帧位置恰好一致；这证明该输入协议下计算链路匹配，不证明真实机械臂执行8动作后就会到达录制的第8n帧，也不消除真实 HOLD 时钟的差异。

| 实测项目 | IQL actor | TD3+BC actor | IQL Action DiT LoRA |
| --- | ---: | ---: | ---: |
| 启动并可连接（秒，含 preflight/加载） | 178.30 | 169.91 | 176.88 |
| 37个请求平均延迟（秒） | 0.815 | 0.791 | 0.948 |
| 第37个请求，完整6帧 history（秒） | 1.006 | 0.995 | 1.143 |
| 完整 history 请求 PyTorch allocated 峰值（MiB） | 18,353.84 | 18,353.84 | 18,367.10 |
| 同请求 reserved（MiB） | 18,732 | 18,732 | 18,754 |

三者可分别放在单张 A100 40 GB。GPU 0/1/2 同时测试成功，没有启动机械臂客户端；测试程序结束时已关闭自己的 TCP server/worker，释放显存和端口。当前提供的是可由上述命令启动的配置，未留下长期运行服务。

8动作按15Hz执行需要0.533秒，而本次同步请求平均/完整 history 延迟超过它，所以会有推理等待。独立重算 history 和生成 reference 是外部 actor 的主要成本，小 actor 本身不会消除这些计算。当前配置用于先比较已训练策略，不能称为稳定15Hz无等待部署。若后续要复用异步流水线，需另行明确动作消费计数、观察时间与 planner 条件同步，不能只更改 backend 标志。
