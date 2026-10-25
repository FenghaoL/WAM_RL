# 候选实现运行与验收

所有命令从 AHA-WAM 仓库根目录运行，不连接机械臂。输出默认示例在 `runs/offline_rl`；数据目录只读。正式 checkpoint、时序和接管目标仍需按主文档讨论确定。

## 1. 环境与代码

本轮使用 `/home/fenghao/miniconda3/envs/ahawam/bin/python`，PyTorch 2.7.1；新增核心只依赖已有 torch/numpy/pandas。无需安装 HALO 的 Python 3.11、Ray、RLinf 或另一套机器人环境。

| 文件 | 职责 |
| --- | --- |
| `src/ahawam/offline_rl/data.py` | 标注身份校验、tick/macro 转移、成功截断、多数据目录 |
| `src/ahawam/offline_rl/features.py` | 既有部署包装器上的 causal KV/reference 导出 |
| `src/ahawam/offline_rl/networks.py` | 小 actor、有序候选动作双 Q、V |
| `src/ahawam/offline_rl/learner.py` | IQL、TD3+BC、BC，target、optimizer、恢复 |
| `scripts/so101_pipeline/offline_rl/run.py` | prepare / export / readout / train |
| `tests/test_offline_rl.py` | 时序、奖励、表征与真实梯度更新测试 |

## 2. 先对同一批生成两个候选

```bash
export PYTHONPATH=src:.
PYTHON=/home/fenghao/miniconda3/envs/ahawam/bin/python
RL_RUN=data/ahawam_hil_raw/so101_ahawam_cylinder_hil_v2/20260911_183057

$PYTHON scripts/so101_pipeline/offline_rl/run.py prepare \
  --roots "$RL_RUN" --temporal-mode tick --chunk-size 8 --stride 2 \
  --output runs/offline_rl/cylinder_review/tick

$PYTHON scripts/so101_pipeline/offline_rl/run.py prepare \
  --roots "$RL_RUN" --temporal-mode macro --chunk-size 8 \
  --output runs/offline_rl/cylinder_review/macro

$PYTHON scripts/so101_pipeline/offline_rl/run.py prepare \
  --roots "$RL_RUN" --temporal-mode tick --chunk-size 1 --stride 1 \
  --output runs/offline_rl/cylinder_review/tick_c1
```

每个目录产生 `audit.json` 与 `transitions.pt`。后者包含 actual action/state、reward、discount、duration、terminal/truncated、control_source、action_valid，以及按 episode 的原始身份和标注 SHA。训练 state 不包含未来 success、source 序列、实际 duration 或 outcome。

默认 `--intervention-mode mixed`；对照用 `--intervention-mode censor` 输出到另一目录。不能覆盖后仍把两种实验视为同一数据合同。

其他采集批次标好后，可 `--roots 路径1 路径2 ...`；待标注成功 episode 记录为 pending。建议显式列目录以便复现。本轮没有读取其他批次做测试或训练。

## 3. GPU 上导出真实 WAM 特征

先选择一个确认的 DAgger 部署 YAML。不要把示例当作本轮已经决定的基座：

```bash
$PYTHON scripts/so101_pipeline/offline_rl/run.py export \
  --transitions runs/offline_rl/cylinder_review/macro/transitions.pt \
  --deploy-config scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_action_dit_only.yaml \
  --history-count 0 --num-tokens 64 --layers -1 --limit 2 \
  --output runs/offline_rl/cylinder_review/features_preview.pt
```

先用 `--limit 2` 检查 dtype、token shape、reference 和状态 normalization，再移除 limit 做完整 cache。partial cache 不能直接训练所有 transition，训练端会因缺 observation key 而拒绝。

默认 source：当前推理产生的 `_chunk_video_kv_cache`，选定层的 `v`，按原 token 顺序 adaptive 1D pool 到 64 个 token。选择多层时沿 feature 维拼接。它是当前 action branch 读取的 **post-OVCR 表征**，不是未来真实图像编码，也不是去噪过程中已经依赖候选动作的 hidden state。

这个 pooling 是便于先验证的基线，会模糊相机区域边界；view-aware 2D pooling 与腕部局部 token 读取仍需后续比较。直接整层/所有层 KV 会显著增加存储和训练输入规模，不建议一开始全保存。

当前 cache 保存在 CPU，direct-attention 训练会展开 current/next features；完整多批数据可能需要较大主存。优先 reader 路线可在训练小 actor 前压缩为单个 256D token。后续大规模 direct-attention 应改成按唯一 observation key 索引/分片流式读取，当前并未实现 mmap replay。

`history-count 0` 表示 reset-per-state 对照，不能代表原模型完整历史能力；H>0 会回放当前帧之前的 H 个观察，间隔由 `history-stride` 指定。真实 history/cache 的更新频率还服从原部署 wrapper 的 prefill 行为，并非“每回放一帧一定存一帧 KV”。导出每个状态独立 reset 和固定 seed，避免受遍历顺序影响。

provenance 保存 checkpoint/stats/config/code SHA、source Parquet SHA、seed、token source 和 cache protocol。state/reference/action 坐标复用部署 processor 的实际 global affine normalizer，包含其 [-5,5] clamp；不重新用 RL 训练集拟合 stats。当前 exporter 针对本项目绝对 6D 目标、无 action-state 额外变换的 AHA 配置；不要用于相对动作/其他 embodiment 后认为无需修改。

本轮无可用 CUDA，完整 exporter 未运行；该阶段必须在 GPU 上完成真实 2-state smoke、保存/重载比对和吞吐测量。程序会在无 CUDA 时直接说明条件，避免意外在 CPU 加载并执行完整 WAM。

## 4. 接近 RLT 的 reader 路线

```bash
$PYTHON scripts/so101_pipeline/offline_rl/run.py readout \
  --transitions runs/offline_rl/cylinder_review/macro/transitions.pt \
  --features runs/offline_rl/cylinder_review/features.pt \
  --output runs/offline_rl/cylinder_review/readout \
  --width 256 --steps 2000 --batch-size 32 --device cuda --seed 42
```

复用已有 `RLTokenReadout`：追加可训练 query、encoder 压缩、因果 decoder 重建、冻结原始输入。训练只用 train episodes，之后冻结 reader，导出单 token cache；decoder 不进入 RL learner。生成 `reader.pt`、`features.pt`、`validation.json`。

这些是 RLT-inspired 的小宽度 reader，不是原版 VLA 的原宽度 token。不能仅凭重建 loss 低就认定对精细控制有用：teacher-forcing 也能让 decoder 绕过 token 使用相邻 embeddings，需要再做 token 打乱/置零、微小偏差 probe 和实机消融。

## 5. 外部小网络训练

```bash
$PYTHON scripts/so101_pipeline/offline_rl/run.py train \
  --transitions runs/offline_rl/cylinder_review/macro/transitions.pt \
  --features runs/offline_rl/cylinder_review/readout/features.pt \
  --algorithm iql --architecture mlp --width 256 \
  --steps 2000 --bc-warmup 200 --batch-size 64 --seed 42 --device cuda \
  --output runs/offline_rl/cylinder_review/iql
```

reader 与 learner 必须用同一 seed/split；缓存含 reader validation episode 列表，改变 seed 会被拒绝。直接用原多 token cache 时，建议 `--architecture attention`；`mlp` 对多 token 仅做均值池化，属于弱诊断对照，并不自动成为 RL token。

同样命令替换 `--algorithm bc` 或 `td3_bc`，输出到各自目录。BC、IQL、TD3+BC 应共享表征、数据切分、训练预算和初始化种子。`bc-warmup` 是先对实际数据动作训练外部 actor；不是假设随机小 actor 能保持原 DAgger 策略。

目前起始超参：IQL expectile=.7、inverse temperature=3、weight cap=20、Adam lr=3e-4、target tau=.005、两次 critic 一次 actor、reference dropout=.1、额外 reference penalty=0。它们都是候选参数，未在 SO101 调优。原 RLT reference dropout 是 .5；本实现没有声称复现该配方。若要改变算法超参，使用 `LearnerConfig` 并保存配置；CLI 只开放了当前必要参数。

2026-10-24 频率核查补充：当前 IQL 的 2:1 是候选适配，原版 IQL 为 V/Q/A 每轮各一次；`bc-warmup` 只训 actor，不是 Q/V 预热。正式比较建议加 actor_delay=1 基线；target 更新频率的实现区别见 [更新调度专项](06_ACTOR_CRITIC_UPDATE_SCHEDULES_ZH.md)。本轮未改默认行为。

输出 `metrics.jsonl`、`last.pt`、`provenance.json`。恢复时加 `--resume 路径/last.pt`，并保持原参数，会核验数据/特征 hash、split、warmup 和网络/learner 配置。保存模型、target、optimizer、step、CPU RNG；当前不是 CUDA 完全逐位恢复保证，也没有恢复异步机器人 collector。

## 6. 本轮已经执行的验证

```bash
PYTHONPATH=src:. /home/fenghao/miniconda3/envs/ahawam/bin/python \
  -m unittest discover -s tests -p test_offline_rl.py -v
```

16 项测试通过，包括：成功观察的 incoming reward、terminal prefix、全零失败、不跨缺口、pending 不标负、hold 的宏动作奖励/折扣、接管 censor、成功观察先于新控制源动作、过期 annotation 拒绝、因果历史、episode 分割、Q 动作顺序敏感且双网络独立、target 无梯度、实际 IQL 学到合成终止成功差异、TD3 延迟 actor/动作梯度、checkpoint 恢复和 KV pooling。

指定真实批次完成三种 prepare。另以 **真实转移＋随机 synthetic token** 测试 train/readout CLI；fixture provenance 明确写 `synthetic=true`，train 必须显式 `--allow-synthetic-smoke` 才能消费它。此 smoke 验证接线和序列化，不能报告成机器人 critic 的训练成果。

详细机器可读结果见 [audit_20261023.json](audit_20261023.json)。运行输出在 `runs/offline_rl/review_20261023/`，不作为源数据发布。

## 7. 正式实验的验收顺序

1. 决定动作时间语义、DAgger 基座、混合/接管信用目标；补齐其他数据标注。
2. GPU 两状态真实 feature export → 同状态重复导出一致 → canonical reference baseline。
3. reader 与 BC 同数据训练；先确认外部 actor 能保持合理行为，再比较 IQL。
4. 固定 offline 训练预算，按 episode/批次留出，监控 Q 和权重退化。
5. 独立实机评估无接管成功率、耗时、接管频率、动作平滑性；训练 Q 上升不等于效果改善。
6. 再实现在线 collector/learner 接线、固定 latency 协议和真实特征日志。当前文件没有自动把实验 checkpoint 接入生产动作服务器。
