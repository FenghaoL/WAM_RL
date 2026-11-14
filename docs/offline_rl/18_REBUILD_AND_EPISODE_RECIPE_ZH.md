# 2026-10-26：连续 episode 重建、特殊样本与两项重训

本次所有训练转移与 KV/reference cache 从原始数据重新构建，不读取旧特征缓存，不恢复旧 actor/critic 权重。原始录像、parquet 和人工标注均不修改。重建目录为 `runs/offline_rl/cylinder103_rebuild_20261026`。

## 固定划分和特殊处理

选择在新模型评分前固定，未按旧 Q 的高低挑样本。episode 编号均为 **从 0 开始的 `episode_index`**，与文件名一致。

两条失败 episode 完整留作验证，不参与 actor、Q、V 的任何优化更新：

| 原始目录 | 文件 | 帧数 | 选择原因 |
|---|---|---:|---|
| `20260911_183057` | `episode_000000` | 389 | 包含 HUMAN 接管的失败 |
| `20260911_225809` | `episode_000001` | 167 | 纯 POLICY 的失败 |

原 `_val` 中三个有成功帧标注的 episode 1、2、3 继续验证。原 `_val` 的 episode 0 没有录到成功，按用户要求纳入训练；**本次划分以 recipe 为准，不能再靠目录名推断 train/val**。

四条未标注成功帧的成功 episode 只取前半段，按保存帧数 `floor(N/2)` 定义，不按去掉 HOLD 后的动作数计算一半：

| 原始目录 | episode 文件 | 原始帧数 | 保留帧范围 | 训练转移数 |
|---|---|---:|---|---:|
| `20260911_232927` | `episode_000007` | 437 | 0–217，共 218 帧 | 113 |
| `20260911_232927` | `episode_000015` | 352 | 0–175，共 176 帧 | 90 |
| `20260912_001855` | `episode_000002` | 416 | 0–207，共 208 帧 | 98 |
| `20260912_001855` | `episode_000009` | 430 | 0–214，共 215 帧 | 111 |

两次采集各选四条中的两条做前半段处理，其余保留完整录像。额外输出这四条的 fixed/wrist 两路前半段 MP4，共 8 个文件，位于运行目录的 `prefix_videos/`，可直接查看。模型仍从原始视频读取相同范围的帧，避免把二次压缩引入特征；截断后半段既不导出 cache，也不构建转移。

另外五条完整纳入训练：

- `20260911_232927/episode_000013`、`episode_000025`。
- `20260912_001855/episode_000005`、`episode_000017`。
- `20260912_010747_val/episode_000000`。

所有 9 条均已纳入训练，不再使用旧的“排除未观测成功录像”规则。

## 没录到成功时，奖励与结尾如何处理

用户已确认这些实验最终成功，但所录画面只到半插入；因此没有可用的成功时刻。完整录像和人为保留前半段的四条，均使用 **观察到的奖励为 0、未终止、保留自举**：

```text
target = 0 + gamma^8 * V(next_observation)       # IQL
target = 0 + gamma^8 * min(target_Q1, target_Q2) # TD3+BC，使用 target actor 动作
```

不在最后一帧臆造 +1，也不把片段末端设成失败的 `discount=0`。如果一个完整 8 动作窗口抵达最后一个可用动作，metadata 记为 `censored_recording_end`、`truncated=True`、`terminated=False`，**discount 仍为 `gamma^8`**。其余不满 8 个动作的尾窗直接不用，不补动作、不越过截断位置读取后半段。每条训练记录只引用保留范围内的当前/后继观察和动作。

这并不能提供截断后的真实回报监督。片段末端的 bootstrap 依赖网络泛化，尤其 TD3 会评价后继 actor 动作；这是本次按要求纳入不完整录像后的实际学习假设。所有这些 episode 在 recipe 中明确标为 censored，后续诊断的已记录回报仍显示未知，不把“目前没看见奖励”当作“未来回报确定为零”。

有成功标注的轨迹继续沿用已确认的奖励语义：成功画面对应上一条动作结果，HOLD 中的成功按规则前移；输入 Q/actor 的自然 8 动作块可跨成功，但包含成功的窗口不自举。POLICY＋HUMAN 保留，仅去掉 HOLD 动作。history 仍包含保存的 HOLD 画面。

## 本轮数据与模型参数

| 项目 | 本轮 |
|---|---:|
| 训练 episode | 98：87 个报告成功，其中 9 个未观测到成功；11 个失败 |
| 验证 episode | 5：3 个有标注成功＋2 个失败 |
| 训练转移 | 18,872 |
| 验证转移 | 858 |
| 全部转移 | 19,730 |
| 导出 cache 的保留观察数 | 42,862 |
| 四个后半段排除的观察数 | 818 |
| 成功奖励窗口（含训练及验证） | 580 |

重叠窗口不等于独立成功事件。动作块 8、窗口 stride **1**、每保留动作 gamma=0.997，连续时间轴 `episode_frames`。单个 episode 内任何 tick 跳号、接管或暂停均不清空历史，也不拆转移；不同 episode 仍隔离。planner anchor 每 8 个保存帧，最多 6 个历史画面，历史间隔 48 个保存帧，RoPE 使用保存帧坐标。

冻结采集用 SFT checkpoint 仍为 `formal-cylinderv1-sft-ahawambase-bs4-acc1-4ep-3gpu-20260905-022135/.../step_006572.pt`，SHA256 `1249707347cb117ac28039faa0f826e05b44b2698e27197f9427f23a500c9a51`。cache 为 BF16 最后一层 post-OVCR K/V＋routed/projected-query，state/reference 为 FP32；约 74.6 GiB，实际大小由 `cache/complete.json` 记录。

IQL 与 TD3+BC 都从头初始化同样的外部小网络，不接着旧权重训练。各 10,000 更新、batch 64、Adam lr=1e-4、先 200 步普通 BC；网络 width 256、depth 2、4 attention heads。IQL expectile=0.7、优势温度系数 3、权重上限 20、每步更新 actor/Q/V；TD3 alpha=0.1、target noise=0.1、clip=0.2、每 2 步更新 actor/targets。两者 tau=0.005、reference dropout=0.1。每 100 步评估 **全部 858 个验证转移**，每 1,000 步保存 checkpoint。验证样本不进入损失反传；SFT normalization 固定，不使用新验证集拟合统计量。

这是保持此前参数的两项新基线，未因新增片段引入其他算法变化。旧基线实测外部网络约 1.2 GiB、10,000 步约 33–37 分钟；新训练的显存、耗时以各 `profile.json` 为准。缓存计算先预留 2–4 小时量级，由四个 rank 的实测进度修正，不作为完成时间承诺。

## 执行队列与查看位置

准备入口：`scripts/so101_pipeline/offline_rl/prepare_rebuild.py`。它从原始 6 个采集目录重新审计，固定全部 103 条 source identity、划分和帧范围，输出：

- `data/episode_recipe.json`：每条 episode 的 split、完整源身份/SHA、保留帧数、奖励规则和处理原因；这是复现实验的主配置。
- `data/transitions.pt`、`data/audit.json`：重新生成的转移和逐 episode 统计。
- `preparation.json`：特殊名单、8 个视频及 SHA、汇总数字。

同份 recipe 另归档为 [episode_recipe_20261026.json](episode_recipe_20261026.json)，方便独立于运行目录保存与查阅。

后台监督脚本：`scripts/so101_pipeline/offline_rl/rebuild_pipeline.py`。顺序为：

1. GPU 0、1、2、3 各启动一个冻结 WAM，导出全新 cache。每片 128 观察，原子写入。
2. 检查所有需要的 key、42,862 个观察、逐片 tensor SHA、每行 anchor/history/offset、帧范围、源码 hash；确保不含四个后半段，写 `cache/complete.json`。
3. GPU 0、1 分别做 4 步真实 cache 的 IQL/TD3 smoke，无 BC warmup，要求 actor/critic 都实际更新且指标有限。
4. 两卡分别从头跑正式 IQL/TD3+BC，各 10,000 步。任一阶段失败不继续后续阶段，日志保留。

状态入口是运行目录中的 `pipeline.json`、`pipeline.log`。缓存日志在 `cache/worker_rank0.log` 到 `worker_rank3.log`，进度与显存在 `cache/progress_rank*.json`。正式训练日志在 `experiments/iql.log`、`experiments/td3_bc.log`；模型、逐步指标和资源报告在对应子目录的 `last.pt`、`metrics.jsonl`、`profile.json`。仅 `pipeline.json` 的 `status=complete` 才表示两项正式训练完成。本轮不安排 Action DiT 微调，也不替换旧部署 YAML 的权重。

启动命令（本轮已由独立后台进程执行；不要重复启动）：

```bash
PYTHONPATH=src:. OMP_NUM_THREADS=2 /home/fenghao/miniconda3/envs/ahawam/bin/python \
  scripts/so101_pipeline/offline_rl/rebuild_pipeline.py \
  --run-dir runs/offline_rl/cylinder103_rebuild_20261026 \
  --gpus 0 1 2 3 --steps 10000
```

42 项现有及新增 CPU 测试通过，覆盖暂停连续性、缺失成功奖励、截断保留自举、训练/验证显式划分及旧 spec 兼容；真实准备脚本额外断言所有动作/后继观察不越过截止帧、9 条未观测成功片段均 reward=0/terminated=False/discount=gamma^8。

启动后已验证四个新 spec 完全一致，并实际检查 `183057/ep0` 帧 98→99 的 tick 112→161：帧 99 仍使用 planner anchor 96、history `[0,48]`、offset 3。首批分片的 tensor hash 已核对。用 167 个可用训练窗口、67 个验证窗口做了另一次启动测试，GPU 0/1 的 IQL/TD3 各 4 步、batch 8、无 BC warmup，actor/critic 都完成有限梯度更新；IQL 也完成 V 更新。该测试权重不用于正式训练，不能视为学习质量评估。结果保存在 `startup_smoke/passed.json` 和 `startup_verification.json`。此时完整导出仍在运行，正式两项训练等待缓存结束后的完整性检查。

## 本次需要明确记录的异常与决定

见 [19 数据异常与处理记录](19_DATA_ANOMALIES_ZH.md)。后续遇到与采集语义不一致、标注与记录冲突、异常时序或观测变化时，应明确告诉用户原始证据、影响哪些样本/缓存/训练以及拟采用的处理，不再默认把它们当作“数据不连续”过滤掉。
