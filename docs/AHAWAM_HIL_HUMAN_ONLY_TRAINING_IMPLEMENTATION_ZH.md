# AHA-WAM：02+04 Demo 与 HUMAN Correction 两组 5-Epoch 训练实现

> 状态：已实现并完成静态预检、真实视频样本边界测试和采样器测试。  
> 数据日期：2026-08-29。  
> 目标：任何 video/action 训练窗口都不使用 POLICY 或 SYSTEM_HOLD 数据。

## 1. 最终实验定义

两个实验都从官方权重重新开始：

```text
checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt
```

都训练 5 epoch，都使用原 02+04 的同一份 SO101 normalization stats：

```text
runs/so101_ahawam_plug_all/assets/dataset_stats.json
```

区别只有 source sampling：

| 实验 | 每个 nominal epoch | 八卡 optimizer steps/epoch | 5 epoch steps |
| --- | ---: | ---: | ---: |
| natural concat | 23,947 demo + 9,158 HUMAN = 33,105 anchors | 1,035 | 5,175 |
| balanced 2:2 | 23,952 demo + 23,952 HUMAN = 47,904 draws | 1,497 | 7,485 |

natural 中 correction 占 27.66%。balanced 在**每张 GPU 的每个本地 batch**中严格取 `2 demo + 2 correction`；每轮覆盖全部 demo，correction 通过独立 shuffle cycle 重采样约 2.62 次。它不会浪费任何 correction，且每次循环都会重新打乱。

33,105 不能被八卡 global batch 32 整除，Accelerate 为保证各 rank 步数一致，会在 natural 每轮末尾最多补齐 15 个已打乱样本；所以表中 33,105 是 nominal unique-anchor pool，实际 batch capacity 是 `1,035×32=33,120`。balanced 的 47,904 本身能被 32 整除，不需要尾部补齐。

注意：用户要求两者都按 5 epoch，因此两组 optimizer-step 预算本来就不同。论文表格必须同时报告 epoch 和 steps；若以后要隔离“比例”与“额外更新次数”，再增加一个 5,175-step 的 balanced compute-matched control，不能把它暗中混入本轮。

## 2. v1/v2 到底能不能直接补 mask 使用

结论是：**HUMAN 数据能用，但不能把完整 raw episode 当成一个连续训练 episode。**

旧说法“只能使用导出的 correction-only episode”表达得过窄。真正必要的操作不是一定要在磁盘上重新编码 MP4，而是满足如下等价语义：

```text
每一个连续 HUMAN 段 = 一个新的逻辑 episode
```

本次实现采用零拷贝 logical view：

1. HUMAN 行必须同时满足 `control_source==1`、`intervention=true`、`human_action_valid=true`；
2. v3 如果有 `action_supervision_mask`，HUMAN 行还必须为 true；
3. raw episode 改变、`frame_index` 不连续、`tick_index` 不连续或 `dropped_tick=true` 时切段；
4. dataset 的公开索引只包含 HUMAN 行；
5. action/state/image future query 到达 HUMAN 段尾后，复制最后一个 HUMAN 值并设置 pad mask；
6. video history 也按逻辑段边界查询，段前 autonomous 帧不会作为历史输入。

这与“先导出 97 个 correction-only LeRobot episode 再训练”在监督语义上等价，但不需要复制约 762 MB 视频，也避免帧级裁剪和重新编码引入的新误差。

原始 HIL 共 22,314 帧，其中 HUMAN 只有 9,158 帧；其余 13,156 个 POLICY/SYSTEM_HOLD 行不在训练索引中。底层 MP4 decoder 在一次 raw 查询中可能先解出边界外帧，但 logical loader 会在图像增强和模型输入之前把这些位置替换为最后一张 HUMAN 图像并标 pad；它们不会进入模型输入或 loss。

## 3. 97 个 HUMAN 段怎样分配 video/action loss

审计结果：

| 项目 | 数量 |
| --- | ---: |
| HUMAN frames | 9,158 |
| 连续 HUMAN segments | 97 |
| segment 长度 min / median / max | 22 / 80 / 261 |
| 长度 `>=49` | 82 |
| 长度 `<49` | 15 |

监督规则是：

```text
demo                         -> video loss on  + action loss on
HUMAN segment length >= 49   -> video loss on  + action loss on
HUMAN segment length < 49    -> video loss off + action loss on
```

这里用 `>=49`，因为 AHA-WAM 的窗口是当前帧加未来 48 个 raw control frames，49 帧刚好能提供一个完整窗口。

短 correction 并没有被删除。它的所有 HUMAN anchor 仍进入 action 训练，只是 `video_supervision_mask=false`，直接 video-generation loss 为零。若一个本地 batch 恰好全部是短 correction，模型得到保持计算图连通的零 video loss，不会发生分布式 unused-parameter 问题。

这里“action-only”精确指**没有直接 video target loss**，并不表示把视觉分支从计算图中物理拆掉。AHA-WAM 的 action 预测仍以 HUMAN 图像为条件，action loss 也可能经 video/action 交互模块更新共享参数；这是希望保留的视觉动作学习。若要求短段对 VideoDiT 参数连 action 梯度也完全为零，就需要拆成独立 forward 或额外 stop-gradient，那是另一项更强、也更可能损害视觉纠错的消融。

## 4. 原仓库 episode 尾部不足 49 帧时做什么

原仓库不是简单丢掉 episode 最后 48 个 anchor：

1. LeRobot 将越过 episode 末尾的 query clamp 到最后一帧；
2. `action_is_pad` 让越界 action token 不进入 action MSE；
3. `image_is_pad` 传入 video loss；
4. 12 张 future sampled frames 以每 4 张为一组对齐一个 VAE latent time step；
5. 只有一组内 4 张 sampled frames 全部为 pad 时，该 latent step 才被排除。

因此一个长 correction 的尾部仍采用仓库原有的 partial-horizon 规则：还有真实 future 时就学习可用 video latent group；全部 future 都越界时该 anchor 的 video loss 自然为零。短于 49 的整个 correction 则按本轮实验约定强制 action-only，不利用这种不完整 video target。

## 5. safety clamp 帧使用哪个 action

9,158 个 HUMAN 帧中有 305 帧的 `human_action` 与最终 `action` 不同；这 305 帧全部正好是 `safety_clamped=true`。

训练必须使用数据中的 `action`，即机器人实际执行的安全裁剪后目标，而不是 leader 原始 `human_action`。否则模型会学习硬件实际上没有执行、且安全层明确拒绝的命令。本实现保留仓库标准的 `action` key，预检同时要求 v3 HUMAN 行拥有有效 supervision mask。

## 6. 为什么两组都固定旧 stats

新增 HUMAN 数据约占 natural 合集的 27.7%，数量不算小；但它与 demo 来自同一台 SO101、相同 6 关节单位、相同 action 定义。固定旧 stats 的主要理由不只是“方便消融”：

- 已部署的 reference policy 与旧 stats 绑定；
- 改 stats 会同时改变每个关节的标准化坐标和 loss 尺度；
- balanced 与 natural 的比较应只改变 sampler；
- correction 的精密姿态分布改变不等于传感器单位或校准改变。

所以这两组主实验不重算 stats。以后可以单独增加 combined-stats ablation；如果 correction 超出旧 min/max 很多，应先检查校准、关节单位和 safety clamp，而不是用重算 stats 掩盖数据错误。

本次还实际检查了 9,158 个 HUMAN label，而不是只根据“同一台机械臂”推断：各 action 维最大绝对 z-score 为 `[2.48, 2.00, 2.14, 2.92, 2.87, 2.30]`，没有任何维超过 3σ。76/9,158 行略微越过旧数据的逐维 min/max，幅度仍在 3σ 内；state 是 82/9,158 行，最大也低于 3σ。这更像新增精密姿态覆盖，而不是单位或校准错位，因此固定旧 stats 是本轮更稳的选择。

## 7. 为什么八卡 A800 仍使用 ZeRO-2

AHA-WAM checkpoint 约 14.7 GB，模型约 6B 参数。普通 DDP 会在每张卡复制 Adam 的 fp32 master weights、动量和二阶矩，再叠加梯度与大视频激活，80 GB 虽可能在特定设置下勉强运行，但安全余量和碎片风险不好。

八卡均为约 80 GB 显存时，本轮自动使用：

```text
8 x A800
bf16
per-GPU batch = 4
gradient accumulation = 1
ZeRO stage 2
optimizer offload = none
parameter offload = none
```

ZeRO-2 分片 optimizer state 和 gradient；A800 具有足够显存，所以不做 CPU offload，避免 PCIe 与 CPU optimizer 成为吞吐瓶颈。这比直接切回普通 DDP更稳，也比旧 ZeRO-2 offload 配置更快。

“A800”并不自动保证一定是 80 GB 型号。启动器会读取八张卡的 `memory.total`：全部达到约 70,000 MiB 才选择 no-offload；若是 40 GB 型号则自动回退到仓库已有的 ZeRO-2 optimizer-offload 配置，并把最低空闲显存阈值改为 36,000 MiB。也就是说默认仍是八卡 80 GB 的高速方案，但不会在 40 GB A800 上盲目 OOM。可用 `GPU_MEMORY_MODE=no_offload|offload` 显式覆盖自动判断。

## 8. 直接运行

先在训练机器进入仓库并激活已安装 AHA-WAM 依赖的环境，然后二选一运行；脚本不需要参数。

自然拼接：

```bash
scripts/so101_pipeline/hil_human_only_training/run_natural_5ep.sh
```

严格 2:2：

```bash
scripts/so101_pipeline/hil_human_only_training/run_balanced_2plus2_5ep.sh
```

默认使用 GPU `0..7`。80 GB 路径要求每张至少 60,000 MiB 空闲；40 GB offload 路径要求至少 36,000 MiB。可选环境变量只用于机器差异，例如 `GPU_IDS`、`MAIN_PORT`、`PYTHON_BIN`、`RUN_ROOT`；正常八卡机器不需要传任何值。

每次启动先执行 preflight，输出中必须出现：

```text
human_only_view segments=97 frames=9158 ...
video_action_segments=82 action_only_segments=15
autonomous_training_frames=0
status=ready
```

预期 schedule：

```text
natural:      steps_per_epoch=1035 total_steps=5175
balanced_2_2: steps_per_epoch=1497 total_steps=7485
```

## 9. 关键实现文件

- [`hil_human_only_dataset.py`](../scripts/so101_pipeline/hil_human_only_dataset.py)：HUMAN segment 扫描、logical episode、边界 clamp/mask 和 mixed dataset；
- [`samplers.py`](../src/ahawam/utils/samplers.py)：严格 2:2 shuffled-cycle sampler；
- [`ahawam.py`](../src/ahawam/models/wan22/ahawam.py)：逐样本 video supervision mask；
- [`settings.yaml`](../scripts/so101_pipeline/hil_human_only_training/settings.yaml)：两实验共享配置；
- [`run_natural_5ep.sh`](../scripts/so101_pipeline/hil_human_only_training/run_natural_5ep.sh) 与 [`run_balanced_2plus2_5ep.sh`](../scripts/so101_pipeline/hil_human_only_training/run_balanced_2plus2_5ep.sh)：无参数入口。

## 10. 已完成验证

- Python compile 与 shell `bash -n` 通过；
- 全数据 metadata preflight 通过；
- runtime dataset 长度为 `23,947 + 9,158 = 33,105`；
- runtime 得到 97 段、15 个短段；
- 抽查长段首尾、短段首尾，action/state/image 有效 token 数与边界一致；
- 抽查短段，边界外图像在 processor 前被精确替换为最后一张 HUMAN 图像；
- 短段最终输出 `video_supervision_mask=false`，段起 history valid length 为 0；
- 2:2 sampler 每个四样本 batch 都严格包含两类各 2 个，每轮覆盖全部 23,947 demo 和全部 9,158 correction；
- 全 action-only batch 的 masked video loss 可正常 backward。
