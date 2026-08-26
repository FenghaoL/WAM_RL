# AHA-WAM 里的"历史"到底是什么

回答四个问题：

1. 历史是什么形态的东西？
2. 它怎么进到模型里？
3. 部署代码支不支持开关？
4. `run_action_server.sh` 现在用没用它？

**先给结论**：

| | |
| --- | --- |
| 训练时的形态 | **6 张图片**（`video_history` 张量），每步都重新过一遍 VAE |
| 部署时的形态 | **6 份注意力 K/V 缓存**，原图早就扔了，占 253 MiB 显存 |
| 怎么输入 | 不是拼成更长的视频，而是在**注意力的 key/value 维度上接一段前缀** |
| 能不能关 | 能，改 `settings.yaml` 里 `num_history_frames: 6` → `0`，一个数字 |
| 你现在用没用 | **用了**，每 2.67 秒往里塞一条，16 秒攒满 6 条后开始滚动 |

---

## 1. 训练时：6 张图片

数据集在 `_build_video_history_frames()`
（[`robot_video_dataset.py:483`](../../../src/ahawam/datasets/lerobot/robot_video_dataset.py#L483)）
里从**当前时刻往回数**，每隔一个完整窗口取一帧：

```
horizon_stride = (num_frames - 1) * global_sample_stride = 48 个控制帧 = 3.2 秒

                                                          ↓ 当前时刻（要预测未来的起点）
 ┌──────┬──────┬──────┬──────┬──────┬──────┬═══════════════╗
 h1     h2     h3     h4     h5     h6    now  ← 要预测的 3.2 秒未来 ═╝
-19.2s -16.0s -12.8s -9.6s  -6.4s  -3.2s  0s

每格 3.2 秒，6 帧最远回看 19.2 秒
```

给到模型的是三个张量：

| 字段 | 形状 | 内容 |
| --- | --- | --- |
| `video_history` | `[B, 3, 6, 384, 320]` | 6 张 RGB 图，和当前帧一样的拼图版式（上 fixed / 左下 wrist / 右下黑块） |
| `video_history_frame_indices` | `[B, 6]` | 每帧的**时间 RoPE 位置**，间隔 12（= 48 控制帧 / `video_rope_frame_stride` 4） |
| `video_history_valid_len` | `[B]` | 实际有效几帧 |

**episode 开头不够 6 帧怎么办**：不够的位置补零图、RoPE 位置填 0，
然后靠 `video_history_valid_len` 把它们从注意力 mask 里剔掉
（`_build_history_video_frame_mask`）。所以 episode 前 19.2 秒的样本历史是不满的——
这也是我挑评测窗口时要按"历史深度"分层的原因。

## 2. 部署时：6 份 K/V 缓存，图片扔了

这是最容易误解的一点。真机在线跑的时候，**模型手里没有那 6 张图**。

流程是这样：

```
某一时刻观测到一帧图
   │
   ├─ VAE 编码 → latent [48, 1, 24, 20]
   ├─ patchify → 120 个 token
   ├─ 过 30 层 VideoDiT，每层算出这 120 个 token 的 key 和 value
   │
   └─ 把这 30 层 × (k, v) 存进 model._history_kv_entries    ← 只存这个
      原图和 latent 用完即弃，永远不会再编码第二次
```

一条历史条目的大小：

```
30 层 × 2 (k 和 v) × 120 token × 3072 维 × 2 字节(bf16) = 42.2 MiB
6 条 = 253 MiB
```

代码在 [`ahawam.py:494`](../../../src/ahawam/models/wan22/ahawam.py#L494)：

```python
if num_history > 0:
    prior_entries.append(video_kv_cache)          # 追加当前帧的 K/V
    if len(prior_entries) > num_history:
        prior_entries = prior_entries[-num_history:]   # 滚动，只留最近 6 条
    self._history_kv_entries = prior_entries
```

**为什么这么设计**：K/V 是"这一帧被这一层理解成了什么"的中间结果。
存它就不用保留原图、也不用每次重新编码——用 253 MiB 显存换掉了 6 次 VAE + 6 次 30 层前向。
代价是这些 K/V 和当时的权重绑死，换 checkpoint 就得清空重来。

## 3. 怎么输入模型：接在注意力的 key/value 前面

这是理解"历史"的关键。它**不是**把 7 张图拼成一段更长的视频送进去，
而是只在**注意力**这一步动手脚。

正常的自注意力：

```
q, k, v = 当前帧的 120 个 token 各自算出来的 query / key / value
输出 = attention(q, k, v)          ← 120 个 query 看 120 个 key
```

带历史的自注意力（[`mot.py:438`](../../../src/ahawam/models/wan22/mot.py#L438) 和 `mot.py:558`）：

```python
k_video = torch.cat([history_k, k_video], dim=1)   # 历史的 key 拼在前面
v_video = torch.cat([history_v, v_video], dim=1)
输出 = attention(q, k_video, v_video)              ← 120 个 query 看 840 个 key
```

- **query 数量不变**（还是当前帧那 120 个 token），所以输出形状不变；
- **key/value 变多**（120 → 120 + 6×120 = 840），当前帧因此"看得到"过去；
- 信息是**单向**的：当前帧读历史，历史不会被当前帧改写（它是算好的常量）。

模型怎么知道"这一条是多久以前的"？靠 `video_history_frame_indices` 那组
**时间 RoPE 位置**——每条历史在编码的时候就被打上了自己的时间戳，
位置编码烙进了 k 和 v 里。所以间隔对不对很重要，这正是训练 12 / 部署 10 那个
mismatch 的来源（见 [PATH_AUDIT_ZH.md](PATH_AUDIT_ZH.md) 第 5 节）。

## 4. `run_action_server.sh` 用了吗：**用了**

调用链（每次机器人来要动作时）：

```
客户端请求
  └─ AHAWAMPolicy.get_action()                        ahawam_policy.py:490
       │
       ├─ next_chunk_index >= num_chunks ?            num_chunks = 40/8 = 5
       │    └─ _soft_reset_for_new_observation()      → _episode_prefilled = False
       │
       ├─ if not _episode_prefilled:
       │    └─ _prefill_episode()
       │         └─ model.prefill_video()             ahawam.py:372
       │              ├─ 当前帧 → K/V
       │              ├─ 和已有的 _history_kv_entries 拼成前缀一起算注意力
       │              ├─ _history_kv_entries.append(...)，超过 6 条就滚动
       │              └─ _observed_frame_index += 40 // 4 = 10
       │
       └─ _infer_action_chunk() → 动作
```

每 **5 个 chunk = 40 个控制帧 = 2.67 秒**触发一次 `prefill_video`，也就是往历史里塞一条。

| 时刻 | `_history_kv_entries` 长度 |
| --- | --- |
| episode 开始 | 0（`reset_obsrvationwindows()` 会清空） |
| 2.67 s | 1 |
| 5.33 s | 2 |
| … | … |
| **16.0 s** | **6（满）** |
| 之后 | 恒定 6，最老的被挤掉 |

开关就在你的 [`online_deploy/settings.yaml`](../online_deploy/settings.yaml)：

```yaml
hydra_patch:
  model:
    num_history_frames: 6      # ← 就是这一行
    video_rope_frame_stride: 4
```

**注意：历史不只影响视频预览，它确实影响你的动作输出。**
`prefill_video` 算出来的当前帧 K/V 是"在能看到历史的前提下"算的，
这份 K/V 又被 `infer_action_chunk` 通过 chunk KV editor 喂给 ActionDiT。
所以改这个数字，动作会变。

## 5. 怎么关，关了会怎样

把那一行改成 `num_history_frames: 0` 就行。代码里的效果：

```python
num_history = self._configured_num_history_frames()          # 0
prior_entries = (...) if num_history > 0 else []             # []
...
video_kv_cache = self._prefill_action_video_cache(...)        # 走无前缀的分支
...
if num_history > 0:      # 不进，不再累积任何东西
```

- 省 **253 MiB** 显存；
- 每次 `prefill_video` 少算一段 840×840 的注意力（很小，速度上基本感觉不到）；
- **时间 RoPE 位置不受影响**——这点我专门确认过：
  `prefill_video` 是显式传 `temporal_position_ids=torch.full((1,), current_frame_index)` 的
  （`ahawam.py:447`），不走那个在 `num_history_frames <= 0` 时会返回 `None` 的
  `_get_main_video_temporal_position_ids`。所以关掉历史**只是去掉 K/V 前缀**，
  不会顺带把 RoPE 变成从 0 开始的连续编号。

这正好就是我离线测的 `--history-modes none`。六次测量（3 个 checkpoint × 训练集/验证集）
方向完全一致，不给历史都更好：

| | online（部署 cadence） | **none（关掉）** | 差 |
| --- | --- | --- | --- |
| 训练集 step_003440 | 17.481 | **17.725** | +0.24 |
| 验证集 step_003440 | 17.394 | **17.569** | +0.18 |

### ⚠️ 但先别急着改真机

我测的是**视频分支的预测质量**，不是**动作精度**。
两者虽然共享同一份视频 K/V，但不是同一个数：
视频分支好 0.2 dB，不能直接推出机械臂动作会更准。

稳妥的做法：

1. 先跑一遍 `deployment_preflight.py` 确认配置能加载；
2. 真机上做个 A/B：同一个任务、同样的起始位姿，`num_history_frames: 6` 和 `0` 各跑 10 次，
   比成功率；
3. 如果动作没变差，就关掉——省显存、少一个训练/部署 cadence 不一致的隐患
   （训练 RoPE 间隔 12、部署 10，见 [PATH_AUDIT_ZH.md](PATH_AUDIT_ZH.md) 第 5 节）。

### 如果想让历史真正有用

现在没用的原因很可能是**任务太短、信息都在当前帧里**：
一个窗口 3.2 秒，历史最近的一格也是 3.2 秒之前，而"白块在哪、夹爪在哪"
当前帧就全看得到，回看 3.2 秒没有增量信息。

要让它有价值，需要历史里**存在当前帧看不到的信息**，比如：

- 需要记住"刚才抓过一次没抓稳"的重试类任务；
- 物体被遮挡、必须靠之前的观测才知道在哪；
- 长时程多步任务（先做 A 再做 B，当前帧分不出处于哪一步）。

或者把历史窗口拉近：现在一格 = 48 个控制帧 = 3.2 秒，跨度很大；
改成比如 8 或 16 个控制帧，历史就变成"最近 1 秒内的运动轨迹"，
对判断速度和方向可能更有用。**但这要重训**，因为间隔是训练时烙进 RoPE 的。
