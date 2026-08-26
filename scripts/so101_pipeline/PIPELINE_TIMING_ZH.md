# 训练和部署到底各自在算什么、多久算一次

回答四个问题：

1. 生成的视频是多少帧？训练和部署各是多少帧？
2. 是不是"生成一段 3 秒视频，这 3 秒内视频模型就不动了"？
3. 是不是"action DiT 一直吃这次生成的 K/V，等视频算完才继续"？
4. 仓库实现和 AHA-WAM 论文一致吗？

所有耗时都是在本机 A100-40GB 上用 `scripts/so101_pipeline/online_deploy/settings.yaml`
的真实配置实测的（见文末"怎么复现"）。

---

## 0. 一句话先纠正最关键的误解

> **在线部署时，视频 DiT 根本不生成视频。**

它每 2.67 秒跑一次，输入只有**当前这一帧**，输出只有 **K/V 缓存**——
没有去噪循环，没有未来帧，没有 VAE 解码。实测 **70 ms**。

那些 3 秒的视频是我给你做的**离线诊断**（`preview` / `baseline_video_experiments`），
在线控制回路里根本不会跑。论文原文也是这么设计的：

> "At deployment, AHA-WAM removes explicit future-frame decoding:
> the video DiT only refreshes planner video context, and the action DiT
> uses this context to generate closed-loop action chunks."
> —— [AHA-WAM, arXiv 2606.09811](https://arxiv.org/abs/2606.09811)

---

## 1. 帧数：三个场景三个数

| 场景 | 视频帧数 | 覆盖的真实时长 | 说明 |
| --- | --- | --- | --- |
| **训练**（一个样本） | **13 帧** | 3.2 秒 | 49 个控制帧里每 4 帧取 1 帧 |
| **离线诊断视频** | **13 帧** | 3.2 秒 | 刻意和训练完全一致，才能对得上 |
| **在线部署** | **1 帧** | —— | 只编码当前帧，不产生任何未来帧 |

训练/诊断那 13 帧是怎么来的：

```
49 个控制帧 @ 15 Hz  (num_frames=49)
   │  每 4 帧取 1 帧 (action_video_freq_ratio=4)
   ▼
13 个视频帧 @ 3.75 Hz  = 48/15 = 3.2 秒
   │  VAE 时间压 4 倍
   ▼
4 个 latent 帧
```

所以你看到的 mp4 是 13 帧、3.2 秒、384x320，按 3.75 fps 播放正好是真实速度。

---

## 2. 在线一个 episode 的完整时序

配置：`action_horizon=40`、`action_chunk_size=8` → `num_chunks = 40/8 = 5`。

```
机器人客户端每次请求 → policy.get_action() 返回 8 个动作

请求 1  ┌─ next_chunk_index=0 >= 5? 否 … 但 _episode_prefilled=False
        ├─ ① video prefill  : 当前帧 → K/V            70 ms
        └─ ② action chunk 0 : OVCR + ActionDiT 去噪   534 ms   → 动作 0-7
请求 2  └─ ② action chunk 1 : 用新的一帧重新路由      534 ms   → 动作 8-15
请求 3  └─ ② action chunk 2                          534 ms   → 动作 16-23
请求 4  └─ ② action chunk 3                          534 ms   → 动作 24-31
请求 5  └─ ② action chunk 4                          534 ms   → 动作 32-39
请求 6  ┌─ next_chunk_index=5 >= 5 → _soft_reset
        ├─ ① video prefill  : 用**新的**当前帧刷新 K/V  70 ms
        └─ ② action chunk 0 …
```

一次 video prefill 支撑 **5 个 chunk × 8 = 40 个动作 = 2.67 秒**（15 Hz）。
这 2.67 秒里视频 DiT 确实"不动"——但**动作分支一点也没瞎**，见下一节。

---

## 3. 关键：这 2.67 秒里，动作分支每个 chunk 都重新看一眼相机

你的理解里缺的就是这一环。它叫 **OVCR（Observation-Guided Context Routing）**，
是 AHA-WAM 这篇论文的核心贡献之一。

每个 chunk 都会做这么一遍
（[`ahawam.py:_prepare_inference_chunk_conditioning`](../../src/ahawam/models/wan22/ahawam.py#L291)）：

```
拿到当前这一帧真实相机图 chunk_obs_image
   │
   ├─ VAE 编码 (实测 9.8 ms)
   ├─ 生成 32 个 "routing query"  (chunk_kv_editor_num_queries=32)
   └─ 用这 32 个 query 去"改写"视频 K/V 缓存
        build_chunk_updated_video_kv_cache(...)
        = 视频 K/V 里第一帧的 k/v + 门控残差 delta
   │
   ▼
改写后的 K/V 才交给 ActionDiT 去噪出 8 个动作
```

所以准确的说法是：

| 说法 | 对不对 |
| --- | --- |
| "2.67 秒内视频 DiT 不再跑" | ✅ 对 |
| "action DiT 一直吃同一份视频 K/V" | ⚠️ 一半对——**底座**是同一份，但每个 chunk 都被当前帧重新路由过 |
| "action DiT 在等视频结果" | ❌ 不等。视频那步只有 70 ms，而且不产生视频 |
| "这 2.67 秒机器人是开环的" | ❌ 不是。每 0.53 秒（一个 chunk）就重新看一次相机 + 重新读一次关节状态 |

论文的原话是这套上下文"amortizing expensive world-model computation across
multiple action steps"——**摊销**，不是**冻结**。

---

## 4. 训练侧：一次 forward 同时覆盖视频和 5 个 chunk

训练走的是
[`_training_loss_action_offset`](../../src/ahawam/models/wan22/ahawam.py#L939)
（因为 `max_action_offset=7 > 0`）：

```
一个训练样本：
  视频分支：13 帧 → 4 个 latent 帧，随机时间步加噪，第 0 帧钉成干净
           → 预测 flow，只在第 1..3 个 latent 帧上算 loss
  动作分支：40 个动作 = 5 个 chunk × 8
           每个 chunk 配一张自己的观测图 (chunk_obs_images，5 张)
           偏移 δ ~ U{0..7} 随机采样   ← Horizon-Adaptive Offset Training
```

两个要点：

1. **训练时 5 个 chunk 是一次前向并行算完的**（block-diagonal / chunk-causal mask），
   部署时是 5 次串行。因为 mask 是 chunk 对角的，两者数学上等价——
   每个 chunk 都只看得到自己那 8 个动作 + 自己那张观测图路由出来的 K/V。
2. **随机偏移 δ** 就是论文的 Horizon-Adaptive Offset Training：
   动作窗口相对视频窗口随机平移 0~7 个控制帧，让模型习惯
   "视频 K/V 是 0~7 帧之前刷新的"这种相位差——正是异步部署时的真实情况。
   `_chunk_start_offsets = range(0, 40, 8) = [0,8,16,24,32]`，再整体加 δ。

---

## 5. 实测耗时，以及一个需要你注意的问题

| 阶段 | 实测 | 占比 |
| --- | --- | --- |
| video prefill（当前帧 → K/V） | **70.1 ms** | 每 5 个 chunk 一次 |
| **action chunk（8 个动作）** | **533.6 ms** | 每个 chunk |
| ├─ OVCR 条件准备（VAE + 32 query + 改写 K/V） | 36.8 ms | 7% |
| └─ **ActionDiT 10 步去噪** | **496.7 ms** | **93%** |
| 离线诊断视频（13 帧 / 20 步） | 2009.7 ms | 在线不跑 |

### ⚠️ 现在的配置刚好跟不上 15 Hz

```
一轮 = 5 个 chunk + 1 次 prefill = 5 x 533.6 + 70.1 = 2738 ms 计算
覆盖机器人时间 = 40 个动作 / 15 Hz  =            2667 ms
                                      ---------------------
                                      比实时慢 2.7%
```

也就是说动作是**将将来不及生成的**，长时间跑会持续累积延迟。
论文报的是 **41.37 ms/chunk、24.17 Hz**，我们实测 533.6 ms、14.8 Hz，差 13 倍。

**93% 的时间在 ActionDiT 的 10 步去噪上**，而 ActionDiT 只有 1B、每步只处理 8 个 token——
50 ms/步 对这个规模是纯粹的 kernel 启动开销（30 层 × 每层若干个小 kernel），
不是算力不够。这正是我在
[`WHY_SLOW_ZH.md`](baseline_video_experiments/WHY_SLOW_ZH.md) 第 0 节讲的
"token 太少 → 掉到 roofline 的内存/开销瓶颈那一侧"。

**最直接的改法（按性价比排序）：**

| 做法 | 预期 | 代价 |
| --- | --- | --- |
| `num_inference_steps: 10 → 4` | chunk ≈ 235 ms，**34 Hz** | 动作精度可能下降，要 A/B |
| 用仓库自带的 ODE 蒸馏（`configs/model/ahawam_ode.yaml`，`default_action_inference_steps: 1`） | chunk ≈ 87 ms，**92 Hz** | 要重训一个 student |
| 开异步双流（见下节） | 把 70 ms prefill 移出关键路径 | 只省 2.6%，杯水车薪 |

---

## 6. 你现在的部署**没有**开异步

`run_action_server.sh` 用的是 `--policy-worker-process`，走
`WorkerPolicyBackend`——它只是把请求**同步转发**给一个子进程（为了隔离 CUDA 崩溃），
视频 prefill 和动作推理仍然在同一条阻塞链上。

仓库里确实实现了论文那套异步双流，但要另外的开关：

| 开关 | 后端 | 行为 |
| --- | --- | --- |
| `--policy-worker-process`（**你现在用的**） | `WorkerPolicyBackend` | 同步，子进程隔离 |
| `--async-mode` | `AsyncPolicyBackend` + `async_runtime.AsyncAHAWAMRuntime` | 单卡异步 |
| `--dual-gpu-async` | `DualGPUAsyncPolicyBackend` | 视频 worker 和动作 worker 各占一张卡，视频 worker 独立跑 `prefill_video_only` 并把 K/V 发到队列（`dual_video_worker_main`） |

论文说的 "two non-blocking streams … removes the video DiT from the per-update
critical path" 对应的是后两个。不过按上面的实测，视频那步只占 2.6%，
**开异步的收益远小于把去噪步数降下来**。

---

## 7. 实现 vs 论文：逐条对照

| 论文设计 | 仓库实现 | 一致？ |
| --- | --- | --- |
| 部署时视频 DiT 不解码未来帧，只刷新 planner context | `prefill_video` 只跑一次 `pre_dit` + `prefill_video_cache`，无 scheduler 循环 | ✅ |
| 滚动 FIFO K/V memory，最多 6 帧历史观测 | `_history_kv_entries`，`num_history_frames=6`，超了丢最老的 | ✅ |
| OVCR：每个 action chunk 用 32 个 observation-guided routing query 改写 K/V | `chunk_kv_editor_num_queries=32`，`build_chunk_updated_video_kv_cache` | ✅ |
| Horizon-Adaptive Offset Training：δ ~ U{0..ha-1} | `max_action_offset=7`，`_training_loss_action_offset` | ✅ |
| 一次 planner context 支撑 hv/ha 个 chunk | 论文 64/16=4；本配置 40/8=**5** | ✅ 结构一致，数值按你的任务调过 |
| 异步双流，视频不在关键路径上 | 实现了（`--dual-gpu-async`），但**你的启动脚本没开** | ⚠️ 见第 6 节 |
| 41.37 ms/chunk，24.17 Hz | 实测 **533.6 ms/chunk，14.8 Hz** | ❌ 差 13 倍，93% 在 10 步去噪上，见第 5 节 |

结论：**架构和训练策略和论文完全一致，差距全在推理配置上**——
论文那个速度对应的是少步/蒸馏后的采样器，而你现在跑的是 10 步。

---

## 怎么复现这些耗时

```bash
conda activate ahawam && source /home/fenghao/use-nvidia-580159.sh
export PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES=1 \
  DIFFSYNTH_MODEL_BASE_PATH=$PWD/checkpoints DIFFSYNTH_SKIP_DOWNLOAD=true

python - <<'PY'
import sys, time, numpy as np, torch; sys.path.insert(0, ".")
from deploy.server.ahawam_policy import AHAWAMPolicy
p = AHAWAMPolicy("scripts/so101_pipeline/online_deploy/settings.yaml")
p.set_language_instruction("Pick up the white block.")
img = (np.random.rand(p.video_height, p.video_width, 3)*255).astype(np.uint8)
st0 = np.zeros(p.proprio_dim, dtype=np.float32)
p._hard_reset(); p.update_observation_window([img], st0)
it, pro = p._observation["image_tensor"], p._normalize_state(st0)
def t(fn, n=12, warm=3):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); ts=[]
    for _ in range(n):
        torch.cuda.synchronize(); a=time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter()-a)*1000)
    return np.mean(ts)
print("video prefill ms:", t(lambda: p.prefill_video_only(it, None, False)))
s = p.prefill_video_only(it, None, True)
def one():
    s["next_chunk_index"] = 0; p.infer_action_chunk_only(s, it, pro)
print("action chunk ms:", t(one))
PY
```
