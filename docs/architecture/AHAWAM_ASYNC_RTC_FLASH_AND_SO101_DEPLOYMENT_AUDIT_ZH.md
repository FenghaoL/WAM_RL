# AHA-WAM 异步推理、RTC、两步去噪与 SO101 部署审计

> 更新日期：2026-09-03  
> 审计对象：当前仓库 `13a0669`、官方 AHA-WAM 论文/项目页、`deploy/` 服务端与客户端，以及当前 SO101 部署配置。  
> 目标：回答“仓库是否自带 RTC”“异步推理与两步去噪到底是什么”“能否只改 `run_action_server.sh` 打开”“机械臂端需要改什么”。

## 0. 先给结论

结论分成六句：

1. **AHA-WAM 确实自带 video/action 异步推理框架，但它不是 RTC。**它把慢速 Video DiT 的 planner prefill 与快速 Action DiT 的闭环更新解耦，让 Action DiT 使用最近可用的 video K/V，并通过 OVCR 读取每次 action request 的当前图像和 proprio。
2. **“动作连续去噪两次”指的应当是 AHA-WAM-Flash 的 2-step ODE-distilled action sampler。**这是另一个需要蒸馏训练得到 student 权重的机制，不是打开异步开关后普通 checkpoint 自动只去噪两次。
3. 当前 [`run_action_server.sh`](../../scripts/so101_pipeline/online_deploy/run_action_server.sh) 没有打开异步：它使用普通 `infer`、`--policy-worker-process` 和 `--no-threaded-connections`。
4. 服务端可以加 `--async-mode`，但**不能只加这一项**：必须去掉 `--no-threaded-connections`，机械臂端也必须从单连接 `infer` 改为双连接的 `image` + `action_request` 协议。否则仍走同步 fallback，几乎得不到异步收益。
5. 仓库当前 async client 在等待 Action DiT 返回时仍然阻塞，然后才执行 action chunk。它异步的是 **Video DiT planner**，不是“执行旧 chunk 的同时生成新 chunk”。因此它本身不能消除你当前约 652 ms Action DiT latency 引起的 `SYSTEM_HOLD`。
6. 当前公开部署代码更像**参考实现**而不是论文中完整的高性能运行时。论文的 41.37 ms 依赖 TensorRT、CUDA Graph、VAE TRT、skip-copy 等优化；当前仓库检索不到这些 TRT/CG 实现。现有 async 路径还有若干需要实测或向作者确认的状态管理问题，所以不建议直接替换正在采集数据的稳定 v3 executor。

一句话概括：

> AHA-WAM 内置的是“慢 planner / 快 executor 解耦”；RTC 是“旧动作执行与新动作生成重叠，并对重叠动作做 inpainting”；AHA-WAM-Flash 是“把 Action DiT 蒸馏成 2-step”。三者互补，但不是同一个按钮。

---

## 1. 三个容易混淆的机制

| 机制 | 解决的问题 | 是否需要训练 | 当前仓库是否有入口 | 是否消除当前 SO101 的块间 HOLD |
|---|---|---:|---:|---:|
| AHA 异步 planner–executor | 不要每次动作更新都重跑 Video DiT | 主模型本身按异步接口训练；部署开 async 协议 | 有：`--async-mode` / `--dual-gpu-async` | **不能单独消除** Action DiT 等待 |
| AHA-WAM-Flash 2-step ODE 蒸馏 | 缩短 Action DiT 去噪时间 | **需要蒸馏 student** | 有训练代码和 config | 若 latency 降到执行预算内，可以显著减少 HOLD |
| RTC | 推理慢时仍执行旧 chunk；平滑衔接新旧 chunk | 原始 RTC 不要求重训 | 本仓库没有 RTC sampler/controller | 可以针对推理等待消除 HOLD，但不是 drop-in |

### 1.1 官方论文如何区分它们

论文明确写道：

- AHA-WAM 部署时将 video planner 与 action executor 作为两个 non-blocking streams；Video DiT 不在每次 action update 的关键路径中，但 Action DiT 仍必须满足闭环频率。
- 标准 AHA-WAM 的 action sampler 是 10-step；论文先用 TensorRT/CUDA Graph 等将 action-chunk latency 从 415.77 ms 降到 41.37 ms。
- AHA-WAM-Flash 再冻结 Video DiT、只蒸馏 action denoising path，把 10-step 降到 2-step，达到 17.56 ms。
- 真实机器人实验中，论文把 `RTC-style non-blocking execution + action interpolation` 用于慢速 Motus/Fast-WAM baseline；AHA-WAM 本身则写成使用其 asynchronous planner–executor interface。

所以“论文用了 RTC-style”不能推导为“AHA-WAM 自己的 async runtime 就是 RTC”。项目页甚至分别展示 AHA-WAM 有无 action interpolation 的视频，说明 action interpolation 也不是 AHA 架构定义的一部分。

### 1.2 RTC 多做了哪一步

假设旧 chunk 是

\[
A^{old}=(a_t,a_{t+1},\ldots,a_{t+H-1}).
\]

RTC 在机器人仍执行其中一部分时，后台对新 observation 生成新 chunk。预计新推理完成前必然会执行的前缀被视为已提交约束，新 chunk 的其余部分通过 flow/diffusion inpainting 与这个前缀保持连续。

AHA 当前公开 async client 并没有这种“committed prefix + inpainting”调用。它的 action 通道执行的是：

```text
取当前 frame/state
  -> 发送 action_request
  -> 阻塞等待整个 Action DiT 返回
  -> 执行返回 chunk 的前 N 步
  -> 再发送下一次 action_request
```

因此它没有 RTC 的重叠执行，也没有 RTC 的去噪 guidance/mask。

---

## 2. 当前 SO101 实际走的同步路径

当前 [`settings.yaml`](../../scripts/so101_pipeline/online_deploy/settings.yaml) 的关键值是：

```yaml
action_horizon: 40
action_chunk_size: 8
num_inference_steps: 10
input_control_hz: 15.0
action_video_read_mode: current_only
```

当前 [`run_action_server.sh`](../../scripts/so101_pipeline/online_deploy/run_action_server.sh) 使用：

```bash
--max-return-actions 8
--no-threaded-connections
--policy-worker-process
```

没有 `--async-mode` 或 `--dual-gpu-async`，所以每次 `infer` 的执行关系是：

```text
robot observation/state
  -> TCP infer
  -> 必要时 Video DiT prefill
  -> OVCR + Action DiT 10-step denoising
  -> 返回 8 actions
  -> 机械臂以 15 Hz 执行约 0.533 s
  -> 再请求下一块
```

在正常 AHA 路径中，一个 40-action planner horizon 对应 5 个 8-action chunks。同步 policy 在 `next_chunk_index >= 5` 时才触发新的 video prefill，因此 Video DiT 不是每个 action chunk 都重跑；但 action request 自身仍同步等待。

rich-v3 实测 request latency 中位数约 652 ms，而 8 actions 在 15 Hz 下只能覆盖：

\[
T_{exec}=8/15\approx 0.533\text{ s}.
\]

即便实现普通预取，652 ms 仍大于 533 ms，平均还有约 119 ms 的理论缺口；考虑 P90、网络和安全层后会更大。内置 planner async 最多移走周期性的 Video DiT prefill，不能把 10-step Action DiT 的 652 ms 自动藏起来。

---

## 3. 仓库自带 async runtime 到底做什么

### 3.1 单 GPU：`--async-mode`

代码入口：

- [`deploy/server/wam_policy_server.py`](../../deploy/server/wam_policy_server.py)：`AsyncPolicyBackend`；
- [`deploy/server/async_runtime.py`](../../deploy/server/async_runtime.py)：`AsyncAHAWAMRuntime`；
- [`deploy/client/wam_remote_client_node.py`](../../deploy/client/wam_remote_client_node.py)：`--async-policy-protocol`。

服务端维护：

```text
latest-wins image queue (size 2)
        |
        v
Video DiT prefill -> double-buffered latest inference_state / video K/V
                                      |
current image + current proprio ------+--> OVCR + Action DiT -> action chunk
```

GPU worker的调度优先级是：

1. `action_request`；
2. `video_prefill`；
3. idle。

其含义是：

- 相机可以持续推送当前图像；队列满时丢旧帧、保留最新帧；
- Video DiT 有机会在 action 请求之间刷新 planner state；
- Action DiT 使用最近发布的 planner state，同时仍使用该次 action request 自带的当前图像和 state；
- OVCR 的职责就是用当前 observation 对缓存 planner context 做路由/适配。

但单 GPU 不是两个 CUDA 模型真正并行运行。一个 GPU thread 顺序处理两类任务；已经开始的 Video DiT prefill 也不能被中途抢占。Action request 太密时，action priority 还可能使 video refresh 饥饿。

### 3.2 双 GPU：`--dual-gpu-async`

双 GPU 路径各加载一份完整 policy：

- video worker 在 `--video-gpu` 上做 prefill；
- action worker 在 `--action-gpu` 上用最新 video state 做 action inference；
- generation id 用于 reset 后丢弃旧 worker message；
- action worker会 `deepcopy(latest_state)`，避免一次 action inference 原地污染共享 planner state。

这更接近论文的两条 stream，但公开实现把整个嵌套 video inference state 转 CPU 后通过 multiprocessing queue 传给 action GPU，再搬回 GPU。它不是论文描述的 TensorRT video-KV skip-copy hot path，实际开销和显存必须测量。两个 worker 还各自加载完整约 7B 模型，至少要预留两张大显存卡。

### 3.3 为什么 action request 仍看到当前图像

异步不是“Action DiT 只看旧图”。一次 action update 实际使用两类视觉信息：

\[
\underbrace{C^p_{\tau}}_{\text{最近完成的慢速 planner K/V}}
\quad+\quad
\underbrace{I_t,s_t}_{\text{当前 action request 图像和 proprio}}.
\]

`I_t` 经当前 observation encoder/OVCR 形成查询，路由 `C^p_\tau` 后供 Action DiT 使用。所以 planner 可以旧一点，但 reactive observation 应当是当前请求时刻的。这正是 AHA 与“盲目复用旧 chunk”不同的地方。

---

## 4. 能不能直接改 `run_action_server.sh` 打开

### 4.1 服务端最小实验配置

可以做一个**实验副本**，不要先覆盖稳定生产脚本。单 GPU server 的最小形态是：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID" python deploy/server/wam_policy_server.py \
  --policy-path "$DEPLOY_CONFIG" \
  --instruction "$INSTRUCTION" \
  --action-dim 6 \
  --host "$SERVER_HOST" \
  --port "$PORT" \
  --max-return-actions 8 \
  --async-mode \
  --no-policy-worker-process
```

相对当前脚本必须满足：

- 新增 `--async-mode`；
- 删除 `--no-threaded-connections`，使用默认 `--threaded-connections`；
- 把原来的 `--policy-worker-process` 改成 `--no-policy-worker-process` 以使命令含义明确。实际上 backend 选择顺序使 `--async-mode` 优先，无论该 flag 如何设置，async policy 都会在 server 进程内加载；
- 暂时不要加 `--async-timing-config`：参数帮助提到的 `calibrate_async_timing.py` 在当前仓库中并不存在。

为什么必须保留 threaded connections：官方 async client 会建立一条 image TCP connection 和一条 action TCP connection。若继续 `--no-threaded-connections`，server 会长期服务第一条连接，第二条连接即使 TCP 建立也得不到处理。

### 4.2 只加 `--async-mode`、机械臂端完全不改会怎样

不会得到真正的 AHA async：

- 旧客户端仍发送 `type="infer"`；
- `AsyncPolicyBackend` 对 `infer` 走 `_handle_sync_infer()`；
- 该路径仍调用普通 `adapter.infer()`，不会经过 background image queue + cached planner state。

因此不能把它理解为 server-only 开关。

### 4.3 不应直接把 `num_inference_steps` 改成 2

当前 task-SFT checkpoint 是标准 AHA-WAM 权重，配置为 10-step。把 YAML 改为：

```yaml
num_inference_steps: 2
```

只会让普通 flow sampler 用两个很大的数值步长运行，**不等于 AHA-WAM-Flash**。它可能变快，但精度和动作分布没有论文保证，必须单独作为 un-distilled 2-step ablation 验证，不能宣称复现 Flash。

论文的 Flash 路线是：

```text
task teacher checkpoint
  -> 16-step teacher trajectory
  -> anchors {0,1,2,4,8,12,16}
  -> freeze Video DiT
  -> distill Action path student
  -> 2-step inference
```

当前仓库有 `configs/model/ahawam_ode.yaml`、`robotwin_ahawam_ode`/`robotwin_ahawam_offset_ode` 和 `src/ahawam/models/wan22/distill/`，但你的 SO101 task-SFT 尚未经过这一步。

另一个必须注意的不一致是：论文 Table 5 写 student 2-step，而发布 config 的 `default_action_inference_steps` 当前为 1；底层 student 支持 1/2/4 三档。应显式指定并验证 2-step，不能依赖默认值。

此外，公开的 RoboTwin ODE task config 是 14D 双臂、不同 action horizon/chunk 设定。SO101 蒸馏必须复用当前 6D、`action_horizon=40`、`action_chunk_size=8`、两相机 composite 和对应 stats，不能直接启动 RoboTwin ODE config。

### 4.4 双 GPU 不能沿用当前单卡环境变量写法

若实验 `--dual-gpu-async --video-gpu 0 --action-gpu 1`，不要再给整个 Python 进程套：

```bash
CUDA_VISIBLE_DEVICES="$GPU_ID"
```

这会在父进程层面只暴露一张 GPU，与子 worker 再按 physical GPU id 绑定的逻辑冲突。双 GPU 应让两个目标设备对父进程可见，再显式传 `--video-gpu` 和 `--action-gpu`。仓库归档的 [`run_dual_gpu_async.sh`](../../scripts/so101_pipeline/archive/online_deploy/run_dual_gpu_async.sh) 还引用旧 config/旧路径，不能直接拿来运行。

---

## 5. 机械臂控制端必须改什么

当前 [`ahawam_tcp_client.py`](../../scripts/so101_pipeline/online_deploy/ahawam_tcp_client.py) 是“一条 stateful connection + `infer`”客户端。若使用 async server，至少需要以下改动。

### 5.1 两条持久连接

```text
image_connection:
  按相机有效频率发送最新 composed front
  request type = image

action_connection:
  使用动作请求瞬间的 composed front + 6D model state
  request type = action_request
  等待 action_chunk reply
```

建议从真实 15 Hz 相机开始用约 15 Hz image stream，而不是机械套用官方 ROS 示例的 30 Hz。服务端本来就是 latest-wins，发送速度高于采集速度只会重复图像、增加序列化和网络负担。

### 5.2 async 消息现在既可传原始双相机，也可传拼好的 `images["front"]`

当前 SO101 async 服务端的 `image` 和 `action_request` 路径都已经统一调用
`prepare_camera_images()`，所以推荐直接传：

```python
images={"fixed": fixed_rgb, "wrist": wrist_rgb}
```

服务端会按训练时布局生成 384×320 composite。也可以由控制端传：

```python
images={"front": composed_rgb_uint8}
```

此时控制端必须继续使用与训练完全一致的 `compose_so101_cameras(fixed, wrist)`：
上方 fixed、左下 wrist、右下黑块。无论选择哪一种传法，image/action 两条通道都必须
来自同一套相机和预处理；若直接把 fixed 单图命名成 `front`，仍会造成严重视觉分布错误。

### 5.3 当前 observation/state 的时刻必须绑定

每个 `action_request` 应原子快照：

- 当前 fixed frame；
- 与其尽量同步的 wrist frame；
- 当前 follower state；
- monotonic capture/send timestamp；
- request id 与本地 policy generation。

不能让 action image 来自一个时刻而 state 来自明显更晚或更早的 tick。background `image` 只负责 planner；action request 自己仍必须携带当前图像和 state。

### 5.4 HIL takeover 的 fencing 不能删除

换 async 协议后仍须保留 v3 的：

- takeover 时清 action FIFO；
- 本地 `policy_generation += 1`；
- 断开/取消在途 action request，或收到后按 generation 丢弃；
- HUMAN 期间任何 late reply 都不能重新获得执行权；
- resume 前 remote reset / hard reset；
- 实际执行动作、source、SYSTEM_HOLD 和 request/reply latency 继续写 raw。

单 GPU async reply 只有 `kv_version`，没有 request id、generation 或 `kv_age_ms`。所以 HIL 安全不能依赖服务端替你完成，控制端必须保留本地 generation fencing；更稳妥的实现还应给协议回包补 request id/generation。

### 5.5 reset/reconnect 需要额外小心

当前 TCP server 对每个新 connection 都会隐式 reset 非 dual-GPU backend。官方 async client 使用两条连接，因此第二条连接建立时也会触发一次 reset；初始阶段通常尚未 prefill，问题不大，但 image channel 中途重连会意外清掉整个 planner state。

生产实现应二选一：

1. 修改 server，使 async mode 不因每条 transport connection 自动 reset，只接受显式 episode/session reset；或
2. 一旦任一 channel 重连，控制端进入 SYSTEM_HOLD，整体重建两条连接并显式 reset，绝不能静默继续旧 episode。

当前 `AsyncPolicyBackend` 也未实现 `health` message；原 SO101 client 的 `health()` 不能直接用于异步 backend，需要补 health handler 或使用专门 dry-run handshake。

---

## 6. 打开内置 async 后会不会消除卡顿

### 6.1 对你当前系统：大概率只能减少一小部分

内置 async 会移走 Video DiT prefill，但 Action DiT 仍同步阻塞。你当前 652 ms 的中位 request latency 与 533 ms 的 chunk execution budget 不匹配，因此：

```text
内置 AHA async
  = 减少/隐藏 Video DiT planner refresh
  != 执行 chunk 时并行请求下一 chunk
  != RTC
  != 2-step Flash
```

它是否有可见收益取决于 652 ms 中 Video prefill 的占比。最直接的判定方法不是猜，而是分别记录：

- `video_prefill_ms`；
- `action_chunk_ms`；
- end-to-end RTT；
- `kv_version` / `kv_age_ms`；
- `SYSTEM_HOLD` 帧数；
- action request 时图像年龄；
- planner refresh rate 与 stale reject 次数。

### 6.2 为什么论文看起来不需要 RTC 也很顺

论文不是只打开 Python async 开关。它把标准 10-step action latency 从 eager 415.77 ms 优化到 41.37 ms，又把蒸馏后的 2-step 降到 17.56 ms。Action branch 已经快于常见控制周期或远小于一个 action chunk 的执行时间，所以不必依赖长时间旧 chunk 才能掩盖推理。

你当前公开 Python eager path 没有这些 TensorRT/CUDA Graph 优化，实测约 652 ms，不能直接期待论文的 24.17/56.95 Hz。

### 6.3 如果目标是真正连续执行，需要哪类路线

至少满足下面之一：

1. 将 action inference 降到可接受预算内：编译/engine、减少去噪步数并验证、或训练 ODE-distilled student；
2. 实现 action inference 与当前 chunk 执行重叠，并解决 request observation 陈旧、late reply 和新旧 chunk 跳变；
3. 对 flow action sampler 实现真正 RTC 的 committed-prefix inpainting；
4. 使用快速 residual/current-observation correction head，在慢 base chunk 之上高频闭环。

这几项可以组合。内置 AHA async 只覆盖其中“planner background refresh”这一层。

---

## 7. 当前发布代码中的风险与未确认点

下面不是说代码一定不可用，而是说这些点仅凭静态代码无法把它等同于论文运行时。

### 7.1 单 GPU cached state 可能被 action inference 原地修改

`StateBuffer.consume()` 直接返回保存的 dict；而 `infer_action_chunk()` 会把去噪后的 `noisy_chunk` 原地写入 `inference_state["start_latents"]`。`infer_action_chunk_only()` 最后恢复 model 原先 state，却没有把返回的 `next_chunk_index` 写回 buffer。

结果可能是：同一个 planner version 上的下一次 action request 仍看到 `next_chunk_index=0`，但 chunk-0 初始 latent 已被上一次完整去噪改写。双 GPU worker显式 `deepcopy(latest_state)`，单 GPU runtime 没有。这种差异不像论文定义的“两次去噪”，也没有文档说明是预期行为。

建议在真机前：

- 对同一 `kv_version` 连续请求两次 action；
- 比较调用前后 buffer 内 `next_chunk_index` 与 `start_latents` hash；
- 与每次 deepcopy state 的结果对照；
- 向作者确认 single-GPU runtime 的正确语义。

### 7.2 公开 ODE config 与论文 2-step 不一致

- 论文：student denoising steps = 2；
- 当前 config：`default_action_inference_steps: 1`；
- 底层实现：只接受 1、2 或 4。

这不一定是模型错误，可能是发布 config 选择了更激进默认值；但复现实验时必须显式记录，而不是统称“两步”。

### 7.3 通用 real-robot wrapper 对 ODE container 的识别可疑

`AHAWAMPolicy` 通过 `inspect.signature(model.infer_action)` 是否显式含 `phase` 来判断 AHA/FastWAM。`AHAWAMODE.infer_action` 是 `*args, **kwargs` 代理，静态上会被识别成 `fastwam`；随后 wrapper 的 FastWAM 路径又没有传 AHA student 所需的 `phase` 或 `sample`。

因此不建议直接把 task 改成 `robotwin_ahawam_ode` 就启动现有 real-robot server。较可能的部署办法是把蒸馏保存的 student/model 权重加载回标准 AHA phase wrapper，并显式使用 2-step；但仓库没有把这条 real-robot 命令写清楚，必须先做离线 smoke test，最好向作者确认正式加载方式。

### 7.4 论文 CUDA 优化没有在当前 Python tree 中出现

对 `README.md`、`deploy/`、`src/`、`configs/` 检索 `TensorRT`/`CUDA Graph`，只有论文说明和本项目文档，没有对应 engine builder/runtime wrapper。也就是说：

- 开源了 Python async reference runtime；
- 开源了 ODE distillation training code；
- 不能从当前仓库直接复现论文完整 41.37 ms TRT/CG pipeline。

### 7.5 single-GPU 网络线程共享 policy observation

两条 TCP connection 的 handler 都会调用同一个 policy 的 `update_observation_window()` 并读取 `_observation`，但当前代码未对这段共享状态加锁。大多数时候 latest-frame 语义可能仍合理，但 image/action thread 可能互相覆盖临时 observation；这在要求图像/state严格配对的插入任务上值得补锁或改为纯局部预处理函数。

---

## 8. 截稿期建议：不要直接替换 v3，按三阶段验证

### 阶段 A：只做离线/桌面 async smoke test

1. 复制而不是覆盖 `run_action_server.sh`；
2. 使用 `--async-mode`、threaded connections 和当前标准 10-step checkpoint；
3. 先用 async dummy client，再用 SO101 双相机录制帧 dry-run；
4. 连续发 image、action、reset、断线重连；
5. 检查 action shape/范围、camera composite、kv version、state mutation 和显存；
6. 不给 follower 下发动作。

通过条件：连续 100 次 action request 无 crash/timeout/stale contamination，reset 后第一块可复现，固定同一输入的 sync/async action 分布差异可解释。

### 阶段 B：真机 SYSTEM_HOLD 安全测试

1. 极低速度和严格 delta limit；
2. 仍然使用每次 reply 后才执行 chunk 的模式；
3. 保留 takeover generation fencing；
4. 对比 sync/async 的 Action latency、Video refresh 和 HOLD 比例；
5. 如果只少了周期性 prefill spike、HOLD 中位数仍接近原值，就停止扩大改动。

### 阶段 C：再决定是否做 Flash/overlap/RTC

优先级建议：

1. **先把论文加速链路的问题问清楚**，尤其是是否能拿到 TRT/CG runtime 或 SO101 teacher 的蒸馏部署方式；
2. 若拿不到 TRT，先试 task checkpoint 的 4/6/8-step 精度–延迟曲线，但把它当普通 sampler ablation；
3. 时间允许再蒸馏 SO101 action student，显式评估 1/2/4-step；
4. 只有 Action DiT 仍显著慢于 chunk budget 时，再做 overlap/RTC；
5. 不让这条系统优化阻塞当前 rich-v3.1 HIL 数据和 adapter 主贡献。

当前论文消融必须固定 executor mode。不能让 baseline 用 `SYNC_HOLD`，新方法同时得到 async/Flash/RTC 后把全部收益归于 adapter。

---

## 9. 建议直接问 AHA-WAM 作者的问题

1. 公开的 `AsyncAHAWAMRuntime` 是否就是论文真实机器人使用的 runtime，还是仅参考实现？
2. single-GPU async state 中 `start_latents` 原地更新但 `next_chunk_index` 不回写，是否符合预期？同一 planner version 的每个 action update 是否都应从 chunk 0 和相同 noise 重新生成？
3. 论文 Table 5 使用 2-step，但发布 `ahawam_ode.yaml` 默认 1-step；发布的 Flash 结果具体使用哪个 override？
4. real-robot `AHAWAMPolicy` 应如何加载 ODE student checkpoint？是否应将 student 权重作为标准 AHA model 加载并设 `num_inference_steps=2`？
5. 论文所用 TensorRT/CUDA Graph/VAE TRT/video-KV skip-copy runtime 是否计划发布？当前 GitHub 是否缺少该部分？
6. 论文真实机器人 AHA-WAM 是执行完整 action chunk、固定前缀，还是配有 action interpolation？项目页两种 AHA 视频各对应什么正式实验设置？
7. planner refresh 的实际频率、action request 频率和允许的最大 planner K/V age 分别是多少？
8. 双 GPU论文实现是否也把完整 inference state 经 CPU queue 搬运，还是使用 CUDA IPC/共享静态 buffer？
9. 使用两条 TCP connections 时，image channel 重连导致非 dual backend 自动 reset 是否是预期行为？推荐的 session/reset 协议是什么？

---

## 10. 主要证据

- [AHA-WAM 论文](https://arxiv.org/abs/2606.09811)：异步 planner–executor、OVCR、10-step、2-step ODE distillation、真实机器人执行和 latency ablation。
- [AHA-WAM 官方项目页](https://serene-sivy.github.io/aha-wam/)：异步接口、AHA-WAM-Flash 频率，以及有无 action interpolation 的部署视频。
- [AHA-WAM 官方仓库](https://github.com/serene-sivy/AHA-WAM)：公开 async server/client、dual-GPU reference runtime 和 ODE distillation code。
- [RTC 原论文](https://arxiv.org/abs/2506.07339)：执行旧 chunk 时生成新 chunk、冻结 committed prefix 并对其余动作 inpaint 的定义。

本文件对公开代码的判断均基于当前工作树静态审计；没有在本轮加载 7B checkpoint 做 GPU 真机测试。所有涉及 latency、state mutation 和 ODE real-robot loading 的结论，在改生产控制端前都应通过 smoke test 或作者回复确认。
