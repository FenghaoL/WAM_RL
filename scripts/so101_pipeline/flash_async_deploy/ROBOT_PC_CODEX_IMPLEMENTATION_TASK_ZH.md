# 给 SO101 本地控制电脑 Codex 的实现任务书

> 请把本文件完整交给控制电脑上的 Codex。该 Codex 必须先阅读本地现有部署/采集代码，
> 再按下面的数据契约做最小增量修改。不要重写已经稳定工作的相机、SO101 控制、HIL
> 状态机和数据落盘代码。

## 1. 目标和已经确定的服务器接口

目标是在本地 SO101 控制程序中新增一种可切换的 `flash_async` policy mode，对接 GPU
服务器上的 AHA-WAM-Flash 服务。必须保留原有同步模式作为回退和 A/B baseline。

服务端已经完成真实 A100 测试：

- 地址：SSH tunnel 后的 `127.0.0.1:12000`；
- 输入：fixed RGB、wrist RGB、6D model-coordinate state；
- 输出：最多 8 个 6D model-coordinate position targets；
- Action-DiT：Flash student 两步去噪；
- warm 模型计算约 145--153 ms，局域/本机测试 RTT 约 155--157 ms；
- 8 个动作按 15 Hz 执行约 533 ms；
- server async 是后台刷新 Video-DiT K/V，不是 RTC，也不会自动让机器人边执行边请求下一块。

任务文字必须逐字符使用：

```text
Remove the power adapter from the holder and plug it into the power strip.
```

## 2. 修改原则

1. 不删除或覆盖旧的同步 `infer` 客户端，新增 `flash_async` 模式或新 client class。
2. 复用现有相机采集、follower state 读取、`arm_to_model`/`model_to_arm`、动作限幅、
   急停、HIL takeover 和 episode recorder。
3. 第一版只实现安全的 chunk-end request，不做 RTC、action inpainting、动作 blending
   或 eager prefetch。
4. 网络线程绝不能阻塞 15 Hz 机器人控制/HIL 线程；任何网络错误都只能让机器人进入
   HOLD/HUMAN，不能让控制线程卡住。
5. 协议使用 pickle，只能通过可信内网或 SSH tunnel，不得把端口直接暴露到公网。

## 3. 先检查本地代码

实现前先定位以下现有对象，记录文件和函数名：

- 相机线程以及取得同步 `fixed_rgb`、`wrist_rgb` 快照的接口；
- 读取 follower 6D state 的接口；
- 当前同步 policy request 的位置；
- action FIFO/当前 chunk 的消费位置；
- 15 Hz command send loop；
- `SYSTEM_HOLD`、`POLICY`、`HUMAN` 状态和 HIL takeover/release 入口；
- `arm_to_model` 和 `model_to_arm` 的调用位置；
- 当前 episode/reset/reconnect 逻辑；
- raw episode、request ledger、chunk ledger 的写入位置。

先画出这些现有调用关系，再修改。不要因为本任务书中的示例类名与本地不同就新建一套
重复控制栈。

## 4. TCP framing

两条连接使用完全相同的 framing：

```text
4-byte big-endian unsigned payload length + pickle payload
```

如果本地已有 AHA-WAM repo，直接复用：

```python
from deploy.common.tcp_protocol import send_message, recv_message
```

如果本地没有这个模块，可以把这两个短 helper 复制到本地项目；不要改变 framing。
每次 `send_message()` 后必须在同一 socket 上恰好调用一次 `recv_message()`。单个 socket
只能由一个 owner thread 使用，禁止多个线程交叉收发。

## 5. 必须使用两条持久连接

实现一个 `AsyncAHAWAMClient`（名称可依本地风格调整），至少管理：

```python
action_sock       # reset + action_request，严格最多一个 request in flight
image_sock        # image + image_ack
stop_event
image_thread
client_generation # 本地单调递增
request_id        # 本地单调递增
last_image_ack_ns
connected
```

不能共用一条 socket。action request 阻塞等待时，image socket 必须仍能送最新图像，
否则就没有 planner/executor 异步。

连接顺序必须是：

1. 建立 `action_sock`；
2. 在 action socket 上发送一次 episode `reset` 并等待 `reset_ack`；
3. 建立 `image_sock`；
4. 启动 image thread，立即送第一帧；
5. 在至少一个有效 `image_ack` 后才允许请求动作；
6. 第一条合法 action chunk 到达前，机械臂保持 HOLD。

不要因为建立第二条 socket 再 reset；不要每个 action chunk reset。一次 reset 对应一个
policy episode。任一 socket 断开时，关闭两条 socket，进入 HOLD，并以新 generation
整体重连和 reset。

## 6. 图像线程

第一版使用 2 Hz，确认稳定后可测试 5 Hz。不要在本地先拼图，优先发送原始双相机：

```python
message = {
    "type": "image",
    "images": {
        "fixed": fixed_rgb.copy(),  # uint8 HWC，当前系统通常为 480x480x3
        "wrist": wrist_rgb.copy(),  # uint8 HWC，当前系统通常为 480x640x3
    },
    "timestamp": time.time(),
    "capture_monotonic_ns": capture_ns,
    "episode_generation": local_generation,
}
```

服务端负责按训练布局拼成 384×320：fixed 在上，wrist 在左下，右下补黑。image thread
必须等待并检查：

```python
reply["ok"] is True
reply["type"] == "image_ack"
```

相机快照必须从现有 thread-safe latest-frame buffer 中复制，不能让采集线程在 pickle
序列化期间原地改数组。如果图像缺失、dtype/shape 错、ack 超时或 socket 断开，立即通知
主控制状态机进入 HOLD；不能继续无限执行旧 policy chunk。

## 7. 动作请求

动作请求时原子冻结一份 observation snapshot：两张图、与其尽量同步的 follower state、
capture timestamp、本地 generation/request id。请求格式：

```python
message = {
    "type": "action_request",
    "images": {
        "fixed": fixed_rgb.copy(),
        "wrist": wrist_rgb.copy(),
    },
    "state": state_model.astype(np.float32),  # shape (6,)
    "timestamp": time.time(),
    "capture_monotonic_ns": capture_ns,
    "episode_generation": generation,
    "request_id": request_id,
}
```

这里的 `state_model` 必须与旧同步部署发给服务器的数值约定完全相同。不要因为换了协议
再次修改 signs/offsets。通常正确链路应当是：

```text
follower state --arm_to_model 一次--> request state
reply action   --model_to_arm 一次--> follower target
```

绝不能双重转换。

服务端成功回复：

```python
{
    "ok": True,
    "type": "action_chunk",
    "action_chunk": np.ndarray,  # (1..8, 6), model coordinates
    "model_latency_ms": float,
    "kv_version": int,
    "kv_age_ms": float,
}
```

当前单 GPU 服务端不保证回显客户端 `request_id/generation`，因此本地 request worker 必须
把收到的 reply 与发请求时捕获的 `(generation, request_id)` 放在同一个 result object 中。
主线程只接收仍等于当前 generation 的结果。

入 action FIFO 前必须依次检查：

- `ok is True` 且 `type == "action_chunk"`；
- shape 必须是 `(N,6)` 且 `1 <= N <= 8`；
- 所有数值 finite；
- `kv_version > 0`；
- `0 <= kv_age_ms <= 2000`；
- 每关节绝对范围与相邻 target delta 通过已有安全限制；
- reply 所属本地 generation 仍然有效；
- 当前控制权仍为 POLICY，而不是 HUMAN/ESTOP。

任何一项失败都不能部分执行，整块拒绝并进入 HOLD。

## 8. 控制线程与第一版执行模式

保留 15 Hz 控制 tick。不要把 8 个 target 一次性下发：

```text
收到 chunk -> 每 66.7 ms 取一个 action -> model_to_arm -> safety clamp -> follower write
```

第一版 `SYNC_HOLD_SAFE`：

1. 执行当前 chunk 的 8 个 action；
2. FIFO 用完后保持最后一个实际下发的 position target，并记录 `SYSTEM_HOLD`；
3. 用此刻最新 image/state 发下一次 action request；
4. 新块完整通过校验后恢复 `POLICY`。

实测 warm RTT 约 0.155 s，所以该模式可能留下约 2--3 个 15 Hz HOLD tick。这是预期的
安全 baseline，不要删掉这些真实时间，也不要把 HOLD 伪装成 policy action。

第二阶段才允许新增可关闭的 `PREFETCH_AFTER_SLOT_5` 实验：第 5 个动作实际发出后用当时
的新 observation 请求下一块，让余下约 0.20 s 覆盖网络/推理。它会引入约 0.20 s 的
observation-to-execution age，并且不是 RTC。必须在保守模式成功后单独测试和记录，不能
作为第一版默认值。

## 9. HIL/急停必须高于网络和 policy

takeover/ESTOP 触发时同步执行以下操作，不能等待 GPU 回复：

1. 立即停止消费 action FIFO；
2. `client_generation += 1`；
3. 把 FIFO 剩余 slots 标记为 skipped 并清空；
4. 当前 in-flight request 标记 stale；
5. HUMAN/ESTOP 获得唯一 follower 写权限；
6. 关闭或作废阻塞中的 action socket，使晚到 reply 不可能重新进入 FIFO；
7. raw 时间轴继续真实记录 HUMAN/SYSTEM_HOLD，不剪掉等待段。

释放 HUMAN 不代表自动恢复 policy。恢复时必须新建连接、显式 reset、等待新 image ack 和
第一块新动作，再由现有状态机切回 POLICY。任何旧 generation reply 必须零执行。

## 10. 推荐状态机

```text
DISCONNECTED
    -> CONNECTING
    -> HOLD_WAIT_IMAGE_ACK
    -> HOLD_WAIT_FIRST_ACTION
    -> POLICY_EXECUTING
          -> SYSTEM_HOLD_WAIT_ACTION  (FIFO 空)
          -> HUMAN                    (takeover，任何时刻可抢占)
          -> DISCONNECTED             (任一网络/协议错误)

HUMAN -> explicit resume -> CONNECTING（新 generation + 新 reset）
ESTOP -> 只能由现有人工安全流程解除
```

不要让 socket callback 直接写机械臂；只有原来的 15 Hz control owner 能写 follower。

## 11. 数据记录不得退化

在现有 rich-v3/v4-compatible raw 记录基础上，append-only 增加或保留：

- `policy_protocol = flash_async`；
- client generation、local request id；
- request observation frame/capture timestamp；
- request send/reply monotonic timestamp、RTT、server model latency；
- `kv_version`、`kv_age_ms`；
- 完整 returned chunk 和 8 个 slot 的 executed/skipped 状态；
- 每个实际 follower command 的 scheduled/sent timestamp；
- `POLICY/HUMAN/SYSTEM_HOLD`；
- takeover requested、human started/stopped、resume/reset；
- safety clamp/reject reason、socket error。

原有 demo/HIL 数据格式和相机视频不能被覆盖或破坏。若暂时不做新字段，也必须保证旧的
真实控制时间轴和 HUMAN correction 记录保持完整。

## 12. 必须实现的运行模式

至少提供下面四种 CLI/config 模式，默认不能直接全速 autonomous：

1. `network_dry_run`：没有机械臂，假图/零 state，只检查协议；
2. `hold_only`：使用真相机/state 请求动作，但绝不下发动作；
3. `single_chunk_low_speed`：只允许一块，低速/严限幅；
4. `autonomous_sync_hold`：保守 chunk-end request；

`prefetch_after_slot_5` 必须是后续独立显式开关，默认 false。

## 13. 验收测试

提交实现前按顺序完成并保存日志：

1. 双 socket 连接、reset、20 次 HOLD-only action request 均成功；
2. action shape 始终 `(1..8,6)` 且 finite；
3. `kv_version` 随 image stream 增长，2 Hz 时 `kv_age_ms` 通常小于约 500 ms，绝不能
   超过 2000 ms 后继续执行；
4. image socket 断开：控制线程进入 HOLD，旧 chunk 不无限续跑；
5. action socket 断开/超时：进入 HOLD，程序不崩溃；
6. action request in-flight 时按 takeover：晚到 reply 为 stale、执行数为 0；
7. chunk 执行到 3/8 时 takeover：后 5 个 slot 标 skipped，不能执行；
8. release 后旧 generation 不可复活，必须 reset 后才能恢复；
9. 单 chunk 低速测试通过急停、绝对范围和 delta clamp；
10. 至少 5 个 conservative chunks 后，再决定是否测试 slot-5 prefetch。

## 14. 禁止事项

- 不要继续给 async server 发旧的 `type="infer"`；那会退回同步 fallback。
- 不要用同一 TCP connection 同时推图和阻塞等待动作。
- 不要把 fixed 单图重命名为 `front`；会造成训练/部署视觉布局错位。
- 不要每个 chunk reset，否则时序上下文被清空。
- 不要在 response callback 中直接控制机器人。
- 不要收到新 chunk 后回退重放旧动作或已执行 slot。
- 不要为了消除停顿删掉真实 SYSTEM_HOLD 时间。
- 不要把普通预取称作 RTC，也不要在未验证时默认打开预取。

## 15. 本地 Codex 最终应交付

请在完成后向用户明确报告：

1. 修改/新增的文件及每个文件职责；
2. 原同步模式是否仍可运行；
3. 两条 socket、reset、HIL generation fencing 的具体实现位置；
4. state/action 坐标转换各发生几次；
5. 四种运行模式的启动命令；
6. 上述验收测试的实际结果和日志位置；
7. 尚未实现的项目，尤其 RTC/blending/prefetch；
8. 在未通过 HOLD-only 和 single-chunk 测试前，不要声称可以直接全自动运行。

