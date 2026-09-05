# SO101 控制端交接：AHA-WAM-Flash 两步动作去噪 + 异步 planner/executor

## 0. 服务器端完成状态（2026-09-04）

当前无参数启动器已经锁定 latest Flash run，并解析到：

```text
runs/so101_ahawam_padapter_v2/flash_distillation/
formal-padapterv2-flash-ode2-from-step005328-4xa100-20260903-124708/
model/checkpoints/weights/step_004995.pt
```

启动前会验证 checkpoint 内的 ODE 元数据必须为 16-step teacher，且包含
`0/8/16` 锚点；普通 SFT checkpoint 会被拒绝。实际在 A100-40GB GPU 3 上完成了模型
加载和 raw SO101 双相机协议测试：权重 `missing=0/unexpected=0`，3/3 action request
成功，返回 `(8,6)` finite 动作；warm 后模型耗时约 145--153 ms，RTT 约 155--157 ms。
首次完整冷启动约 326 s，属于模型/权重加载时间，不是在线控制延迟。

服务器启动：

```bash
bash scripts/so101_pipeline/flash_async_deploy/run_action_server_async_flash.sh
```

看到 `Listening on 127.0.0.1:12000` 后，在第二个终端运行无机械臂测试：

```bash
bash scripts/so101_pipeline/flash_async_deploy/smoke_async_protocol.sh
```

只有看到 `RESULT: PASS` 才进入 HOLD-only 真机检查。

## 1. 本次服务器端的确定接口

- 模型仍按标准 AHA-WAM 6D checkpoint 加载，但 checkpoint 必须来自
  `flash_distillation/`；动作采样固定为 **2-step ODE student**。
- 动作维数 6，控制频率沿用训练数据的 15 Hz。
- `action_horizon=40`、`action_chunk_size=8`；服务器每次最多返回 8 条动作，
  即约 0.533 s 的命令。
- 服务地址默认 `127.0.0.1:12000`。pickle 协议不要直接暴露公网；控制电脑通过
  SSH tunnel 访问，例如将本地 12000 转发到 GPU 服务器的 127.0.0.1:12000。
- 这是 AHA 的 Video-DiT K/V refresh / Action-DiT 解耦，不是 RTC action
  inpainting。当前 `phase=video` 部署路径编码干净当前帧并刷新 K/V，不执行完整未来视频
  去噪/解码；动作端对当前 action request 做两步去噪。

## 2. 为什么必须有两条 TCP 连接

一条 `image_conn` 只发送最新图像，另一条 `action_conn` 发送 episode reset 和
动作请求。如果共用单连接，`action_request` 等待回复期间就无法继续送图，Video-DiT
得不到后台观测。服务器已启用 threaded connections，并修复了“每次新连接自动
reset”的问题；**现在只接受控制端显式 episode reset**。

两个连接均使用 `deploy/common/tcp_protocol.py` 的 length-prefixed pickle 协议：
`send_message(sock, dict)`，随后必须 `recv_message(sock)` 取回对应 ack/reply。
同一 socket 不要让多个线程并发读写。

## 3. episode 启动顺序

1. 建立 `action_conn`。
2. 发送：

   ```python
   {"type": "reset", "instruction": TASK_TEXT}
   ```

   等到 `{"ok": True, "type": "reset_ack"}`。
3. 建立 `image_conn` 并启动图像线程。
4. 图像线程立即发第一帧，等待 `image_ack`。冷启动 Video-DiT prefill 尚未完成时，
   动作请求会等待最多 5 秒，因此控制端应把首次动作超时设得更长（建议 300 秒），
   并保持机械臂 HOLD。
5. 得到首个合法 action chunk 后才允许进入 EXECUTING。

不要在建立第二条连接后再自动 reset；不要每个 chunk reset。一次 reset 对应一个
真实 episode。

## 4. 图像通道

建议每次发送原始双相机：

```python
{
    "type": "image",
    "images": {
        "fixed": fixed_rgb_uint8,   # HWC, 480x480x3
        "wrist": wrist_rgb_uint8,   # HWC, 480x640x3
    },
    "timestamp": time.time(),
    "episode_generation": generation,
}
```

服务器会用与训练相同的布局拼成 384x320：fixed 在上；wrist 缩放到左下；右下补黑。
也允许控制端自行发送已经拼好的 `images={"front": uint8[384,320,3]}`，但两条
通道必须使用同一套拼接逻辑，不能一边 raw 双相机、一边不同预处理的 front。

图像采用 latest-wins queue；旧帧被覆盖是预期行为。先以 2--5 Hz 图像推送实测
Video-DiT K/V refresh latency，再调频，不要一开始假定 15/30 Hz prefill 能跑完。每次发送后读取
`image_ack`，断线时进入 HOLD，禁止静默继续旧命令。

## 5. 动作请求和执行

请求必须使用“发请求这一刻”的相机 snapshot 与当前 6D 关节/夹爪 state：

```python
{
    "type": "action_request",
    "images": {"fixed": fixed, "wrist": wrist},
    "state": state_float32_6,
    "timestamp": time.time(),
    "episode_generation": generation,
}
```

成功回复：

```python
{
    "ok": True,
    "type": "action_chunk",
    "action_chunk": float32[<=8, 6],
    "model_latency_ms": ...,
    "kv_version": ...,
    "kv_age_ms": ...,
}
```

执行前必须检查：`ok`、shape、finite、每关节绝对范围、相邻 action delta、generation
仍匹配。以 15 Hz 顺序发送整块动作；切勿把 8 个动作一次性写成同一时刻的目标。
第一版在 chunk 执行完后用最新 observation 请求下一块即可。Flash 的目标是让 2-step
请求尽量短，从而缩短/消除 HOLD；本提交没有把新块和旧块重叠，也没有做 RTC blending。

本机 smoke 的 warm RTT 约 155--157 ms，而 8 个动作以 15 Hz 执行约 533 ms。因此：

- 最保守的 chunk-end request 仍会留下约 0.15 s HOLD，但最适合先验证 student 行为质量；
- 若真机检查稳定，可在旧块第 5 个动作发出后，用**当时新拍摄**的 image/state 预取下块，
  让剩余约 0.20 s 覆盖推理；这只是普通预取，观测到执行约有 0.20 s 延迟，不是 RTC；
- 未做跳变、late reply 和 HIL fencing 测试前，不要直接打开预取模式。

如果 ActionDiT latency 仍大于 0.533 s，可以在第 6--7 个动作时预取下一块，但这会
引入 snapshot 到真正执行之间的延迟。未测量动作误差前不要默认启用，更不能收到新块
后回退重放已执行动作。

## 6. HIL 接管与过期回复 fencing

维护单调递增的 `episode_generation` 和本地 `request_id`。发生以下任一事件立即：

- 停止消费当前 action chunk；
- generation 加一；
- 清空本地待执行/待回复队列；
- 切换 HUMAN 并保持人类控制优先；

事件包括人类接管、急停、相机/state 无效、任一 socket 断开、服务端 error、动作越界。
接管前已经发出的慢回复即使后来成功返回，只要 generation/request_id 已过期就丢弃，
绝不能在 HUMAN 期间或释放接管后补执行。新 episode/恢复自动策略时重新显式 reset，
等待新的首个 prefill。

## 7. 需要记录的时间戳

至少记录：camera capture、image send/ack、action request send/reply、`kv_version`、
`kv_age_ms`、每条 command scheduled/sent、measured state、HIL takeover/release、
generation/request_id。否则无法区分 Flash 动作误差、旧 planner state、网络等待与机械臂
执行器延迟。

## 8. 上机门槛

1. 先运行服务器和 `smoke_async_protocol.sh`，3 次请求均 PASS。
2. HOLD 状态跑 20 个请求，确认 action 形状/范围、`kv_version` 能更新、没有跨连接 reset。
3. 低速、限幅、软垫环境执行 5 个 chunk；急停和人类接管优先级必须已验证。
4. 与原 step_005328 的 10-step 同初始条件对照。蒸馏 student 未经真机成功率验证前，
   “2-step 更快”不能等同于“行为质量不下降”。

## 9. 控制端 Codex 最小实现清单

- 复用现有 TCP framing，不重写协议。
- 新增 image/action 两连接和独立锁；显式 reset；断线 HOLD。
- 复用当前相机采集和 6D state/action 安全限幅。
- 加 generation/request fencing 和日志字段。
- 保持 15 Hz 物理执行；先 chunk-end request，不在第一版加入 RTC/blending。
- 提供 dry-run、HOLD-only、单 chunk 三种测试模式，再开放完整 autonomous。
