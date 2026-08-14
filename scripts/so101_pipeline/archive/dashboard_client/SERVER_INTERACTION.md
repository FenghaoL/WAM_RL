# AHA-WAM 服务器交互契约

这份文档用于让本地 Codex 理解 Windows SO101 Dashboard 与 GPU 服务器之间的固定
契约。不要根据旧 G0.5 客户端重新引入坐标变换或旧相机拼接。

## 服务入口

服务器仓库：

```text
/home/fenghao/workspace/AHA-WAM
```

动作服务：

```bash
bash scripts/so101_pipeline/online_deploy/run_action_server.sh
```

- 服务器地址：`127.0.0.1:12000`
- 只允许通过 SSH local forwarding 访问
- action dim：6
- action chunk：8
- action horizon：40
- 控制频率：15 Hz
- 明确禁用未来视频请求

视频诊断服务：

```bash
bash scripts/so101_pipeline/online_deploy/run_video_server.sh
```

- 服务器地址：`127.0.0.1:12001`
- 支持 `video_preview` 和 `infer_with_video`
- 生成可能持续数分钟，客户端 timeout 使用 1800 秒
- 单卡时不要与动作服务同时运行两个完整模型

Windows 动作通道：

```powershell
ssh -N -L 12000:127.0.0.1:12000 fenghao@chickadee.engin.umich.edu
```

## 序列化协议

协议是受信任回环地址上的 length-prefixed pickle TCP：

```text
4-byte unsigned big-endian payload length
pickle payload
```

请求和响应都是 Python `dict`，图像和向量使用 NumPy array。pickle 不能暴露到公网
或不受信任网络。

推荐会话顺序：

```text
connect -> health -> reset(task) -> infer... -> reset/close
```

切换任务、停止一轮控制或扭矩关闭后都应发送完整 `reset`。服务器会在多次 infer
之间保留 AHA-WAM 的 video prefill、rolling history/KV 和 chunk index；不要为每个
动作块重新建立 TCP 连接或 reset。

## 请求与响应

Health：

```python
{"type": "health"}
```

成功响应至少包含：

```python
{
    "ok": True,
    "type": "health",
    "action_dim": 6,
    "proprio_dim": 6,
    "action_chunk_size": 8,
    "action_horizon": 40,
    "model_image_shape_hwc": [384, 320, 3],
    "text_mode": "cached",
    "video_preview_enabled": False,  # 12001 上为 True
}
```

Reset：

```python
{"type": "reset", "instruction": "Pick up the white block."}
```

成功响应：

```python
{"ok": True, "type": "reset_ack"}
```

Action inference：

```python
{
    "type": "infer",
    "instruction": "Pick up the white block.",
    "state": np.ndarray((6,), dtype=np.float32),
    "images": {"front": np.ndarray((384, 320, 3), dtype=np.uint8)},
}
```

也可以发送原始相机，由服务器做 canonical composition：

```python
"images": {
    "fixed": np.ndarray((480, 480, 3), dtype=np.uint8),
    "wrist": np.ndarray((480, 640, 3), dtype=np.uint8),
}
```

成功响应：

```python
{
    "ok": True,
    "type": "action_chunk",
    "action_chunk": np.ndarray((8, 6), dtype=np.float32),
    "model_latency_ms": float,
    "server_inference_step": int,
    "cuda_memory": dict,
}
```

视频请求把 `type` 改为 `video_preview` 或 `infer_with_video`。前者响应类型为
`future_video`；后者为 `action_chunk_with_video`。`future_video.data` 是 MP4 bytes。
诊断模式返回的动作只用于展示，不应控制机械臂。

## 坐标契约

当前训练和部署配置均为：

```yaml
action_state_transforms: null
action_output_dim: 6
proprio_output_dim: 6
```

因此：

- 请求 state 是 SO101 follower `use_degrees=True` 返回的原生 6D calibrated degrees。
- 响应 action 已由服务器使用本次 SO101 stats 反归一化到同一个 degree frame。
- 客户端必须使用 identity mapping；不要应用旧的
  `[1,-1,1,1,1,1]` signs 或 `[0,90,90,0,0,0]` offsets。
- 维度顺序固定为 `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
  wrist_roll, gripper`。

## 图像契约

模型 front 是 RGB `uint8[384,320,3]`：

```text
rows   0:256   fixed camera, 320x256
rows 256:384   wrist-left 160x128 | black-right 160x128
```

原始 fixed 是 `640x480`，客户端先裁掉右侧 160 像素得到 `480x480`。训练和服务端
都先把每路相机缩放为 `320x240`，再把 fixed 缩放为 `320x256`、wrist 缩放为
`160x128`。发送 `front` 时必须已经包含右下黑块；发送 raw 两路相机时服务器负责。

## 文本缓存

服务器没有在线加载 UMT5，只接受已有 cache 的精确 prompt：

```text
Pick up the white block.
Put the white block on the paper roll.
```

大小写、空格和句号不同都会得到 `No cached text embedding`。

## 本地执行约束

- action chunk 必须严格为 finite `float32[8,6]`。
- 以 15 Hz 顺序执行完整 8 步；服务器推理期间保持当前姿态，不重放旧动作。
- 每 tick 最大关节变化 3 度。
- follower calibration limits 只在 Dashboard 显示，不参与动作裁剪或拒绝；这是当前
  操作者明确选择的本地行为。shape、finite 检查与每 tick 3 度变化限制仍然生效。
- Home 只是一项可选操作，Run 不依赖 Home；任何起始姿态仍应机械上有效，并尽量
  位于训练数据覆盖范围内。
- 真机默认 dry-run。切换到 live motion 前必须人工确认相机布局、状态 frame、动作
  连续性和急停可用。

## 当前服务器资产

部署配置：

```text
scripts/so101_pipeline/online_deploy/settings.yaml
```

当前关键路径：

```text
checkpoint: runs/so101_ahawam_square_v1/full_zero2_offload/
            formal-fastwam-zero2offload-bs4-acc1-4ep-20260813-220551/
            model/checkpoints/weights/step_003440.pt
stats:      runs/so101_ahawam_square_v1/assets/dataset_stats.json
text cache: data/so101_ahawam_square_v1_text_embeds_cache
```
