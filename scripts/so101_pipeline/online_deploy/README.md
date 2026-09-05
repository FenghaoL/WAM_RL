# SO101 在线部署

这个目录提供两种服务：普通动作推理，以及动作加未来视频诊断。两者使用相同模型、
观测历史和动作生成链路，区别只是第二种额外执行视频去噪和 MP4 编码。

## 文件

| 文件 | 作用 |
| --- | --- |
| `settings.yaml` | checkpoint、stats、模型形状和视频参数，主要配置文件 |
| `run_action_server.sh` | GPU 3、端口 12000，只返回动作，当前同步 DAgger 采集使用 |
| `run_video_server.sh` | GPU 0、端口 12001，可同时返回动作和未来视频 |
| `ahawam_tcp_client.py` | 机械臂端客户端、关节坐标转换和双相机拼接 |
| `deployment_preflight.py` | 服务启动前检查 checkpoint、stats 和配置 |

服务核心实现在 `deploy/server/ahawam_policy.py` 和
`deploy/server/wam_policy_server.py`，一般不需要修改。

## 1. 修改配置

编辑 `settings.yaml`，最常改的是：

```yaml
checkpoint_path: runs/.../weights/step_005328.pt
dataset_stats_path: runs/so101_ahawam_padapter_v2/assets/dataset_stats.json
text_embedding_cache_dir: data/so101_ahawam_square_v1_text_embeds_cache
```

checkpoint 的模型形状必须与 `hydra_patch` 一致。当前是 SO101 6D、动作 horizon 40、
chunk 8、历史 6 帧、三槽相机布局。

## 2. 启动服务

正常控制机械臂：

```bash
bash scripts/so101_pipeline/online_deploy/run_action_server.sh
```

观察未来视频：

```bash
bash scripts/so101_pipeline/online_deploy/run_video_server.sh
```

动作脚本当前使用 GPU 3，视频脚本的 GPU 设置以其文件为准，不要同时启动。服务只监听
`127.0.0.1`，Windows 端通过
SSH 隧道访问，例如动作端口：

```powershell
ssh -N -L 12000:127.0.0.1:12000 fenghao@chickadee.engin.umich.edu
```

## 3. 请求和返回

客户端保持一条 TCP 长连接。每条消息是四字节网络序长度加 pickle 字典。pickle 只
适合可信连接，因此必须保持 localhost + SSH tunnel。

动作请求：

```python
{
    "type": "infer",
    "instruction": "Remove the power adapter from the holder and plug it into the power strip.",
    "state": np.ndarray((6,), np.float32),
    "images": {
        "fixed": np.ndarray((H, W, 3), np.uint8),
        "wrist": np.ndarray((H, W, 3), np.uint8),
    },
}
```

返回 `action_chunk`，默认截取为 `(8, 6)`。这些值已经反归一化到训练数据的 G05/model
坐标；写入 SO101 前使用 `model_to_arm()` 转回 follower-arm 角度。

视频服务还接受：

- `video_preview`：只返回 `future_video`，不消费动作 chunk；
- `infer_with_video`：同时返回 `action_chunk` 和 `future_video`，动作可以正常用于控制；
- `reset`：新 episode 或任务变化时清空历史；
- `health`：检查服务状态。

`future_video["data"]` 是 MP4 bytes，同时包含 `frame_count`、`fps` 和生成延迟。客户端
的命令行入口可用于单张图片连通测试，但真实控制循环应直接导入 `AHAWAMTCPClient`。
