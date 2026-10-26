# SO101 在线部署

这个目录提供两种服务：普通动作推理，以及动作加未来视频诊断。两者使用相同模型、
观测历史和动作生成链路，区别只是第二种额外执行视频去噪和 MP4 编码。

## 文件

| 文件 | 作用 |
| --- | --- |
| `settings.yaml` | checkpoint、stats、模型形状和视频参数，主要配置文件 |
| `run_action_server.sh` | 同步动作服务；取消注释一行 `DEPLOY_CONFIG` 或传入 YAML 路径选模型，GPU/端口/指令及策略类读自 YAML 的 `serving:` |
| `run_video_server.sh` | GPU 0、端口 12001，可同时返回动作和未来视频 |
| `ahawam_tcp_client.py` | 机械臂端客户端、关节坐标转换和双相机拼接 |
| `deployment_preflight.py` | 服务启动前检查 checkpoint、stats 和配置 |

服务核心实现在 `deploy/server/ahawam_policy.py` 和
`deploy/server/wam_policy_server.py`，一般不需要修改。

离线 RL 三项对比使用新增的 `deploy/server/offline_rl_policy.py`。它复用相机、坐标与 TCP 协议，负责加载外部 actor 或 Action DiT LoRA，按训练的单 8 动作条件推理；不支持异步/未来视频服务。详细说明见 [离线 RL 部署](../../../docs/offline_rl/16_ONLINE_DEPLOYMENT_ZH.md)。

## 离线 RL 三项对比

| YAML（均在 `deployments/`） | 策略权重 | GPU | 端口 |
| --- | --- | ---: | ---: |
| `cylinder_offline_iql.yaml` | IQL 外部 actor，10,000步 | 0 | 12130 |
| `cylinder_offline_td3_bc.yaml` | TD3+BC 外部 actor，10,000步 | 1 | 12131 |
| `cylinder_offline_iql_action_dit.yaml` | IQL 加权 FM 的 Action DiT LoRA，2,000步 | 2 | 12132 |

三个终端分别启动，不用来回修改脚本：

```bash
bash scripts/so101_pipeline/online_deploy/run_action_server.sh scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_iql.yaml
bash scripts/so101_pipeline/online_deploy/run_action_server.sh scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_td3_bc.yaml
bash scripts/so101_pipeline/online_deploy/run_action_server.sh scripts/so101_pipeline/online_deploy/deployments/cylinder_offline_iql_action_dit.yaml
```

机器人端沿用 `AHAWAMTCPClient`、`arm_to_model`、`model_to_arm`，切换服务端口即可。请求指令为 `Pick up the cylinder and insert it into the matching hole.`，每次返回 `(8,6)` 的绝对 G05/model 坐标目标。每次假设执行完整8动作；新 episode、人工接管或中途丢弃部分动作块后，需要重新 `reset`，避免沿用错误的历史。

SSH 转发可在同一连接中带三个端口：

```powershell
ssh -N -L 12130:127.0.0.1:12130 -L 12131:127.0.0.1:12131 -L 12132:127.0.0.1:12132 fenghao@chickadee.engin.umich.edu
```

三个方案均使用采集用 SFT `step_006572.pt`；YAML 顶层 `checkpoint_path` 是完整基座，`offline_rl.checkpoint_path` 是小 actor 或 LoRA sidecar，不能将二者互换。每次请求刷新独立 planner 条件、使用 offset=0 / chunk index=0。现有客户端每 chunk 只提供一个观察，因此 history 按每48个请求执行动作取样，无法重建含 HOLD 的原始控制 tick；这是明确的在线近似，见详细文档。

## 1. 修改配置

编辑 `settings.yaml`，最常改的是：

```yaml
checkpoint_path: runs/.../weights/step_006572.pt
dataset_stats_path: runs/so101_ahawam_cylinder_v1/assets/dataset_stats.json
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

GPU 以所选 YAML 的 `serving.gpu_id` 为准；视频脚本另有 GPU 设置，同一 GPU 的服务需考虑显存。服务只监听
`127.0.0.1`，Windows 端通过
SSH 隧道访问，端口换成所选 yaml 的 `serving.port`，例如 cylinder SFT：

```powershell
ssh -N -L 12100:127.0.0.1:12100 fenghao@chickadee.engin.umich.edu
```

## 3. 请求和返回

客户端保持一条 TCP 长连接。每条消息是四字节网络序长度加 pickle 字典。pickle 只
适合可信连接，因此必须保持 localhost + SSH tunnel。

动作请求：

```python
{
    "type": "infer",
    "instruction": "Pick up the cylinder and insert it into the matching hole.",
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
