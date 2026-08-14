# FastWAM 权重审计与部署

## 先说结论

SharePoint 中的 `step_137500.pt` 是原生 FastWAM 权重，不是 AHA-WAM
权重。它使用 RoboTwin 的 3 相机、14 维双臂 state/action 约定：

```text
step: 137500
torch_dtype: torch.bfloat16
action_dim: 14
proprio_dim: 14
aha_only_payload_keys: []
```

本仓库的 `model=fastwam` 可以直接加载并运行它。当前 SO101 服务则使用
`model=ahawam`、6 维 state/action 和 AHA 的异步模块，所以不能只把
`online_deploy/settings.yaml` 中的 `checkpoint_path` 换成这个文件。

## 目录内容

- `check_checkpoint.py`：使用 `torch.load(..., mmap=True)` 检查 checkpoint，
  不把 11.2 GiB 张量实际分配到内存或显存。
- `settings_native_robotwin.yaml`：重建原生 FastWAM RoboTwin 配置：3 相机、
  14 维 state/action、32 步动作 horizon、384x320 组合图像。
- `run_native_robotwin_server.sh`：在 12010 端口启动原生 14 维 FastWAM
  TCP 服务。
- `smoke_test_policy.py`：不开放端口、不连接机械臂，执行一次合成输入推理。

下载文件位于：

```text
checkpoints/FastWAM-SharePoint/step_137500.pt
checkpoints/FastWAM-SharePoint/robotwin_uncond_3cam_384_dataset_stats.json
```

权重的 step、大小、张量结构和维度都与官方 RoboTwin 3 相机发布版一致；
配套 `dataset_stats.json` 与本仓库已有的 RoboTwin stats 的 SHA-256 完全
相同。

## 原生 FastWAM 服务

运行：

```bash
bash scripts/so101_pipeline/fastwam_deploy/run_native_robotwin_server.sh
```

服务协议与现有 AHA-WAM 服务相同，仍然是 length-prefixed pickle TCP，区别是：

- 端口为 `12010`。
- 输入 state 必须是 14 维。
- 图像可以直接提供已经拼好的 `images["front"]`，也可以提供
  `cam_high/cam_left_wrist/cam_right_wrist` 三张图；server 会按上方主相机、
  下方左右腕部相机的官方布局拼成 384x320。
- 输出是 `[最多 8, 14]`；模型内部一次预测完整的 `[32, 14]` horizon，
  server 再按 `MAX_RETURN_ACTIONS=8` 截取。
- FastWAM 推理时只把当前组合图像经过一次 VideoDiT，再用当前帧的视觉 KV
  表征进行 10 步 ActionDiT 去噪。
- 它不生成未来视频，也没有 AHA-WAM 的历史 KV、低频 planner、短 action
  chunk、OVCR 或 offset 推理。

这个服务用于验证权重和原生 FastWAM，不得连接 SO101。启动前的维度检查会
接受 14 维配置；如果把预期维度改为 6，会在模型加载前直接退出。

## 离线实测

```bash
source /home/fenghao/miniconda3/etc/profile.d/conda.sh
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh
CUDA_VISIBLE_DEVICES=0 python \
  scripts/so101_pipeline/fastwam_deploy/smoke_test_policy.py
```

本服务器使用 10 步动作去噪的结果：

```text
action_shape: (32, 14)
action_finite: True
inference_seconds: 1.169
peak_allocated_mib: 13160.78
peak_reserved_mib: 13504.00
```

这是黑色合成图像上的连通性测试，只证明模型加载和推理链路正确，不代表任务
成功率或输出动作适合真实机器人。

官方 checkpoint 没有本地代码中的 `mot.action_branch_embedding`。本地参数初始化
为零，而且只被 AHA chunk-routing 路径读取；原生 FastWAM `infer_action` 不读取
它，所以不会改变上面实测的 FastWAM 推理路径。

## 能否作为 AHA-WAM 权重

可以作为训练初始化，不能当成训练完成的 AHA policy 直接部署。在现有
`full_finetune_zero3/settings.yaml` 中修改：

```yaml
init_checkpoint: checkpoints/FastWAM-SharePoint/step_137500.pt

model:
  checkpoint_shape_adapt: true
```

可以复用：

- 5B VideoDiT。
- ActionDiT 中与动作维度无关的 Transformer 主体。
- 语言交叉注意力、时间嵌入和共享 MoT 参数。

不能直接复用或根本不存在：

- `14 -> 6` 后形状不匹配的 ActionDiT `action_encoder.weight`。
- 形状不匹配的动作 `head.weight` 和 `head.bias`。
- AHA 的 `action_obs_visual_proj`。
- AHA 的 `chunk_obs_query_encoder` 和 MoT chunk-KV editor。
- AHA 的异步历史、horizon-adaptive offset 和 OVCR 行为。

我还把这个 FastWAM checkpoint 实际加载进当前 6 维 AHA 配置做了审计，结果是：

```text
3 个张量因 14-D -> 6-D 形状变化被跳过
mot missing_keys: 305
mot unexpected_keys: 0
proprio_encoder: 14 -> 6，自适应平均池化后加载
action_obs_visual_proj: checkpoint 中不存在
chunk_obs_query_encoder: checkpoint 中不存在
```

`305` 是张量键数量，不是 305 个网络模块。其中包括 3 个动作维度张量、
`action_branch_embedding`，以及 FastWAM 没有的 layer-wise chunk-KV editor 参数。
这说明共享 VideoDiT/ActionDiT 主体确实兼容，但 AHA 的关键异步适配部分不是从该
权重学到的。

因此，把缺失参数补成随机值再另存为“6 维 AHA checkpoint”，只能让文件通过
加载，不能让它突然理解 SO101 的 6 个关节。正确流程是把它设为
`init_checkpoint`，在 SO101 数据上继续微调，然后将训练产出的 AHA checkpoint
写回 `online_deploy/settings.yaml`，继续运行原来的 `run_action_server.sh`。

## 若要训练原生 FastWAM-SO101

另一条路线是保持 `model=fastwam`，使用 SO101 的 6 维数据微调。初始化时同样要
过滤三个 14 维 ActionDiT 张量，并重新学习 6 维动作输入/输出层。训练得到真正的
6 维 FastWAM checkpoint 后，可以继续使用这里新增的 `FastWAMPolicy`，把部署
Hydra override 设成 `model=fastwam` 并使用 SO101 自己的 stats。

在这个 6 维微调权重产生以前，真正能够连接当前 SO101 的仍然只有已经完成微调的
AHA-WAM 服务。

## 参考

- [FastWAM 官方代码](https://github.com/yuantianyuan01/FastWAM)
- [FastWAM 官方 RoboTwin/LIBERO 权重](https://huggingface.co/zrgong/fastwam)
- [Fast-WAM 论文](https://arxiv.org/abs/2603.16666)
- [AHA-WAM 论文](https://arxiv.org/abs/2606.09811)
- [AHA-WAM 仓库提供的 FastWAM 复现权重](https://huggingface.co/SereneC/AHA-WAM-Pretrained)
