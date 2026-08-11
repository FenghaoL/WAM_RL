# AHA-WAM 代码入门与训练/部署指南

本文档按“完全新手也能顺着走”的方式解释这个 repo。目标场景是：

- 你已经创建了 `conda` 环境 `ahawam`，并安装了 `torch==2.7.1+cu128`、`torchvision==0.22.1+cu128`、`pip install -e .`、`ffmpeg`。
- 服务器因为驱动/库版本更新，需要先执行 `source /home/fenghao/use-nvidia-580159.sh` 才能让 Python 看到 GPU。
- 你最终想把 AHA-WAM 微调后，通过 TCP/ROS 一类的通道部署到自己的 SO101 机械臂。
- 你的 SO101 有一个全局摄像机和一个腕部摄像机。

这份文档不是论文解读，而是“怎么读这个代码、该跑哪些命令、哪些地方要改”的工程指南。

## 0. 先说结论

当前 repo 自带的主线是 RoboTwin2.0 双臂数据和双臂部署模板，不是 SO101 的即插即用工程。

你可以直接试 repo 自带训练流程，但前提是准备好：

- Wan2.2-TI2V-5B 相关大模型组件。
- RoboTwin2.0 的 LeRobot 格式数据。
- 文本 embedding cache。
- `dataset_stats.json`。
- ActionDiT backbone 或完整的 AHA-WAM 初始化 checkpoint。

服务器 GPU 情况很好。提权实测后可见：

```text
4 x NVIDIA A100-SXM4-40GB
driver_version: 580.159.03
torch: 2.7.1+cu128
torch.cuda.is_available(): True
torch.cuda.device_count(): 4
```

所以 4*A100 40GB 足够作为 AHA-WAM 的单机多卡训练/评估起点。真正会不会 OOM，取决于 batch size、是否开启 gradient checkpointing、任务配置、数据尺寸和 ZeRO stage。建议先从小 batch、小步数 smoke test 开始。

训练不是只有“从零全量训练”。它支持：

- 从 Wan + ActionDiT backbone 初始化后训练。
- 用 `init_checkpoint` 从发布的完整 checkpoint 开始微调。
- 用 `resume` 从中断训练继续。
- 用 `freeze_video_dit=true` 冻住视频 DiT，只训练 action 相关模块和额外模块。
- ODE distillation，把训练好的 teacher 蒸馏成更快的 student。

但它没有内置 LoRA/QLoRA/adapter 这类参数高效微调脚本。默认训练方式仍然是全参或“冻一部分模块”的训练。

对 SO101 来说，最推荐的路线是：

1. 先在 repo 自带 RoboTwin 数据上跑通小步训练，验证环境、GPU、checkpoint 保存都正常。
2. 再把 SO101 数据整理成 LeRobot 格式。
3. 新增 SO101 的 `configs/data/*.yaml` 和 `configs/task/*.yaml`。
4. 用 RoboCOIN-pretrained 的 `AHA-WAM-pretrained.pt` 做 `init_checkpoint` 微调。
5. 修改部署 wrapper，让它支持“全局摄像机 + 腕部摄像机”和 SO101 的 state/action 维度。
6. 先 dummy client，再 dry-run，再低速实机闭环。

## 1. 每次开新终端先做什么

建议每个终端都先执行：

```bash
cd /home/fenghao/workspace/AHA-WAM
source /home/fenghao/use-nvidia-580159.sh
conda activate ahawam
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

解释：

- `cd ...`：进入 repo 根目录。
- `source /home/fenghao/use-nvidia-580159.sh`：使用服务器当前能用的 NVIDIA 580.159.03 兼容库，否则 Python 可能看不到 GPU。
- `conda activate ahawam`：进入你已经装好依赖的 Python 环境。
- `export PYTHONPATH=...`：让 `deploy/`、`experiments/` 这些没有被 `pip install -e .` 安装进去的顶层目录也能被 Python import 到。

验证命令：

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
PY
```

如果你在 Codex/沙箱里直接跑，可能看不到 GPU；但提权实测服务器本身是能看到 4 张 A100 的。所以你在真实 shell 训练时，以真实 shell 的结果为准。

## 2. 这个 repo 在做什么

AHA-WAM 是一个机器人操作策略模型。它把“看视频、理解未来世界变化”和“快速输出机器人动作”拆成两个节奏：

- video DiT：慢一些，负责长时域世界/视觉上下文，相当于 planner。
- action DiT：快一些，负责短动作 chunk，相当于 executor。
- OVCR：用最新 observation 修正/路由已经缓存的视觉上下文，让动作分支不用每一步都重跑完整 video DiT。
- offset training：训练时故意让视觉上下文和动作时刻有偏移，增强异步部署鲁棒性。
- ODE distillation：把多步 action denoise 的 teacher 蒸馏成更快的 student。

你可以把它想成：

```text
语言指令 + 摄像机图像 + 机器人状态
        -> AHA-WAM
        -> 一段未来动作 action chunk
        -> 机器人执行其中若干步后再请求下一段
```

## 3. 目录怎么读

主目录：

```text
configs/
  train.yaml                 基础训练配置
  data/robotwin.yaml          RoboTwin 数据配置
  model/*.yaml                AHA-WAM/Fast-WAM/ODE 模型配置
  task/*.yaml                 训练任务配置
  sim_robotwin.yaml           RoboTwin 仿真评估配置
  deploy.yml                  真机部署 Hydra 配置

scripts/
  train.py                    训练入口，Hydra 配置驱动
  train_zero1.sh              DeepSpeed ZeRO-1 启动脚本
  train_zero2.sh              DeepSpeed ZeRO-2 启动脚本
  precompute_text_embeds.py   预计算文本 embedding
  preprocess_action_dit_backbone.py
                              从 Wan video DiT 生成 ActionDiT backbone

experiments/robotwin/
  run_robotwin_manager.py     多任务 RoboTwin 评估管理器
  eval_robotwin_single.py     单任务 RoboTwin 评估入口
  ahawam_policy/              RoboTwin policy adapter

deploy/
  README.md                   真机部署说明
  deploy_example.yml          真机部署配置模板
  server/wam_policy_server.py TCP policy server
  server/ahawam_policy.py     AHA-WAM 部署 wrapper
  client/wam_policy_dummy_client.py
                              不依赖 ROS 的同步测试 client
  client/wam_policy_async_dummy_client.py
                              不依赖 ROS 的异步测试 client
  client/wam_remote_client_node.py
                              ROS 机器人侧 client 模板

src/ahawam/
  runtime.py                  创建模型、构建数据集、进入训练
  trainer.py                  训练循环、checkpoint、resume
  datasets/                   LeRobot 数据读取和预处理
  models/wan22/               模型结构、推理、checkpoint 逻辑

third_party/RoboTwin/
  vendored RoboTwin 代码       仿真环境依赖/参考，不是 AHA-WAM 主训练入口
```

## 4. README 后半部分逐段解释

### 4.1 Environment

README 里的环境安装你已经做了：

```bash
conda create -n ahawam python=3.10 -y
conda activate ahawam
pip install -U pip setuptools wheel
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
conda install -c conda-forge ffmpeg -y
```

这里每一步的作用：

- `python=3.10`：repo 要求 Python `>=3.10,<3.11`。
- `torch==2.7.1+cu128`：匹配 CUDA 12.8 的 PyTorch。
- `pip install -e .`：把 `src/ahawam` 安装成可 import 的包，同时安装 `pyproject.toml` 里的依赖，如 `accelerate`、`deepspeed`、`hydra-core`、`modelscope`、`huggingface-hub`。
- `ffmpeg`：处理视频数据和 mp4。

注意：`pip install -e .` 只安装 `src/ahawam`，不会安装顶层的 `deploy` 包。所以跑部署脚本时加上：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

或者用：

```bash
python -m deploy.server.wam_policy_server --help
```

### 4.2 Model Assets & Checkpoints

README 说有三类模型资产，实际可以理解成四类。

第一类：Wan base components。

这些是 AHA-WAM 底座需要的大模型组件：

```text
Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors
DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors
DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/
```

它们分别大致对应：

- video DiT 权重。
- 文本 encoder。
- VAE，负责图像/视频和 latent 之间的编码解码。
- tokenizer。

默认代码会用 DiffSynth/ModelScope 风格的 loader 自动下载，默认缓存到 `./checkpoints/` 下。你也可以设置：

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/wan_models
export DIFFSYNTH_DOWNLOAD_SOURCE=modelscope
```

如果服务器不能联网，提前把文件放到对应目录，然后：

```bash
export DIFFSYNTH_SKIP_DOWNLOAD=true
```

什么时候你需要它？

- 从零训练需要。
- 生成 ActionDiT backbone 需要。
- 评估/部署完整 AHA-WAM checkpoint 通常仍需要 VAE/text encoder/tokenizer，video/action DiT 权重会从 checkpoint 加载。

第二类：ActionDiT backbone。

命令：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/ahawam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

它做什么：

- 读取 Wan video DiT。
- 创建一个 action DiT。
- 把 video DiT 中能迁移的 backbone 权重复制/插值到 action DiT。
- 保存成 `.pt` 文件，供训练时 `model.action_dit_pretrained_path` 使用。

什么时候需要？

- 你从 Wan 初始化 AHA-WAM 或 Fast-WAM 训练时需要。
- 如果你使用一个完整的 AHA-WAM checkpoint 做 `init_checkpoint`，并且该 checkpoint 已包含 action DiT 权重，可以把 `model.skip_dit_load_from_pretrain=true` 和 `model.action_dit_pretrained_path=null` 一起传入，避免先读取不存在的 backbone。

第三类：RoboTwin checkpoint。

命令：

```bash
huggingface-cli download SereneC/AHA-WAM-RoboTwin2.0 \
  robotwin_ahawam.pt dataset_stats.json \
  --local-dir checkpoints/AHA-WAM-RoboTwin2.0
```

它是什么：

- 已经在 RoboTwin2.0 上训练好的 AHA-WAM 任务 checkpoint。
- `dataset_stats.json` 是训练数据的 action/state 归一化统计，部署和评估必须用。

什么时候用？

- 想快速跑 RoboTwin 评估。
- 想 smoke-test 部署 server 和 dummy client。
- 不建议直接拿它上 SO101 真机，因为它是 RoboTwin 双臂 14 维动作和三相机输入分布。

第四类：RoboCOIN pretrained checkpoint。

命令：

```bash
huggingface-cli download SereneC/AHA-WAM-Pretrained \
  AHA-WAM-pretrained.pt Fast-WAM-pretrained.pt \
  --local-dir checkpoints/AHA-WAM-Pretrained
```

它是什么：

- 在 RoboCOIN 子集上预训练过 45k steps 的初始化权重。

什么时候用？

- 你要迁移到自己的 SO101 数据时，它比 RoboTwin task checkpoint 更适合作为下游微调初始化。
- 用法是训练时传 `init_checkpoint=...`。

### 4.3 Data Preparation

repo 默认训练数据是 LeRobot 风格的 RoboTwin2.0 数据，不是普通图片文件夹。

README 给了两种方式。

Option A：下载别人预处理好的 RoboTwin2.0 数据。

```bash
huggingface-cli download yuanty/robotwin2.0-fastwam \
  --repo-type dataset \
  --local-dir data/robotwin2.0-fastwam

cd data/robotwin2.0-fastwam
cat robotwin2.0.tar.gz.part-* | tar -xzf -
cd -
```

解压后期望结构类似：

```text
data/robotwin2.0/
└── robotwin2.0/
    ├── data/
    ├── meta/
    └── videos/
```

Option B：从原始 RoboTwin 数据转换。

代码要求的 key 来自 `configs/data/robotwin.yaml`：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.state.default        # 14-D
action.default                   # 14-D
task
```

LeRobot 元信息通常包括：

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
data/*.parquet
videos/**/*.mp4
```

注意：

- AHA-WAM 训练时会读 `meta/tasks.jsonl`，把 task 文本转成 prompt。
- 数据加载时会读取 `text_embedding_cache_dir` 中预计算好的文本 embedding。
- 如果没有 `dataset_stats.json`，训练数据集会尝试计算 stats。多卡新手阶段建议先用已有 stats 或先单进程生成，避免多进程时序问题。

### 4.4 Precompute Text Embeddings

命令：

```bash
python scripts/precompute_text_embeds.py task=robotwin_ahawam
```

offset 版本：

```bash
python scripts/precompute_text_embeds.py task=robotwin_ahawam_offset
```

多 GPU：

```bash
torchrun --standalone --nproc_per_node=4 \
  scripts/precompute_text_embeds.py task=robotwin_ahawam
```

它做什么：

- 扫描数据集的 `meta/tasks.jsonl`。
- 把每个 task 包装成 prompt：

```text
A video recorded from a robot's point of view executing the following instruction: {task}
```

- 用 Wan/T5 文本 encoder 编成 embedding。
- 存到 `data.train.text_embedding_cache_dir` 和 `data.val.text_embedding_cache_dir`。

它是不是必须？

对当前训练流程来说，基本必须。`RobotVideoDataset` 在 `__getitem__` 时会按 prompt hash 去 cache 目录找：

```text
<sha256>.t5_len128.wan22ti2v5b.pt
```

找不到就报错：

```text
Missing text embedding cache
```

### 4.5 Training

README 默认命令：

```bash
bash scripts/train_zero1.sh 8
```

作者默认写的是 8 卡。你现在是 4 卡，所以单机起步应改成：

```bash
bash scripts/train_zero1.sh 4 task=robotwin_ahawam
```

这个脚本做什么：

- 自动补默认 `task=robotwin_ahawam`。
- 自动从 task yaml 里解析默认 model。
- 调用 `accelerate launch`。
- 使用 `scripts/accelerate_configs/accelerate_zero1_ds.yaml`。
- DeepSpeed config 是 `scripts/ds_configs/ds_zero1_config.json`。

ZeRO-1 和 ZeRO-2 的区别：

- ZeRO-1：切 optimizer state，通常更稳，通信相对少。
- ZeRO-2：进一步切 gradient，可能省显存，但通信和兼容性压力更大。
- 这个 repo 的 `train_zero1.sh` 多机逻辑更完整。`train_zero2.sh` 更适合先单机试，不建议新手一上来用它跑多机。

## 5. 这个 repo 自带哪些训练/推理入口

### 5.1 训练入口

主训练入口：

```bash
python scripts/train.py task=robotwin_ahawam model=ahawam
```

更推荐用启动脚本：

```bash
bash scripts/train_zero1.sh 4 task=robotwin_ahawam
bash scripts/train_zero2.sh 4 task=robotwin_ahawam
```

可选 task：

```text
robotwin_ahawam
robotwin_ahawam_offset
robotwin_ahawam_ode
robotwin_ahawam_offset_ode
```

可选 model：

```text
ahawam
fastwam
ahawam_ode
fastwam_ode
```

Hydra help：

```bash
python scripts/train.py --help
```

我实测 help 能看到这些配置组：

```text
data: robotwin
model: ahawam, ahawam_ode, fastwam, fastwam_ode
task: robotwin_ahawam, robotwin_ahawam_ode, robotwin_ahawam_offset, robotwin_ahawam_offset_ode
```

### 5.2 数据/预处理入口

文本 embedding：

```bash
python scripts/precompute_text_embeds.py task=robotwin_ahawam
```

ActionDiT backbone：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/ahawam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

### 5.3 RoboTwin 评估入口

单任务：

```bash
PYTHONPATH=. python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_ahawam \
  ckpt=checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json \
  EVALUATION.robotwin_root=/path/to/RoboTwin \
  gpu_id=0
```

多任务 manager：

```bash
PYTHONPATH=. python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_ahawam \
  ckpt=checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt \
  EVALUATION.dataset_stats_path=checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json \
  EVALUATION.robotwin_root=/path/to/RoboTwin \
  MULTIRUN.num_gpus=4
```

注意：

- `third_party/RoboTwin` 是 vendored 代码，但我检查到当前 vendored 目录里没有完整 `task_config/_eval_step_limit.yml`。所以如果直接用 manager，可能需要按 RoboTwin 官方方式准备完整 RoboTwin checkout 和资产。
- README 的 symlink 命令是把 `experiments/robotwin/ahawam_policy` 链到 RoboTwin 的 `policy/` 目录里。

### 5.4 真机/部署推理入口

部署 server：

```bash
PYTHONPATH=. python deploy/server/wam_policy_server.py \
  --policy-path deploy/deploy.yml \
  --policy-module deploy.server.ahawam_policy \
  --policy-class AHAWAMPolicy \
  --instruction "your task instruction" \
  --action-dim 14 \
  --host 0.0.0.0 \
  --port 10000
```

同步 dummy client：

```bash
PYTHONPATH=. python deploy/client/wam_policy_dummy_client.py \
  --host 127.0.0.1 \
  --port 10000 \
  --instruction "your task instruction" \
  --num-requests 1
```

异步 server，单进程：

```bash
PYTHONPATH=. python deploy/server/wam_policy_server.py \
  --async-mode \
  --policy-path deploy/deploy.yml \
  --policy-module deploy.server.ahawam_policy \
  --policy-class AHAWAMPolicy \
  --instruction "your task instruction" \
  --action-dim 14
```

双 GPU 异步 server：

```bash
PYTHONPATH=. python deploy/server/wam_policy_server.py \
  --dual-gpu-async \
  --video-gpu 0 \
  --action-gpu 1 \
  --policy-path deploy/deploy.yml \
  --policy-module deploy.server.ahawam_policy \
  --policy-class AHAWAMPolicy \
  --instruction "your task instruction" \
  --action-dim 14
```

异步 dummy client：

```bash
PYTHONPATH=. python deploy/client/wam_policy_async_dummy_client.py \
  --host 127.0.0.1 \
  --port 10000 \
  --instruction "your task instruction" \
  --num-action-requests 10
```

ROS client 模板：

```bash
PYTHONPATH=. python deploy/client/wam_remote_client_node.py \
  --server-ip <GPU_WORKSTATION_IP> \
  --server-port 10000 \
  --instruction "your task instruction" \
  --dry-run
```

异步 ROS client：

```bash
PYTHONPATH=. python deploy/client/wam_remote_client_node.py \
  --server-ip <GPU_WORKSTATION_IP> \
  --server-port 10000 \
  --instruction "your task instruction" \
  --async-policy-protocol \
  --image-stream-rate 30 \
  --dry-run
```

重要限制：

- 当前 `deploy/server/ahawam_policy.py` 只使用 `img_arr[0]`，也就是只用一张 front image。
- 当前 `deploy/client/wam_remote_client_node.py` 是双臂 14 维模板，订阅 left/right puppet joint，发布 left/right command。
- 所以它不是 SO101 现成部署脚本，只是 TCP/ROS 架构模板。

## 6. 默认训练用的是什么数据集

默认数据配置是 `configs/data/robotwin.yaml`。

默认输入：

```yaml
images:
  - cam_high
  - cam_left_wrist
  - cam_right_wrist
state:
  - default  # 14-D
action:
  - default  # 14-D
```

默认相机处理：

- 每个相机原始 shape 认为是 `[3, 480, 640]`。
- transform 后是 `[3, 240, 320]`。
- `concat_multi_camera: robotwin`。
- 三个相机被拼成一张 `[384, 320]` 的图：
  - 顶部：head/high camera resize 到 `256x320`。
  - 底部：两个 wrist camera 各 resize 到 `128x160` 后左右拼接。

默认时序：

- `num_frames: 65`。
- `action_video_freq_ratio: 8`。
- 注释里说是 32 action, 9 video frames。
- action/state 是 14 维，符合 RoboTwin 双臂配置。

对 SO101 意味着：

- 你不能直接复用 `cam_high + left_wrist + right_wrist`。
- 你大概率要改成 `global + wrist` 两个 camera key。
- `concat_multi_camera` 可以考虑 `horizontal` 或 `vertical`，也可以自己写 SO101 专属拼接。
- `action.default` 和 `observation.state.default` 的 raw_shape/shape 要改成 SO101 实际维度。
- `processor.action_output_dim` 和 `processor.proprio_output_dim` 也要跟着改。

## 7. 推荐的 RoboTwin smoke test 路线

先不要一上来跑 5 epoch 全量训练。先验证路径、权重、数据、显存。

### 7.1 准备路径变量

```bash
export ROBOTWIN_DATA=/abs/path/to/data/robotwin2.0/robotwin2.0
export TEXT_CACHE=/abs/path/to/data/robotwin2.0/text_embeds_cache
export ACTION_DIT=checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
export OUT=runs/smoke_robotwin_ahawam
```

### 7.2 如果没有 ActionDiT backbone，先生成

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/ahawam.yaml \
  --output "$ACTION_DIT" \
  --device cuda \
  --dtype bfloat16
```

### 7.3 预计算文本 embedding

```bash
python scripts/precompute_text_embeds.py \
  task=robotwin_ahawam \
  data.train.dataset_dirs=[$ROBOTWIN_DATA] \
  data.val.dataset_dirs=[$ROBOTWIN_DATA] \
  data.train.text_embedding_cache_dir=$TEXT_CACHE \
  data.val.text_embedding_cache_dir=$TEXT_CACHE
```

如果 shell 对方括号解析不舒服，可以给参数加引号：

```bash
python scripts/precompute_text_embeds.py \
  task=robotwin_ahawam \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  "data.val.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.train.text_embedding_cache_dir=$TEXT_CACHE \
  data.val.text_embedding_cache_dir=$TEXT_CACHE
```

### 7.4 小步训练

如果你已经有 RoboTwin 的 `dataset_stats.json`：

```bash
bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam \
  model.action_dit_pretrained_path=$ACTION_DIT \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  "data.val.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.train.pretrained_norm_stats=/abs/path/to/dataset_stats.json \
  data.val.pretrained_norm_stats=/abs/path/to/dataset_stats.json \
  data.train.text_embedding_cache_dir=$TEXT_CACHE \
  data.val.text_embedding_cache_dir=$TEXT_CACHE \
  output_dir=$OUT \
  wandb.enabled=false \
  batch_size=1 \
  num_workers=4 \
  max_steps=5 \
  save_every=5 \
  eval_every=0
```

如果这个能跑完，再逐步：

- `max_steps=100`。
- `batch_size=2`。
- `batch_size=4`。
- 最后考虑回到 task 默认 `batch_size=6`。

如果 OOM：

```bash
batch_size=1
gradient_accumulation_steps=2
model.mot_checkpoint_mixed_attn=true
```

`gradient_accumulation_steps=2` 的意思是显存上仍然每次只吃小 batch，但累积两次梯度再更新一次，等效全局 batch 变大。

## 8. 4*A100 是否够，分布式要改什么

### 8.1 单机 4 卡

你不需要改 accelerate 配置文件。直接把第一个参数写成 `4`：

```bash
bash scripts/train_zero1.sh 4 task=robotwin_ahawam
```

脚本会调用：

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero1_ds.yaml \
  --num_processes 4 \
  --num_machines 1 \
  ...
```

### 8.2 多机

多机才需要设置：

```bash
export NNODES=2
export NODE_RANK=0                 # 第二台写 1
export MASTER_ADDR=<rank0_ip>
export MASTER_PORT=29500
export DEEPSPEED_HOSTFILE=/path/to/hostfile
export RUN_ID=some-shared-run-id
```

hostfile 示例：

```text
192.168.254.114 slots=4
192.168.254.116 slots=4
```

然后每台机器运行同一个命令：

```bash
bash scripts/train_zero1.sh 4 task=robotwin_ahawam
```

你目前只有一台 4*A100，不需要这些多机变量。

### 8.3 ZeRO-1 vs ZeRO-2 怎么选

建议顺序：

1. 先 `train_zero1.sh`。
2. 如果显存不够，再试 `train_zero2.sh`。
3. 如果 ZeRO-2 不稳定，回到 ZeRO-1 并降低 batch size 或开启 gradient checkpointing。

## 9. checkpoint、init、resume 怎么理解

训练会保存两类 checkpoint：

```text
<output_dir>/checkpoints/weights/step_XXXXXX.pt
<output_dir>/checkpoints/state/step_XXXXXX/
```

区别：

- `weights/step_XXXXXX.pt`：模型权重，用于评估、部署、作为 `init_checkpoint`。
- `state/step_XXXXXX/`：Accelerate/DeepSpeed 训练状态，用于真正断点续训。

从预训练权重开始一个新训练：

```bash
bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam \
  init_checkpoint=checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt
```

建议使用完整 checkpoint 初始化时加上：

```bash
model.skip_dit_load_from_pretrain=true
model.action_dit_pretrained_path=null
```

这样可以避免配置里残留 `path/to/ActionDiT...` 导致还没 load checkpoint 就失败。

从中断训练继续：

```bash
bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam \
  resume=/path/to/output_dir/checkpoints/state/step_002500
```

如果你只有 `.pt` 权重，也可以：

```bash
bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam \
  resume=/path/to/output_dir/checkpoints/weights/step_002500.pt
```

但 `.pt` 是 weight-only resume，optimizer/scheduler 状态不能完全恢复。

## 10. 是否只有全量微调

不是只有“从零训练”，但默认没有 LoRA。

代码里训练前会执行类似逻辑：

- `model.requires_grad_(False)`。
- `model.dit.requires_grad_(True)`。
- 如果 `freeze_video_dit=true`，再把 video DiT 冻住。
- 额外训练 `get_additional_trainable_modules()` 返回的模块，例如 proprio encoder、AHA-WAM 的 chunk/context routing 相关模块。

可用策略：

### 10.1 默认全参训练/微调

```bash
bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam \
  init_checkpoint=checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt
```

### 10.2 冻住 video DiT

```bash
bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam_offset \
  init_checkpoint=checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt \
  model.skip_dit_load_from_pretrain=true \
  model.action_dit_pretrained_path=null \
  freeze_video_dit=true
```

这更适合：

- 数据量比较小。
- 想降低显存和过拟合风险。
- 真机初期只想让 action executor 适应自己的机器人。

### 10.3 ODE distillation

ODE 任务：

```text
robotwin_ahawam_ode
robotwin_ahawam_offset_ode
```

它需要 teacher checkpoint：

```yaml
model.teacher.checkpoint_path: path/to/step_045000.pt
```

不要一开始就跑 ODE。它更像第二阶段加速：

1. 先训练/微调一个普通 AHA-WAM teacher。
2. 确认 teacher 在评估或真机 dry-run 上可用。
3. 再用 ODE 配置训练 student。

## 11. SO101 适配要准备什么

这是你的最终目标，也是工作量最大的部分。

### 11.1 数据

你需要采集 SO101 的 demonstration 数据，并整理成 LeRobot 风格。

最少需要每个时间步包含：

```text
observation.images.global
observation.images.wrist
observation.state.default
action.default
task
```

具体 state/action 维度取决于你的控制方式：

- 如果用 joint position 控制：state/action 可能是关节角 + gripper。
- 如果用 delta joint：action 可以是相对当前位置的增量。
- 如果用末端位姿：action 可能是 xyz + rotation + gripper。

关键原则：

- 训练时 `action.default` 表示什么，部署时 server 输出的 action 就必须按同样含义执行。
- 训练时 state/action 怎么归一化，部署时必须用同一个 `dataset_stats.json` 反归一化。
- 训练时相机怎么拼图，部署时必须同样拼图。

### 11.2 新增 SO101 data config

建议新增：

```text
configs/data/so101.yaml
```

大概形态：

```yaml
train:
  _target_: ahawam.datasets.lerobot.robot_video_dataset.RobotVideoDataset
  dataset_dirs:
    - /path/to/so101_lerobot
  shape_meta:
    images:
      - key: global
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
      - key: wrist
        raw_shape: [3, 480, 640]
        shape: [3, 240, 320]
    action:
      - key: default
        raw_shape: <SO101_ACTION_DIM>
        shape: <SO101_ACTION_DIM>
    state:
      - key: default
        raw_shape: <SO101_STATE_DIM>
        shape: <SO101_STATE_DIM>
  num_frames: 65
  global_sample_stride: 1
  action_video_freq_ratio: 8
  video_size: [384, 320]
  concat_multi_camera: horizontal
  pretrained_norm_stats: /path/to/so101_dataset_stats.json
  text_embedding_cache_dir: /path/to/so101_text_embeds_cache
  processor:
    _target_: ahawam.datasets.lerobot.processors.ahawam_processor.AHAWAMProcessor
    shape_meta: ${data.train.shape_meta}
    num_obs_steps: ${data.train.num_frames}
    num_output_cameras: 2
    action_output_dim: <SO101_ACTION_DIM>
    proprio_output_dim: <SO101_STATE_DIM>
    action_state_transforms: null
    use_stepwise_action_norm: false
    norm_default_mode: z-score
    norm_exception_mode: null
    action_state_merger:
      _target_: ahawam.datasets.lerobot.transforms.action_state_merger.ConcatLeftAlign
    train_transforms:
      - _target_: ahawam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [240, 320]
    val_transforms:
      - _target_: ahawam.datasets.lerobot.transforms.image.ToTensor
      - _target_: torchvision.transforms.Resize
        size: [240, 320]
  context_len: 128
```

然后 `val:` 可以复制 train 并改：

```yaml
is_training_set: false
```

### 11.3 新增 SO101 task config

建议新增：

```text
configs/task/so101_ahawam_offset.yaml
```

大概形态：

```yaml
# @package _global_

defaults:
  - override /data: so101
  - override /model: ahawam
  - _self_

batch_size: 1
num_workers: 4
history_aware_batching: true

model:
  mot_checkpoint_mixed_attn: true
  skip_dit_load_from_pretrain: true
  action_dit_pretrained_path: null
  action_horizon: 80
  action_chunk_size: 16
  max_action_offset: 15
  action_obs_downsample_factor: 1
  chunk_kv_editor_num_queries: 32
  num_history_frames: 6
  action_video_read_mode: current_only
  video_rope_frame_stride: ${data.train.action_video_freq_ratio}
  loss:
    lambda_action_prior: 1.0

output_dir: runs/so101/${hydra:runtime.choices.task}-${now:%Y-%m-%d}_${now:%H-%M-%S}

wandb:
  enabled: false
  project: ahawam
  name: so101_ahawam_offset

learning_rate: 2e-5
num_epochs: 5
save_every: 500
eval_every: 100
gradient_accumulation_steps: 1
weight_decay: 1e-2
resume: null
```

SO101 初期建议：

- 从 `batch_size=1` 开始。
- 用 `init_checkpoint=checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt`。
- 先 `freeze_video_dit=true`。
- 先 `max_steps=20 eval_every=0 save_every=20`。

### 11.4 SO101 部署 wrapper 要改

当前 server adapter 要求请求格式：

```python
{
    "type": "infer",
    "instruction": "...",
    "state": np.ndarray,
    "images": {"front": image},
}
```

而 `deploy/server/ahawam_policy.py` 当前只取：

```python
front_img = np.asarray(img_arr[0])
```

SO101 两路相机建议改成：

```python
images = {
    "global": global_img,
    "wrist": wrist_img,
}
```

并在 policy wrapper 中按训练时相同规则拼接。例如 horizontal：

```python
image = np.concatenate([global_resized, wrist_resized], axis=1)
```

或者写成跟训练数据 `concat_multi_camera` 完全一致的函数。

同时还要改：

- `--action-dim`。
- state shape 检查。
- `_normalize_state()` 的 shape meta。
- `_denormalize_action()` 的输出解释。
- ROS topic 订阅。
- action 发布接口。
- 安全限幅。
- reset/init 姿态。
- emergency stop 流程。

## 12. 真机部署前安全清单

部署真实机械臂前，至少确认：

- 模型输出 action 的单位是什么：rad、meter、delta、absolute。
- action rate 和控制器 rate 是否匹配。
- 每维 joint limit、gripper limit 是否正确。
- 每步最大变化量是否限制。
- 网络中断时机器人怎么停。
- server 报错时机器人怎么停。
- 反归一化后的 action 是否在合理范围。
- dry-run 只打印 action，不发布控制。
- 第一次实机只执行很少步，低速，手在急停附近。

当前模板里的 `SafetyFilter` 是 Piper 双臂 14 维限幅，不适合直接用于 SO101，需要重写。

## 13. 常用命令速查

环境：

```bash
cd /home/fenghao/workspace/AHA-WAM
source /home/fenghao/use-nvidia-580159.sh
conda activate ahawam
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

看训练配置：

```bash
python scripts/train.py --help
python scripts/train.py task=robotwin_ahawam --cfg job
```

生成 ActionDiT backbone：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/ahawam.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda \
  --dtype bfloat16
```

预计算文本：

```bash
python scripts/precompute_text_embeds.py task=robotwin_ahawam
torchrun --standalone --nproc_per_node=4 scripts/precompute_text_embeds.py task=robotwin_ahawam
```

单机 4 卡训练：

```bash
bash scripts/train_zero1.sh 4 task=robotwin_ahawam
```

小步 smoke test：

```bash
bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam \
  batch_size=1 \
  num_workers=4 \
  max_steps=5 \
  save_every=5 \
  eval_every=0 \
  wandb.enabled=false
```

从完整预训练 checkpoint 微调：

```bash
bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam_offset \
  init_checkpoint=checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt \
  model.skip_dit_load_from_pretrain=true \
  model.action_dit_pretrained_path=null \
  freeze_video_dit=true
```

RoboTwin 单任务评估：

```bash
PYTHONPATH=. python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_ahawam \
  ckpt=checkpoints/AHA-WAM-RoboTwin2.0/robotwin_ahawam.pt \
  EVALUATION.task_name=click_alarmclock \
  EVALUATION.dataset_stats_path=checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json \
  EVALUATION.robotwin_root=/path/to/RoboTwin \
  gpu_id=0
```

部署 server help：

```bash
PYTHONPATH=. python deploy/server/wam_policy_server.py --help
```

同步部署 smoke test：

```bash
cp deploy/deploy_example.yml deploy/deploy.yml
# 编辑 deploy/deploy.yml 里的 checkpoint_path 和 dataset_stats_path

PYTHONPATH=. python deploy/server/wam_policy_server.py \
  --policy-path deploy/deploy.yml \
  --policy-module deploy.server.ahawam_policy \
  --policy-class AHAWAMPolicy \
  --instruction "pick up the object" \
  --action-dim 14 \
  --host 0.0.0.0 \
  --port 10000
```

另开一个终端：

```bash
PYTHONPATH=. python deploy/client/wam_policy_dummy_client.py \
  --host 127.0.0.1 \
  --port 10000 \
  --instruction "pick up the object" \
  --num-requests 1
```

双 GPU 异步部署 smoke test：

```bash
PYTHONPATH=. python deploy/server/wam_policy_server.py \
  --dual-gpu-async \
  --video-gpu 0 \
  --action-gpu 1 \
  --policy-path deploy/deploy.yml \
  --policy-module deploy.server.ahawam_policy \
  --policy-class AHAWAMPolicy \
  --instruction "pick up the object" \
  --action-dim 14
```

另开一个终端：

```bash
PYTHONPATH=. python deploy/client/wam_policy_async_dummy_client.py \
  --host 127.0.0.1 \
  --port 10000 \
  --instruction "pick up the object" \
  --num-action-requests 10
```

## 14. 当前 repo 里我实测/检查到的注意点

- 服务器真实 GPU 可用，4*A100 40GB，驱动 580.159.03。
- 在普通 Codex 沙箱里看不到 GPU 是沙箱限制，不代表服务器不可用。
- `deploy` 脚本建议加 `PYTHONPATH=.`，否则可能 `ModuleNotFoundError: No module named 'deploy'`。
- 当前 workspace 没看到本地 `data/`、`checkpoints/`、`runs/` 下已有数据或权重文件。
- `third_party/RoboTwin` 是 vendored 代码，但不像完整可运行的 RoboTwin checkout。跑评估前按 RoboTwin 官方说明准备完整环境更稳。
- 默认部署 wrapper 和 ROS client 是双臂 14 维模板，不是 SO101 现成模板。
- 默认训练配置里很多路径还是 `path/to/...`，必须用命令行 override 或创建自己的 yaml。

## 15. 推荐你下一步实际执行

最小闭环建议是：

1. 下载或准备 RoboTwin2.0 LeRobot 数据。
2. 下载 AHA-WAM-RoboTwin2.0 checkpoint 和 `dataset_stats.json`。
3. 先跑 `precompute_text_embeds.py`。
4. 跑 `max_steps=5` 的 4 卡 smoke test。
5. 用 release checkpoint 跑部署 server + dummy client。
6. 确认这些都通后，再开始 SO101 数据 config 和部署 wrapper 改造。

如果只想验证 GPU/环境，不要直接跑完整训练。用：

```bash
source /home/fenghao/use-nvidia-580159.sh
conda activate ahawam
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
PY
```

如果只想验证代码入口：

```bash
python scripts/train.py --help
python scripts/precompute_text_embeds.py --help
PYTHONPATH=. python deploy/server/wam_policy_server.py --help
```
