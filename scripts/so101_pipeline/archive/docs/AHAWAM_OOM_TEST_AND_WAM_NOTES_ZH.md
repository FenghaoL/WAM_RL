# AHA-WAM 报错分析、OOM 测试路线与 WAM/Zero-Shot 说明

本文记录这次 `preprocess_action_dit_backbone.py` 报错的原因、已做的修复，以及接下来如何一步一步测试训练是否 OOM。写法会尽量把“命令背后发生了什么”讲清楚。

## 1. 这次报错是什么原因

你运行的是：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/ahawam.yaml \
  --output "$ACTION_DIT" \
  --device cuda \
  --dtype bfloat16
```

下载阶段其实已经成功了。日志显示这些文件都已经放到本地：

```text
checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors
checkpoints/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors
checkpoints/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/...
checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors
checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors
checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors
```

真正失败的是这里：

```text
ValueError: invalid literal for int() with base 10: '${model.action_chunk_size}'
```

### 1.1 为什么会出现 `${model.action_chunk_size}`

`configs/model/ahawam.yaml` 里有：

```yaml
action_chunk_size: 16

action_dit_config:
  action_chunk_size: ${model.action_chunk_size}
```

在正常训练时，Hydra 会先组合：

```text
configs/train.yaml
configs/task/robotwin_ahawam.yaml
configs/model/ahawam.yaml
configs/data/robotwin.yaml
```

组合之后，`ahawam.yaml` 会挂在全局配置的 `model` 下面，所以 `${model.action_chunk_size}` 能解析到 `16`。

但 `preprocess_action_dit_backbone.py` 原本不是用 Hydra 组合任务配置，而是直接：

```python
cfg = OmegaConf.load("configs/model/ahawam.yaml")
```

这时候 `action_chunk_size` 在 yaml 顶层，不在 `cfg.model.action_chunk_size`。所以 `${model.action_chunk_size}` 还保持原样，传进 `ActionDiT` 后，代码执行：

```python
self.action_chunk_size = int(action_chunk_size)
```

于是 Python 试图执行：

```python
int("${model.action_chunk_size}")
```

这当然失败。

### 1.2 为什么 `action_dim` 也会 warning，但不是错误

脚本会打印：

```text
[WARN] `video_dit_config.action_dim` is unresolved; defaulting to 7 for preprocessing.
[WARN] `action_dit_config.action_dim` is unresolved; defaulting to 7 for preprocessing.
```

这是因为模型配置里还有：

```yaml
action_dim: ${data.train.processor.action_output_dim}
```

直接加载 `configs/model/ahawam.yaml` 时，没有 `data.train...`，所以解析不了。脚本原本就知道这个情况，并默认用 `7`。

这在生成 ActionDiT backbone 时通常没问题，因为要迁移的是 DiT 主干层，`action_encoder` 和 `head` 这两个和 action 维度直接相关的层被故意跳过了：

```python
ACTION_BACKBONE_SKIP_PREFIXES = ("action_encoder.", "head.")
```

也就是说，生成 backbone 时 `action_dim=7` 只是为了临时构建一个 ActionDiT 实例；真正跟你机器人 action 维度有关的输入头和输出头会在训练/加载完整 checkpoint 时处理。

## 2. 我已经对脚本做了什么修复

修改文件：

```text
scripts/preprocess_action_dit_backbone.py
```

修复内容：

- 让脚本在直接读取 `configs/model/*.yaml` 时，也能把 `${model.xxx}` 回退解析到同一个 yaml 顶层的 `xxx`。
- 例如 `${model.action_chunk_size}` 会解析成顶层 `action_chunk_size: 16`。
- 顺手优化了加载逻辑：这个预处理脚本只需要 Wan video DiT，不需要把 VAE 和 T5 text encoder 也加载进显存。

轻量验证已经通过：

```text
action_chunk_size 16 int
use_gradient_checkpointing True bool
```

完整 GPU 预处理也已经在这台服务器上跑通：

```text
[INFO] Saved ActionDiT backbone payload to checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt (copied=300, interpolated=520, skipped=4).
```

生成的文件大小约为：

```text
2.0G checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

## 3. 现在重新生成 ActionDiT backbone

先设置环境：

```bash
cd /home/fenghao/workspace/AHA-WAM
source /home/fenghao/use-nvidia-580159.sh
conda activate ahawam
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

设置输出路径：

```bash
export ACTION_DIT=checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

确认变量不是空的：

```bash
echo "$ACTION_DIT"
```

然后运行：

```bash
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/ahawam.yaml \
  --output "$ACTION_DIT" \
  --device cuda \
  --dtype bfloat16
```

这个命令做什么：

1. 读取 `configs/model/ahawam.yaml`。
2. 找到 Wan2.2 video DiT 配置。
3. 加载本地 `checkpoints/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-*.safetensors`。
4. 构造一个较小 hidden dim 的 ActionDiT。
5. 从 video DiT 里拷贝可复用的 Transformer 主干权重。
6. 如果 shape 不一样，就做线性插值。
7. 跳过 action 输入层和输出层。
8. 保存到 `$ACTION_DIT`。

这台服务器上生成成功后看到的是：

```text
[INFO] Saved ActionDiT backbone payload to checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt (copied=300, interpolated=520, skipped=4).
```

检查文件：

```bash
ls -lh "$ACTION_DIT"
```

目前这个文件已经存在，大小约为 `2.0G`。

## 4. 训练 OOM 测试前必须先解决的数据问题

当前 workspace 里还没有看到训练数据：

```text
data/ 目前没有可用 LeRobot 数据文件
```

AHA-WAM 训练不是拿一张图就能跑。它需要 LeRobot 格式轨迹数据，至少包含：

```text
meta/info.json
meta/tasks.jsonl
meta/episodes.jsonl
data/*.parquet
videos/**/*.mp4
```

默认 RoboTwin 配置需要这些字段：

```text
observation.images.cam_high
observation.images.cam_left_wrist
observation.images.cam_right_wrist
observation.state.default
action.default
task
```

所以 OOM 测试有两个层次：

- 只测模型是否能加载：需要模型权重，不需要数据。
- 测训练是否 OOM：必须有训练数据、文本 embedding cache、dataset stats。

## 5. 第一步：只测模型能不能加载到 GPU

这一步不训练，只验证模型实例化和 checkpoint/权重加载路径。它比训练轻，适合作为第一关。

如果你只想测试从 Wan + ActionDiT backbone 初始化：

```bash
python - <<'PY'
from pathlib import Path
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

config_dir = str(Path("configs").resolve())
with initialize_config_dir(version_base="1.3", config_dir=config_dir):
    cfg = compose(
        config_name="train.yaml",
        overrides=[
            "task=robotwin_ahawam",
            "model.action_dit_pretrained_path=checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt",
        ],
    )

model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
print("model loaded")
print("device:", model.device)
print("dtype:", model.torch_dtype)
PY
```

它做什么：

- 用 Hydra 组合完整训练配置。
- 创建 AHA-WAM 模型。
- 加载 Wan video DiT。
- 加载 ActionDiT backbone。
- 不构建数据集、不前向、不反向、不保存。

如果这里都 OOM，后面训练肯定 OOM，需要先降低加载压力或改策略。

注意：这个命令会用默认 `configs/task/robotwin_ahawam.yaml`，里面仍然有 `path/to/...` 的数据路径，但这里没有构建 dataset，所以不会立刻访问数据。

## 6. 第二步：准备 RoboTwin 数据

下载预处理数据：

```bash
huggingface-cli download yuanty/robotwin2.0-fastwam \
  --repo-type dataset \
  --local-dir data/robotwin2.0-fastwam

cd data/robotwin2.0-fastwam
cat robotwin2.0.tar.gz.part-* | tar -xzf -
cd /home/fenghao/workspace/AHA-WAM
```

然后找到真正包含 `meta/ data/ videos/` 的目录，例如：

```bash
find data/robotwin2.0-fastwam -maxdepth 4 -type d -name meta
```

假设找到的是：

```text
data/robotwin2.0-fastwam/robotwin2.0/meta
```

那数据根目录就是：

```bash
export ROBOTWIN_DATA=$PWD/data/robotwin2.0-fastwam/robotwin2.0
```

确认：

```bash
ls "$ROBOTWIN_DATA/meta"
ls "$ROBOTWIN_DATA/data" | head
ls "$ROBOTWIN_DATA/videos" | head
```

这一步做什么：

- 准备训练样本。
- 每个样本是一段机器人轨迹，不是一张独立图片。
- 模型会从轨迹里切出一个时间窗口，例如 65 帧 observation/state/action。

## 7. 第三步：准备 dataset_stats.json

训练和部署都需要归一化统计。

归一化的意思是：机器人关节角、夹爪、动作值的尺度差异很大。模型不直接学原始数值，而是先把它们变成均值附近、尺度相近的数。

默认配置里：

```yaml
norm_default_mode: z-score
```

就是：

```text
normalized = (raw - mean) / std
```

训练输出时也会把模型输出反归一化回真实 action。

### 7.1 什么时候需要下载 `robotwin_ahawam.pt`

这个命令：

```bash
huggingface-cli download SereneC/AHA-WAM-RoboTwin2.0 \
  robotwin_ahawam.pt dataset_stats.json \
  --local-dir checkpoints/AHA-WAM-RoboTwin2.0
```

会下载两个东西：

- `robotwin_ahawam.pt`：作者已经在 RoboTwin2.0 上训练好的完整 AHA-WAM checkpoint。
- `dataset_stats.json`：作者训练/部署这个 RoboTwin checkpoint 时用的 action/state 归一化统计。

如果你的目标是“直接加载作者发布的 RoboTwin checkpoint 做推理或继续微调”，就下载两个。

如果你的目标只是“用我们刚生成的 ActionDiT 初始化 backbone 跑训练，测试会不会 OOM”，那不需要 `robotwin_ahawam.pt`。此时你只需要一个合适的 `dataset_stats.json`，或者让训练第一次自己算。

只下载 stats 可以这样：

```bash
huggingface-cli download SereneC/AHA-WAM-RoboTwin2.0 \
  dataset_stats.json \
  --local-dir checkpoints/AHA-WAM-RoboTwin2.0
```

然后：

```bash
export ROBOTWIN_STATS=$PWD/checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json
```

这适合一种情况：你正在用和作者 RoboTwin release 基本一致的数据格式、action/state 维度、归一化配置。

### 7.2 如果用自己的数据或 SO101 数据，不要直接套 RoboTwin stats

`dataset_stats.json` 不是模型权重，它是数据尺度表。它记录的是：

- action 每一维的均值、标准差、最小值、最大值等。
- state/proprio 每一维的均值、标准差、最小值、最大值等。

如果换成 SO101，action 维度、关节范围、夹爪范围、单位、动作定义都可能不一样。这时候继续用 RoboTwin 的 `dataset_stats.json` 会让模型看到错误尺度的数据，轻则训练不稳定，重则部署动作完全不对。

所以：

- 用作者 RoboTwin checkpoint/作者 RoboTwin 数据：可以用作者发布的 `dataset_stats.json`。
- 用当前初始化出来的 ActionDiT backbone 在 RoboTwin 格式数据上做 OOM 测试：可以下载 release stats，也可以现算。
- 用 SO101 本地数据：应该从 SO101 训练数据现算自己的 `dataset_stats.json`。

### 7.3 不下载 stats，让第一次训练自己生成

训练集没有给 `pretrained_norm_stats` 时，代码会遍历训练集，计算 action/state 的统计量，并把结果保存为当前 run 目录下的：

```text
dataset_stats.json
```

关键 override 是：

```bash
data.train.pretrained_norm_stats=null
data.val.pretrained_norm_stats=null
```

训练代码的顺序是：

1. 先构建 train dataset。
2. 如果 `data.train.pretrained_norm_stats=null`，train dataset 会计算 stats。
3. rank 0 把 stats 保存到 `output_dir/dataset_stats.json`。
4. 如果此时 `torch.distributed` 已经初始化，rank 0 会把 stats 广播给其他 rank。
5. 但在当前 `train_zero1.sh 4` 这条路径里，dataset 构建发生得比较早；实际日志显示非 0 号 rank 可能在广播前拿到 `dataset_stats=None`，然后在 normalizer 里报错。

所以不要直接用 4 卡 `train_zero1.sh` 来第一次现算 stats。更稳的流程是：

1. 单进程、只实例化 dataset，生成 `dataset_stats.json`。
2. 之后多卡训练时显式传入这个 stats 文件。

注意：现算 stats 可能比 5 step 训练本身还慢，因为它要扫描训练 split 的 episode。这个阶段不是模型在训练，而是在统计数据尺度。

## 8. 第四步：预计算文本 embedding

设置 cache：

```bash
export TEXT_CACHE=$PWD/data/robotwin_text_embeds_cache
mkdir -p "$TEXT_CACHE"
```

运行：

```bash
python scripts/precompute_text_embeds.py \
  task=robotwin_ahawam \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  "data.val.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  data.val.text_embedding_cache_dir="$TEXT_CACHE"
```

如果你采用后面路线 B，准备第一次训练时临时 `data.val=null`，也可以这样预计算：

```bash
python scripts/precompute_text_embeds.py \
  task=robotwin_ahawam \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.val=null \
  data.train.text_embedding_cache_dir="$TEXT_CACHE"
```

这一步做什么：

1. 读 `$ROBOTWIN_DATA/meta/tasks.jsonl`。
2. 找出所有 task 文本，例如 `place the can into the basket`。
3. 包装成 prompt：

```text
A video recorded from a robot's point of view executing the following instruction: ...
```

4. 用 T5 text encoder 编成张量。
5. 按 prompt 的 hash 保存成 `.pt`。

为什么训练前必须做：

- 默认训练时 `model.load_text_encoder=false`。
- 意思是训练过程不加载 12.7GB 的文本 encoder。
- 所以 dataloader 必须提前从 cache 里拿到已经算好的文本 embedding。

如果你看到：

```text
Missing text embedding cache
```

就说明这一步没跑、路径不对，或者 task prompt 和 cache 不匹配。

## 9. 第五步：先跑最小训练 smoke test

先不要跑完整 5 epoch。先跑 5 个 optimizer step。

### 9.1 路线 A：使用已有 `dataset_stats.json`

如果你已经有 `$ROBOTWIN_STATS`，用这一版：

```bash
export OUT=$PWD/runs/smoke_robotwin_ahawam

bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam \
  model.action_dit_pretrained_path="$ACTION_DIT" \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  "data.val.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.train.pretrained_norm_stats="$ROBOTWIN_STATS" \
  data.val.pretrained_norm_stats="$ROBOTWIN_STATS" \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  data.val.text_embedding_cache_dir="$TEXT_CACHE" \
  output_dir="$OUT" \
  wandb.enabled=false \
  batch_size=1 \
  num_workers=4 \
  max_steps=5 \
  save_every=5 \
  eval_every=0
```

这条命令具体做什么：

- `bash scripts/train_zero1.sh 4`：启动 4 个训练进程，一张 A100 一个进程。
- `task=robotwin_ahawam`：选择基础 AHA-WAM 训练配置。
- `model.action_dit_pretrained_path=...`：告诉 action DiT 用刚生成的 backbone 初始化。
- `data.*.dataset_dirs=...`：告诉训练数据在哪里。
- `data.*.pretrained_norm_stats=...`：告诉 action/state 的归一化统计在哪里。
- `data.*.text_embedding_cache_dir=...`：告诉文本 embedding cache 在哪里。
- `output_dir=...`：训练输出目录。
- `wandb.enabled=false`：先不连 wandb，减少变量。
- `batch_size=1`：每张卡每次只放 1 个样本，优先测能不能跑。
- `max_steps=5`：只做 5 次参数更新。
- `eval_every=0`：先不做验证，因为验证会额外跑推理，增加显存和时间。
- `save_every=5`：第 5 step 保存 checkpoint，确认保存链路正常。

### 9.2 路线 B：不用发布 stats，先单进程生成 stats，再多卡训练

如果你想“就用当前初始化出来的 ActionDiT backbone 跑一下训练试试”，并且不想下载作者的 RoboTwin checkpoint/stats，不要直接让 `train_zero1.sh 4` 现算 stats。先单进程生成：

```bash
export STATS_OUT=$PWD/runs/make_robotwin_stats
mkdir -p "$STATS_OUT"

python - <<'PY'
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from ahawam.utils import misc
from ahawam.utils.config_resolvers import register_default_resolvers

repo = Path("/home/fenghao/workspace/AHA-WAM")
register_default_resolvers()

with initialize_config_dir(version_base="1.3", config_dir=str(repo / "configs")):
    cfg = compose(
        config_name="train",
        overrides=[
            "task=robotwin_ahawam",
            f"data.train.dataset_dirs=[{os.environ['ROBOTWIN_DATA']}]",
            "data.train.pretrained_norm_stats=null",
            "data.val=null",
            f"output_dir={os.environ['STATS_OUT']}",
        ],
    )

misc.register_work_dir(cfg.output_dir)
dataset = instantiate(cfg.data.train)
stats_path = Path(cfg.output_dir) / "dataset_stats.json"
print(f"dataset_len={len(dataset)}")
print(f"stats_path={stats_path}")
PY

export ROBOTWIN_STATS=$STATS_OUT/dataset_stats.json
ls -lh "$ROBOTWIN_STATS"
```

这一步只构建 dataset，不加载 AHA-WAM 模型，所以它主要花 CPU/IO 时间，不吃 6B 模型显存。

然后确认几个关键变量都不是空的：

```bash
export ACTION_DIT=$PWD/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
export TEXT_CACHE=$PWD/data/robotwin_text_embeds_cache
export OUT=$PWD/runs/smoke_robotwin_ahawam

ls -lh "$ACTION_DIT"
ls -lh "$ROBOTWIN_STATS"
test -d "$TEXT_CACHE" && find "$TEXT_CACHE" -maxdepth 1 -name '*.pt' | head
```

最后多卡 smoke train：

```bash
bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam \
  model.action_dit_pretrained_path="$ACTION_DIT" \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.val=null \
  data.train.pretrained_norm_stats="$ROBOTWIN_STATS" \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  output_dir="$OUT" \
  wandb.enabled=false \
  batch_size=1 \
  num_workers=2 \
  max_steps=5 \
  save_every=5 \
  eval_every=0
```

这版命令的关键区别：

- `data.val=null`：不构建验证集，先只测训练链路，减少变量。
- `data.train.pretrained_norm_stats="$ROBOTWIN_STATS"`：多卡训练不再现算 stats，而是直接读取单进程提前生成的 stats。
- `model.action_dit_pretrained_path="$ACTION_DIT"` 和 `data.train.text_embedding_cache_dir="$TEXT_CACHE"` 必须确认不是空字符串。

成功后检查：

```bash
find "$OUT" -maxdepth 4 -type f | sort | head -50
```

你应该能看到：

```text
config.yaml
dataset_stats.json
checkpoints/weights/step_000005.pt
checkpoints/state/step_000005/...
```

## 10. 如果 `train_zero1.sh` 报 `CUDA_HOME does not exist`

你看到的错误：

```text
deepspeed.ops.op_builder.builder.MissingCUDAException:
CUDA_HOME does not exist, unable to compile CUDA op(s)
```

发生在真正训练之前。调用链是：

```text
train_zero1.sh
accelerate launch
import deepspeed
deepspeed.ops.op_builder.fp_quantizer.is_compatible()
installed_cuda_version()
需要 CUDA_HOME/bin/nvcc
```

这里要区分三个东西：

- NVIDIA driver：负责让系统和 GPU 通信。
- CUDA runtime：PyTorch `torch==2.7.1+cu128` 自带一部分运行时库，所以普通 PyTorch CUDA 可以工作。
- CUDA toolkit：包含 `nvcc`、CUDA 头文件、编译链接用的库。DeepSpeed 编译/检查 CUDA op 时需要它。

`source /home/fenghao/use-nvidia-580159.sh` 只解决 driver/library mismatch，让进程使用 580.159.03 驱动库；它不会提供 `nvcc`，也不会自动设置 `CUDA_HOME`。

先检查：

```bash
cd /home/fenghao/workspace/AHA-WAM
source /home/fenghao/use-nvidia-580159.sh
conda activate ahawam

python - <<'PY'
import os, shutil, torch
import torch.utils.cpp_extension as ext
print("torch.cuda.is_available =", torch.cuda.is_available())
print("torch.version.cuda =", torch.version.cuda)
print("CUDA_HOME env =", os.environ.get("CUDA_HOME"))
print("torch cpp CUDA_HOME =", ext.CUDA_HOME)
print("nvcc =", shutil.which("nvcc"))
PY
```

如果看到：

```text
torch.version.cuda = 12.8
CUDA_HOME env = None
torch cpp CUDA_HOME = None
nvcc = None
```

就说明问题不是 GPU 没了，而是当前 conda 环境没有 CUDA toolkit。

推荐安装和 PyTorch 匹配的 CUDA 12.8 toolkit：

```bash
conda activate ahawam
conda install -c nvidia cuda-toolkit=12.8 -y
```

然后每次训练前设置：

```bash
source /home/fenghao/use-nvidia-580159.sh
conda activate ahawam
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
```

确认：

```bash
which nvcc
nvcc -V
python - <<'PY'
import torch.utils.cpp_extension as ext
print(ext.CUDA_HOME)
PY
```

不要优先用 `/usr/local/cuda-11.7`，因为当前 PyTorch 是 CUDA 12.8 构建的。CUDA 11.7 和 PyTorch 12.8 是大版本不匹配，后面即使绕过检查，也容易遇到编译或运行期问题。

如果只是想绕开 DeepSpeed 先做一个更粗的训练链路测试，可以试 plain DDP：

```bash
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_ddp.yaml \
  --num_processes 4 \
  scripts/train.py \
  task=robotwin_ahawam \
  model=ahawam \
  model.action_dit_pretrained_path="$ACTION_DIT" \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.val=null \
  data.train.pretrained_norm_stats=null \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  output_dir="$OUT" \
  wandb.enabled=false \
  batch_size=1 \
  num_workers=2 \
  max_steps=5 \
  save_every=5 \
  eval_every=0
```

但 DDP 不做 ZeRO optimizer state 分片，显存压力会比 `train_zero1.sh` 更大。它适合检查数据/模型链路，不适合作为最终多卡训练方案。

日志里的：

```text
df: /home/fenghao/.triton/autotune: No such file or directory
```

不是这次退出的主因。可以创建目录消掉这行噪音：

```bash
mkdir -p /home/fenghao/.triton/autotune
```

## 11. 训练时怎么判断是不是 OOM

另开一个终端：

```bash
cd /home/fenghao/workspace/AHA-WAM
source /home/fenghao/use-nvidia-580159.sh
watch -n 1 nvidia-smi
```

OOM 常见报错：

```text
torch.cuda.OutOfMemoryError
CUDA out of memory
```

也可能表现为某个 rank 退出，然后 accelerate/deepspeed 报子进程失败。

如果 OOM，按这个顺序降压力：

1. 保持 `batch_size=1`。
2. 加 `model.mot_checkpoint_mixed_attn=true`。
3. 加 `gradient_accumulation_steps=2` 或 `4`。
4. 把 `num_workers` 降到 `2`。
5. 先用 `freeze_video_dit=true`。
6. 再试 `scripts/train_zero2.sh 4`。

解释：

- `batch_size` 影响显存最直接。
- gradient checkpointing 用更多计算换更少激活显存。
- gradient accumulation 不会减少单次前向显存，但能让你用小 batch 模拟较大总 batch。
- `freeze_video_dit=true` 会减少需要保存梯度/优化器状态的参数。
- ZeRO-2 比 ZeRO-1 更省一部分训练状态，但通信更复杂。

## 12. 第六步：逐步放大训练压力

如果 5 step 成功，再按这个顺序试：

### 11.1 20 step

```bash
bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam \
  model.action_dit_pretrained_path="$ACTION_DIT" \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  "data.val.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.train.pretrained_norm_stats="$ROBOTWIN_STATS" \
  data.val.pretrained_norm_stats="$ROBOTWIN_STATS" \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  data.val.text_embedding_cache_dir="$TEXT_CACHE" \
  output_dir="$PWD/runs/smoke_robotwin_20steps" \
  wandb.enabled=false \
  batch_size=1 \
  num_workers=4 \
  max_steps=20 \
  save_every=20 \
  eval_every=0
```

目的：

- 观察显存是否稳定。
- 观察 step time。
- 确认不是第 2 到第 10 step 才爆显存。

### 11.2 开启一次 eval

```bash
bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam \
  model.action_dit_pretrained_path="$ACTION_DIT" \
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" \
  "data.val.dataset_dirs=[$ROBOTWIN_DATA]" \
  data.train.pretrained_norm_stats="$ROBOTWIN_STATS" \
  data.val.pretrained_norm_stats="$ROBOTWIN_STATS" \
  data.train.text_embedding_cache_dir="$TEXT_CACHE" \
  data.val.text_embedding_cache_dir="$TEXT_CACHE" \
  output_dir="$PWD/runs/smoke_robotwin_eval" \
  wandb.enabled=false \
  batch_size=1 \
  num_workers=4 \
  max_steps=20 \
  save_every=20 \
  eval_every=10 \
  eval_num_inference_steps=4
```

目的：

- eval 会跑验证 forward/inference。
- `eval_num_inference_steps=4` 比默认 10 轻，适合 smoke test。

### 11.3 提高 batch size

如果还有很多空闲显存：

```bash
batch_size=2
```

再试：

```bash
batch_size=4
```

默认 `robotwin_ahawam.yaml` 是：

```yaml
batch_size: 6
num_workers: 32
```

这对新手 smoke test 太激进，不建议一开始就用。

## 13. AHA-WAM 里有哪些模块，各自做什么

### 12.1 VAE

作用：

- 把图片/视频压缩成 latent。
- 把 latent 解码回视频。

为什么需要：

- 直接在像素空间训练视频模型太贵。
- latent 空间更小，更适合 DiT。

通常微调吗：

- 这个 repo 默认不训练 VAE。
- VAE 主要是冻结的编码/解码工具。

### 12.2 Text encoder 和 tokenizer

作用：

- tokenizer 把语言指令切成 token。
- text encoder 把 token 变成向量。

训练时怎么用：

- 默认训练 `model.load_text_encoder=false`。
- 文本向量提前由 `scripts/precompute_text_embeds.py` 算好。

通常微调吗：

- 这个 repo 默认不微调 text encoder。
- 对你的 SO101 初期也不建议微调 text encoder。

### 12.3 Video DiT

作用：

- 学视觉世界怎么变化。
- 在 AHA-WAM 里作为低频 long-horizon planner。
- 维护 rolling K/V memory，保存长时域视觉上下文。

可不可以微调：

- 默认可以训练。
- 如果数据少或显存紧，可以 `freeze_video_dit=true` 冻住它。

什么时候训练它：

- 你的相机视角、场景、任务和原始训练差异大。
- 你有足够数据和显存。

什么时候先冻住：

- 初期 SO101 数据少。
- 只是想让 action 分支适配本体。
- 先测 OOM 和 pipeline。

### 12.4 Action DiT

作用：

- 输入当前 observation/state 和 video planner 的上下文。
- 输出未来一段动作，也就是 action chunk。

它是部署时最关键的执行器。

可不可以微调：

- 可以，默认训练中它属于 `model.dit` 的一部分。
- 如果 `freeze_video_dit=true`，action DiT 仍然会训练。

### 12.5 MoT / mixed attention

MoT 可以理解成把 video expert 和 action expert 放在一个可交互的 Transformer 框架里。

作用：

- video token 和 action token 可以通过设计好的 attention mask 交互。
- action 分支能读 video 分支的层级上下文。

### 12.6 Proprio encoder

proprio 就是机器人自身状态，例如关节角、夹爪、末端状态。

作用：

- 把低维 state 映射到和 text/context 同一类的 hidden 表示。
- 让模型知道“现在机械臂自己在哪”。

可不可以微调：

- 可以。代码里 `get_additional_trainable_modules()` 会把 proprio encoder 加进训练参数。
- 对跨机器人迁移很重要，因为 SO101 的 state/action 维度和 RoboTwin 双臂不同。

### 12.7 AHA-WAM 的 chunk/context routing 模块

作用：

- 让 action 分支在每个 chunk 查询、修正、复用 video planner 的上下文。
- 这是 AHA-WAM 相比普通同步 WAM 的关键。

可不可以微调：

- 可以。它属于 AHA-WAM 额外可训练模块/DiT 相关参数。
- SO101 初期通常应保留训练。

## 14. 哪些模块可以分开微调

这个 repo 没有内置 LoRA/QLoRA，但有粗粒度冻结策略。

### 13.1 默认训练

```bash
freeze_video_dit=false
```

会训练：

- MoT 中的 DiT 相关参数。
- video expert。
- action expert。
- proprio encoder。
- AHA-WAM 额外 chunk/context 模块。

不训练：

- VAE。
- text encoder。
- tokenizer。

### 13.2 冻住 video DiT

```bash
freeze_video_dit=true
```

会训练：

- action expert。
- proprio encoder。
- AHA-WAM 额外 chunk/context 模块。

不训练：

- video expert。
- VAE。
- text encoder。

适合：

- SO101 初期。
- 小数据。
- 降低显存。
- 先把动作分布适配起来。

### 13.3 ODE distillation

训练对象：

- teacher 冻住。
- student 训练。

作用：

- 把多步 denoise 的 teacher 压成更快的 student。
- 更适合最后部署加速，不适合第一阶段。

## 15. WAM 的 zero-shot 到底是什么意思

先给直觉结论：

WAM 的 zero-shot 不是“完全不用任何机器人数据，直接把下载的模型丢到任意本地机械臂上就能干活”。更准确地说，它通常指：

- 模型已经在大量视频/机器人数据上训练过。
- 部署或评估时遇到未见过的新任务、新物体、新场景，可以不针对这个具体任务再训练。
- 对新机器人本体，通常仍然需要少量适配数据或微调，尤其是 action/state 维度、控制频率、相机位置都变了的时候。

### 14.1 WAM 为什么会有 zero-shot 潜力

WAM 同时学习：

- 世界会怎么变：future visual dynamics。
- 机器人应该怎么动：action prediction。

NVIDIA 对 WAM 的解释是：WAM 学习世界变化和动作如何影响世界，不只是把图像和语言直接映射到动作。因为它从大规模视频里学到物体运动和交互规律，所以对未见任务更有泛化潜力。

DreamZero 论文的摘要也强调，它基于预训练 video diffusion backbone，通过 joint video/action modeling，在新任务和新环境上比 VLA 有更强泛化，并展示了跨本体 transfer；但它仍然提到新本体的 few-shot adaptation，例如 30 分钟 play data。

### 14.2 AHA-WAM 里的 zero-shot/泛化该怎么理解

AHA-WAM arXiv 摘要说它在 RoboTwin 和真实机器人任务上取得好结果，并强调 RoboTwin 结果是 “without any robot-data pretraining”。这句话容易误读。

它不是说：

```text
不用训练就能部署到任意机器人
```

更接近：

```text
不依赖额外大规模机器人数据预训练，也能在 RoboTwin 训练后取得高成功率
```

AHA-WAM repo 同时释放了 RoboCOIN-pretrained checkpoint，说明作者也认为下游真实机器人适配会受益于 embodied/robot data pretraining。

### 14.3 对你的 SO101 意味着什么

你有三种选择：

1. 直接拿 RoboTwin checkpoint 部署 SO101：不推荐。相机、动作维度、控制对象都不匹配。
2. 用 RoboCOIN/AHA-WAM pretrained checkpoint，在 SO101 数据上微调：推荐。
3. 完全从 Wan + ActionDiT backbone 开始训练 SO101：可行但数据需求更高。

对本地泛化能力的现实判断：

- 同一 SO101、同一相机、类似桌面任务：有机会通过少量任务数据泛化到相似新物体/新摆放。
- 同一 SO101、完全不同场景或光照：需要域随机化或更多数据。
- 换机械臂/换 action 表达：通常必须重新适配 action/state。
- 零训练直接上真机：不要期待可靠，也不安全。

## 16. 参考资料

- AHA-WAM project page: https://serene-sivy.github.io/aha-wam/
- AHA-WAM arXiv: https://arxiv.org/abs/2606.09811
- DreamZero / World Action Models are Zero-shot Policies arXiv: https://arxiv.org/abs/2602.15922
- DreamZero GitHub: https://github.com/dreamzero0/dreamzero
- Fast-WAM arXiv: https://arxiv.org/abs/2603.16666
- Hugging Face LeRobot FastWAM docs: https://huggingface.co/docs/lerobot/main/en/fastwam
- NVIDIA WAM glossary: https://www.nvidia.com/en-us/glossary/world-action-model/
