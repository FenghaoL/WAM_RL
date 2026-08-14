目前我准备在这个服务器上微调这个ahawam，并通过通道的方式在我自己的so101机械臂上部署这个ahawam，目前我的机械臂又一个全局摄像机和一个腕部摄像机，现在这个服务器更新导致版本不匹配，需要用source /home/fenghao/use-nvidia-580159.sh来访问gpu。目前对于这个repo我已经运行了“conda create -n ahawam python=3.10 -y
conda activate ahawam
pip install -U pip setuptools wheel
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 
  --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
conda install -c conda-forge ffmpeg -y”但对于readme后面的内容我有点没看懂，一方面需要你为我解释（假设我没啥基础）这个readme里后面的一系列操作分别是干嘛的，对于我来说是否应该用，或者要在哪些情况下用，除此之外还有什么我可以用的命令，或者需要做的准备，这个repo自带的能直接跑的训练/推理脚本都有哪些？用的都是什么数据集？虽然我的最终目标是在我的so101机械臂上部署，不过我也想先试试repo自带的训练脚本，顺便看看目前这个服务器上的gpu是否足够（4*A100)如果想要分布式是不是还需要做什么调整，另外这个repo里的训练脚本是只有全量微调吗？把这些问题都为我解释清楚，给我一个全面详细的代码入门指南，假设我完全是个小白。可以放在一个md文档里




Show more

Aug 4, 4:50 PM






Worked for 10m 12s



已整理成中文 Markdown 指南：
[docs/AHAWAM_CODE_GUIDE_ZH.md](/home/fenghao/workspace/AHA-WAM/docs/AHAWAM_CODE_GUIDE_ZH.md)
里面覆盖了你问的这些点：README 后半段每个命令是干嘛的、哪些现在该用、repo 自带训练/评估/部署入口、默认数据集格式、4*A100 怎么跑单机多卡、ZeRO-1/ZeRO-2、init_checkpoint/resume/freeze_video_dit、是否支持 LoRA、以及后续 SO101 双相机真机适配要改哪些地方。
我也实测确认了环境：提权后 source /home/fenghao/use-nvidia-580159.sh 能看到 4 张 NVIDIA A100-SXM4-40GB，torch 2.7.1+cu128 也能看到 4 个 CUDA 设备。普通沙箱里看不到 GPU 是沙箱限制。另一个小坑也写进文档了：部署脚本建议加 PYTHONPATH=.，否则可能找不到顶层 deploy 包。
没有实际启动训练，因为当前 workspace 里还没看到本地数据和 checkpoint。下一步最稳的是先准备 RoboTwin2.0 LeRobot 数据和 release checkpoint，然后按文档里的 max_steps=5 smoke test 跑一遍。

Edited AHAWAM_CODE_GUIDE_ZH.md+1,302-3Review changesUndoReview





Aug 4, 5:00 PM








运行这个命令“python scripts/preprocess_action_dit_backbone.py 
  --model-config configs/model/ahawam.yaml 
  --output "$ACTION_DIT" 
  --device cuda 
  --dtype bfloat16“，遇到了报错，为我分析一下原因，另外我目前打算测试一下模型训练，看看会不会oom，给我接下来具体的步骤，同时也要解释清楚每一步都是干什么的，我不希望运行一个命令的时候不清楚它具体做了什么，所以你要把细节都解释到位，假定我完全没有前置知识，如果解释一个步骤需要讲解很多模型相关的知识的话，我也不介意你都讲清楚，我需要对这个模型建立起系统明确的认知，同时也要知道有哪些模块可以分开微调，分别起着什么作用，你可以在网络上先检索了解一下这一系列wam相关的知识再告诉我，而且我听说wam有一定的zero shot的能力，不知道这个zero shot指的是在公开数据集训练本地部署，还是真的不用训练，有没有跨本地的泛化能力“Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:32
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:32
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:32
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:32
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:32
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:32
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:32
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:32
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:33
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:33
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:33
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:33
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:33
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:33
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:33
Downloading [models_t5_umt5-xxl-enc-bf16.safetensors]: 12.7GB [4:33:06, 834kB/s]
Processing 1 items: 100%|████████████████████████████████████████████████████████████████████████| 1.00/1.00 [4:36:17<00:00, 16.6ks/itProcessing 1 items: 100%|████████████████████████████████████████████████████████████████████████| 1.00/1.00 [4:36:17<00:00, 16.6ks/it]
2026-08-05 16:48:10,287 - modelscope - INFO - Download model 'DiffSynth-Studio/Wan-Series-Converted-Safetensors' successfully.
Downloading Model from  to directory: /home/fenghao/workspace/AHA-WAM/checkpoints/Wan-AI/Wan2.1-T2V-1.3B
2026-08-05 16:48:13,026 - modelscope - INFO - Got 4 files, start to download ...
Downloading [google/umt5-xxl/special_tokens_map.json]: 100%|█| 6.47
Downloading [google/umt5-xxl/tokenizer_config.json]: 100%|█| 60.3k/
Downloading [google/umt5-xxl/tokenizer.json]: 100%|█| 16.1M/16.1M [
Downloading [google/umt5-xxl/spiece.model]: 100%|█| 4.34M/4.34M [00
Processing 4 items: 100%|███████| 4.00/4.00 [00:02<00:00, 1.54it/s]
2026-08-05 16:48:15,628 - modelscope - INFO - Download model 'Wan-AI/Wan2.1-T2V-1.3B' successfully.
2026-08-05 16:48:15,628 - modelscope - INFO - Target directory already exists, skipping creation.
Downloading Model from  to directory: /home/fenghao/workspace/AHA-WAM/checkpoints/Wan-AI/Wan2.2-TI2V-5B
2026-08-05 16:48:18,442 - modelscope - INFO - Got 3 files, start to download ...
Downloading [diffusion_pytorch_model-00003-of-00003.safetensors]: 1
Downloading [diffusion_pytorch_model-00002-of-00003.safetensors]: 1
Downloading [diffusion_pytorch_model-00001-of-00003.safetensors]: 1
Processing 3 items: 100%|████| 3.00/3.00 [1:30:37<00:00, 1.81ks/it]
2026-08-05 18:18:55,591 - modelscope - INFO - Download model 'Wan-AI/Wan2.2-TI2V-5B' successfully.
2026-08-05 18:18:55,591 - modelscope - INFO - Target directory already exists, skipping creation.
Traceback (most recent call last):
  File "/home/fenghao/workspace/AHA-WAM/scripts/preprocess_action_dit_backbone.py", line 240, in <module>
    main()
  File "/home/fenghao/workspace/AHA-WAM/scripts/preprocess_action_dit_backbone.py", line 181, in main
    action_expert = ActionDiT(**action_cfg).to(device=args.device, dtype=torch_dtype)
  File "/home/fenghao/workspace/AHA-WAM/src/ahawam/models/wan22/action_dit.py", line 73, in init
    self.action_chunk_size = int(action_chunk_size)
ValueError: invalid literal for int() with base 10: '${model.action_chunk_size}'”




Show more

Aug 5, 6:33 PM






Worked for 10m 59s



我已经把这次报错处理完了，并写成了中文入门文档：[AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md](/home/fenghao/workspace/AHA-WAM/docs/AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md)。之前那份更完整的 repo 入门指南也在：[AHAWAM_CODE_GUIDE_ZH.md](/home/fenghao/workspace/AHA-WAM/docs/AHAWAM_CODE_GUIDE_ZH.md)。
核心原因很明确：下载没有失败，CUDA 也不是根因。scripts/preprocess_action_dit_backbone.py 原来直接读取 configs/model/ahawam.yaml，没有经过 Hydra 组合配置，所以 ${model.action_chunk_size} 没被解析成 16，最后 ActionDiT 初始化时执行 int("${model.action_chunk_size}") 才报错。我已经改了 [scripts/preprocess_action_dit_backbone.py (line 66)](/home/fenghao/workspace/AHA-WAM/scripts/preprocess_action_dit_backbone.py:66)，让这个独立预处理脚本能正确解析 ${model.xxx}，并且顺手减少了不必要的 VAE/T5 加载。
我也实际重跑了你的预处理命令，已经成功：
[INFO] Saved ActionDiT backbone payload to checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt (copied=300, interpolated=520, skipped=4).
生成文件约 2.0G：
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
验证结果：
py_compile: OK
action_chunk_size: 16 int
use_gradient_checkpointing: True bool
CUDA: True
GPU: 4 x NVIDIA A100-SXM4-40GB
每张卡空闲约 40GB
接下来测试训练是否 OOM，按文档里的路线走：先 source /home/fenghao/use-nvidia-580159.sh，再激活 ahawam，然后准备 RoboTwin/LeRobot 格式数据、数据统计文件、文本 embedding cache，最后用 bash scripts/train_zero1.sh 4 ... 做 batch_size=1 max_steps=5 eval_every=0 save_every=5 wandb.enabled=false 的最小冒烟训练。这个顺序是为了先确认“模型能加载、数据能进 batch、反传能跑”，再逐步加 batch/eval/save，而不是一上来跑长训练。
关于 WAM/zero-shot，我也在文档里系统解释了。简短结论是：WAM 的 zero-shot 通常不是“完全不用你本地机器人数据就能直接控制 SO101”，而是指在已有大规模视频/机器人数据训练后的模型，对未见任务、环境、物体有一定泛化能力。换到你的 SO101、本地相机、动作空间和标定后，最稳妥路线仍然是用已有 checkpoint 做适配/微调。相关背景我参考了 AHA-WAM、DreamZero、Fast-WAM 和 LeRobot FastWAM 文档：
AHA-WAM arXiv、DreamZero arXiv、Fast-WAM arXiv、LeRobot FastWAM docs、NVIDIA WAM glossary。

Edited 2 files+885-9Review changesUndoReview

scripts/preprocess_action_dit_backbone.pyscripts/preprocess_action_dit_backbone.py+75-6
docs/AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.mddocs/AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md+810-3





Aug 5, 6:44 PM








你给的“7. 第三步：准备 dataset_stats.json”这一步里面，是要用“huggingface-cli download SereneC/AHA-WAM-RoboTwin2.0 
  robotwin_ahawam.pt dataset_stats.json 
  --local-dir checkpoints/AHA-WAM-RoboTwin2.0”这个下载它给checkpoint吗？如果我想拿目前初始化出来的这个模型来跑一下训练试试，可以怎么弄这个stats？这部分的教程讲的有点模糊






Aug 5, 11:32 PM






Worked for 3m 43s



你抓到的点是对的：我之前那节把“下载作者 checkpoint”和“准备 stats”放得太近，容易误会。我已经把文档第 7、8、9 节改清楚了：[AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md (line 314)](/home/fenghao/workspace/AHA-WAM/docs/AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md:314)。
简单说：
robotwin_ahawam.pt 是作者已经训练好的完整 checkpoint。
dataset_stats.json 不是模型权重，它只是 action/state/proprio 的归一化统计表。
所以如果你只是想用我们刚初始化出来的：
checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
来试训练、测 OOM，不需要下载 robotwin_ahawam.pt。训练时用：
model.action_dit_pretrained_path="$ACTION_DIT"
就行。
如果你想只下载作者的 stats，可以这样：
huggingface-cli download SereneC/AHA-WAM-RoboTwin2.0 \
  dataset_stats.json \
  --local-dir checkpoints/AHA-WAM-RoboTwin2.0

export ROBOTWIN_STATS=$PWD/checkpoints/AHA-WAM-RoboTwin2.0/dataset_stats.json
但如果你想完全不用作者 stats，让当前训练数据自己生成 stats，就用这个 smoke test：
export OUT=$PWD/runs/smoke_robotwin_ahawam_make_stats

bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam \
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
这里最关键的是：
data.train.pretrained_norm_stats=null
data.val=null
意思是：训练集自己扫描数据，计算 action/state 的 mean/std 等统计量，并保存到：
$OUT/dataset_stats.json
第一次跑完后，后面就可以固定使用它：
export ROBOTWIN_STATS=$OUT/dataset_stats.json
之后再训练就把：
data.train.pretrained_norm_stats="$ROBOTWIN_STATS"
data.val.pretrained_norm_stats="$ROBOTWIN_STATS"
显式传进去，避免每次都重新扫描数据。
另外提醒一下：SO101 本地数据以后不要直接套 RoboTwin 的 dataset_stats.json，因为动作维度、关节范围、夹爪定义可能都不一样。SO101 应该用你自己的 SO101 轨迹数据现算 stats。

Edited AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md+118-3Review changesUndoReview





Aug 5, 11:35 PM








帮我检查讲解一下为什么会有这个报错“(ahawam) fenghao@chickadee:~/workspace/AHA-WAM$ bash scripts/train_zero1.sh 4 \
  task=robotwin_ahawam 
  model.action_dit_pretrained_path="$ACTION_DIT" 
  "data.train.dataset_dirs=[$ROBOTWIN_DATA]" 
  data.val=null 
  data.train.pretrained_norm_stats=null 
  data.train.text_embedding_cache_dir="$TEXT_CACHE" 
  output_dir="$OUT" 
  wandb.enabled=false 
  batch_size=1 
  num_workers=2 
  max_steps=5 
  save_every=5 
  eval_every=0
[model] resolved from task yaml (robotwin_ahawam.yaml): model=ahawam
[launch] nproc_per_node=4 num_processes=4 num_machines=1 machine_rank=0 main_process_ip=127.0.0.1 main_process_port=29500 deepspeed_hostfile=none task=robotwin_ahawam run_id=2026-08-06_00-04-45
[env] hostname=chickadee.engin.umich.edu SLURM_NODEID=unset SLURM_PROCID=unset LOCAL_RANK=unset RANK=unset WORLD_SIZE=unset
[env] NNODES=unset NODE_RANK=unset MASTER_ADDR=unset MASTER_PORT=unset HOSTFILE=unset DS_HOSTFILE=unset DEEPSPEED_HOSTFILE=unset
[env] resolved_master_addr=<unset>-><unset> local_ips=['141.212.3.148']
df: /home/fenghao/.triton/autotune: No such file or directory
Traceback (most recent call last):
  File "/home/fenghao/miniconda3/envs/ahawam/bin/accelerate", line 6, in <module>
    sys.exit(main())
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/commands/accelerate_cli.py", line 50, in main
    args.func(args)
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/commands/launch.py", line 1266, in launch_command
    deepspeed_launcher(args)
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/accelerate/commands/launch.py", line 915, in deepspeed_launcher
    from deepspeed.launcher.runner import DEEPSPEED_ENVIRONMENT_NAME
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/init.py", line 25, in <module>
    from . import ops
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/ops/init.py", line 15, in <module>
    from ..git_version_info import compatible_ops as compatible_ops
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/git_version_info.py", line 29, in <module>
    op_compatible = builder.is_compatible()
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/ops/op_builder/fp_quantizer.py", line 35, in is_compatible
    sys_cuda_major, _ = installed_cuda_version()
  File "/home/fenghao/miniconda3/envs/ahawam/lib/python3.10/site-packages/deepspeed/ops/op_builder/builder.py", line 51, in installed_cuda_version
    raise MissingCUDAException("CUDA_HOME does not exist, unable to compile CUDA op(s)")
deepspeed.ops.op_builder.builder.MissingCUDAException: CUDA_HOME does not exist, unable to compile CUDA op(s)”




Show more

Thursday 12:08 AM






Worked for 3m 5s



这个报错不是 OOM，也还没跑到模型训练。它是在 accelerate launch 导入 deepspeed 的阶段就挂了。
原因是：你的 PyTorch 是 torch==2.7.1+cu128，它自带 CUDA 12.8 runtime，所以普通 torch.cuda 可以用；但 DeepSpeed 需要编译/检查 CUDA 自定义 op，它要找完整 CUDA toolkit，也就是 CUDA_HOME/bin/nvcc。现在环境里：
CUDA_HOME = None
nvcc = None
torch.version.cuda = 12.8
所以 DeepSpeed 报：
CUDA_HOME does not exist, unable to compile CUDA op(s)
source /home/fenghao/use-nvidia-580159.sh 只解决 NVIDIA driver/library mismatch，让进程能用 580.159.03 驱动库；它不提供 nvcc，也不会自动设置 CUDA_HOME。
推荐修复：
conda activate ahawam
conda install -c nvidia cuda-toolkit=12.8 -y
之后每次训练前：
cd /home/fenghao/workspace/AHA-WAM
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh

export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="/home/fenghao/nvidia-compat-580159/root/usr/lib/x86_64-linux-gnu:$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
检查：
which nvcc
nvcc -V
python - <<'PY'
import torch.utils.cpp_extension as ext
print(ext.CUDA_HOME)
PY
不要优先用 /usr/local/cuda-11.7，因为你现在 PyTorch 是 CUDA 12.8 构建的，CUDA 11.7 大版本不匹配。
另外这行：
df: /home/fenghao/.triton/autotune: No such file or directory
不是主因，只是 Triton cache 目录不存在的噪音，可以：
mkdir -p /home/fenghao/.triton/autotune
我也把这部分补进文档了：[AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md (line 573)](/home/fenghao/workspace/AHA-WAM/docs/AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md:573)。

Edited AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md+130-6Review changesUndoReview





Thursday 12:11 AM








(ahawam) fenghao@chickadee:~/workspace/AHA-WAM$ bash scripts/train_zero1.sh 4 t…

这次又遇到了报错，为我分析一下原因






Thursday 1:34 AM






Worked for 3m 6s



这次已经不是 CUDA_HOME 问题了，也不是 OOM。它已经成功启动 4 卡、加载 Wan2.2/AHA-WAM 模型，并进入数据集构建阶段。真正崩的是这里：
TypeError: 'NoneType' object is not subscriptable
...
src/ahawam/datasets/lerobot/utils/normalizer.py", line 38
cur_stats = ... stats["action"][key] ...
根因是：你用了 4 卡训练，同时让训练现场自动计算 stats：
data.train.pretrained_norm_stats=null
当前代码在多进程下的逻辑是：rank 0 去扫描数据并计算 dataset_stats.json，其他 rank 暂时拿到 dataset_stats=None。但这时候分布式 broadcast 还没正常发生，非 0 号 rank 就继续调用 normalizer，于是执行：
stats["action"]
但 stats 实际是 None，所以报：
'NoneType' object is not subscriptable
所以这次错误可以理解为：不能直接用 train_zero1.sh 4 第一次现算 dataset_stats.json。要先单进程生成 stats，再 4 卡训练。
还有两个隐藏问题也在日志里暴露了：
model.action_dit_pretrained_path=
data.train.text_embedding_cache_dir=
这说明你当前 shell 里的 $ACTION_DIT 和 $TEXT_CACHE 是空的。日志也明确说了：
No `action_dit_pretrained_path` provided, initializing ActionDiT with random weights.
也就是说，这次并没有用你之前生成的 2.0G ActionDiT 初始化文件。后面即使 stats 修好了，TEXT_CACHE 为空也会继续报缺 text embedding cache。
下一步建议这样做：
cd /home/fenghao/workspace/AHA-WAM
conda activate ahawam
source /home/fenghao/use-nvidia-580159.sh

export ACTION_DIT=$PWD/checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
export ROBOTWIN_DATA=$PWD/data/robotwin2.0-fastwam/robotwin2.0
export TEXT_CACHE=$PWD/data/robotwin_text_embeds_cache
export OUT=$PWD/runs/smoke_robotwin_ahawam

ls -lh "$ACTION_DIT"
ls "$ROBOTWIN_DATA/meta"
echo "$TEXT_CACHE"
然后先单进程生成 dataset_stats.json，再把生成的 stats 显式传给 train_zero1.sh 4。我已经把文档改成这个正确流程了：[AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md (line 386)](/home/fenghao/workspace/AHA-WAM/docs/AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md:386)。

Edited AHAWAM_OOM_TEST_AND_WAM_NOTES_ZH.md+65-25Review changesUndoReview





Thursday 1:37 AM