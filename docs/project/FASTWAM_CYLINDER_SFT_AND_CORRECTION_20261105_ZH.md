# FastWAM 圆柱任务：公开权重核查、四项训练与存储策略

日期：2026-11-05。用户最终选择：**Franka 真机四任务联合初始化为主线，另做 AHA 作者发布的 Fast-WAM RoboCOIN 初始化对照；全部使用原生 FastWAM，不输入历史帧。**

## 1. 找到的公开权重及来源可信度

以下区分“官方论文发布”“其他作者团队复现”“第三方下游训练”，不将不同来源统称为官方。清单按本次公开文件/API/model card/config 实查整理；不声称覆盖全网所有上传，也不根据文件夹名称臆测缺失的训练 provenance。

| 来源与权重 | 训练数据/任务 | 真机或模拟 | 单臂/双臂 | 骨干与兼容性 | 本次用途 |
|---|---|---|---|---|---|
| FastWAM 官方 `yuanty/fastwam/libero_uncond_2cam224.pt` | LIBERO Spatial/Object/Goal/Long 操作任务 | 模拟，MuJoCo/robosuite | 单臂 Panda | Wan2.2 5B video＋Action DiT；需要适配 SO101 动作/状态与相机 | 已列出，不下载 |
| 同上 `robotwin_uncond_3cam_384.pt` | RoboTwin 2.0 | 模拟 | 发布接口为双臂，14D | 三相机、384×320；本仓库已有原生部署适配记录 | 已列出，不下载 |
| 同上 `libero_optional_idm_2cam224.pt` | 同系列 LIBERO 任务 | 模拟 | 单臂 | Optional IDM 新变体；支持先想象再动作或仅首帧模式，非本轮普通 FastWAM 基线 | 已列出，不下载 |
| AHA-WAM 作者 `SereneC/AHA-WAM-Pretrained/Fast-WAM-pretrained.pt`；model card 宣称 45,000 step，**文件内部为 7,500，未解释** | 与其 AHA 权重相同的选定 RoboCOIN 子集 | 真机 | 发布接口为双臂，14D state/action，头部＋双腕相机；完整 RoboCOIN 的其他构型不等于此 checkpoint 的训练组成 | 本仓库 FastWAM 复现实现；65-frame、ratio 8；与当前代码直接相容的候选 | **独立 demo SFT 对照** |
| `aabyaneh/fasterwam/joint/fastwam/train90_joint_job4415/.../step_058620.pt` | fruit_pick、dish_utensil、place_cube、real Push-T，联合训练；360 train/40 held-out | 真机 | 单臂 Franka Panda | Wan2.2 5B＋1024 hidden Action DiT；action 7D、proprio 8D；224×448 双相机；原始动作是 delta/夹爪约定，需重新学习 SO101 接口 | **主线初始化** |
| 同仓库的单任务 FastWAM，以及归档 `SleepMastger/fruit-picking-fastwam`、`pusht-fastwam`、`dish-utensil-fastwam-v2` 等 | 单独果实抓取、餐具、真实 Push-T；place_cube 也有单任务权重 | 真机 | 数据说明为单臂 Franka | 下游任务窄、历史划分不一定等同修正后的联合 run；各自元数据不同 | 备选，不混入主线 |
| `typoverflow/fastwam` 中 `pretrain_droid*`、`cotrain-droid` 等 | config 显示 DreamZero/DROID、teleop 及混采设置；目录间配方不同 | DROID 部分为真机；自定义 teleop 的完整来源未充分说明 | DROID 为单臂 Franka；额外 teleop 的构型不能仅凭仓库名确定 | 可见自定义 10D action、8D proprio，以及不同 Action DiT 宽度/归一化；缺少足够 model card/评测信息 | 低确定性备选，本次不采用 |
| `angelic123/fastwam_droid_wan21_1p3b_step100k` | DreamZero-DROID-Data，step 100,000 | 真机 | DROID 单臂 Franka | **Wan2.1-T2V-1.3B**，video hidden 1536、合计约 2.06B，7D；不是本轮 Wan2.2 5B 架构，不能直接加载其 Transformer 主干 | 另一个架构候选，不采用 |

官方入口现在是 [FastWAM 官方仓库](https://github.com/yuantianyuan01/FastWAM) 链接的 [yuanty/fastwam](https://huggingface.co/yuanty/fastwam)。旧文档中的 `zrgong/fastwam` 本次访问未成功，不再将它作为可下载入口。LeRobot 还有[官方集成与转换格式](https://huggingface.co/docs/lerobot/main/en/fastwam)，这些不是另一份不同数据预训练的独立证据。

RoboCOIN 权重来自 [AHA 作者 model card](https://huggingface.co/SereneC/AHA-WAM-Pretrained)，对方明确标注 Fast-WAM 是其**复现版**、同一选定 RoboCOIN 子集、非原 FastWAM 官方权重。没有据此添加未经公布的精确子任务数或轨迹规模。

Franka 联合权重来自 [job4415 说明](https://huggingface.co/aabyaneh/fasterwam/blob/main/joint/fastwam/train90_joint_job4415/README.md)，配套有配置、源码快照、数据 revision、划分、归一化和 hash inventory。[数据卡](https://huggingface.co/datasets/aabyaneh/fasterwam)说明为 Franka Panda 遥操作四任务，每任务 100 episode。该 run 因时间限制在 epoch 26 中断，最新发布的是 **epoch 20、step 58,620**，不是完整 50 epoch 或不存在的 epoch-30。修正后的 place_cube crop revision 为 `4ad42310b4408ff08370124e680e3a20ad267b1e`。其验证指标不等于真机成功率，本次选择依据是来源可追溯及真机域接近，不是已有 SO101 效果保证。

DROID 的真机来源见 [DROID 官方站](https://droid-dataset.github.io/)，对应模型及配置分别见 [typoverflow 文件列表](https://huggingface.co/typoverflow/fastwam/tree/main)、[Wan2.1 DROID 模型说明](https://huggingface.co/angelic123/fastwam_droid_wan21_1p3b_step100k)。这些为第三方训练结果，未把它们认作 FastWAM 原作者的 DROID 官方发布。

## 2. 本次固定的下载与适配

Franka 固定模型仓库 revision `692b2912246655435f97801219f289c71bc6c417`，只下载修正后 joint FastWAM 的 epoch-20 权重和必要配置，避免把 FasterWAM/FlashWAM 当成 FastWAM。

RoboCOIN 固定 revision `dd11c34ce319c411b62317f0808729f343530eca`。权重路径分别为：

```text
checkpoints/FastWAM-Franka-Joint/joint/fastwam/train90_joint_job4415/checkpoints/weights/step_058620.pt
checkpoints/FastWAM-RoboCOIN/Fast-WAM-pretrained.pt
```

下载后对照 Hugging Face LFS SHA256，结果写各权重旁的 `*.source.json`。初始串行下载改为有范围验证的并行下载，复用已下前缀、不下载第二份完整文件；最终校验通过才发布正式 `.pt`。

两份下载已完成并通过校验：Franka `b763a2cb81026ec36e99b1f45083c7d295cc64526b8712b7197d5eeabda7677d`，RoboCOIN `ff953284f669b6fbf551c5e7ab88aa1c3004a0bd8b314df1068eecb68cadffc8`。两份都通过 meta skeleton 全骨干检查，实际载入 1,646 个匹配张量，另有三个动作接口张量及本地兼容 embedding 不直接载入。

**来源异常：RoboCOIN 的 model card/config 宣称 Fast-WAM 45,000 step，但已校验文件内嵌 `step=7500`。** 本轮未改写该字段，也未推断两者是否为 local/global/续训计数；已明确告知用户。权重身份固定为上列 revision＋SHA256，结构可用，训练总步数不能当作已验证事实。Franka 内嵌 step=58,620 与发布说明一致。审计见 `robocoin_checkpoint_audit.json` 及 `queue.json`。

从 7D/14D action 迁移到 6D SO101 时，仅允许重新初始化动作输入和输出的三个不匹配张量：`action_encoder.weight`、`head.weight`、`head.bias`。Franka 8D / RoboCOIN 14D proprio 与 SO101 6D 不共享关节语义，因此重新初始化整个 proprio encoder，**不以插值/平均池化猜测关节对应**。其他 Transformer 张量形状或键名不匹配会报错。模型建立前使用 meta device 审计全部 backbone shape，避免仅凭模型名认定兼容。

纠正训练从已适配的 Franka demo SFT 启动，要求相同架构/接口严格匹配，不再允许形状跳过。归一化使用 SO101 demo 训练统计，不沿用 Franka delta action / min-max 或 RoboCOIN 14D 统计。

## 3. 四项实验与依赖关系

运行根目录：`runs/fastwam_cylinder_20261105`。队列依次运行，使用物理 GPU **0、1、2**；每阶段先做 2 步真实 smoke，再从原始阶段初始化重新开始正式训练，smoke 不作为正式模型。

| 队列项 | 初始化 | 数据 | 更新范围 | 正式更新步数 |
|---|---|---|---|---:|
| `franka_demo_sft` | Franka 四任务 joint epoch-20 | cylinder insertion v1 纯 demo | Video DiT＋Action DiT＋proprio encoder | 6,572，4 个 demo epoch |
| `robocoin_demo_sft` | AHA 作者 Fast-WAM RoboCOIN 权重 | **与第一项完全相同的纯 demo** | 与第一项相同 | 6,572 |
| `franka_action_only` | **本轮 Franka demo SFT 最终权重** | demo＋HIL v2 HUMAN，严格 1∶1 | Action DiT＋proprio encoder；冻结 Video DiT，video loss=0 | 3,285 |
| `franka_full_vidext` | **同一个 Franka demo SFT 最终权重** | 与上一项相同；成功 rollout 的未来视频可越过交还边界 | Video DiT＋Action DiT＋proprio encoder，video/action loss 各 1 | 3,285 |

两条纠正路线是**共同 SFT checkpoint 的独立分叉**，全量路线不从 action-only 的结果继续训练。按用户后续要求，执行顺序调整为原编号 **1→4→2→3**，RoboCOIN demo 对照提前到第二个运行，不覆盖 Franka 主线。

每 GPU microbatch=4、3 GPU、accumulation=1，全局 batch=12；BF16、ZeRO-2 optimizer CPU offload、混合 attention 与分支 gradient checkpointing，AdamW lr=1e-5、weight_decay=1e-2、betas=(0.9,0.95)、cosine＋5% warmup、grad clip=1。沿用被引用 AHA 配方的数据曝光预算，不因换模型默默增加纠正轮数。

## 4. 原三份 YAML 实际对应哪些数据

`cylinder_v1_sft.yaml` 的 demo 训练目录是：

```text
data/so101_ahawam_cylinder_insertion_v1/01_pick_up_and_insert_cylinder/
  20260904_014410
  20260904_020841
  20260904_023259
  20260904_024404
```

合计 19,708 个训练帧；独立 demo 验证为 `20260904_024009_val`，661 帧。

`cylinder_v2_p1_action_dit_only.yaml` 不是仅用 HUMAN 的 correction-only 训练：原配方为原 demo 与 HIL v2 HUMAN **1∶1 混采**、step 3,285，从 task SFT 开始；Video DiT 冻结、视频损失关闭。AHA 的 action 路径还包括 OVCR/观测编码器，而 FastWAM 没有这些 AHA 模块；本轮对应 Action DiT＋proprio encoder。

`cylinder_v2_p1_full_balanced_video_vidext.yaml` 同样是 1∶1 demo/HUMAN，但联合更新 video/action，且 HUMAN 起点的未来视频能继续到成功 rollout 的后续非 HOLD 执行画面；**动作/state 标签仍截止 HUMAN 段**。没有把 POLICY 的后续动作当成人类监督。

本轮复用这份已冻结的纠正 train/val episode 选择：5 个训练采集目录、3,460 HUMAN 训练帧；`20260912_010747_val` 的 episode 0–3 用作 HUMAN 验证，共 169 帧。demo＋HUMAN 的自然索引数为 23,168，但 sampler 每个本地 batch 固定 2 demo＋2 HUMAN，以相同预算重复采样较小的 HUMAN 池。

**本次 SFT/HG-DAgger 对照不使用 10 月离线 RL 的特殊 recipe**：不把两条失败移到新的 val，不对四条成功片段截半，也不把旧 `_val` 的未观测成功 episode 移进训练。这次目标是复用原三份部署配置对应的训练对照。逐目录 episode 名单已固化到 `manifest.json` 和各 YAML。

## 5. FastWAM 与 AHA 的时序差异

用户已确认不需要历史帧。原生 FastWAM 仅用当前观察的视觉 KV 对动作建模；没有 AHA 的长历史 prefill、异步 offset、OVCR query 或 chunk-causal teacher forcing。训练时仍联合预测未来视频，但 action attention 只读取当前首帧视觉 token。

为保留原始 49 个控制帧的视频时域，采样 `[0,4,...,48]` 共 13 张视频图像，原生 FastWAM 对应 **48 个动作目标**，执行端仍可每次取前 8 个。AHA 原配置则在 49 帧窗口中训练 40 个动作、offset 0–7。不能直接复制 40 动作给原生 FastWAM：其训练要求动作长度能整除视频 transition 数 12，而 40/12 不成立。

因此本轮明确选择：49 原始帧、13 视频帧、48 action horizon、8 执行动作、无 offset、无 history。保持场景、相机、数据划分、采样比例和更新预算；这是两个架构的必要接口差异，不声称所有张量布局逐项相同。

双相机图像仍采用 SO101 的 384×320 拼接：上方 fixed，下方左 wrist、右黑色空槽，与现有 SO101 服务保持一致；Franka 原来的 224×448 横向图像布局不直接复用，需通过 demo SFT 适配。

未来视频 extension 沿用现有 `SO101DemoHumanCorrectionDataset`：成功轨迹越过 HUMAN 边界时只扩展图像；失败不延伸。末尾 padding 有 image/action mask；无完整视频标签的纠正仍可提供动作监督，但不计直接 video loss。本地普通 FastWAM 原先没有读取 `video_supervision_mask`，所以新 wrapper 补上与 AHA 对照一致的 eligible-sample 视频损失归一化，避免把短片段的重复末帧当作真实预测目标。

已逐个审计全部 3,460 个 HUMAN 训练起点：3,052 个可延伸未来视频、1,899 个拥有完整 49 帧时域；动作标签跨 HUMAN 边界、延伸尾部纳入 HOLD、跨 episode 均为 0。结果在 `boundary_audit.json`。这不要求所有短纠正都有视频损失；它们仍有被 mask 正确限制的动作监督。

## 6. 第二轮训练是否继承 optimizer

仓库现有 trainer 有三种入口：

| 配置 | 模型权重 | Adam 动量/方差 | scheduler/训练进度 |
|---|---|---|---|
| `init_checkpoint: xxx.pt` | 加载 | **不恢复** | 从新任务的第 0 步开始 |
| `resume: xxx.pt` | 加载 | 不恢复 | 从文件名/配置推算进度并跳过已读数据，不等于完整恢复 |
| `resume: checkpoints/state/...` | 加载 | DeepSpeed-compatible 同后端 state 可恢复 | scheduler、RNG 和进度一起恢复；某些跨后端兼容路径仍只提取权重 |

这几轮 AHA 的 task-SFT→correction 本来使用 `init_checkpoint`，`resume=null`、`zero2_save_training_state=false`，所以没有继承前一阶段优化器。模型的 `load_checkpoint(..., optimizer=None)` 在构建新优化器前执行，DeepSpeed 的 FP32 master weights 才能从正确模型参数创建。

本次也**不继承 optimizer**：从 demo-only 转为重采样的 HUMAN 混合分布，尤其 action-only 冻结范围变化，旧 Adam 统计和旧 schedule 不适合被视为同一次连续训练。重新 warmup 并使用新 schedule 更清楚，也节省 optimizer checkpoint 空间。

同一训练阶段因系统中断而继续时，完整 optimizer state 有助于精确恢复，但用户选择空间优先：本轮只保留权重，重启需重新初始化优化器，不能声称 bitwise 等价续训。保留少量 config、normalization、进度、loss/显存和 source hash 文本文件，便于部署与复现；它们不是大体积 optimizer state。

## 7. 存储与运行入口

四阶段均仅保留各自**最新一个模型权重**；新权重写临时文件并原子发布后删除同阶段旧权重。最终约四份 11.2 GiB 模型，加两份初始化共约 67 GiB，另有临时写盘峰值和已有共享 Wan 组件。依赖中的 Franka SFT 最终权重保留，两条分支都从它读入。Smoke 不写模型，不写 optimizer/FP32 master/ZeRO state；每个 run 的小型日志、配置与统计继续保存。

| 入口/文件 | 内容 |
|---|---|
| `scripts/so101_pipeline/fastwam_cylinder/download_assets.py` | 固定 revision、分段下载、LFS SHA256 核对 |
| `prepare.py` | 从原 demo/纠正配置生成四份 FastWAM YAML 和数据 manifest |
| `checkpoint_audit.py` | CPU mmap＋meta skeleton 核对全部骨干 shape 和重建接口 |
| `model.py` | 原生 FastWAM wrapper，受限跨机器人加载＋视频监督 mask；无 AHA 模块 |
| `trainer.py` | 权重保存上限、双验证集、进度/显存；基于已有 ZeRO-2 trainer |
| `run.py` | data-only、2-step smoke、正式训练 |
| `experiment_suite.py` | 四项顺序队列，依赖与错误停止 |
| `deploy_configs.py` | 为四项最终权重生成原 action server 可用的 FastWAM YAML |
| `runs/fastwam_cylinder_20261105/queue.json` | 队列状态、命令、PID、兼容性审计与各阶段结果 |
| 同目录 `*_smoke.log` / `*.log` | 实际训练日志 |
| 各 run 的 `model/progress.json` | 最新 step、显存、forward/backward/optimizer 时间 |
| 各 run 的 `model/initialization.json` | 载入/重建张量、训练参数量、数据量 |
| 各 run 的 `model/checkpoints/weights/step_*.pt` | 该阶段唯一保留的模型权重 |

队列启动命令如下；本轮已注册队列，**不要重复启动**：

```bash
bash scripts/so101_pipeline/fastwam_cylinder/launch_suite.sh \
  --root runs/fastwam_cylinder_20261105 --gpus 0 1 2
```

队列等待每个正式初始化 `.pt` 原子发布，先审计、再 smoke、再正式训练。任一审计或进程失败立即停止，不拿随机初始化继续下一项。所有时间/显存结果以实际文件为准，不能用 2-step smoke 预测真机成功率。

**启动验证：** 5 项针对接口适配、video mask、冻结范围、权重保存和配置分叉的 CPU 测试通过；完整 HUMAN 边界审计通过。Franka demo 的三卡 full-model smoke 已完成两个真实更新，loss 分别为 3.5130 / 4.3596，action/video 两项均有限，梯度范数为 17.8812 / 18.4178（clip 前）。峰值 allocated 21,060 MiB，约 20.57 GiB；reserved 峰值 24,312 MiB。第二步 forward 0.537s、backward 6.412s，不据两步估计最终学习质量。正式 6,572 步 SFT 已接续启动，其余三项待依赖完成。Smoke 权重未保存、未复用。

正式进程已完成 step 1/6,572，loss=3.5139，action=2.4097、video=1.1041，梯度有限。读取时 `queue.json` 为 running；这是实际参数更新验证，不只是进程启动或显存分配。实时 step 继续读取 `franka_demo_sft/model/progress.json`，不要把本文的启动快照当作最新进度。

### 后续调整：1→4→2→3 与 W&B

用户要求后，已将队列调整为 **Franka demo SFT → RoboCOIN demo SFT → Franka action-only → Franka full-vidext**。调度器接管原训练进程 `2812510`，没有重启 trainer、重建 optimizer 或回退训练进度。操作记录为 `reorder.json`、`wandb_enable.json`；当前 supervisor PID 以 `queue.json` 为准。

首次启动时 W&B 关闭；用户随后要求启用。当前 Franka SFT 无需重启，使用 `wandb_bridge.py` 从已有完整控制台记录补传，再每 5 秒跟踪一次训练日志、validation JSONL 及匹配 step 的显存快照。历史 loss/梯度等只能恢复**控制台打印精度**；未记录的历史逐步显存不伪造。后续三项直接启用原生 SDK，项目为 `fastwam-so101-cylinder`，group 为 `cylinder-20261105`，name 与各实验目录名相同；smoke 保持不上传。

项目：[W&B 项目入口](https://wandb.ai/fenghaoliu8-university-of-michigan/fastwam-so101-cylinder)。当前 run：[franka_demo_sft](https://wandb.ai/fenghaoliu8-university-of-michigan/fastwam-so101-cylinder/runs/fwc1105frankademo)。本地跟踪状态在 `wandb_bridge/franka_demo_sft.json`，云端核验在 `wandb_cloud_verified.json`。记录训练 loss、action/video loss、梯度、学习率、吞吐、前后向耗时以及验证指标，不上传模型、optimizer 或原始视频/数据 artifact。

使用自定义 `global_step` 作为横轴，允许训练和验证在同一 optimizer step 分别记录，避免重复使用 W&B 内部 step 时丢弃验证数据。原始 `model/config.yaml` 仍保留当前进程启动时 W&B=false 的事实；运行配置目录中的 YAML 已改为未来重新启动时启用，桥接 run 的 config 额外注明当前观测方式。

四项训练完成后对应部署配置已提前生成，当前尚需等待目标权重产生：

| 部署 YAML（均在 `online_deploy/deployments/`） | GPU | 端口 |
|---|---:|---:|
| `cylinder_fastwam_franka_demo_sft.yaml` | 0 | 12140 |
| `cylinder_fastwam_franka_action_only.yaml` | 1 | 12141 |
| `cylinder_fastwam_franka_full_vidext.yaml` | 2 | 12142 |
| `cylinder_fastwam_robocoin_demo_sft.yaml` | 3 | 12143 |

复用 `bash scripts/so101_pipeline/online_deploy/run_action_server.sh <YAML路径>`；注释区也列出四项。配置使用 `FastWAMPolicy`、48 动作 horizon、只返回前 8 动作、10 步去噪，训练用 SO101 stats；不加载任何 optimizer。当前没有启动机器人控制或额外部署服务。

## 8. 本轮已发现并显式处理的问题

- 原“action-only”并不是 HUMAN-only 数据：已纠正为 demo/HUMAN 混采，保留原对照预算。
- FastWAM 没有 history/offset/OVCR，不能把 AHA YAML 的这些开关直接复制并宣称生效；用户确认无 history。
- 原生 13 帧视频与 40 动作不相容：已改为 48 动作，并保留 8 动作执行长度。
- Franka 动作与状态语义不同：重建 6D 接口，不假装可以零样本控制 SO101。
- `job4415` 没有 epoch-30 权重，选择其最新真实保存的 epoch-20。
- RoboCOIN 宣传步数 45,000 与已校验文件内的 7,500 不一致；保留证据并提示，不擅自解释。
- FastWAM 原始 loss 忽略 HUMAN 短窗口 video mask：增加受控 wrapper 修复，不改其他 AHA/RL 模型。
- 新脚本最初名为 `queue.py` 时遮蔽 Python 标准库，已在 GPU 启动前更名为 `experiment_suite.py`；数据检查随后通过。
- 沙盒数据检查写入 Hugging Face 全局缓存被阻止后，改用 `/tmp/fastwam_hf_datasets`，不以此解释为数据损坏。
- 首次 GPU launcher 缺少 `CUDA_HOME`，在进入模型训练前退出；已设置 conda CUDA toolkit 并沿用既有驱动兼容环境重启。失败记录保存在 `initial_cuda_env_failure.log` 与 `queue_initial_cuda_env_failure.json`，不算作训练结果。
- 随后加载权重成功，但 CPUAdam 找不到 conda 环境里的 ninja（进程 PATH 没有完整激活 conda）；已补全 PATH、复用已有 CPUAdam 扩展，并在大型模型启动前加入 `CPU_ADAM_READY` 工具链检查。该次未进行 optimizer 更新，记录在 `initial_path_failure.log`。新入口 `launch_suite.sh` 统一环境，避免用户重复遇到此问题。
- 网络盘 mmap 权重直接复制到 CUDA 的加载阶段较慢；真实模型加载改为普通 CPU storage，结构审计仍用 mmap。临时 CPU 权重在 optimizer 建立前释放，不增加持久磁盘文件。

部署与 RLT 后续工作应使用这些原生 FastWAM 的真实 task-SFT 权重重新评估。FastWAM 的当前视觉 KV 更易作为外部 actor–critic 的状态接口，但这次四项训练仍是 SFT/HG-DAgger 对照，不额外启动 RL。
