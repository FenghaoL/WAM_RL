# SO101 AHA-WAM 视频预览与微调策略核查

记录日期：2026-08-08；历史 K/V 实验与 preview 修正：2026-08-09（America/Detroit）

本文回答四个问题：动作与视频服务怎样运行、当前视频为什么失真、官方和相关 WAM 实际训练哪些模块、下一轮 SO101 训练应怎样选初始化和冻结边界。

## 1. 先给结论

1. 当前 `infer_with_video` 可以返回动作，但不能边返回动作边生成视频。服务先算动作、再同步生成完整视频，约 2 秒后一起返回，只适合低频诊断，不适合 15 Hz 连续控制。
2. 真正并行需要两个独立 GPU 进程和两个端口。动作服务持续控制，视频服务异步预览；两者默认不共享状态，视频不会影响动作。
3. 当前显式视频链路的 flow scheduler、固定首帧、Video DiT 和 VAE 解码与 FastWAM 的联合采样视频侧一致，没有发现会凭空引入“别的桌面”的符号、shape 或 flow 方向错误。
4. 该端点仍是 AHA-WAM 官方接口之外的实验扩展。2026-08-09 后已接入 AHA rolling-history K/V 和全局 RoPE 时间坐标；但视频配置 `action_conditioned=false`，所以它仍看不到本次 Action DiT 生成的 6D 动作。
5. 最终 SO101 checkpoint 的 825 个 Video expert 参数张量与 RoboCOIN 初始化权重全部相同。当前预览不是“SO101 微调后的视频模型”，而是未经 SO101 适配的 RoboCOIN 双臂、三相机视频先验。
6. 第 0 帧正确、未来逐步崩坏，且 10 步增加到 20 步没有改善，说明输入布局、VAE 编解码和首帧条件基本正确；主要问题是视觉域、机器人形态、相机布局和时间配置不匹配，而不是采样步数太少。
7. 不建议用当前 125 条、约 23 分钟数据从 Wan2.2 完全从头训练 AHA-WAM。实际可行的基座仍是 RoboCOIN AHA checkpoint，然后为 Video DiT 增加受控适配，再联合校准 Action/OVCR。

## 2. 三种请求的真实时序

### 2.1 `infer`

```text
当前双相机图像 + 6D state + language
  -> VAE encoder
  -> 必要时 Video DiT planner prefill / rolling K/V
  -> OVCR 用最新图像修正上下文
  -> Action DiT 10 步去噪
  -> 返回 [8, 6] 动作块
```

它不采样未来视频，也不调用 VAE decoder。这是默认真机路径。

### 2.2 `video_preview`

```text
当前双相机图像 + language
  -> 当前帧 VAE latent + 随机未来 latent
  -> 可用时读取过去 planner observation 的 rolling-history K/V
  -> Video DiT 20 步 flow sampling
  -> VAE decoder
  -> 返回并保存 13 帧 MP4
```

它不生成动作，不推进 action chunk index，但会把请求中的图像写入 policy 当前 observation。

### 2.3 `infer_with_video`

当前代码顺序是：

```text
1. 完整执行 infer，生成并消费一个动作块
2. 在同一进程中完整执行 video_preview 的采样
3. 动作和 MP4 一起返回
```

因此有两种用法：

- 客户端收到结果后执行动作：能控制，但每块动作前都要等待视频，控制频率约为 0.4 Hz 量级，机械臂会明显走走停停。
- 客户端只保存视频、丢弃动作：不能控制，而且该服务内部仍消耗了一个 chunk，之后再用同一进程控制会造成时序错位。

纯诊断应使用 `video_preview`。`infer_with_video` 主要用于验证“同一观测能同时得到两种输出”和记录成对结果，不是并发控制实现。

## 3. 两张 GPU 到底怎样分工

推荐架构是两个完全独立的进程：

```text
机械臂客户端
  |-- 每个控制周期 -> 12000 / GPU A / infer -> 立即执行动作
  `-- 每隔若干秒  -> 12001 / GPU B / video_preview -> 异步显示 MP4
```

二者的关系由客户端建立：给请求附同一个观测时间戳或 id，然后把实际执行录像与预测视频按时间对齐。默认情况下它们不共享：

- 模型参数所在显存；
- AHA planner rolling K/V；
- action chunk index；
- TCP 连接和端口；
- CUDA stream 或 worker。

这有一个重要好处：即使视频生成失败或耗时 3 秒，动作服务仍可继续控制。代价是显存里要加载两份约 7.23B 的 AHA 模型。

2026-08-08 检查时，GPU0 总显存 40 GiB、空闲约 28.4 GiB，但已有其他用户进程占约 12.1 GiB；GPU1-3 只剩约 11-13 GiB。单个本项目服务实测峰值约 19-20 GiB，所以当时只能在 GPU0 风险可控地运行一个服务，无法加载第二份模型。其他进程显存继续增长时，GPU0 也可能 OOM。

## 4. 视频采样链路核查

### 4.1 已对齐的部分

本地 `generate_future_video()` 做了以下事情：

1. 使用训练相同的 `384x320` 三槽布局：上方固定相机、左下腕部、右下缺失相机黑槽。旧批量结果使用的是腕部铺满底栏的旧布局，详见 `video_path_experiments/batch/README_ZH.md`。
2. 用同一个 Wan2.2 VAE encoder 编码当前帧。
3. 未来 latent 从固定 seed 的高斯噪声初始化。
4. 每个 flow step 都将第 0 个 latent 恢复为当前帧 latent，避免首帧漂移。
5. 使用模型的 `infer_video_scheduler.build_inference_schedule()` 和 `step()`。
6. 使用 Video expert 的 `pre_dit -> 30 blocks -> post_dit`。
7. 使用与训练相符的 causal video mask、clean prefix 和带全局偏移的 temporal ids，例如当前索引为 48 时使用 `[48, 52, 56, 60]`。
8. 可用时把过去 planner observation 的逐层 K/V 作为 attention prefix；图像哈希用于避免当前 first frame 被重复放进历史。
9. 最后使用同一个 Wan VAE decoder 还原 RGB。

FastWAM 仓库中的 `infer_joint()` 也是随机初始化 video latent、固定第 0 帧、逐步调用 video scheduler、最后 VAE decode。当前实现的 flow 符号和更新方向与它一致。

这里需要精确区分“没有调用训练包装函数”和“没有走同一个视频网络”。当前预览没有调用顶层 `training_loss()`，但使用与联合 forward 视频侧相同的 `video_pre -> 带历史前缀的 Video blocks -> video post_dit`。Video blocks 先于 Action blocks 计算，Video 输出不读取 Action 输出。因此当前预览是正确的**带可用历史的自由采样视频侧**，不是另一套陌生网络。

训练与自由采样的真正区别是：训练只随机选择一个噪声时刻，将 ground-truth future latent 与噪声混合后预测 flow；它不会在每个 batch 中从纯噪声完整去噪出 MP4。预览则从纯噪声开始连续积分 20 步。这意味着训练 loss 正常不等于自由生成视频一定正常。

### 4.2 不是官方 AHA 接口的部分

AHA-WAM 官方默认部署只公开动作推理，不公开未来 RGB 采样 API。当前端点参考 FastWAM 将 AHA 的 Video expert 单独拿出来采样，因此仍有一个核心限制：

- `action_conditioned=false`。视频只根据当前图像和语言预测一个可能未来，不根据本次动作块模拟机械臂后果。

第二点尤其重要：即使真实机械臂执行了返回动作，预测视频也不保证出现相同运动。它不是 action-conditioned simulator，也不能用于碰撞安全判断。

### 4.3 为什么判定主要是模型/域问题

已有证据形成了比较完整的闭环：

- 第 0 帧正确重建两个 SO101 相机，说明图像通道、尺寸、拼接和 VAE 路径没有整体错位。
- 约 1.6 秒开始出现机械臂重影、桌面颜色变化和视角漂移，约 3.2 秒严重崩坏，符合扩散视频模型在分布外条件上逐步回到预训练先验的现象。
- 10 个采样步改为 20 个采样步后现象没有本质变化，排除了最直接的欠采样解释。
- SO101 使用 2 个相机、6D 单臂、13 个视频帧；RoboCOIN checkpoint 使用 3 个相机、14D 双臂、9 个视频帧。外观、形态、布局和时间长度同时变化。
- 最终 checkpoint 的 Video expert 与 RoboCOIN 初始化逐张量相同，因此 SO101 训练不可能改善显式视频。

2026-08-09 已完成训练式历史 K/V 对照：SO101 有历史版本相对无历史版本 MSE 降低约 32%，但机械臂运动和腕部画面仍明显失真；RoboCOIN 原生域有历史版本达到 20.276 dB PSNR，场景结构稳定。完整输入、结果和脚本见 `scripts/so101_pipeline/video_path_experiments/RESULTS_ZH.md`。

## 5. 为什么训练日志有 `loss_video`，视频却没学

训练前 `freeze_video_dit=true` 会把 Video expert 的 `requires_grad` 关闭。前向过程仍计算视频 flow 预测，所以日志还能显示 `loss_video`；但这个 loss 没有可更新的视频参数路径。

当前配置又是 `action_conditioned=false`，Action expert 不参与视频预测。因此视频 loss 在这次训练中主要是监控指标，不是有效的视频 co-training 梯度。真正更新的是：

- Action DiT；
- 14D 到 6D 后重新初始化的 action 输入/输出层；
- proprio encoder；
- OVCR、chunk K/V editor 和相关路由模块。

这能形成一个有意义的“固定视觉特征 + 训练执行器”baseline，但不能据此评价 SO101 视频预测训练是否成功。

## 6. 官方和相关工作实际怎么训练

### 6.1 AHA-WAM

官方论文以 Wan2.2-TI2V-5B 初始化 Video DiT、文本编码器和 VAE，并对世界建模和动作预测都使用 flow matching。论文明确写道：两分支联合优化时，video loss 与 action loss 等权。仓库通用 `configs/train.yaml` 默认 `freeze_video_dit=false`。

论文只在 AHA-WAM-Flash 的动作 ODE 蒸馏阶段明确冻结 Video DiT，因为那一阶段只蒸馏动作采样器。不要把这个冻结规则外推为主模型 SFT 的标准策略。

RoboTwin 主实验从 Wan2.2 初始化，不做 embodied pretraining，但使用 50 个任务、2,500 条 clean 和 25,000 条 randomized demonstrations、全局 batch 512、5 epochs。这一数据规模远大于当前 SO101。

真实机器人实验则先在 RoboCOIN 子集 24,600 条轨迹、约 165 小时上预训练，再在每个目标任务约 120 条 demonstration 上微调。官方模型卡也把 RoboCOIN checkpoint 定义为 downstream real-robot fine-tuning 的初始化权重，而不是通用 zero-shot policy。

来源：[AHA-WAM 论文](https://serene-sivy.github.io/aha-wam/static/aha-wam.pdf)、[RoboCOIN 预训练模型卡](https://huggingface.co/SereneC/AHA-WAM-Pretrained)。

### 6.2 FastWAM

FastWAM 在训练时保留视频 co-training，推理时才删除显式未来视频。官方 trainer 冻结 VAE/text 等非 DiT 模块，但将整个 `model.dit` 加入 AdamW；这个 DiT 包含 Video 和 Action experts。论文和官方实现没有把 LoRA 或冻结 Video DiT 作为主训练方案。

其消融很有参考价值：在 RoboTwin 上，FastWAM 从 91.8% 降到无视频 co-training 的 83.8%；真实任务里删除视频 co-training 后成功率降到 10%。这说明“不在推理时生成 MP4”可以成立，但训练时让 Video DiT 学目标数据仍然重要。

这个 `83.8%` 不能直接当作当前冻结方案的预期下限。FastWAM 的“w/o video co-train”只删除 video loss，整个 DiT 仍在 optimizer 中；action loss 仍可通过 Action 对当前帧 Video K/V 的读取更新 Video DiT。当前 SO101 设置则把 Video DiT 完全 `requires_grad=false`，连 action loss 也不能更新它，约束更强。

来源：[FastWAM 论文](https://arxiv.org/pdf/2603.16666)、[官方 trainer](https://github.com/yuantianyuan01/FastWAM/blob/main/src/fastwam/trainer.py)。

### 6.3 Motus 与 Efficient-WAM

Motus 使用三阶段：先单独训练视频生成模型，再联合预训练全部专家，最后在目标机器人轨迹上微调全部专家。它支持“先把视频世界模型适配好，再学动作”的思路，但不是冻结原始通用视频模型直接完成全部目标机器人适配。

Efficient-WAM 的阶段划分更直接：Stage 1 蒸馏紧凑 Video expert，Stage 2 冻结已经适配过的视频分支训练动作，Stage 3 再做视频-动作联合 refinement。它说明冻结视频可以是合理的中间阶段，前提是视频 expert 已经对目标数据建立了有用表示，且最终还有联合校准。

来源：[Motus 官方项目](https://motus-robotics.github.io/motus)、[Efficient-WAM 官方仓库](https://github.com/jiajun613/Efficient-WAM)。

### 6.4 LoRA 的位置

AHA-WAM 和 FastWAM 的官方主配方都没有 LoRA。LingBot-VA 的官方 post-training 采用 FSDP；Hugging Face LeRobot 的 LingBot-VA 集成则为显存有限场景提供 PEFT/LoRA。这说明 LoRA 是合理的工程折中，但目前没有证据表明它天然优于 WAM 的联合全量训练。

当前 AHA-WAM 仓库没有 LoRA 注入、保存、恢复和部署加载实现。它不能靠一个 YAML 开关启用。

## 7. 基座应该怎样选

### 7.1 推荐：RoboCOIN AHA checkpoint

它仍是当前最合适的主基座：

- 与 AHA 架构、OVCR 和 checkpoint key 完全匹配；
- 已包含真实机器人视频和动作先验；
- 官方真实机器人实验就是先 RoboCOIN 预训练再做 task-specific fine-tuning；
- 比 RoboTwin 模拟双臂 checkpoint 更接近真实相机分布。

它的双臂、三相机先验会在未适配视频中产生重影，但这更支持“适配 Video DiT”，并不支持丢掉全部预训练。

### 7.2 不推荐当前数据直接从 Wan2.2 起步

纯 Wan2.2 提供通用视频生成能力，但没有 SO101 动作专家、OVCR 和真实机器人 action semantics。AHA 在 RoboTwin 从 Wan 起步成功依赖约 27,500 条多任务 demonstrations；当前数据只有 125 条、20,756 帧，按 15 Hz 约 23.1 分钟，平均每任务约 30 条。

从 Wan 起步可作为科研对照，不应作为当前最可能成功的部署路线。

### 7.3 RoboTwin checkpoint

它经过大规模 AHA 联合训练，但视觉是模拟环境、动作为 14D 双臂、相机为三视角。对单臂真实 SO101，域差通常大于 RoboCOIN，因此不作为首选。

### 7.4 当前 SO101 final checkpoint

下一阶段可从当前 final checkpoint 继续，因为 Action/OVCR 已学过 6D SO101。先适配 Video 分支后，必须再做一次联合 action refinement；否则 Video 表示变化后，原 Action/OVCR 所读取的 K/V 分布也会变化。

Fast-WAM checkpoint 是不同模型结构，不能直接塞进 AHA-WAM 当基座。若选择它，就是切换模型方案，而不是替换一个初始化文件。

## 8. 下一轮最有信息量的实验顺序

### 阶段 A：先把动作问题和视频问题分开

在不上电或限速条件下，对完整 7-episode held-out 集做离线 action evaluation：

- 比较预测 `[8,6]` 与 ground truth；
- 记录每个关节 `predicted - current`、反归一化结果和 G05 逆变换结果；
- 记录客户端限幅前后实际命令；
- 不只看当前 trainer 每次随机抽少量样本的 `action_l1/action_l2`。

如果真实机械臂完全不动，要先判断是模型输出接近当前 state、3 tick 安全限制全部截断、坐标逆变换错误，还是客户端根本没有执行返回动作。坏视频不能替代这项检查。

### 阶段 B：视频链路可学习性 sanity check

选择 1-8 条训练 episode，做短时过拟合实验：

- 从当前 final checkpoint 或 RoboCOIN checkpoint 初始化；
- 冻结 Action/OVCR；
- 只训练 Video expert 的最后 2-4 个 blocks，或给上层 attention/FFN 加 rank 8/16 LoRA；
- Video LR 从 `1e-5` 或更低开始；
- 每 50-100 steps 用固定首帧、固定 seed 生成 9 帧和 13 帧。

若训练 loss 和可视结果都不能记住极小训练集，再继续查 history、mask、temporal ids 或优化器链路；若能记住，当前采样端点就具备可学习性，主要问题是冻结和数据规模。

### 阶段 C：实用的 SO101 联合适配

更适合当前 4x40GB 资源和小数据的方案是：

```text
基座：当前 SO101 final checkpoint
Action/OVCR：全量可训练，LR 约 5e-5
Video：上层 partial unfreeze 或 LoRA，LR 约 5e-6 到 1e-5
VAE / UMT5：冻结
目标：action loss + video loss，保留 held-out episode 验证
```

需要为 trainer 增加明确的参数组和保存/加载逻辑。LoRA 还需要推理端重建 adapter 或训练后 merge。

### 阶段 D：全量联合训练只作为后续对照

这是最接近官方 AHA/FastWAM 的训练方式，但当前 3 卡 ZeRO-2 冻结 Video DiT 已达约 37.4 GiB/卡。直接解冻 5B Video expert 必然增加 gradient、optimizer state 和 activation，3 卡 ZeRO-2 不现实，4 卡 ZeRO-2 也很可能 OOM。

若坚持全量训练，应改用 ZeRO-3 或 FSDP full-shard、gradient checkpointing，必要时 optimizer/parameter CPU offload，并先做 5-20 steps 冒烟。它是分布式内存工程调整，不改 AHA 算法目标；但训练会更慢，也更容易在 125 条轨迹上过拟合。

## 9. 最终建议

当前 frozen-video checkpoint 应保留为动作 baseline，不要因为预览差就直接删除。下一步优先级是：

1. 完成全验证集动作离线评估和真机命令链路日志，解释“为什么不动”。
2. 做 Video upper-block/LoRA 的极小集过拟合试验，验证视频链路可学习性。
3. 继续使用 RoboCOIN AHA 或当前 SO101 final 为基座，进行低 LR 的 Video 适配和 Action/OVCR 联合校准。
4. 数据增加到覆盖更多起始位置、光照、物体姿态和失败恢复后，再比较 partial/LoRA 与全量 ZeRO-3。

不建议用 `infer_with_video` 同步控制来判断训练策略。它会把视频采样延迟混入控制，并且视频不读取动作。更干净的实验是动作服务独立控制、视频服务异步观察；在第二张空闲 GPU 出现之前，分别启动两种服务、用相同离线帧做对照即可。
