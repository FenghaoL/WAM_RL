# AHA-WAM 架构全解：从向量数学到训练、推理与源码

> 适用代码版本：本仓库当前工作树，检查时 HEAD 为 `f30640c`。SO101 数字另根据当前未提交的训练/部署 yaml 复核。
>
> 这是一份“从零开始、以当前源码为准”的架构教材。论文用于解释设计意图，源码用于判定实际行为。凡是论文表述与当前代码不完全一致的地方，本文都会明确区分。

## 0. 先给出一句话答案

AHA-WAM 不是“先生成一段未来 RGB 视频，再从视频反推动作”的流水线。它是一个双 DiT 系统：

- 慢速的 **Video DiT** 在训练时学习视频 latent 的流匹配，因此其逐层 K/V 被迫包含场景如何演化的世界知识；
- 快速的 **Action DiT** 直接从噪声动作生成短动作块；
- 在线时通常不解码未来视频，而是缓存 Video DiT 对当前观测算出的逐层 K/V；
- 每次生成动作块前，**OVCR** 用最新相机图像把这份旧 K/V 改写成当前动作块专用的 K/V；
- Action DiT 每一层都让动作 query 同时读取“自己的动作 K/V”和“OVCR 更新后的视觉规划 K/V”；
- Video DiT 可以低频刷新，Action DiT 可以高频闭环执行，两者因此可以异步。

最短的数据流可以写成：

```text
语言 ──UMT5──> 文本 token ─────────────┬────────> Video DiT cross-attention
                                      └────────> Action DiT cross-attention

当前图像 ──Wan VAE──> 当前 latent ──Video DiT──> 每层 planner K/V（慢速缓存）
    │
    └──Wan VAE──> 最新观测 latent ──投影/查询池化──> OVCR ──> 每块更新后的 K/V
                                                               │
当前本体状态 ──Linear──> proprio token ─────────────────────────┤
高斯噪声动作 ──Action DiT + 多步流积分 <─────────────────────────┘
                         │
                         └──> 归一化动作块 ──反归一化──> 机械臂命令
```

这里最容易混淆的三件事是：

1. “horizon-adaptive”不是在线自动改变 horizon，而是训练模型适应 planner 与 executor 的不同 horizon 和相位偏移。
2. “world model”不等于在线显式播放未来视频；未来视频主要提供训练监督。
3. “asynchronous”是可采用的运行调度，不是单次 `model.infer()` 天然自动并行。仓库另外提供了线程/双进程双 GPU 的异步服务实现。

---

## 1. 阅读地图：各文件负责什么

建议按下列顺序读源码：

| 层次 | 文件 | 作用 |
|---|---|---|
| 配置 | [`configs/model/ahawam.yaml`](../../configs/model/ahawam.yaml) | 两个 DiT、scheduler、chunk 和 OVCR 的默认超参数 |
| 实例化 | [`src/ahawam/runtime.py`](../../src/ahawam/runtime.py) | 组装 VAE、文本编码器、Video DiT、Action DiT、MoT 和 AHA-WAM |
| 数据 | [`robot_video_dataset.py`](../../src/ahawam/datasets/lerobot/robot_video_dataset.py) | 采样视频/动作、拼相机、offset、history、文本缓存 |
| 预处理 | [`ahawam_processor.py`](../../src/ahawam/datasets/lerobot/processors/ahawam_processor.py) | 图像变换、动作/状态归一化与维度合并 |
| 视频 VAE | [`wan_video_vae.py`](../../src/ahawam/models/wan22/wan_video_vae.py) | RGB 与 48 通道时空 latent 之间的映射 |
| Video DiT | [`wan_video_dit.py`](../../src/ahawam/models/wan22/wan_video_dit.py) | 视频 patch、3D RoPE、时间调制、视频流预测 |
| Action DiT | [`action_dit.py`](../../src/ahawam/models/wan22/action_dit.py) | 动作 token、1D RoPE、chunk mask、动作流预测 |
| 双专家耦合 | [`mot.py`](../../src/ahawam/models/wan22/mot.py) | 逐层 Q/K/V、缓存、混合注意力及 OVCR K/V editor |
| 公共 WAM | [`base_wam.py`](../../src/ahawam/models/wan22/base_wam.py) | 输入构造、VAE、文本、本体状态、scheduler、通用推理 |
| chunk 框架 | [`ahawam_chunk_base.py`](../../src/ahawam/models/wan22/ahawam_chunk_base.py) | chunk 观测、query encoder、两阶段推理、动作去噪 |
| AHA-WAM 本体 | [`ahawam.py`](../../src/ahawam/models/wan22/ahawam.py) | OVCR 主训练、offset、rolling memory 与部署语义 |
| scheduler | [`scheduler_continuous.py`](../../src/ahawam/models/wan22/schedulers/scheduler_continuous.py) | 线性 flow path、shift、损失权重、Euler 更新 |
| 训练器 | [`trainer.py`](../../src/ahawam/trainer.py) | 冻结规则、AdamW、Accelerate/DeepSpeed、训练循环 |
| 在线封装 | [`deploy/server/ahawam_policy.py`](../../deploy/server/ahawam_policy.py) | 图像/状态归一化、prefill、动作块、反归一化 |
| 异步运行 | [`deploy/server/async_runtime.py`](../../deploy/server/async_runtime.py) | 视频刷新与动作请求的线程调度 |
| 双 GPU 服务 | [`wam_policy_server.py`](../../deploy/server/wam_policy_server.py) | video worker 和 action worker 通过最新 K/V 状态通信 |

---

## 2. 必要的数学前置知识

### 2.1 标量、向量、矩阵和张量

- 标量 $x\in\mathbb R$：一个数。
- 向量 $\mathbf x\in\mathbb R^D$：$D$ 个数，例如 6 个关节位置。
- 矩阵 $X\in\mathbb R^{T\times D}$：$T$ 个 $D$ 维向量，例如 40 步、每步 6 维动作。
- 张量：更多轴的数组。例如视频批次
  $$
  X^v\in\mathbb R^{B\times 3\times T\times H\times W}.
  $$

本文统一使用：

| 符号 | 含义 |
|---|---|
| $B$ | batch size |
| $T_v$ | 输入 RGB 视频帧数 |
| $F_v$ | VAE 压缩后的视频 latent 帧数 |
| $H,W$ | RGB 高和宽 |
| $H_z,W_z$ | VAE latent 高和宽 |
| $S_v$ | Video DiT token 数 |
| $T_a$ | 动作 horizon，即动作 token 数 |
| $c$ | action chunk size |
| $N=T_a/c$ | chunk 数 |
| $D_v=3072$ | Video DiT hidden dimension |
| $D_a=1024$ | Action DiT hidden dimension |
| $D_t=4096$ | UMT5/context dimension |
| $H_h=24$ | 注意力头数，避免与图像高度混淆 |
| $d_h=128$ | 每个注意力头维度 |
| $D_{qkv}=H_h d_h=3072$ | Q/K/V 总维度 |
| $L=30$ | Transformer 层数 |
| $Q=32$ | 每个 chunk 的 OVCR routing query 数 |

轴的次序非常重要。`[B,C,T,H,W]` 与 `[B,T,C,H,W]` 包含相同数量的数，却有完全不同的语义。源码在 VAE 前使用前者，在数据处理过程中有时先得到后者再 `permute`。

### 2.2 线性层不是“只做缩放”

PyTorch 的 `nn.Linear(D_in, D_out)` 对每个 token 做

$$
\mathbf y=\mathbf xW^\top+\mathbf b,
\quad
W\in\mathbb R^{D_{out}\times D_{in}}.
$$

若输入是 $X\in\mathbb R^{B\times T\times D_{in}}$，线性层独立作用于前两个轴中的每个位置，输出为

$$
Y\in\mathbb R^{B\times T\times D_{out}}.
$$

AHA-WAM 中的例子：

- 6 维 SO101 动作经 `Linear(6,1024)` 成为动作 token；
- 6 维 proprio 经 `Linear(6,4096)` 成为 context token；
- 48 维 VAE 空间向量经 MLP 成为 4096 维观测 token。

### 2.3 点积注意力的全部向量运算

令 query、key、value 为

$$
Q\in\mathbb R^{B\times H_h\times S_q\times d_h},\quad
K,V\in\mathbb R^{B\times H_h\times S_k\times d_h}.
$$

注意力为

$$
\operatorname{Attn}(Q,K,V)
=\operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_h}}+M\right)V.
$$

逐步看：

1. $QK^\top$ 得到 `[B,heads,Sq,Sk]`。每个值是一个 query 和一个 key 的点积，相似度越大越值得读取。
2. 除以 $\sqrt{d_h}$ 防止维度大时点积数值过大、softmax 饱和。
3. mask $M$ 把不允许看的位置变成 $-\infty$；本代码传布尔 mask，`True` 表示允许。
4. softmax 沿 key 轴把相似度变成和为 1 的权重。
5. 权重乘 $V$，得到每个 query 对 value 的加权平均。

源码入口是 `flash_attention()`。虽然函数名叫 flash attention，当前兼容路径实际调用 PyTorch `scaled_dot_product_attention`。

### 2.4 多头注意力为什么要求两个 DiT 头配置一致

Video DiT hidden 是 3072，Action DiT hidden 是 1024，但二者都把自身 hidden 投影成

$$
D_{qkv}=24\times128=3072.
$$

所以动作 query 可以直接与视频 key 做点积。两个 expert 可以有不同 residual hidden 维度，但 Q/K/V 的头数与 head dimension 必须一致。`BaseWAM.from_wan22_pretrained()` 对层数、头数和 head dimension 都有硬检查。

### 2.5 normalization、residual、MLP 和 gate

RMSNorm 对最后一维做

$$
\operatorname{RMSNorm}(\mathbf x)
=\gamma\odot\frac{\mathbf x}
{\sqrt{D^{-1}\sum_i x_i^2+\epsilon}}.
$$

LayerNorm 还会减均值。源码中 Q/K 使用 RMSNorm，Transformer 主分支使用 LayerNorm。

残差连接形如

$$
X' = X + g\odot F(X).
$$

它保留原始信息并让深网络更容易训练。这里的 $g$ 不是固定 1，而来自扩散时间调制。

FFN/MLP 逐 token 计算

$$
\operatorname{FFN}(x)=W_2\,\operatorname{GELU}(W_1x+b_1)+b_2.
$$

Video expert 的宽度是 `3072 -> 14336 -> 3072`，Action expert 是 `1024 -> 4096 -> 1024`。

### 2.6 DiT 的时间条件：sinusoidal embedding + AdaLN 式调制

flow time $t$ 先变成正余弦向量：

$$
e_{2i}(t)=\cos(t\omega_i),\qquad
e_{2i+1}(t)=\sin(t\omega_i).
$$

再经 MLP 产生每层使用的六组向量：

$$
(\Delta_{sa},S_{sa},G_{sa},\Delta_{ff},S_{ff},G_{ff}).
$$

归一化后的 token 被调制为

$$
\operatorname{modulate}(x;\Delta,S)=x\odot(1+S)+\Delta.
$$

自注意力和 FFN 输出再分别乘 gate $G_{sa},G_{ff}$ 后进入残差。这就是代码里的 `t_mod`、`modulate()` 和 `GateModule`。

Action DiT 支持每个 chunk 甚至每个 token 使用不同时间；Video DiT 则把干净的条件帧时间设为 0，未来噪声 latent 使用采样时间。

### 2.7 RoPE：不把位置相加，而是旋转 Q/K

对一对通道 $(x_{2i},x_{2i+1})$，位置 $p$ 的旋转是

$$
\begin{bmatrix}x'_{2i}\\x'_{2i+1}\end{bmatrix}
=
\begin{bmatrix}
\cos(p\omega_i)&-\sin(p\omega_i)\\
\sin(p\omega_i)& \cos(p\omega_i)
\end{bmatrix}
\begin{bmatrix}x_{2i}\\x_{2i+1}\end{bmatrix}.
$$

同样旋转 Q 与 K 后，它们的点积自然含有相对位置 $p_q-p_k$。

- Action DiT 使用 1D RoPE，位置就是动作步编号。
- Video DiT 把 128 维 head 拆成时间、高、宽三部分，分别旋转后拼接，形成 3D RoPE。
- history/streaming 时不能每次把新帧都当位置 0，否则不同刷新会位置重叠。因此数据集生成全局 frame index，`video_rope_frame_stride` 控制从动作时间到视频位置的尺度。

### 2.8 VAE、latent 和 patch token 是三个不同层级

RGB 视频先由冻结的 Wan2.2 VAE 压缩：

$$
[B,3,T,H,W]
\longrightarrow
[B,48,F,H/16,W/16],
$$

其中

$$
F=1+\frac{T-1}{4},\qquad T\bmod4=1.
$$

第一帧单独保留，之后每 4 个 RGB 时间步压成一个 latent 时间步。这是因果视频 VAE 的边界约定。

VAE latent 还不是 Transformer token。Video DiT 再用 `Conv3d(kernel=stride=(1,2,2))` patchify：

$$
[B,48,F,H_z,W_z]
\to[B,3072,F,H_z/2,W_z/2]
\to[B,S_v,3072].
$$

所以

$$
S_v=F\frac{H}{32}\frac{W}{32}.
$$

在 $384\times320$ 下，每个 latent frame 有 $12\times10=120$ 个 Video DiT token。

### 2.9 flow matching：模型预测的不是“干净动作”，而是速度

令数据为 $y$，高斯噪声为 $\epsilon\sim\mathcal N(0,I)$，代码定义

$$
y_\sigma=(1-\sigma)y+\sigma\epsilon,
\quad \sigma\in[0,1].
$$

对 $\sigma$ 求导：

$$
\frac{dy_\sigma}{d\sigma}=\epsilon-y.
$$

因此训练 target 是恒定速度

$$
v^*=\epsilon-y,
$$

模型学习

$$
\mathcal L_{FM}=\|v_\theta(y_\sigma,\sigma,c)-(\epsilon-y)\|_2^2.
$$

注意代码的生成方向是从 \(\sigma=1\) 的噪声走向 \(\sigma=0\) 的数据。推理 schedule 的 \(\Delta\sigma<0\)，显式 Euler 更新为

$$
y_{k+1}=y_k+\Delta\sigma_k\,v_\theta(y_k,\sigma_k,c).
$$

因为 \(\Delta\sigma\) 为负，预测 \(\epsilon-y\) 会把样本从噪声方向拉回数据方向。

---

## 3. 模型是怎样被组装出来的

`create_ahawam()` 的实际组装顺序为：

1. 读取 Wan2.2-TI2V-5B 的 Video DiT、Wan2.2 VAE、UMT5 encoder/tokenizer；
2. 创建 1.02B 级 Action DiT；
3. 用 `MoT(mixtures={video,action})` 统一管理逐层交互；
4. 创建 `AHAWAM`，分别安装 video/action flow scheduler；
5. 配置 action horizon/chunk；
6. 配置观测 visual projector、32-query encoder 和逐层 K/V editor；
7. 配置 rolling history、video read mode 和 RoPE stride；
8. 配置最大 action offset。

默认核心尺寸为：

| 模块 | 配置 |
|---|---|
| Wan VAE | 48 latent channels，stride `(4,16,16)` |
| Video DiT | 30 层，hidden 3072，FFN 14336，24×128 heads |
| Action DiT | 30 层，hidden 1024，FFN 4096，24×128 heads |
| 文本 context | 4096 维，数据配置通常缓存 128 tokens |
| Video patch | `(1,2,2)` |
| action horizon / chunk | 默认 64 / 16 |
| OVCR query | 每 chunk 32 个，query 初始维度 4096 |
| scheduler | 1000 标度、shift 5 |

Video 和 Action hidden 不同，但 Q/K/V 都是 3072 维，因此能够 joint attention。层数必须同为 30，因为第 \(\ell\) 层动作必须读取第 \(\ell\) 层视频上下文。

### 3.1 Action DiT 如何从 Video DiT 初始化

[`preprocess_action_dit_backbone.py`](../../scripts/preprocess_action_dit_backbone.py) 不是简单随机初始化：

- `action_encoder` 和最终 `head` 不从视频模型复制，因为输入/输出是动作维度；
- 其余同名 backbone 张量从 Video DiT 复制；
- 形状不同的轴做逐轴 1D 线性插值；
- 最后一维改变时默认乘
  $$
  \alpha=\sqrt{d_v/d_a}
  $$
  以补偿维度变化对幅度的影响。

这给 Action DiT 一个来自视频生成 Transformer 的初始化，但它仍需机器人数据训练才能成为动作策略。

---

## 4. 数据怎样变成模型输入

### 4.1 原始 LeRobot 样本

RoboTwin 默认包含三路图像、14 维状态和 14 维动作。`AHAWAMProcessor` 做：

1. 每路图像变为 tensor 并 resize；
2. 动作与状态执行可选坐标变换；
3. 用训练集统计量归一化；
4. 多个 action/state 字段合并为单一向量；
5. 输出 `pixel_values`、`action`、`proprio` 与 padding mask。

z-score 模式可抽象为

$$
\hat x_i=\frac{x_i-\mu_i}{s_i}.
$$

模型预测的是归一化动作；部署端必须用完全相同的训练统计量做反变换

$$
x_i=\hat x_i s_i+\mu_i.
$$

统计文件不匹配会让网络输出看似正常的数值，却转成错误的关节幅度。

### 4.2 多相机不是多个独立 Transformer 流

数据集先把相机画面拼成一张 $384\times320$ 图：

- RoboTwin：顶部 `cam_high` 为 \(256\times320\)，底部左右腕各 \(128\times160\)；
- 当前 SO101 适配：顶部 fixed camera，底部左侧 wrist，底部右侧用黑色填充缺少的第二腕部相机。

拼接后模型只看到一张三通道图。相机身份来自固定空间布局，而不是额外 camera-ID token。因此训练与部署必须保持相同布局、顺序、裁剪和黑边。

### 4.3 视频采样和动作采样的频率不同

先把“帧”分成三种，否则很容易把 49、13 和 4 当成同一个时间轴：

1. **控制时间步/原始相机帧**：SO101 数据以 15 Hz 记录，索引为 $n=0,1,2,\ldots$。相邻两个状态之间有一个动作。
2. **送入主 Video DiT 监督片段的 RGB 帧**：数据 loader 每隔 $r$ 个原始时间步取一帧。
3. **VAE latent frame**：因果视频 VAE 在时间上又以 4 倍压缩 RGB 帧。

所以“每 $r$ 个动作时间步取一帧”不是说相机只会拍这些帧，也不是说机械臂每 $r$ 步才执行一次动作。它是数据 loader 对**主视频生成监督**的降采样。源码直接构造：

```python
video_sample_indices = list(range(0, num_frames, action_video_freq_ratio))
```

设原始窗口有 $T_{raw}$ 个状态/图像时刻，其中有 $T_{raw}-1$ 个动作，则

$$
T_{RGB}=1+\frac{T_{raw}-1}{r},
\qquad
F_v=1+\frac{T_{RGB}-1}{4}
=1+\frac{T_{raw}-1}{4r}.
$$

仓库用 assert 要求这些除法整除。例如原配置 `num_frames=65,r=8`：

```text
原始时间轴       0,1,2,...,64       -> 65 个 state/image 时刻
动作区间         [0->1],...,[63->64] -> 64 个 action
主视频采样索引   0,8,16,...,64      -> 9 帧 RGB
VAE latent 时间长度  1+(9-1)/4         -> 3 latent frames
Video DiT token       3*120             -> 360 tokens
```

这里的 `0,8,...,64` 就是“选取视频帧下标”：在已经记录的高频时间轴上，选哪些时刻组成 Video DiT 的低频训练 clip。这样做是为了在保持几秒物理 horizon 的同时降低 Video DiT token 数和显存/计算量。

#### 4.3.1 当前 SO101 的真实数字

当前训练和部署配置的关键量是：

```text
控制频率 f_c              15 Hz
num_frames                 49
action_video_freq_ratio r   4
action_horizon T_a         40
action_chunk_size c         8
max_action_offset           7
```

于是一个 loader 原始窗口的时间关系是：

```text
原始 state/image        49 个：0..48
原始 action             48 个：0..47
整个原始窗口物理时长  48/15 = 3.2 s
主视频 RGB            13 帧：0,4,8,...,48
主视频有效帧率       15/4 = 3.75 Hz
相邻采样 RGB 的间隔    4/15 ≈ 0.267 s
VAE latent frames          1+(13-1)/4 = 4
Video DiT tokens          4*120 = 480
```

注意：4 个 latent frames 不是 4 张普通 RGB 图。它们是能被 VAE 解码回 13 张 RGB 的压缩时空表示。第一个 latent 对应已知首帧的条件表示；之后每新增一个 latent 时间位置，VAE 时间上覆盖一组 4 个低频 RGB transition。因此不应说“一个 latent frame 就是一张真实图像”。

#### 4.3.2 action window 和 action chunk 是什么

**action window/horizon** 是一串待学习的高频动作向量，不是一个动作。当前 SO101 每步是 6 维，所以目标窗口为

$$
A\in\mathbb R^{40\times6}.
$$

训练时先采样

$$
\delta\sim\operatorname{Uniform}\{0,1,\ldots,7\},
$$

再从 48 个原始动作中取

$$
A^{\delta}=A[\delta:\delta+40].
$$

例如 $\delta=3$ 时，监督的是原时间轴上索引 `3..42` 的 40 个动作，物理时长约 $40/15=2.67$ s。它再分成 5 个 chunk：

$$
[3:11],\ [11:19],\ [19:27],\ [27:35],\ [35:43].
$$

每个 chunk 包含 8 个 6 维动作，对应 $8/15\approx0.533$ s。在线接口的一次 `phase="action"` 返回的正是这样一个 `[8,6]` 动作块，而不是一次返回全部 40 步。

### 4.4 文本 context 为何从缓存读取

这里的“文本 context 缓存”不是 Transformer 逐层推理用的 K/V cache。它是**冻结 UMT5 文本编码器的最终 hidden states**：

$$
\text{prompt string}
\xrightarrow{\text{tokenizer+UMT5}}
C_{text}\in\mathbb R^{128\times4096}.
$$

数据集把完整 prompt 做 SHA-256，用 hash 找到

```text
{sha256}.t5_len128.wan22ti2v5b.pt
```

文件中的 `context` 和 `mask`。这个 context 之后同时供 Video DiT 和 Action DiT 的 cross-attention 使用。

当前 dataset 读出后还会先用原 mask 把 padding 位置的 embedding 清零，然后把传给 DiT 的 mask 改为全 1。这是为了保持 Wan2.2 的历史条件行为：注意力可“看到”固定长度 128 的位置，但 padding 位置的 value 是零。因此要比较 cache 与 online 路径，不能只比 UMT5 输出，也要确保这一步后处理一致。

这**不是 WAM 或 AHA-WAM 在数学上必须具有的特性**，而是当前工程配置中一个等价的计算/显存优化：

- UMT5 是冻结的，同一 prompt 在 `eval` 下的编码结果不会变；
- 机器人数据中同一任务文本会重复几千次，每个 batch 重跑 UMT5 没有信息收益；
- `configs/model/ahawam.yaml` 默认 `load_text_encoder: false`，因而训练/部署不需要把这个大文本编码器常驻 GPU；
- 从磁盘读一份 128×4096 的已编码张量，并在进程内做小型 LRU/字典缓存，比重复文本前向便宜。

VLA 也不是“不需要文本编码”。很多 VLA 把 tokenizer/LLM/VLM 包在模型里在线前向，因为它们的语言主干要微调、prompt 高度多变，或需要让视觉 token 和文本 token 在同一 LLM 中联合演化。如果一个 VLA 也使用冻结、独立的文本 encoder，并且任务文本集合有限，它同样可以预计算。

缓存只在以下前提下与在线编码等价：UMT5 权重、tokenizer、prompt 模板、`context_len` 和预处理都不变。如果要微调文本 encoder，旧 cache 不仅过期，还会切断到 encoder 的梯度，就不能用。

部署封装因此提供三种模式：

- `cache`：只允许已缓存的精确 prompt，可不加载 UMT5；
- `online`：保留 UMT5，每条指令在线编码；
- `hybrid`：命中则读 cache，未命中则在线编码并可回写。

### 4.5 chunk 对齐观测：它与主视频降采样不是一回事

每个 action chunk 开始时需要当时的最新图像和 proprio。无 offset 的简单例子中，若 $T_a=64,c=16$，chunk 起点为 `0,16,32,48`。若 64 个动作对应 8 个主视频 transition，每个 transition 有 8 个动作，则这四个时刻在降采样视频中是帧下标 `0,2,4,6`。

但当前 SO101 启用了 action offset，loader 不强迫 chunk 观测落在 `0,4,8,...,48` 这些主视频帧上。它会预先向底层数据集申请所有可能的 chunk-start 原始图像。对某个已采样的 $\delta$，真正用的是

$$
\delta+[0,8,16,24,32].
$$

例如 $\delta=3$ 时，`chunk_obs_images` 来自原时间轴 `3,11,19,27,35`。这些图像分别给 5 个 action chunk 做 OVCR 条件。同一训练样本的主 Video DiT 监督 clip 仍然固定用 `0,4,...,48`。两者的职责不同：

- `0,4,...,48`：低频未来视频生成/世界建模监督；
- `delta+[0,8,...,32]`：每个动作块开始时的高频闭环观测。

这些 chunk 图像各自经 VAE 变成 `[B,N,48,1,24,20]`，再经视觉投影和 query pooling 变成 OVCR 使用的观测 token。

### 4.6 `video transition` 到底是什么

`video transition` 只是**两张相邻的已采样 RGB 帧之间的时间区间**，不是 Video DiT 一次调用的输出，也不是一个 predicted latent。

例如 13 张采样 RGB 有 12 个 transition：

$$
(0\to4),(4\to8),\ldots,(44\to48).
$$

每个 transition 跨过 4 个 15 Hz 控制区间，约 0.267 s。VAE 每 4 个这样的低频 transition 新增一个 latent 时间位置，因此 12 transitions 对应首 latent 之后的 3 个 latent 时间位置。这是时间计数与压缩关系，不代表 Video DiT 只做 3 次前向。

---

## 5. Video DiT：世界规划分支

### 5.1 输入和干净首帧

训练时完整视频 latent $z_0$ 加噪：

$$
z_\sigma=(1-\sigma)z_0+\sigma\epsilon_v.
$$

随后代码强制覆盖第一 latent frame：

```python
latents[:, :, 0:1] = first_frame_latents
```

于是第一帧是干净条件，未来 latent 是噪声状态。`clean_prefix_frames=1` 还把第一帧的时间调制设为 0。模型输出与 $\epsilon_v-z_0$ 做 MSE，但 video loss 会去掉第一 latent step，只监督未来部分。

### 5.2 patchify 和 3D RoPE

以 `[B,48,3,24,20]` 为例：

```text
Conv3d patch (1,2,2)
    -> [B,3072,3,12,10]
flatten
    -> [B,360,3072]
```

每个 token 的 128 维 head RoPE 被分给时间、高、宽三个坐标。空间位置在每段 clip 中保持 `0..H'-1/0..W'-1`，时间位置可使用样本全局 ID，以便 rolling history 与当前帧保持时间次序。

### 5.3 每个 DiT block

第 $\ell$ 层可写成：

$$
\begin{aligned}
U &= \operatorname{LN}(X)\odot(1+S_{sa})+\Delta_{sa},\\
Q &= \operatorname{RoPE}(\operatorname{RMSNorm}(UW_Q)),\\
K &= \operatorname{RoPE}(\operatorname{RMSNorm}(UW_K)),\\
V &= UW_V,\\
X_1 &= X+G_{sa}\odot W_O\operatorname{Attn}(Q,K,V),\\
X_2 &= X_1+\operatorname{CrossAttn}(\operatorname{LN}(X_1),C_{text}),\\
X' &= X_2+G_{ff}\odot\operatorname{FFN}(\operatorname{LN}(X_2)\odot(1+S_{ff})+\Delta_{ff}).
\end{aligned}
$$

### 5.4 视频因果 mask

默认 `per_frame_causal` 不是逐像素自回归。设每帧 120 tokens：

- 同一帧内 120 个 token 相互可见；
- 第 2 latent frame 可以读取第 1、2 帧；
- 第 1 帧不能读取未来帧。

帧级 mask 是下三角矩阵，再在行列上各 repeat 120 次。

这个 mask 有两个作用：

1. 强迫 planner 表征符合时间向前的动态；
2. 避免 Action DiT 在线依赖训练时才有的未来视觉信息。

### 5.5 当前配置不是 action-conditioned 视频模拟器

`video_dit_config.action_conditioned: false`。因此当前主配置的 Video DiT 不以候选动作作为条件去回答“执行这个动作会发生什么”。视频分支学习的是语言和视觉条件下数据分布中的未来动态。源码保留 action-conditioned 路径，但当前 AHA-WAM 配置没有启用。

这意味着不能把当前模型严格解释为 MPC 中可对不同候选动作做 counterfactual rollout 的动力学模型。

---

## 6. Action DiT：动作生成分支

### 6.1 动作 token

归一化动作

$$
A\in\mathbb R^{B\times T_a\times D_{act}}
$$

加噪后由 `Linear(D_act,1024)` 编码为

$$
X^a\in\mathbb R^{B\times T_a\times1024}.
$$

最终 `Linear(1024,D_act)` 输出的是动作 flow velocity，不是动作本身。

### 6.2 为什么 Action hidden 1024，Q/K/V 却是 3072

每个 Action block 的 Q/K/V 投影是

$$
W_Q,W_K,W_V:\mathbb R^{1024}\to\mathbb R^{3072}.
$$

3072 再 reshape 为 24 头 × 128。注意力结果重新经 $W_O:\mathbb R^{3072}\to\mathbb R^{1024}$ 回到动作 residual stream。

### 6.3 chunk 因果 mask

当前 AHA 主训练用 `single_branch_chunk_causal=True`：第 $j$ 个 chunk 可以读取自己以及更早 chunk，不能读取更晚 chunk。

若 $T_a=8,c=2$，chunk ID 为 `[0,0,1,1,2,2,3,3]`，mask 是

$$
M_{ij}=[\operatorname{chunk}(j)\le\operatorname{chunk}(i)].
$$

同一个 chunk 内是双向注意力，因为一个动作块作为整体并行生成；chunk 之间是因果的。

代码还保留 teacher-forcing 路径：clean chunk 可作为前缀，noisy chunk 只能读更早 clean chunks，并且 noisy chunks 之间块对角隔离。但当前 `AHAWAM.training_loss()` 明确走“no teacher forcing”的 prior-only 路径。不要因配置仍写着 `autoregressive_teacher_forcing: true` 就误以为主损失一定拼接了 clean action；这个布尔值在 chunk 配置校验中仍是硬要求，但当前叶类覆盖了训练实现。

### 6.4 proprio 如何进入

每个 chunk 起点的 proprio

$$
s_j\in\mathbb R^{D_s}
$$

经 `Linear(D_s,4096)` 成为一个 token。它与文本 context 合并，但 mask 只允许对应 action chunk 读取对应 proprio token。视觉 token 不直接拼进 Action DiT cross-attention；它们走 OVCR。

---

## 7. OVCR：最新观测怎样改写 planner K/V

这是 AHA-WAM 最关键也最容易被笼统描述的模块。

### 7.1 从最新图像得到视觉 token

每个 chunk 的最新图像经 VAE：

$$
O_j\to Z_j\in\mathbb R^{B\times48\times1\times24\times20}.
$$

去掉时间轴并展平空间：

$$
[B,48,24,20]\to[B,480,48].
$$

若 `action_obs_downsample_factor>1`，会先做 average pooling，减少 token。默认当前配置为 1。

每个 48 维 token 经

```text
LayerNorm(48) -> Linear(48,4096) -> GELU -> Linear(4096,4096)
```

成为

$$
X_j^v\in\mathbb R^{B\times480\times4096}.
$$

### 7.2 32 个 base queries 做观测池化

学习参数

$$
B_q\in\mathbb R^{1\times1\times Q\times4096},\quad Q=32
$$

广播到 batch 和 chunk 后，与观测 token 做注意力：

$$
S=\frac{B_qK_o^\top}{\sqrt{4096}},
\quad W=\operatorname{softmax}(S),
\quad Z^q=WV_o.
$$

形状为：

```text
queries       [B,N,32,4096]
obs keys      [B,N,480,4096]
scores        [B,N,32,480]
obs weights   [B,N,32,480]
guided query  [B,N,32,4096]
```

这 32 个 query 可以理解成 32 个可学习的信息槽，但代码没有预先规定哪个槽代表孔位、夹爪或目标。

### 7.3 每层 K/V editor 的两次注意力

对 Video DiT 第 $\ell$ 层，当前代码取 planner cache 的**第一帧** K/V：

$$
K_0^\ell,V_0^\ell\in
\mathbb R^{B\times S_f\times3072},\quad S_f=120.
$$

这是重要的源码细节：论文用一般 planner context 记号，当前实现为防止动作读取未来 token，editor 明确切 `:video_tokens_per_frame`。

先把 4096 维 guided query 投影到 3072，并 reshape 成 24×128：

$$
Q_j^\ell\in\mathbb R^{B\times N\times32\times24\times128}.
$$

第一次注意力从 planner token 读到 query 槽：

$$
R_{bnqhd}=\sum_s
\operatorname{softmax}_s
\left(\frac{Q_{bnqhd}K_{bshd}}{\sqrt{128}}\right)V_{bshd}.
$$

第二次把 query 槽“解码”回 120 个 planner token 位置：

$$
D_{bnshd}=\sum_q
\operatorname{softmax}_q
\left(\frac{K_{bshd}R_{bnqhd}}{\sqrt{128}}\right)R_{bnqhd}.
$$

然后每层独立 MLP：

```text
LayerNorm(3072)
-> Linear(3072,3072)
-> GELU
-> Linear(3072,6144)
-> split(delta_k, delta_v)
```

得到

$$
\Delta K_j^\ell,\Delta V_j^\ell
\in\mathbb R^{B\times N\times120\times3072}.
$$

最终：

$$
\tilde K_j^\ell=K_0^\ell+\sigma(g_\ell)\Delta K_j^\ell,
\quad
\tilde V_j^\ell=V_0^\ell+\sigma(g_\ell)\Delta V_j^\ell.
$$

其中 $g_\ell$ 初始为 $-4$，所以初始 gate

$$
\sigma(-4)\approx0.018.
$$

最后一个 delta linear 还被零初始化，因此训练一开始严格近似原 planner K/V，随后逐渐学会改写。这是稳定训练的关键工程设计。

### 7.4 Action DiT 如何消费改写后的 K/V

每个 chunk 独立形成：

```text
query: 当前 chunk 的 c 个 action queries
key:   120 个 updated video keys + c 个当前 action keys
value: 120 个 updated video values + c 个当前 action values
```

即

$$
\bar H_j^{a,\ell}
=\operatorname{Attn}
\left(Q_j^{a,\ell},
[\tilde K_j^{v,\ell};K_j^{a,\ell}],
[\tilde V_j^{v,\ell};V_j^{a,\ell}]
\right).
$$

训练函数把 `[B,T_a,*]` reshape 成 `[B,N,c,*]`，每个 chunk 分别做上述 attention，再拼回动作序列。

### 7.5 为什么 context router 会有约 1.22B 参数

每层 editor 都有独立的大矩阵，粗略计数：

- query `4096 -> 3072`：约 12.6M；
- delta `3072 -> 3072 -> 6144`：约 $9.4+18.9=28.3$M；
- 每层约 40.9M，30 层约 1.23B。

因此它不是一个几百万参数的小 adapter。“lightweight”是相对 5B Video DiT 的计算路径和 token 数而言，不表示参数绝对很小。

---

## 8. MoT 在这里究竟是什么

代码里的 `MoT` 更接近“mixture of token experts / 双专家逐层混合注意力”，不是通常带 top-k router 的稀疏 Mixture-of-Experts FFN。

Video 和 Action 各自拥有：

- 独立 hidden size；
- 独立 Q/K/V/O；
- 独立 AdaLN 调制；
- 独立 cross-attention 和 FFN。

它们的耦合发生在逐层 attention 空间。通用 `MoT.forward()` 会把两个 expert 的 Q/K/V 沿序列轴拼接后统一注意力，再把输出切回各 expert。AHA-WAM 主训练为了保证未来信息不泄漏并实现 chunk routing，没有直接使用最一般的全 joint path，而走 `forward_prior_action_with_chunk_updated_kv()`：

- video 只做自己的因果 attention；
- OVCR 只更新第一帧 video K/V；
- action 每个 chunk 只读其 updated K/V 和当前 chunk 动作 K/V。

`action_branch_embedding` 在注释中被明确标为旧实验遗留、为 checkpoint 兼容保留的参数，不应赋予它重要理论含义。

---

## 9. 完整训练前向传播

以下以当前 `AHAWAM.training_loss()` 为准。

### 9.1 输入

得到：

```text
input_latents     [B,48,F,24,20]
action            [B,Ta,Da]
text context      [B,Lt,4096]
chunk visual obs  [B,N,480,4096]
chunk proprio     [B,N,1,4096]
padding masks
```

视觉 obs 生成 OVCR queries；proprio 与视觉 obs 被拆开，proprio 直接交给 Action DiT，视觉只用来改写 K/V。

### 9.2 视频 flow 分支

$$
\begin{aligned}
\epsilon_v&\sim\mathcal N(0,I),\\
t_v&\sim p_{code}(t),\\
z_t&=(1-\sigma_v)z+\sigma_v\epsilon_v,\\
z_t[:,:,0]&\leftarrow z[:,:,0],\\
u_v^*&=\epsilon_v-z.
\end{aligned}
$$

Video DiT 处理带干净首帧的长 clip。若启用 history，历史观测先在 `no_grad` 下变成逐层 K/V，当前 video query 能读取它们。

### 9.3 动作 flow 分支

$$
\begin{aligned}
\epsilon_a&\sim\mathcal N(0,I),\\
t_a&\sim p_{code}(t),\\
A_t&=(1-\sigma_a)A+\sigma_a\epsilon_a,\\
u_a^*&=\epsilon_a-A.
\end{aligned}
$$

`per_chunk` 模式为每个 chunk 独立采一个时间，再 repeat 到该 chunk 的所有 action tokens：

$$
t_a\in\mathbb R^{B\times N}
\to\mathbb R^{B\times T_a}.
$$

这样同一 chunk 处于相同噪声强度，不同 chunk 可处于不同噪声强度，迫使模型学习 chunk-wise 生成。

### 9.4 逐层执行

每一层做：

1. 当前视频 token 计算 video Q/K/V；
2. video query 读取 history K/V + 当前 video K/V；
3. 只取当前 clip 第一帧 K/V；
4. OVCR 根据每 chunk 最新观测生成该层 updated K/V；
5. action token 计算 action Q/K/V；
6. 每个 chunk 的 action query 读取 updated video K/V + 本 chunk action K/V；
7. video/action 分别做文本 cross-attention 和 FFN。

### 9.5 损失

视频先对通道和空间平均，随后用 padding mask 对 latent 时间平均：

$$
\ell_v^{(b)}=
\frac{\sum_f m_{bf}\operatorname{mean}_{c,h,w}
(\hat u^v-u_v^*)^2}{\sum_fm_{bf}}.
$$

动作先对动作维平均，再按有效时间步平均：

$$
\ell_a^{(b)}=
\frac{\sum_tm_{bt}\operatorname{mean}_{d}
(\hat u^a-u_a^*)^2}{\sum_tm_{bt}}.
$$

代码还为不同 flow time 乘 `training_weight(t)`：

$$
w(t)\propto
\exp\left[-2\left(\frac{t-T/2}{T}\right)^2\right]-w_{min},
$$

并归一化使网格均值约为 1。最终当前 AHA 路径为

$$
\mathcal L
=\lambda_v\mathcal L_v
+\lambda_{prior}\mathcal L_{action\ prior}.
$$

日志仍把 action prior 项记为 `loss_action`。

### 9.6 scheduler 的 shift

代码先采 $u\sim U(0,1)$，再定义

$$
\sigma=\phi(u;s)=\frac{su}{1+(s-1)u}.
$$

当 $s=5$ 时，中间的 $u$ 会映射到更大的 $\sigma$，改变噪声时间分布。训练时间标量实际传给 DiT 的是

$$
t=1000\sigma.
$$

论文写“logit-normal noise time”，而本次检查的当前 scheduler 是“uniform $u$ + rational shift + time weight”。这是实现差异，不应把两者混写。

### 9.7 训练时两个 DiT 要不要重复去噪？

**当前主训练路径不做多步去噪 loop。**一个 minibatch/一个 optimizer step 中，代码对已知 GT 视频 latent 和 GT 动作分别采一个随机噪声时间，合成该时间点的带噪样本，然后让联合 MoT 主干做一次前向：

$$
\begin{aligned}
z_{\sigma_v}&=(1-\sigma_v)z_0+\sigma_v\epsilon_v,\\
a_{\sigma_a}&=(1-\sigma_a)a_0+\sigma_a\epsilon_a,\\
(\hat v_v,\hat v_a)&=f_\theta(z_{\sigma_v},a_{\sigma_a},
\sigma_v,\sigma_a,C),\\
\mathcal L&=\|\hat v_v-(\epsilon_v-z_0)\|^2
+\|\hat v_a-(\epsilon_a-a_0)\|^2.
\end{aligned}
$$

从统计上看，不同 batch 会抽到从接近干净到接近纯噪声的不同 $\sigma$，经过许多个 optimizer step 学到整条路径上的速度场。训练不需要把这个 batch 从噪声真正还原到 GT，因为 GT 已知，目标速度 $\epsilon-y$ 可直接计算。

对“一次前向”需要两个限定：

- 它指当前视频 clip 与当前 action window 穿过 30 层 Video/Action experts 的一次联合主前向，不是“只算一层”；
- 启用 `num_history_frames>0` 时，历史观测还会先在 `no_grad` 下另做一次 Video DiT cache prefill。它是历史条件构建，不是对当前噪声样本反复去噪。

Action 的 `per_chunk` 模式会让 5 个 chunk 抽到不同 $\sigma_a$，但仍是把整个 `[B,40,6]` 动作窗口一次送进主干，不是 5 个 chunk 各自跑一个训练去噪 loop。

---

## 10. horizon-adaptive offset training 到底做什么

设 chunk size $c=16$，代码从

$$
\delta\sim\operatorname{Uniform}\{0,1,\ldots,15\}
$$

取偏移。动作目标从原序列的 $\delta$ 开始切：

$$
A^\delta=A[\delta:\delta+T_a].
$$

每个 chunk 的最新图像和 proprio 同样平移到

$$
\delta, \delta+c, \delta+2c,\ldots
$$

因此不是“只把动作标签平移但图像不动”。数据集专门返回 `chunk_obs_images` 和 offset 后的 chunk-start proprio。

为什么只需 (0\ldots c-1)？因为 chunk 网格每 (c) 步重复一次，相位只有 (c) 种。

它解决的是：异步部署中，慢 planner 的刷新时刻与快 action chunk 的开始时刻未必对齐。模型不能只会“planner 刚刷新时的第 0 块”。

它不解决：

- 自动选择动作 horizon；
- 根据任务难度动态改变 chunk size；
- planner 极度陈旧时无限复用；
- 数据分布之外的视觉/接触误差。

---

## 11. rolling planner memory 与 planner context 不是一回事

有两类 K/V，必须分清：

1. **rolling history K/V**：历史 planner 刷新的观测状态，供下一次 Video DiT prefill 内部读取；
2. **current planner K/V**：当前 prefill 产生，供 OVCR 改写后让 Action DiT 读取。

在线 `prefill_video()` 中：

- 当前图像只编码成一个 latent frame；
- 若已有历史 entries，当前 Video DiT query 读取 `[history K/V; current K/V]`；
- 返回 cache 只存当前这一帧的新 K/V；
- entry 加入 FIFO，超过 `num_history_frames` 则丢弃最旧项。

因此 rolling memory 是让当前 planner representation 具有历史感；Action DiT 并不会直接拿完整历史 FIFO 做注意力。默认 `action_video_read_mode=current_only` 也表达了这个意图。

历史训练样本不是相邻 RGB 帧，而是按一个完整规划窗口向前取观测：

$$
\text{history stride}=(\text{num\_frames}-1)\times\text{global stride}.
$$

这让 FIFO 更像过去几次 planner refresh，而不是普通短视频的连续帧。

---

## 12. 推理：为什么只跑当前图像也能用“视频世界模型”

### 12.1 phase=`video`：planner prefill

输入当前图像和语言：

1. 初始化整个 action horizon 的高斯噪声 `[B,T_a,D_a]`；
2. 当前图像经 VAE 得到 `[B,48,1,24,20]`；
3. Video DiT 以 (t=0) 对这一帧做一次前向；
4. 每一层缓存这一帧的 K/V；
5. 若有 rolling history，当前帧形成 K/V 时可以读取历史前缀；
6. 保存 inference state。

这里没有生成未来 latent，也没有 VAE decode。世界预测能力通过训练塑造 Video DiT 的层级表征，在线使用的是表征接口。

### 12.2 phase=`action`：每个 chunk 的闭环生成

对 chunk (j)：

1. 获取刚采到的最新图像和 proprio；
2. 图像经 VAE、visual projector、32-query encoder；
3. OVCR 逐层改写 planner 当前帧 K/V；
4. 从 inference state 中取该 chunk 的噪声动作；
5. 建立从 $\sigma=1$ 到 0 的 $K$ 步 schedule；
6. 每一步 Action DiT 预测 flow velocity；
7. Euler 更新动作 latent；
8. 最终动作块转 CPU，并由 policy wrapper 反归一化。

默认论文/部署常用 10 步。每步都必须重算 Action DiT Q/K/V，因为动作 latent 在变；planner K/V 在同一 chunk 内不变，可复用。

### 12.3 为什么推理要多次，训练却只要一次？

推理时没有 GT $y_0$，只有初始噪声 $y_{\sigma_0}\sim\mathcal N(0,I)$。模型只能在“当前位置、当前时间”预测局部速度，因此要数值积分 ODE：

$$
y_{k+1}=y_k+(\sigma_{k+1}-\sigma_k)
v_\theta(y_k,\sigma_k,C),
\qquad \sigma_0=1>\cdots>\sigma_K=0.
$$

每更新一次 $y_k$，下一次都要在新位置重新问模型速度。所以多步不是“推理噪声比训练噪声更大”这一个原因，更准确地说，是训练在学速度场，推理在用这个速度场求一条从噪声到数据的轨迹。

所以训练和推理的**网络函数与 flow 方向是一致的**，但**调用程序不一样**：训练用 GT 构造一个随机路径点来监督；生成推理则把同一网络当成 ODE 速度函数反复调用。而常规 AHA action 部署又进一步把未来视频生成省略了，只用 Video DiT 的 current-frame 表征缓存。

步数越多通常积分误差越小，但延迟线性增长，而且并不保证在分布外观测上单调变好。当前动作设为 10 步，是速度与质量折中。

#### 12.3.1 当前部署到底调用几次两个 DiT

| 场景 | Video DiT | Action DiT | 输出 |
|---|---:|---:|---|
| 主训练的当前 clip/window | 1 次联合主前向 | 1 次联合主前向 | video/action flow 预测与 loss |
| 普通在线 `phase="video"` | 1 次 current-frame prefill | 0 | 当前帧逐层 planner K/V |
| 普通在线一个 `phase="action"` | 0 次完整 Video DiT；OVCR 仅改 cache | 10 次（当前 yaml） | `[8,6]` 动作块 |
| 显式调用未来视频 preview | 20 次（当前 yaml） | 0 | 13 张 RGB，含已知当前帧 |

上表的“Video DiT 1 次”是一次穿过完整 30 层的前向，不是一个 block。显式 preview 则初始化 4 个 latent 时间位置，把第一个覆盖为已知当前图像 latent，只让其余未来部分从噪声逐步改变：

$$
[z_{current},\epsilon_1,\epsilon_2,\epsilon_3]
\xrightarrow{20\text{ steps}}
[z_{current},\hat z_1,\hat z_2,\hat z_3]
\xrightarrow{VAE\ decode}13\text{ RGB frames}.
$$

每一步 Video DiT 更新后代码都重新把第一 latent 覆盖为干净 `first_frame_latents`，保证已观测首帧不被去噪漂移。这个 preview 是诊断接口，普通 SO101 action server 不会在每个动作块前执行它。

### 12.4 当前 AHA 类不累计 action history K/V

基类具备“把已生成 clean action chunk 缓存给下一块”的设施，但 `AHAWAM._should_update_action_history()` 返回 `False`。当前 AHA 的 chunk 间依赖主要由：

- 每块最新视觉；
- 当前 proprio；
- 共享 planner context；
- 训练时 chunk-causal action structure

承担，而部署的逐块路径不保留已生成 action K/V。阅读基类时不要误把备用能力当成当前叶类行为。

### 12.5 同步、单 GPU 异步和双 GPU 异步

- 同步 policy：同一 Python 流中先 prefill，再逐 chunk action。
- `AsyncAHAWAMRuntime`：一个运行时循环维护最新 video state 和动作请求，使用 latest-state buffer。
- 双 GPU server：video worker 在一张卡持续发布 CPU 化的最新 inference state；action worker 在另一张卡消费最新版本，并检查 `kv_max_age_ms`。

双 GPU 路径会 deep-copy state，避免 Action 推理原地更新噪声张量污染共享的 latest planner state。

论文的 TensorRT、CUDA Graph 静态 buffer 是其优化系统的一部分；当前公开 Python 部署代码提供了异步线程/进程/双 GPU 结构，但不能仅因为调用了这些 Python 类就假定已获得论文中 41.37 ms 的 TensorRT 延迟。

---

## 13. 两个完整形状例子

### 13.1 官方式 RoboTwin offset 配置

配置：RGB $384\times320$，原始窗口 97，video ratio 8，目标 action horizon 80，chunk 16，offset 0–15。

```text
原动作序列                 [B,96,14]
offset 后动作目标           [B,80,14]
chunk 数                    80/16 = 5
采样 RGB 帧                 0,8,...,96 -> 13 帧
VAE video latent            [B,48,4,24,20]
Video token/frame           12*10 = 120
Video tokens                [B,480,3072]

每 chunk 最新图像 latent    [B,5,48,1,24,20]
视觉 obs tokens             [B,5,480,4096]
OVCR queries                [B,5,32,4096]
每层 updated video K/V      [B,5,120,3072]

Action hidden               [B,80,1024]
按 chunk reshape            [B,5,16,1024]
每块 attention key 长度     120+16 = 136
模型输出 flow               [B,80,14]
```

### 13.2 当前 SO101 插头任务配置

训练配置：6 维状态/动作，15 Hz，窗口 49，ratio 4，action horizon 40，chunk 8，offset 0–7，两相机拼接。

```text
原动作序列                 [B,48,6]
offset 后动作目标           [B,40,6]
chunk 数                    40/8 = 5
采样 RGB 帧                 0,4,...,48 -> 13 帧
VAE video latent            [B,48,4,24,20]
Video tokens                [B,480,3072]
每块视觉 obs                [B,5,480,4096]
每块 proprio                [B,5,1,4096]
OVCR queries                [B,5,32,4096]
Action hidden               [B,40,1024]
输出 flow                   [B,40,6]
```

对一个具体的 $\delta=3$ 样本，四条时间轴可以对在一起看：

| 用途 | 原始 15 Hz 索引 | 数量 | 解释 |
|---|---|---:|---|
| 完整 loader 窗口 | `0,1,...,48` | 49 时刻 | 覆盖 3.2 s |
| 主 Video RGB | `0,4,8,...,48` | 13 帧 | 给 video flow 监督 |
| action 监督 | `3,4,...,42` | 40 动作 | $A[3:43]$ |
| chunk-start 图像/state | `3,11,19,27,35` | 5 组 | 给 5 个 chunk 的 OVCR/proprio |
| VAE 主 clip latent 时间位置 | 对应原始 `0,16,32,48` | 4 | 能 decode 回上述 13 帧 RGB |

最后一行的 `0,4,8,12` 是 Video RoPE 使用的 latent 时间 offset（单位是降采样视频时间），不是原始 15 Hz 下的四帧索引。它们来自原始索引 `0,16,32,48` 再除以 ratio 4。

当前打开的训练配置与部署 yaml 均为 `num_history_frames: 6`，因此本次检查未发现该项的 train–deploy 不一致。但 checkpoint 所用 resolved config 仍应以该 run 目录的保存副本为准，不能用后来手工改过的部署 yaml 反推训练条件。

### 13.3 训练 clip 长度与在线一次预测帧数不能直接画等号

训练时 `num_frames=49` 意味着 loader 需要 49 个原始 15 Hz 时刻，主视频分支从中取 13 帧并编成 4 latent frames。它定义的是**训练监督 horizon**，不是说当前在线 Video DiT 每次都生成 49 帧或 13 帧。

当前三种路径分别是：

- **训练**：看 13 张采样 RGB，VAE 得到 4 latent frames，第一帧干净、未来部分加噪，一次主前向估计 flow loss。
- **普通 action 部署**：Video DiT 只看当前实拍图，即 1 RGB $\to$ 1 latent frame $\to$ 120 video tokens，做 cache prefill；不生成未来视频。
- **显式 future-video preview**：`preview_num_frames=13`，输出 13 张采样 RGB（含当前帧），在 `preview_fps=15/4=3.75` 下覆盖 $(13-1)/3.75=3.2$ s。这正好与训练主 video clip 的物理 horizon 对齐，但它是单独调用的诊断生成路径。

---

## 14. 参数是否训练、谁被冻结

训练器先执行：

```text
model.eval()
model.requires_grad_(False)
model.dit.train()
model.dit.requires_grad_(True)
```

`model.dit` 指向 `MoT`，因此默认训练：

- Video DiT；
- Action DiT；
- MoT 中的 K/V editor；
- 兼容参数。

额外重新打开训练的模块包括：

- `action_obs_visual_proj`；
- `chunk_obs_query_encoder`；
- `proprio_encoder` 通过父类 additional modules。

VAE 和 UMT5 默认冻结。若 `freeze_video_dit=true`，Video DiT 额外冻结，但 Action DiT/OVCR 仍可训练。

优化器为 AdamW，betas `(0.9,0.95)`，带 cosine/warmup、gradient clipping 和 bf16/DeepSpeed 支持。任务 yaml 可覆盖基础学习率；当前 RoboTwin task 文件是 `5e-5`，论文主表述为 `1e-4`，要以实际启动时 compose 后配置和日志为准。

---

## 15. 论文、配置与代码的关键差异

| 问题 | 论文/概念描述 | 当前代码实际行为 |
|---|---|---|
| horizon-adaptive | 适应 planner/executor 相位 | 固定 horizon/chunk，随机 offset；不是在线选长度 |
| future video | 长期 world plan | 训练预测未来 latent；常规动作推理不显式生成/解码未来帧 |
| action-conditioned video | WAM 易被理解为动作条件世界模型 | 默认 `action_conditioned=false` |
| OVCR planner context | 记为一般每层 planner K/V | editor 具体只改写当前 clip 第一帧的 K/V |
| noise time | 论文写 logit-normal | 当前 scheduler 为 uniform $u$ + rational shift + 显式 time weight |
| action teacher forcing | ActionDiT 支持且配置要求开启 | 当前 AHA 主 loss 传 `clean_action_tokens=None`，使用 prior-only chunk-causal 路径 |
| action history | 基类支持 clean action K/V cache | AHA 叶类返回 False，不更新 action history |
| 异步 | 双 non-blocking streams | 需要使用 async/dual-GPU server；普通调用仍可同步 |
| CUDA 加速 | 论文含 TensorRT/CUDA graph | 公开 repo 有 Python 异步框架，不能自动等同论文 TRT 性能 |

这些不是简单的“谁对谁错”。论文抽象用于说明方法，代码分支则体现当前 checkpoint 兼容、训练稳定性和部署路径。做实验时必须以实际 commit 和 compose 后配置为准。

---

## 16. 对插入任务应怎样理解这套架构

AHA-WAM 对插入任务潜在有价值的部分：

- wrist + global camera 的最新观测能在每个 chunk 通过 OVCR 修正 planner context；
- action chunk 比整个视频 horizon 短，允许重复闭环纠偏；
- proprio 直接进入动作分支，不必等待 Video DiT 刷新；
- offset training 让动作块不依赖完美的 planner 刷新相位；
- video co-training 给视觉表示加入运动/接触前后的动态监督。

但它本身没有：

- 力/力矩或触觉输入；
- 显式孔位几何、阻抗控制或搜索策略；
- 在线成功判别器；
- 碰撞/接触物理模拟；
- 自动保证亚毫米定位精度。

还要注意 $384\times320$ 图像经 VAE 和 DiT 后，planner 每个 token 对应较大空间区域。虽然 latent 通道可以编码亚 patch 信息，OVCR 也直接使用 24×20 的 VAE 网格，但精密插入仍高度依赖原始成像尺度、腕部视角、标定、低层控制和数据是否覆盖“孔口附近的小误差纠正”。不能仅凭 Video DiT 很大就推断它一定保留足够的微小间隙信息。

针对当前 SO101，最应该先验证的不是抽象的“video 好不好”，而是以下可测量链路：

1. 同一孔位发生 1–5 mm / 1–5° 偏差时，VAE latent 与 OVCR query 是否可区分；
2. `chunk=8` 在 15 Hz 下约 0.53 s，是否长到在接触阶段来不及纠错；
3. 腕相机在拼图中只有 $128\times160$，插口像素是否足够大；
4. checkpoint 的 resolved `num_history_frames` 是否与部署一致，以及 history=6 相对 history=0 是否真有收益；
5. 10 步 Action DiT 延迟加机械臂通信后，真实闭环频率是多少；
6. 动作 z-score 是否让少量精密动作维度被大范围粗移动统计量淹没。

这些问题可以通过 representation probe、局部偏差数据集和阶段性 success/error 指标回答，而不必先修改整个架构。

---

## 17. 常见误解与问答

### Q1：Video DiT 在线只吃一帧，怎么还能叫“长 horizon planner”？

长 horizon 主要来自训练时的未来视频 flow supervision，以及 rolling memory 提供的历史上下文。在线一帧 prefill 取得的是被这种训练塑造过的逐层 K/V，不是重新生成完整未来视频。

### Q2：OVCR 是不是又完整跑了一遍 Video DiT？

不是。最新图像仍需跑冻结 VAE，但之后只走 visual projector、32-query pooling 和逐层 K/V editor；不重新执行 5B Video DiT 的 30 层主干。

### Q3：为什么每个 chunk 还要给最新图像，planner 不就失去意义了吗？

最新图像告诉系统“现在实际到了哪里”；planner K/V 提供语言条件、世界动态和历史规划语境。OVCR 做的是用前者选择/修正后者，而不是用小观测编码器完全替代 planner。

### Q4：`action_horizon=64` 与 `action_chunk_size=16` 是不是一次执行 64 步？

不是必然。模型初始化 64 步噪声并分成 4 块，但闭环接口每次返回一个 16 步块。控制端也可以只执行其中前若干步再请求新块，具体由 client/action queue 决定。

### Q5：flow matching 与 diffusion policy 是同一件事吗？

都是从噪声逐步得到动作的生成策略，但目标参数化和路径可不同。这里用线性 probability path 的 velocity regression；推理用确定性 ODE/Euler 式更新。不能直接套用 DDPM 的“预测噪声 epsilon”解释。

### Q6：为什么模型 target 是 `noise - data`，不是 `data - noise`？

因为代码把 path 写成从数据 \(\sigma=0\) 到噪声 \(\sigma=1\)，导数是 `noise-data`；生成时 schedule 从 1 走到 0，delta 为负，所以方向自动反转。

### Q7：rolling K/V 会无限增长吗？

不会。`num_history_frames` 是固定 FIFO 长度。每次只保留当前 prefill 的一帧 K/V entry，超过长度丢最旧项。

### Q8：AHA-WAM 是不是强化学习算法？

不是。当前核心训练是监督数据上的 video/action flow matching。以后可以加 RL，但那是额外的 post-training objective 或 critic/reward 结构。

### Q9：文本 cache 是因为 DiT 不能直接读文字吗？

不是。DiT 本来就不直接读 Python 字符串，它需要数值 context token。这些 token 既可以在线由 UMT5 计算，也可以在 UMT5 冻结时预计算后读 cache。当前仓库默认后者是为了节省重复计算和显存，不是模型理论要求。

### Q10：两个 DiT 在推理时是不是都要去噪多次？

对一个“同时真正生成未来视频和动作”的一般生成设想，两个生成变量都可以需要多步积分。但当前 AHA-WAM 的常规 SO101 部署没有这样做：Video DiT 对当前干净图做一次 prefill，Action DiT 才对每个动作块做 10 步去噪。只有显式 future-video preview 调用 Video DiT 的 20 步去噪。

### Q11：`action_video_freq_ratio=4` 是不是规定部署时每 4 个动作才看一次图？

不能这样理解。它首先规定训练主视频 clip 怎样从 15 Hz 时间轴降采样，并影响 video RoPE 时间尺度和 preview fps。在线简单 policy 中，每次生成 8 步 action chunk 都会使用当时最新的 chunk observation 做 OVCR；而完整 Video DiT planner cache 何时刷新是服务调度问题。同步 wrapper 在 5 个 chunk 用完后重新 prefill，异步/双 GPU runtime 则可以按更新的观测独立发布 planner state。

---

## 18. 最小伪代码

### 18.1 训练

```python
# 数据与条件
video, action, proprio, text = batch
z = frozen_vae.encode(video)
obs_z = frozen_vae.encode(chunk_start_images)
visual_obs = visual_projector(flatten(obs_z))
chunk_queries = query_pool(base_queries, visual_obs)
proprio_tokens = proprio_encoder(chunk_start_proprio)

# flow corruption
z_t = (1 - sigma_v) * z + sigma_v * eps_v
z_t[:, :, :1] = z[:, :, :1]
a_t = (1 - sigma_a) * action + sigma_a * eps_a

# Video DiT 与逐层 OVCR + Action DiT
video_tokens = video_dit.pre(z_t, text)
action_tokens = action_dit.pre(a_t, text, proprio_tokens)
for layer in range(30):
    video_tokens, K0, V0 = video_layer(video_tokens, history_kv, text)
    K_tilde, V_tilde = ovcr[layer](chunk_queries, K0, V0)
    action_tokens = action_layer_by_chunk(
        action_tokens, K_tilde, V_tilde, text, proprio_tokens
    )

pred_v = video_head(video_tokens)[:, :, 1:]
pred_a = action_head(action_tokens)
loss = weighted_mse(pred_v, eps_v - z) \
     + weighted_mse(pred_a, eps_a - action)
```

### 18.2 在线动作块

```python
# 慢速刷新，可异步
planner_state = prefill_video(current_image, text, rolling_history)

# 快速闭环，每个 chunk 重复
latest_obs_z = frozen_vae.encode(latest_image)
queries = query_pool(latest_obs_z)
updated_video_kv = ovcr(queries, planner_state.video_kv)

a = gaussian_noise(chunk_shape)
for t, delta in flow_schedule(num_steps=10):
    velocity = action_dit(a, t, text, proprio, updated_video_kv)
    a = a + delta * velocity

execute(denormalize(a))
```

---

## 19. 外部资料与证据层级

1. **AHA-WAM 论文**：方法动机、公式、实验和论文系统优化。  
   <https://arxiv.org/abs/2606.09811>
2. **AHA-WAM 官方仓库**：当前公开实现与 checkpoint 使用。  
   <https://github.com/serene-sivy/AHA-WAM>
3. **Wan2.2 官方 TI2V-5B 配置**：VAE stride `(4,16,16)`、patch `(1,2,2)`、3072 hidden、24 heads、30 layers。  
   <https://github.com/Wan-Video/Wan2.2/blob/main/wan/configs/wan_ti2v_5B.py>
4. **Flow Matching for Generative Modeling**：velocity field 和 ODE 生成的理论背景。  
   <https://arxiv.org/abs/2210.02747>
5. **Scalable Diffusion Models with Transformers**：latent patch 上的 DiT 与条件调制背景。  
   <https://arxiv.org/abs/2212.09748>
6. **RoFormer**：RoPE 的旋转与相对位置性质。  
   <https://arxiv.org/abs/2104.09864>
7. **T5**：文本 encoder 的基础架构背景。  
   <https://arxiv.org/abs/1910.10683>
8. **Fast-WAM**：视频共训练有价值、测试时不一定需要显式未来想象的直接前序证据。  
   <https://arxiv.org/abs/2603.16666>

证据优先级是：当前源码和 compose 后配置 > 当前仓库 README > AHA-WAM 论文抽象 > 上游论文的一般概念。尤其是做你自己的 SO101 实验时，应保存 commit、完整 Hydra resolved config、dataset stats 和 deployment yaml，避免“同名 AHA-WAM”实际运行了不同模型。

---

## 20. 最终心智模型

可以把 AHA-WAM 想成三层时间尺度：

```text
慢：Video DiT planner refresh
    当前观测 + rolling history -> 一份逐层世界 K/V

中：每个 action chunk 开始
    最新图像 -> OVCR -> 当前 chunk 专用的世界 K/V

快：chunk 内生成与执行
    Action DiT 多步 flow 去噪 -> 短动作块 -> 机器人执行
```

对当前 SO101，再把具体数字压成一行：

```text
49 个 15-Hz 原始时刻(3.2 s)
 -> 主视频监督 13 RGB
 -> VAE 4 latent frames
 -> 训练一次 video/action 联合 flow 前向；

在线则是 1 张当前 RGB
 -> Video DiT 一次 planner prefill
 -> 每块最新 RGB/proprio 经 OVCR
 -> Action DiT 10 步积分
 -> 8×6 动作块(约 0.533 s)。
```

Video DiT 的意义不是“必须把未来视频画出来”，而是通过未来视频监督学习一个有动态含义的视觉规划表征；OVCR 的意义不是替代 planner，而是让这份慢表征跟上最新闭环状态；Action DiT 的意义是把条件分布建模成连续 flow，在短 horizon 内生成协调动作。

这三者共同构成 AHA-WAM。任何只描述其中一个分支的说法——“它就是视频生成模型”“它就是 diffusion policy”“它就是 KV cache 加速”——都不完整。
