# 视频 preview 到底等不等于训练/推理走的 DiT 链路

这份文档回答你的第二个问题：`scripts/so101_pipeline/video_preview/` 里那套视频生成，
是不是训练/推理时真正走的那条 VideoDiT 链路，输入是不是一致。

结论先说：

| 项目 | 结论 |
| --- | --- |
| 视频分支的**计算图** | 完全一致（已数值验证，见第 4 节） |
| 视频分支的**输入张量** | 一致（文本 context、first-frame latent、temporal RoPE、history K/V 全部对得上） |
| 单步训练 vs 多步采样 | 不同，但这是训练目标和采样器的正常关系，不是 bug |
| 历史帧的时间间隔 | **真的不一致**：训练 48 控制帧 / RoPE 12，在线部署 40 控制帧 / RoPE 10 |
| 这条链路在动作推理中的地位 | **在线动作推理根本不生成未来视频**，preview 是旁路诊断，不在动作输出的关键路径上 |

后两条是需要你知道的重点，第 5、6 节展开。

---

## 1. 这个数据窗口长什么样

先把几何摆清楚，后面所有讨论都基于它（数字来自
`runs/so101_g05_square_v1/full_zero3/.../model/config.yaml`）：

```
fps (LeRobot 控制帧)          = 15
num_frames                    = 49 个控制帧
action_video_freq_ratio       = 4          -> 每 4 个控制帧取 1 个视频帧
=> 一个窗口 = 13 个视频帧 @ 3.75 Hz，跨 48 个控制帧 = 3.2 秒
=> VAE 时间下采样 4 倍 -> 4 个 latent 帧
=> video_size = [384, 320]，VAE 空间 16 倍 -> latent 24x20
video_rope_frame_stride       = 4
num_history_frames            = 6
历史帧间隔 horizon_stride     = (49-1) * global_sample_stride = 48 个控制帧
历史帧 RoPE 间隔              = 48 / 4 = 12
```

所以 `run_preview_batch.sh` 里的 `FPS=3.75` 是正确的实时播放速度，
输出的 mp4 是 3.2 秒的真实时长。

SO101 的一帧是个拼图：上方 256 行是 fixed 相机，下方 128 行的左半 160 列是 wrist 相机，
右下角是黑色占位（对应 RoboCOIN 少掉的第二个腕部相机），见
`scripts/so101_pipeline/so101_dataset.py:20`。新脚本因此会分别报 `fixed_rgb_psnr`
和 `wrist_rgb_psnr`。

---

## 2. 训练时，视频分支实际执行了什么

SO101 配置里 `max_action_offset=7 > 0`，dataset 会给 sample 塞 `action_offset`，
所以 `AHAWAM.training_loss` 会转发到
[`_training_loss_action_offset`](../../../src/ahawam/models/wan22/ahawam.py#L939)。
两个函数的**视频那一半是逐行相同的**，动作 offset 只影响动作分支。

视频侧顺序：

1. `build_inputs` → VAE 编码 13 帧 → `input_latents` `[B,48,4,24,20]`；
2. 随机采样时间步 `t`，加噪，然后 `latents[:,:,0:1] = first_frame_latents`
   —— 第一帧 latent 永远被替换成干净的；
3. `video_expert.pre_dit(x=latents, timestep=t, context, context_mask, action=action,
   temporal_position_ids=...)`；
4. 历史 6 帧 → `_encode_training_history_latents` →
   `pre_dit(..., clean_prefix_frames=effective_history)` →
   `mot.prefill_video_cache(...)` 一次性带因果 frame mask 灌进 K/V；
5. `build_video_to_video_mask(per_frame_causal)`，前面再拼上 history 的全 1 前缀 mask；
6. `mot.forward_prior_action_with_chunk_updated_kv(...)`；
7. `video_expert.post_dit(...)`，loss 只算第 1..T-1 帧（第 0 帧是白送的）。

## 3. preview 执行了什么

[`deploy/server/ahawam_policy.py:predict_future_video_flow`](../../../deploy/server/ahawam_policy.py#L63)：

```
pre_dit(x=latents, timestep, context, context_mask, action=None,
        temporal_position_ids, fuse_vae_embedding_in_latents=True,
        clean_prefix_frames=1)
-> build_video_to_video_mask + history 全 1 前缀
-> mot.forward_video_with_history(...)
-> post_dit
```

外层 `denoise_future_video_latents` 用同一个 `infer_video_scheduler` 循环 N 步，
每步之后都把第 0 帧 latent 按回干净值——和训练里 `latents[:,:,0:1] = first_frame_latents`
是同一个约束。

逐项对照：

| 输入 | 训练 | preview | 一致？ |
| --- | --- | --- | --- |
| `clean_prefix_frames` | 没传 → 默认 `1`（`wan_video_dit.py:533`） | 显式传 `1` | ✅ |
| `context` | `inputs["context"]` | `sample["context"]` | ✅ 见下 |
| `temporal_position_ids` | `_get_main_video_temporal_position_ids(sample)` | `sample["video_temporal_position_ids"]` | ✅ 同一个张量 |
| first frame latent | `inputs["first_frame_latents"]` | `_encode_input_image_latents_tensor(video[:,:,0])` | ✅ 同一个 VAE |
| 注意力 mask | `build_video_to_video_mask` + history 前缀 | 同 | ✅ |
| 历史 K/V | `mot.prefill_video_cache` | 训练模式同；在线模式增量 prefill | ✅ 数学等价，见第 4 节 |

`context` 这条值得单独说一句，因为它绕了一圈：
`_build_standard_inputs` 会在 128 个 T5 token 后面**追加**一个 proprio token（变成 129），
紧接着 `AHAWAMChunkBase.build_inputs` 又把最后一个 token **切掉**（`ahawam_chunk_base.py:901`），
所以最终喂给视频 `pre_dit` 的就是原始那 128 个文本 token，
和 preview 直接用 `sample["context"]` 完全相同。
在线部署那边 `prefill_video` 调 `_prepare_action_context(..., proprio=None, ...)`，
`_append_proprio_to_context` 在 `proprio is None` 时原样返回，所以也是 128。三边一致。

---

## 4. 数值验证

代码：[`verify_video_path_equivalence.py`](verify_video_path_equivalence.py)。
它用真实权重、真实样本跑三个断言，结果写在
`runs/so101_g05_square_v1/baseline_experiments/path_equivalence.json`。

**断言 1：`pre_dit` 完全忽略 action。**
配置是 `action_conditioned=false`，`wan_video_dit.py:578` 那个
`if self.action_conditioned and action is not None` 分支进不去，
所以训练传 `action=action`、preview 传 `action=None`，输出的
`tokens / freqs / t_mod / context` 应当逐位相同。

**断言 2：MoT 里加不加动作分支，视频输出不变。**
用两组不同的随机动作张量跑
`forward_prior_action_with_chunk_updated_kv`，再和
`forward_video_with_history` 对比。
看 `mot.py:419-516` 就知道为什么必然成立：视频 block 的 K/V 只由
`history_video_kv_cache + 自己的 k_video/v_video` 拼成，
动作 token 从来没有进过视频的注意力；信息是单向的——
视频第一帧的 K/V 经 `chunk_kv_cache_editor` 流向 ActionDiT，反向没有通路。
所以"先生成动作再生成视频"是不必要的，preview 直接跑视频分支是对的。

**断言 3：一次性 prefill 和逐帧滚动 prefill 得到同一份 K/V。**
训练用 `prefill_video_cache` 把 6 帧历史一次灌进去（帧间因果 mask）；
在线部署 [`AHAWAM.prefill_video`](../../../src/ahawam/models/wan22/ahawam.py#L372)
每个 planner tick 用 `prefill_video_cache_with_prefix` 追加一帧。
两者的可见性语义相同（第 i 帧看得到 ≤ i 的所有帧），所以 K/V 应当相同。
这条成立意味着：**在线 rolling K/V 和训练 prefill 在算术上没有差别，
唯一的差别是喂哪些帧、放在哪个 RoPE 位置**——也就是下面第 5 节。

> bf16 矩阵乘在不同规约顺序下不是逐位可复现的，所以报告打印的是
> max/mean 绝对误差以及相对张量尺度的比值，而不是断言精确相等。

### 实测结果

用 SO101 微调 checkpoint（`step_004896.pt`）在窗口 `sample_index=18014`
（4 帧有效历史，RoPE 位置 `[0,12,24,36]`，当前 clip RoPE `[48,52,56,60]`）上跑出来：

| 断言 | 结果 |
| --- | --- |
| 1. `pre_dit(action=action)` vs `pre_dit(action=None)` | `tokens / freqs / t_mod / context` **全部逐位相同** |
| 2. 联合 MoT 换两组完全不同的随机动作张量 | 视频输出 **逐位相同**（max_abs_diff = 0.0） |
| 2. 联合 MoT vs `predict_future_video_flow` | **逐位相同**（max_abs_diff = 0.0） |
| 3. 一次性 prefill vs 逐帧滚动 prefill，第 0 层 K/V | **逐位相同** |
| 3. 同上，第 29 层（最后一层）K/V | mean_abs_diff 0.0037 / 0.0039，张量均值尺度 1.04 / 0.29 |

第 2 条是决定性的：**联合前向和纯视频前向的输出在 bf16 下逐位相同**，
说明 preview 不是"近似"训练链路，它就是训练链路里的那一半，一个 bit 都没差。

第 3 条第 0 层逐位相同、深层出现 0.3% 量级的漂移，是 30 层 bf16 累积的数值噪声
（两种 prefill 的规约顺序不同），不是语义差异——如果 mask 语义不同，
第 0 层就该先对不上了。

完整报告：`runs/so101_g05_square_v1/baseline_experiments/path_equivalence.json`。

---

## 5. 真实的不一致：历史帧间隔 48 vs 40

- **训练**：dataset 的历史帧取在 `sample_idx - 48*k`
  （`robot_video_dataset.py:493`），RoPE 位置差 `48/4 = 12`。
- **在线部署**：`prefill_video` 每次刷新推进
  `_observed_frame_index += action_horizon // video_rope_frame_stride = 40/4 = 10`
  （`ahawam.py:490`），因为 action server 每 40 个控制帧重规划一次。

也就是说模型训练时见到的历史 RoPE 间隔是 12，真机部署时喂的是 10。
这是 AHA-WAM 代码本身就存在的一处 train/deploy mismatch，不是 preview 引入的。

你原来的 `run_preview_batch.sh` 用的是 `--history-mode online`，
所以它忠实反映的是**部署**链路，不是**训练**链路。原目录的 README 也写明了这一点。

新脚本默认把 `online / training / none` 三种都跑一遍，你可以直接看到这个差值。
在 16 个窗口上实测（AHA-WAM-Pretrained checkpoint）：

```
online   (RoPE 间隔 10) : 12.634 dB
training (RoPE 间隔 12) : 12.676 dB
none     (不给历史)      : 10.763 dB
```

差 0.04 dB。结论：**这个 mismatch 目前不是主要矛盾**，但给不给历史差 1.9 dB，
说明历史 K/V 本身是有效的。

如果以后想对齐，两个方向选一个：
把 `data.train.num_frames` 调成 `40+1=41`（让训练间隔等于 `action_horizon`），
或者把 action server 的重规划间隔改成 48。前者会顺带改掉窗口长度，
后者不用重训，建议先试后者。

---

## 6. 更重要的一点：在线动作推理根本不生成视频

> 完整的时序拆解、实测耗时、以及和论文的逐条对照，见
> [`../PIPELINE_TIMING_ZH.md`](../PIPELINE_TIMING_ZH.md)。


看 [`AHAWAM.prefill_video`](../../../src/ahawam/models/wan22/ahawam.py#L372) 就清楚了：
它只把**当前这一帧**编码进 video K/V cache，然后
`infer_action_chunk` 通过 chunk KV editor 去读它。
整条动作产出链路里**没有任何一次未来视频去噪**。

`AHAWAMPolicy.generate_future_video`（`deploy/server/ahawam_policy.py:665`）
是 policy server 上的诊断/预览功能，默认不参与动作输出。

所以这个 preview 衡量的是：
- ✅ 视频分支的生成质量 —— 也就是 `loss_video` 直接优化的东西；
- ✅ VideoDiT 这个 expert 学到的表征质量（ActionDiT 就是从它的 K/V 里取信息的）；
- ❌ **不是**动作精度的直接代理。

它仍然是很有价值的诊断——视频分支塌了基本意味着表征塌了——
但"视频看起来对"不能直接推出"动作会对"。要评动作，还是得看
`evaluate_validation` 的动作指标或者真机成功率。

---

## 7. 我在新脚本里做的调整

1. **三种 history mode 全跑**，而不是只跑 `online`，这样训练/部署两条 cadence 的差值一眼可见。
2. **分区域指标**：`fixed_rgb_psnr` / `wrist_rgb_psnr`。全图 PSNR 会被右下角那块常黑区域拉高，
   分开看才知道模型是不是只学会了"背景不动"。
3. **样本选择重写**：跨 4 个 task × 跨 episode × 跨历史深度，一共 16 个窗口，
   而不是原来的"每个 task 取历史最长的那个"。原来的选法会系统性偏向 episode 后段。
4. **所有 checkpoint 共用同一批窗口**（`export_reference_samples.py` 先落盘 manifest），
   所以 raw Wan2.2 / AHA-WAM-Pretrained / SO101-finetuned / MiniMax-H3 的数字可以直接对比。
   原来的脚本每次运行都重新挑窗口，跨 checkpoint 的数字其实不可比。
5. **第 0 帧继续排除在指标外**（原 `_frame_metrics` 已经是 `[1:]`，这点没改），
   因为它是所有模型都白送的干净条件帧。

去噪本身仍然调用 `deploy/server/ahawam_policy.py` 的
`denoise_future_video_latents()`，历史 K/V 仍然复用
`archive/video_path_experiments/exact_video_path.py` 里的
`_prepare_history_cache` / `_prepare_online_history_cache`，
所以新旧脚本走的是同一份代码，数字可以和你之前的结果对照。

---

## 8. 顺带发现的一个 dtype 问题（和 preview 无关，但影响动作分支）

写验证脚本时撞到的，值得单独记一笔。

`MoT.configure_chunk_kv_cache_editor`（`mot.py:224`）用
`next(self.parameters()).dtype` 决定 editor 的 dtype。
而 MoT 自己注册的第一个参数是
`action_branch_embedding = nn.Parameter(torch.zeros(2, hidden))`，
它是**默认 float32** 的——`BaseWAM.__init__` 里只有 `self.to(self.device)`，没有 `.to(dtype)`。

结果就是，普通地

```python
model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda:0")
model.load_checkpoint("....pt")
```

之后：

```
mot.chunk_kv_cache_editor : torch.float32     <-- 这两个是 fp32
mot.action_branch_embedding: torch.float32
chunk_obs_query_encoder    : torch.bfloat16
action_expert / video_expert: torch.bfloat16
```

`load_checkpoint` 也不会改（`load_state_dict` 的 `copy_` 保留目标 dtype）。

训练时看不出来，因为 accelerate/DeepSpeed 会把整个 module 统一 cast 成 bf16。
但 `build_layer_updated_cache` 会把 K/V 转成 editor 的 dtype（`mot.py:86-97`），
于是走到 `_forward_chunk_routed_prior_only_attention` 时会变成
`query=bf16, key=float32, value=float32`，SDPA 直接抛：

```
RuntimeError: Expected query, key, and value to have the same dtype,
but got query.dtype: c10::BFloat16 key.dtype: float and value.dtype: float
```

**这只影响动作分支，视频 preview 完全不碰 editor，所以实验 A/B 不受影响。**
我在 `verify_video_path_equivalence.py` 里显式 cast 了一下绕过去。

需要你自己确认的是：`AHAWAMPolicy.__init__` 里只有
`self.model = self.model.to(device).eval()`（`ahawam_policy.py:289`），同样没有 dtype cast。
如果你的 action server 目前跑得通，说明还有别的地方兜住了；
如果哪天报上面那个 dtype 错误，根因就是这里。稳妥的修法是在
`configure_chunk_kv_cache_editor` 里改成用 `self.torch_dtype`，
或者在 `BaseWAM.__init__` 末尾补一句 `self.to(dtype=torch_dtype)`。
