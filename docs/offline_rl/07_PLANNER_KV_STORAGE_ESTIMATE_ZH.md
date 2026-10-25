# Planner 最后一层 KV：全量缓存容量估算

日期：2026-10-24。基于本地配置和源码静态推导，以及逐个 parquet 的 metadata 行数核对；未执行 GPU 推理，尚无实际耗时基准。此次只读全目录 metadata，没有修改原始数据或对其他未完成标注的数据做训练。

## 1. 结论与假设

按当前 cylinder 部署的 384×320 双相机**拼接画面**、Wan2.2 TI2V-5B、batch=1、BF16，单个观察的最后一层 K 与 V 分别为 `[1,120,3072]`，两者合计 **1.40625 MiB**。整个 `data/ahawam_hil_raw/so101_ahawam_cylinder_hil_v2` 共 **43,680 帧、103 条 episode**，逐帧各存一次最后一层 K+V 的纯 tensor 容量为 **59.99 GiB（64.41 GB）**，建议预留 70–80 GiB。保存可行，推理耗时尚需 GPU 小批量实测。

这一结论适用于同形状部署。更换 checkpoint 但不更改模型/输入尺寸不改变容量；改变分辨率、精度、保存层数或历史展开方式会改变容量。

## 2. 形状依据

- [部署配置](../../scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_action_dit_only.yaml)：输入 `video_height=384, video_width=320`，`mixed_precision=bf16`。fixed/wrist 在 [policy](../../deploy/server/ahawam_policy.py) 的 `prepare_camera_images` 内拼成一张图，不能再乘相机数 2。
- [WanVideoVAE38](../../src/ahawam/models/wan22/wan_video_vae.py)：空间缩小 16 倍；单张图 latent 网格为 `24×20`。
- [模型配置](../../configs/model/ahawam.yaml)：DiT patch 为 `[1,2,2]`，所以 token 网格 `12×10=120`；24 heads × head_dim 128 = **3072**。Action DiT 的 hidden_dim=1024 不等于缓存宽度。
- [MoT](../../src/ahawam/models/wan22/mot.py)：`prefill_video_cache` 保存每层 K/V `[B,S,3072]`；`prefill_video_cache_with_prefix` 的返回值仅含当前帧 K/V，历史通过 attention 影响其值，但不会拼进返回张量。`build_layer_updated_cache` 用 32 query 路由后解码回 S 个 token，因此 **32 query 不代表最终 cache 只有 32 token**。
- [AHA-WAM](../../src/ahawam/models/wan22/ahawam.py)：`video_kv_cache` 为原 planner cache，`_chunk_video_kv_cache` 为动作分支使用的 post-OVCR cache；本配置下两者形状相同，语义不同，保存时必须标明选哪一个。

公式：`bytes = 观察数 × 保存层数 × token数 × heads × head_dim × 元素字节数 × (K和V的份数)`。

输入是单帧，不能再乘训练视频长度 49、preview 长度 13、action_horizon 40、动作块 8 或去噪步数 10。最后一层也仍需计算前面的 planner 层才能取得。

## 3. 数据实数

| 子目录 | episode 数 | parquet 行数 |
| --- | ---: | ---: |
| 20260911_183057 | 7 | 2,941 |
| 20260911_225809 | 30 | 12,010 |
| 20260911_232927 | 30 | 11,502 |
| 20260912_001855 | 30 | 14,271 |
| 20260912_010503 | 2 | 1,241 |
| 20260912_010747_val | 4 | 1,715 |
| 合计 | 103 | 43,680 |

包含 val，仅用于容量统计；学习 reader/actor/critic 时仍需隔离验证集。全部帧是成功截断之前的原始规模上界。最初指定的 183057 子集，BF16 最后一层 K+V 约 **4.04 GiB**。

| 保存内容，每观察一次 | 每观察容量 | 全目录纯 tensor 容量 |
| --- | ---: | ---: |
| 最后一层原始 K+V，BF16/FP16 | 1.40625 MiB | 59.99 GiB |
| 最后一层原始 V，BF16/FP16 | 0.703125 MiB | 29.99 GiB |
| 最后一层原始 K+V，FP32 | 2.8125 MiB | 119.97 GiB |
| V 池化成 64 token，FP32（当前 exporter） | 0.75 MiB | 31.99 GiB |
| reader 压成单个 256 维 token，FP32 | 1 KiB | 42.66 MiB |
| 全 30 层原始 K+V，BF16 | 42.1875 MiB | 1,799.56 GiB，约 1.76 TiB |

当前磁盘显示 workspace 所在文件系统可用约 1.6 TiB，最后一层容量足够；文件系统余量会变化。不要将全量缓存一次装进 GPU，亦不建议现有 Python dict 全量驻留再保存；应分 episode/分 shard 写入，训练按需读取。

## 4. 如何保存更适合接下来的 RL

建议保存**未池化的最后一层 post-OVCR K+V，BF16**，逐观察去重，以 episode/frame 标识建立索引；约 60 GiB 能保留以后比较 K、V、K+V 和不同 reader 的余地。只用当前 V 路线也可仅存 V，约 30 GiB。现有 exporter 是 `.float()` 后对 V 做 64-token 平均池化，不能称为原始 KV 导出，且落盘没有 BF16 的空间优势。

同时存少量 metadata：checkpoint/config/hash、cache 类型、层号、dtype、观察索引、拼图方式、历史窗口/采样间距、随机种子、归一化配置；若 actor 需要参考动作，还应存对应 reference。s 和 s' 通过同一观察索引引用，不能对每条 transition 复制两份 KV。

若只存最后一层作为冻结 RL 表征，并不要求存全部历史层；但**不能仅靠最后一层缓存恢复 planner 的完整滚动推理状态**，因为每层 attention 都需要自己的历史 K/V。将当前帧和过去 6 帧的同层张量重复打包到每个样本，最坏会接近单帧容量的 7 倍；用帧索引引用历史可避免该重复。历史条件下算出的当前特征也不等于无历史独立帧特征，提取协议必须固定。

按逐帧推理时简单连续推进历史，与真实每次动作请求之间推进一次历史不是同一时间尺度；需要选择和训练/部署一致的因果窗口。不能声称仅凭 RGB 就还原原采集时异步 planner 的隐藏状态。原始 planner cache 不需要 action 去噪；post-OVCR 需要当前观察/查询相关步骤。现有 exporter 调用完整 `get_action()`，会同时产生 reference；如仅需要 cache，可单独优化特征提取路径，但需验证与部署结果一致。

## 5. 时间估算的限制

没有本机 GPU 实测，以下仅为 `43,680 × 单观察总耗时` 的换算，不是速度预测：0.2 秒/观察约 2.43 小时，0.5 秒约 6.07 小时，1 秒约 12.13 小时，2 秒约 24.27 小时。实际须计入解码、VAE、30 层 planner、OVCR、可选 reference 去噪、传输和写盘。

如果每个样本 reset 后重放 6 个历史观察再处理当前观察，最多约 7 次模型调用/目标观察，耗时可能明显高于一次连续扫描；不能直接把上述单调用耗时当成这种协议的总耗时。建议后续在已授权测试子集 183057 先做短段 shape/dtype/数值一致性与吞吐测试，再确定导出器及耗时预算。本次未启动全量推理。
