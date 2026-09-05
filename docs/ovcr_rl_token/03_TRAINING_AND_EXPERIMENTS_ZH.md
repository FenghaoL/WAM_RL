# 训练起点、数据、命令与实验矩阵

## 1. 从哪个 checkpoint 出发

主实验用 **相同的 SO101 插入任务 SFT checkpoint P0**，并固定与它匹配的 normalization、相机拼接、horizon、history 和 action 定义。这里的“相同”同时适用于 Q32/V32/T4、teacher anchor 和无 adapter 基线。

本轮核查了已有 `settings_mixed_from5988.yaml` 指向的文件，确认存在、保存 step=5988，action encoder 输入确实为 6 维：

```text
runs/so101_ahawam_plug_all/full_zero2_offload_ahawam_base/
  formal-plugall-ahawambase-zero2offload-bs4-acc1-4ep-3gpu-20260827-020601/
    model/checkpoints/weights/step_005988.pt
```

它是**可复现的候选起点**，不是本轮新证明的“最佳 checkpoint”。如果当前另一个 checkpoint 在独立验证中有更好的原始任务表现，就整组改用那个 P0；不能 Q32 从一个已经做过 DAgger 的模型起步，而 V32 从较弱旧模型起步。

不要直接从 `checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt` 启动冻结全基座的主实验：官方预训练模型与 SO101 action/proprio 接口可能需要形状适配和任务学习，冻结这些适配后的层会把不合适的动作语义一起固定。官方 checkpoint 应先经过现有 SO101 SFT，再作为 adapter 基座。

也暂时不与 Flash ODE 蒸馏一起变更。先固定原 10-step action flow，证明 adapter 有效，再评估“先纠正后蒸馏”和“先蒸馏再纠正”。后者用的 base teacher、flow trajectory 与前者不同，属于单独实验。

## 2. 当前实际可用的数据

复用 [settings_mixed_from5988.yaml](../../scripts/so101_pipeline/hil_human_only_training/settings_mixed_from5988.yaml) 和 [SO101DemoHumanCorrectionDataset](../../scripts/so101_pipeline/hil_human_only_dataset.py)。

| 数据 | 角色 | 监督 |
| --- | --- | --- |
| D0：原 02 grasp/move/align ＋ 04 full remove/insert demo | 维持原始技能、覆盖完整任务 | native action flow ＋ P0 anchor |
| DH：四个 2026-08-29 HIL 目录的连续 HUMAN logical episodes | 学恢复与精细纠正 | native action flow，不对齐 P0 |
| 完整 HIL raw 的 POLICY 行 | 未来风险建模／RL transition | 本轮不作为专家 BC target |
| SYSTEM_HOLD 行 | 执行协议、未来时间跨度分析 | 本轮不做 action supervision |
| 独立 04 验证目录 | held-out action 指标 | 不进入训练 |

这四个 HIL 目录为：

```text
data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v1/20260829_004417
data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v1/20260829_110214
data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v1/20260829_112454
data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v1/20260829_155403
```

本次重新运行现有 summary：35 raw episodes、22,314 raw frames → 97 HUMAN segments、9,158 HUMAN frames。不能拿 raw 22,314 当成 HUMAN epoch 长度。D0 的目录定义及原始统计见 [HUMAN-only 实现文档](../hil/AHAWAM_HIL_HUMAN_ONLY_TRAINING_IMPLEMENTATION_ZH.md)；原协议统计为 23,947 demo anchors。

保持原 stats：

```text
runs/so101_ahawam_plug_all/assets/dataset_stats.json
```

它已在本地核实存在。本轮不重算 stats；否则既改变学习数据，也改变关节标准化坐标。动作仍是 6D **绝对关节目标**，不是 TCP delta，也不是 measured state。HUMAN target 使用安全层之后的有效 `action`，不能换成被裁剪前的 `human_action`。

logical episode loader 已按 HUMAN 连续段限定图像、历史与 action 窗口；段尾复制只用于组成张量，`action_is_pad` 屏蔽越界动作。本轮 Video loss=0，短 HUMAN 片段仍可用于 action。离线历史从 HUMAN 段起点构造，与实机 takeover 前的旧 planner cache 并不相同；这是已知分布差异，不宣称是在线 cache 的精确回放。

## 3. 第一轮超参数

以下是实验起始值，尚未经过 SO101 adapter 实机调参。

| 项目 | 默认 |
| --- | --- |
| 改写层 | 20–29，共 10 层，0-based |
| bottleneck rank | 32 |
| T4 tokens | 4 |
| action horizon / chunk | 40 / 8 |
| action offset / history | 最大 7 / 6 帧 |
| flow inference steps | 10，与 P0 相同 |
| 可训练范围 | 只有新增 adapter |
| 直接 Video loss | 0 |
| learning rate | 1e-4；备选 3e-5 |
| optimizer | 原 trainer AdamW，β=(0.9,0.95)，原设置 weight decay=0.01 |
| schedule | 原 cosine + 默认 warmup ratio 0.05 |
| source sampling | 默认 `balanced_2_2`，每卡 batch=4：2 demo＋2 HUMAN |
| loss | native masked action flow ＋ λa=1 的 demo same-noise anchor |
| 第一轮训练预算 | 每个方法 1,000 optimizer steps；每 250 steps 保存 |
| periodic evaluation | 每 250 steps，复用原 trainer 固定验证子集；`--eval-every 0` 可关闭 |
| seed | 首轮 42；有效后再 43、44 |
| 验证 | 相同 episode/seed/noise；真机评测单独固定初始条件 |

先末 10 层是减少自由度和计算图长度的工程假设，不是知道“精细操作只在后 10 层”。若无增益，先检查输出投影是否更新、adapter 对动作是否有可测灵敏度，再扩展为全部 30 层。

`balanced_2_2` 增加 HUMAN 曝光次数；正式比较按 optimizer steps 和采样次数匹配，不按不同 sampler 的 nominal epoch 数比较。不同 GPU 数会改变每 step 样本数，主表还要匹配有效 global batch。八卡 batch4、acc1 时 1,000 steps 是 32,000 draws；其中理想平衡为各 16,000 次 demo/HUMAN，包含重复采样，不能说新增了 16,000 帧人类数据。

## 4. 可直接执行的预检

从仓库根目录运行。当前环境没有 shell 默认 `python`，以下使用已安装的环境路径。

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
OVCR_PYTHON=/home/fenghao/miniconda3/envs/ahawam/bin/python
OVCR_P0=runs/so101_ahawam_plug_all/full_zero2_offload_ahawam_base/formal-plugall-ahawambase-zero2offload-bs4-acc1-4ep-3gpu-20260827-020601/model/checkpoints/weights/step_005988.pt

"$OVCR_PYTHON" -m scripts.so101_pipeline.ovcr_adapter.train \
  --variant scripts/so101_pipeline/ovcr_adapter/value.yaml \
  --init-checkpoint "$OVCR_P0" \
  --output-dir runs/ovcr_adapter/preflight_value \
  --dry-run

"$OVCR_PYTHON" -m unittest discover -s tests -p test_ovcr_adapter.py -v
```

`--dry-run` 解析 Hydra、写 resolved config、检查 P0 路径和基本实验设置，不加载 6B 模型，也不代表已经验证了所有视频解码、GPU 显存、分布式 backward。对所有 variant 分别执行一次即可。

## 5. GPU 恢复后的训练命令

以下复用仓库现有 ZeRO-2 no-offload 配置。GPU 数、设备可用性和显存需按实际服务器调整；不是已经运行过的性能承诺。

```bash
export PYTHONPATH="$PWD/src:$PWD${PYTHONPATH:+:$PYTHONPATH}"
OVCR_PYTHON=/home/fenghao/miniconda3/envs/ahawam/bin/python
OVCR_P0=runs/so101_ahawam_plug_all/full_zero2_offload_ahawam_base/formal-plugall-ahawambase-zero2offload-bs4-acc1-4ep-3gpu-20260827-020601/model/checkpoints/weights/step_005988.pt

"$OVCR_PYTHON" -m accelerate.commands.launch \
  --config_file scripts/so101_pipeline/hil_human_only_training/accelerate.yaml \
  --num_processes 8 \
  -m scripts.so101_pipeline.ovcr_adapter.train \
  --variant scripts/so101_pipeline/ovcr_adapter/value.yaml \
  --init-checkpoint "$OVCR_P0" \
  --output-dir runs/ovcr_adapter/v32_seed42 \
  --steps 1000 --save-every 250 --anchor-weight 1.0
```

首次把 `--steps 1000 --save-every 250` 改为 `--steps 2 --save-every 2`，输出到单独 smoke 目录，核对 trainable count、有限梯度、loss、导出的权重再启动正式运行。Q32/T4 只改 variant 和 output-dir；种子重复用 `--seed 43`、`--seed 44`。不要同时启动三组满卡任务。

入口继承现有 `Zero2WeightOnlyTrainer`，base 权重在 `accelerator.prepare` 之前加载，避免 DeepSpeed master weights 覆盖初始化。权重保存是原生完整 AHA checkpoint，含 adapter config 和 weights；**并非只保存几 MB adapter，也不保存 optimizer state**。继续训练可将新 checkpoint 作为 `--init-checkpoint`，但这是 weight-only continuation，不是精确续接 optimizer/scheduler 的 resume。

默认关闭 W&B，不新增外部数据上传。数据统计、文本 embedding cache 等仍依赖已有项目资产，缺失时报错应先准备资产，不能把干预段换成随机数据凑通流程。

## 6. checkpoint 与部署

在更新后的代码中，`AHAWAMChunkBase.load_checkpoint` 读取 `ovcr_adapter_config` 自动安装相同模块，再严格加载 adapter 权重。原本不带 adapter 的 checkpoint 仍按旧路径加载。若模型已安装的 adapter 配置与 checkpoint 不一致，直接报错；缺失 adapter 元数据或部分权重也报错。

部署必须使用包含本次 loader 改动的服务器代码，以及该训练 run 的 matching config、stats、动作定义。旧服务器使用 `strict=False` 可能把陌生 adapter keys 忽略，产生“以为部署了新模型，实际只跑 base”的错误。上线前检查实际加载日志和 `model.mot.chunk_kv_cache_editor.adapter.config`。

不必改变原客户端 TCP action schema；输出仍是原形状。更换权重时重新启动／reset planner 和 action queues，不能混用新权重和旧 cache。Flash/自定义权重转换器若不调用该 loader，应另做兼容验证；本轮未声称覆盖所有转换路径。

## 7. 先做什么实验，分别回答什么问题

| 组别 | 初始化／数据 | 训练模块 | 想回答的问题 | 状态 |
| --- | --- | --- | --- | --- |
| B0 | P0，不更新 | 无 | 当前真实基线 | 使用现有模型 |
| Q32 | 同一 P0，D0＋DH | query adapter | 重新路由能否学恢复 | 已实现 |
| V32 | 同上 | value adapter | 内容纠正是否已足够 | 已实现，首选 |
| T4 | 同上 | token K/V writer | 显式瓶颈读写是否有帮助 | 已实现 |
| Q/V/T no-anchor | 与对应组一致 | 同上，λa=0 | 保持项是否减少遗忘 | 已实现开关 |
| natural control | 与对应组一致 | 同上，sampling=natural | 改善来自架构还是 HUMAN 曝光 | 已实现开关 |
| Action-only full | P0、匹配数据预算 | 原 Action DiT/少量条件模块 | 新接口是否比直接 SFT 更高效 | 仓库已有 trainer，需单独配置公平预算 |
| Action-LoRA | 同一 P0、同一数据 | action attention/MLP LoRA | 改路由与改动作映射的公平比较 | 本轮未实现，不应称已有结果 |
| RLT-reader＋actor | P0＋重建训练＋在线 transitions | 独立 readout 后冻结，再训 actor/Q | 是否值得由外部小 actor 接管 | 仅 readout 原型已实现；RL 在 04 设计 |

先做 B0、V32、Q32；若二者都无增益，检查数据与 feature 可辨识性，而不是马上大扫超参。若至少一组改善恢复且保持原任务，加入 T4 和 matched-capacity 控制。

第二轮优先消融：rank=8/32/64；层=后10/全部30；T4 tokens=1/4/8；anchor=0/0.1/1。一次只改变一个维度。T4 与 Q/V 的参数量不同，报告实际 trainable 参数和峰值显存；等 rank 不等于等 capacity。

## 8. 评测必须区分“恢复更强”与“原任务遗忘”

设置两类实机起点：完整原任务起点；固定夹具下 near-contact／轻微横向或角度偏差的恢复起点。每类固定任务分布和评测超时，策略顺序交错，减少夹具磨损、光照和操作者熟练度变化的偏差。

首轮每组每类至少 20 次只能用于筛选；最终建议每类至少 50 次并报告置信区间，不能根据 9/10 对 8/10 下强结论。seed 重复与独立 episode 重复分别报告。

主指标：自主成功率、允许接管后的 assisted success、人类接管次数／分钟、成功 episode 时间、卡滞／退回次数。另报成功率和执行时间共同决定的 throughput，不能只删掉失败轨迹后比较速度。

保持指标：D0 同初始条件下成功率，same-seed action chunk drift，长距离运输／对齐阶段错误率。恢复指标：held-out HUMAN episode 的有效 action L1/L2、pad-aware flow loss、恢复起点的自主成功率。

系统指标：request 总耗时 p50/p95、OVCR 耗时、prefill age、hold time ratio、实际执行 chunk 长度、接管响应延迟、参数量和峰值显存。adapter 不会自动解决 15 Hz 下 8 步约 0.533 秒的开环执行问题；缩短执行前缀应作为另一组时序消融，不能与结构同时变更再归因给 adapter。

同一 raw episode 的多个 HUMAN 段必须放在同一 split；不能把相邻帧随机分进 train/val。当前默认验证目录主要覆盖原任务，若要选恢复性能最好的 checkpoint，需要新增独立 DH-validation recording 或在训练前按 raw episode 划分，不能在测试任务上调 λa。

## 9. 只读 readout 原型怎样用

可选接口文件格式：

```python
{
    "tokens": FloatTensor[N, M, D],
    "valid": BoolTensor[N, M],  # True=有效
    "feature_spec": {
        "base_checkpoint": "P0 path and hash",
        "normalization": "stats path and hash",
        "token_source": "例如 chunk_queries，或明确层号的 routed tokens",
        "cache_protocol": "例如 offline logical-HUMAN reset；不能谎称 online snapshot",
    },
}
```

```bash
"$OVCR_PYTHON" -m scripts.so101_pipeline.ovcr_adapter.train_readout \
  --features /path/to/frozen_features.pt \
  --output runs/ovcr_adapter/readout/reader.pt \
  --hidden-dim 256 --steps 2000 --device cpu
```

这是离线 reader 优化器，尚没有本轮新增的实机 feature exporter、replay collector、critic 或 robot actor loop。不要把其 reconstruction loss 下降标注成 RL 成功率。若要走这条路线，先用 held-out feature set 比较 simple mean pooling、learned readout、不同宽度，以及 z 置零／打乱后的任务 probe。
