# 实现范围与验证记录

日期：2026-09-05。以下区分实际执行的检查与仍未执行的实验。

## 1. 文件与职责

| 文件 | 本轮内容 |
| --- | --- |
| [ovcr_adapter.py](../../src/ahawam/models/wan22/ovcr_adapter.py) | Query／Value／Token-KV 三种模块、冻结设置、P0 同噪声 anchor |
| [mot.py](../../src/ahawam/models/wan22/mot.py) | 原 editor 共用方法中的 query 和 output K/V 注入点 |
| [ahawam.py](../../src/ahawam/models/wan22/ahawam.py) | 保持原 native loss 实现，增加可选 teacher/student anchor 包装与 flow 捕获 |
| [ahawam_chunk_base.py](../../src/ahawam/models/wan22/ahawam_chunk_base.py) | checkpoint 保存 adapter config，加载时自动安装并严格校验新增权重 |
| [runtime.py](../../src/ahawam/runtime.py) | factory 可选 `ovcr_adapter`、`ovcr_anchor_weight`；默认无模块，保留原位置参数顺序 |
| [train.py](../../scripts/so101_pipeline/ovcr_adapter/train.py) | 现有 HIL dataset＋ZeRO-2 trainer 的 adapter-only 入口、配置预检、seed 和评估间隔 |
| [query.yaml](../../scripts/so101_pipeline/ovcr_adapter/query.yaml)、[value.yaml](../../scripts/so101_pipeline/ovcr_adapter/value.yaml)、[token_kv.yaml](../../scripts/so101_pipeline/ovcr_adapter/token_kv.yaml) | 三种默认方法配置 |
| [rl_token_readout.py](../../src/ahawam/models/wan22/rl_token_readout.py) | 独立 RLT-inspired 表征重建原型，输入冻结，支持 mask |
| [train_readout.py](../../scripts/so101_pipeline/ovcr_adapter/train_readout.py) | 对明确标注来源的 feature 文件训练 reader |
| [test_ovcr_adapter.py](../../tests/test_ovcr_adapter.py) | 结构、真实小尺寸 DiT、梯度、cache、anchor、checkpoint 和 readout 回归 |

所有新增说明文件集中在本目录；只在 [docs 导航](../README.md) 加入口。任务开始前已有 deploy/server、client、Flash 和其他研究文档的未提交修改，本轮没有回退或覆盖这些改动，也没有提交 Git commit。

## 2. 已实际执行的验证

运行环境：`/home/fenghao/miniconda3/envs/ahawam/bin/python`，PyTorch `2.7.1+cu128`，CPU；测试使用标准库 unittest，不需要额外安装 pytest。

测试命令：

```bash
PYTHONPATH=src:. /home/fenghao/miniconda3/envs/ahawam/bin/python \
  -m unittest discover -s tests -p test_ovcr_adapter.py -v
```

**8 个测试通过。** 部分测试内部对 Q32/V32/T4 逐项运行，覆盖：

1. 三种模块零初始化时与 bypass 输出逐值一致；输入 cache/tensor 不被原地改变。
2. 动作 attention 的损失可反传到 adapter，冻结基座参数不产生梯度；一步更新能改变动作条件；V-only 的 K 保持不变。
3. 原 `build_chunk_updated_video_kv_cache` 的 chunk slicing 与直接 editor 输出一致；原 Action DiT 的实际 pre/forward/post 路径能反传，并开启 MoT checkpointing。
4. 真正的 **WanVideoDiT＋OVCR＋ActionDiT** 小尺寸训练路径可反传；Video 和 Action 参数无梯度。这不是 mock 掉 Action DiT 后只测 MLP。
5. layer 选择和 adapter state-dict round-trip；非法 layer index／重复层／空层报错。
6. 调用实际原生 checkpoint save/load 方法，自动恢复三种 adapter 的元数据和参数；缺失新增权重报错；在已适配实例上重新加载普通 P0 时，将 adapter 输出投影归零，避免残留旧纠正。为避免加载 6B 资产，此项以极小模块承载 serializer/loader。
7. P0/student 共享同一随机 noise，teacher forward 不额外消耗最终 RNG 序列；anchor 只计 demo 的有效位置，不计 HUMAN／padding。
8. readout 对 frozen 输入不反传；读出 query 能得到梯度；padding 中的 NaN 不污染表示和 loss；全无效样本拒绝。

首次新增双 DiT 测试时，fixture 未启用 Wan 要求的 `seperated_timestep`／`fuse_vae_embedding_in_latents`，触发原模型的输入检查；已将 fixture 改为与真实训练相同的模式后通过，没有为通过测试而改动原 Video DiT 行为。

此外已完成：

- 三个 variant 的 Hydra config 构建；保持 chunk=8、Video loss=0、`checkpoint_shape_adapt=false`。
- `train.py --dry-run` 使用真实 step_005988 路径解析并输出配置。
- 使用 `torch.load(..., mmap=True, map_location="meta")` 检查真实 P0 元数据，不执行它的推理；确认保存 step=5988、action encoder shape=(1024,6)。文件大小 14,699,136,953 bytes。
- 重新统计 D0=23,947 帧；DH=97 段／9,158 帧；确认原 stats 文件存在。
- 按生产维度实例化新增模块并计数：Q32=2,744,640；V32=2,058,560；T4=5,511,360。
- readout CLI 用 **明确标为 synthetic-test-only** 的 6×4×12 随机 features 完成 2-step CPU 训练并保存 checkpoint。该 smoke 只验证优化器／序列化，不是任务数据训练成果。
- 检查本地 Accelerate 支持 `-m/--module`；文档训练命令采用已安装 CLI 支持的形式。

## 3. 仍未验证或未实现的内容

| 项目 | 当前状态与下一步 |
| --- | --- |
| full-size GPU backward | 未执行；当前 CUDA 不可用，先恢复驱动再按 03 的 2-step smoke 执行 |
| 大模型 bf16／8卡 ZeRO-2 性能 | 未测；CPU float32 的正确性不替代分布式和显存验证 |
| 实际数据 decode→VAE→loss 整链 | 本轮复用已有经过审计的 loader，但尚未运行 full-size end-to-end batch |
| SO101 成功率、延迟、改善程度 | 没有本轮新结果；需统一 B0/Q32/V32/T4 真机评测 |
| 低 rank／后 10 层是否最优 | 是起始假设，需要参数和层位置消融 |
| intervention risk gate | 未实现；现有 HUMAN-only dataset 不能直接提供无偏 gate 标签 |
| critic、reward learner、在线 RL collector | 未实现；04 给出目标和 replay 契约，而不是声称已经接通 |
| 实机 readout feature exporter | 未实现；readout 脚本消费显式提供的 frozen feature 文件 |
| compact adapter-only 分发文件 | 未实现；本轮保存原生完整模型权重，部署读相同 loader |
| Flash / 任意自定义转换器 | 未覆盖；若转换器丢弃 adapter config，必须增加兼容验证 |

## 4. 一个容易误判的计算成本

Video loss 系数为零意味着该分支没有优化目标，但当前 native training 仍执行 video forward、输出与 loss 计算。本轮没有为省计算重写整个 video training pipeline。所有 base 权重冻结可省 optimizer、parameter gradient；ActionDiT 在 adapter 之后的激活仍要用于反传，P0 anchor 还多一次前向。

因此“只训约 2M 参数”不等于训练只需要一个 2M 模型的显存／时间，也不能据此承诺消费级 GPU 就能训练。先完成 full-size smoke 并量化瓶颈后，才能决定是否缓存 frozen 前缀、复用 teacher cache 或裁剪无用 video loss 计算；这类优化应保持同一数值目标。

## 5. 最小后续验收

依次完成 GPU 2-step smoke → 保存并重新加载推理 → 与 adapter bypass 做同输入比较 → V32/Q32 各 1,000-step 监督训练 → 原任务和恢复任务配对评测。如果这一步没有收益，应先报告诊断结果，再考虑 T4／Action-LoRA／更好的观测；不要直接用未校准 critic 增大优化强度。
