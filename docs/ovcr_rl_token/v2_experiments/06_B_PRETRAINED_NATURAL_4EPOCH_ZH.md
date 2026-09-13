# B 方案：从 AHA 预训练重新进行四轮混合 SFT

2026-09-07 用户明确启动本方案，并确认实际部署模型正确。本轮不再把部署入口作为实验失败原因。

## 已启动配置

| 项目 | 设置 |
|---|---|
| 实验名 | `b_full_pretrained_natural_4ep` |
| 初始化 | `checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt` |
| 更新范围 | video/action DiT、原OVCR及原任务SFT可学习的观测模块；VAE冻结 |
| adapter / anchor | 无 / 0 |
| 数据 | v2 demo train 15,984帧 + v2 HUMAN train 4,437帧 |
| 验证 | 原demo val 1,735帧 + 独立HUMAN val 1,061帧，沿用固定episode划分 |
| 采样 | natural；每轮整体打乱、不放回，分布约78.27% demo / 21.73% HUMAN |
| GPU | 物理1、2、3；GPU0已有其他任务，不占用 |
| batch | 每卡4，accumulation1，全局12 |
| 日程 | 4epoch × 1702step = 6808step |
| 优化器 | AdamW，lr1e-5，weight_decay0.01，betas=(0.9,0.95)，clip1 |
| 调度 | cosine，5% warmup（340step） |
| 精度/分布式 | bf16，ZeRO2 + CPU optimizer offload |
| 保存 | 1702、3404、5106、6808步 |
| 验证频率 | 每426步及起点/终点，固定双验证集，继承修复后的边界mask |

自然数据20421个起点；1702个全局batch容纳20424个起点，Accelerate在epoch尾部补齐少量样本，使三卡步数一致。四轮约为每个来源起点四次曝光，不是1:1过采样的14.4遍HUMAN。

video loss：demo沿用原规则；HUMAN只有当前起点提供完整49原始帧/13采样图像、无视频padding的窗口计算video loss，其他HUMAN窗口仅监督有效动作。两项权重均1。没有把自主运行动作作为专家标签。

从预训练初始化启用 `checkpoint_shape_adapt:true`，与原v2 P0任务SFT初始化方式一致，用于预训练到SO101维度适配。不是从5328继续，也不加载之前任何V/Q/T/full纠正权重。optimizer重新初始化。

## 复用与复现

复用此前已通过真实三卡训练的 v2 `run.py::FullTrainer`、原 `Zero2WeightOnlyTrainer`、HUMAN dataset、manifest及修复后的验证mask。仅新增B配置生成器；无新增算法改动。

```bash
# 已执行；拒绝覆盖现有配置
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/prepare_b.py

# 已在tmux会话 ovcr_b_pretrained_4ep 中启动，请勿重复执行
GPU_IDS=1,2,3 python scripts/so101_pipeline/ovcr_adapter/v2_experiments/suite.py \
  --names b_full_pretrained_natural_4ep
```

配置：`runs/ovcr_adapter_padapter_v2/configs/b_full_pretrained_natural_4ep.yaml`

日志：`runs/ovcr_adapter_padapter_v2/logs/b_full_pretrained_natural_4ep.log`

状态：`runs/ovcr_adapter_padapter_v2/suite_status.json`

输出：`runs/ovcr_adapter_padapter_v2/b_full_pretrained_natural_4ep/model/`

最终权重：`checkpoints/weights/step_006808.pt`（训练完成后生成）。完整权重可通过原action server加载，继续使用v2 stats和power-adapter任务指令。当前未改部署配置。

每轮保存权重，不保存optimizer state；因此中断后不能把weight-only restart当作精确optimizer恢复。前次full运行状态已归档至 `suite_status_before_b_pretrained_4ep.json`。

本任务回答“demo和纠正从任务适配开始共同学习”的表现。它与先前1:1、2664步、从P0出发的full同时存在初始化、采样和预算差别；未来若要严格归因初始化路径，应再配相同natural四轮、从P0出发的A对照。

## 启动核验

实际三卡 sampler 每轮长度均为1702，四轮总步数6808已校验。预训练加载日志确认14维→6维适配：`mixtures.action.action_encoder.weight`、`mixtures.action.head.weight/bias` 尺寸不匹配，保留新初始化；proprio投影通过adaptive average pooling适配14→6，其余兼容权重正常加载。这与原始P0的初始化机制相同，并非恢复SO101已训练动作头。
