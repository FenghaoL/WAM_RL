# AHA-WAM HIL 数据采完以后：首轮训练、RECAP 复刻边界与 RL 路线决策

> 更新时间：2026-08-29  
> 当前任务：SO101 从支架取出电源适配器，并插入插线板  
> 本文回答的是“现有 HIL 数据下一步怎样用”，不替代 [`SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md`](./SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md) 中的数据契约与 mask 规范。

## 0. 先给结论

当前最合理的下一步不是直接实现完整的 π*0.6 / RECAP，也不是只把 97 段人工接管数据单独再训几轮，而是：

1. 把当前采集策略固定为 `P0 = step_007984.pt`，先做一次不含新数据的正式自主评测；
2. 如果目标是对标 RECAP 的“固定预训练初始化 + 累计数据重训”语义，第一主实验应从官方 RoboCOIN `AHA-WAM-pretrained.pt` 重新开始，在 `02 + 04 + 本轮 HIL correction` 上训练；
3. 先跑一个自然累计采样 baseline，再跑分源重加权版本；重加权主实验建议每个本地 batch 取 `2 demo + 2 correction`，并补一个 `3 demo + 1 correction` 的比例消融；
4. 原始 demo batch 保留 AHA-WAM 的 video + action 联合损失；correction batch 第一版只给有效 HUMAN action 算 action loss，不让 SYSTEM_HOLD、占位 action 或片段尾部 padding 进入动作监督；
5. 主实验继续使用训练 `P0` 时的 SO101 `dataset_stats.json`，避免重算 stats 成为额外变量；
6. 可另跑一个从 `P0` 接着训练的 `HG-cont`，用于回答“最低计算成本能否快速修复当前错误”，但它是经典 DAgger continuation 对照，不是最接近 RECAP restart 语义的版本；
7. 暂时不要用现有 35 条全成功、均有人救回的 episode 单独训练完整价值函数。先补无接管 autonomous success / failure / timeout，之后再做 RECAP-like；
8. 如果目标是尽快把精密插入成功率推高，HG/IWR 之后更值得优先考虑的是“末端插入 specialist RL + 成功轨迹蒸馏回 AHA-WAM”，而不是直接对 6B 级完整 WAM 做在线 actor–critic。

一句话概括：**你的理解是对的：`base → (02+04)` 得到采集策略以后，下一轮用 `base → (02+04+HIL)` 重新训练，就是与 RECAP 累计数据重训最接近的 AHA-WAM 版本；它目前仍是 cumulative HG-SFT，而不是完整的 value/advantage RECAP。**

---

## 1. 本次审计到的真实数据状态

### 1.1 采集策略

本轮接管数据由下面的 task-SFT checkpoint 产生：

```text
runs/so101_ahawam_plug_all/
  full_zero2_offload_ahawam_base/
  formal-plugall-ahawambase-zero2offload-bs4-acc1-4ep-3gpu-20260827-020601/
  model/checkpoints/weights/step_007984.pt
```

文件 SHA-256 为：

```text
844ab2a4de0cb8d8822e2d3c8b8046a177b5767b9c3a727778f6bd5bfb12867b
```

四批 `episodes_manifest.jsonl` 中的 `policy_checkpoint_sha256` 当前都是 `null`。不要事后改写原始历史记录；建议在实验归档目录增加一个 immutable sidecar，把上述路径、SHA-256、部署配置 hash、训练配置和 git commit 绑定起来。否则论文复现时无法仅根据 HIL 数据证明它究竟是哪一个 policy 采出来的。

### 1.2 原始 task-SFT 数据

`P0` 从官方 `AHA-WAM-pretrained.pt` 初始化，在以下数据上训练 4 epoch：

| 数据源 | episode | frame | 作用 |
| --- | ---: | ---: | --- |
| `02_grasp_move_and_align` | 32 | 9,351 | 抓取、搬运、粗对准 |
| `04_full_remove_and_insert` | 41 | 14,596 | 完整取出与插入 |
| 合计 `D_demo` | **73** | **23,947** | 当前完整任务先验 |

原配置的关键项是：全量微调、`lr=1e-5`、每 GPU batch 4、3 GPU、4 epoch、action horizon 40、history 6，并使用：

```text
runs/so101_ahawam_plug_all/assets/dataset_stats.json
```

新增的 `01` 没有用于产生 `P0`，且你已观察到加入它的另一个 checkpoint 没有明显改善。因此第一组可解释实验应保持 `D_demo = 02 + 04`；`01` 可以作为单独的数据消融，不能悄悄并入主实验。

### 1.3 HIL 数据

`data/ahawam_hil_raw` 当前共有：

| contract | episode | HUMAN frame | correction segment |
| --- | ---: | ---: | ---: |
| rich-v1 | 1 | 96 | 2 |
| rich-v2（两批） | 16 | 4,441 | 44 |
| rich-v3 | 18 | 4,621 | 51 |
| 合计 | **35** | **9,158** | **97** |

全部已保存 episode 的 outcome 都是 `success`。这说明人工接管系统成功把 rollout 救回来了，但也意味着：

- 这些数据有很强的 correction policy 监督价值；
- 它们不能提供可靠的 success-vs-failure value 边界；
- 不能把“最终被人救成功”解释成“接管前 robot action 是好动作”；
- 它们的初始条件元数据都是 `adapter_position=holder`、`power_strip_position=fixed`、`scene_variant=default`，因此即使训练后成功率提高，也先只能说明固定工作站分布内的提升。

按 frame 粗算，直接把 `D_demo` 与 HUMAN frame 合并时：

```text
demo fraction       = 23,947 / (23,947 + 9,158) = 72.3%
correction fraction =  9,158 / (23,947 + 9,158) = 27.7%
```

这只是 frame 比例，不等于真正的 40-step training-anchor 比例。片段长度、尾部 padding、history、video horizon 和每目录采样方式都会改变最终权重，因此不能仅把目录写进一个列表便声称比例是 72:28。

---

## 2. 能不能从官方 AHA base 直接混合重训

可以，而且如果你的目标是对标 RECAP 的下游迭代语义，它就应当是第一主实验，而不只是次要 baseline。

### 2.1 为什么它正是 RECAP-style restart 的直接对应

从：

```text
checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt
```

重新在累计数据 `D_demo ∪ D_human` 上训练，恰好对应下面的轮次关系：

```text
Round 0:
    AHA RoboCOIN base → train on D0=(02+04) → P0

Collect:
    deploy P0 → obtain HIL corrections H1

Round 1 restart:
    AHA RoboCOIN base → train on D1=(02+04+H1) → P1
```

这与 RECAP Algorithm 1 的两个原则一致：数据集累计，但每轮 policy 从固定 pretrained policy checkpoint 重新微调。它还有三个科研优点：

- 所有对照都能从同一个公开初始化开始；
- 不会把 `P0` 已经经历过的 4 epoch 隐式算成额外训练预算；
- 符合 dataset aggregation 的思想：每轮使用到目前为止的累计专家数据，而非只训新 correction。

这里的“从 RoboCOIN 开始”是指加载已经完成 RoboCOIN 预训练的 `AHA-WAM-pretrained.pt`，不是每一轮把 RoboCOIN 原始数据也重新加入下游训练。RECAP 下游同样是从 `π_pre` checkpoint 初始化，再在任务累计数据 `D_l` 上微调；它并不要求每轮重新执行通用预训练阶段。

由于当前 AHA base 不是 advantage-conditioned `π_pre`，而且本轮只使用了 correction、还没有 value relabel 的 autonomous reward experience，所以它应命名为：

```text
cumulative HG-SFT from RoboCOIN AHA initialization
```

不要命名为 `π*0.6`、`RECAP` 或 `offline RL`，因为这里没有 value、advantage label、positive/negative conditioning，也没有 autonomous reward data。

### 2.2 从 P0 继续仍然有什么价值

本轮 correction 是 `P0` 在真实部署分布中犯错以后，人从其访问到的状态进行纠正得到的。因此以 `P0` 继续训练仍是合理的经典 DAgger 工程分支：

- 与经典 DAgger/HG-DAgger 的迭代更新最一致；
- 不需要重新学习抓取、搬运和寻找插座；
- 计算量更低，能更快检验这批 correction 是否真的修复了目标 failure mode；
- correction 的状态分布与初始化 policy 精确匹配。

它的风险是连续微调可能产生 drift 或遗忘，所以必须 replay `D_demo`，并以更低学习率、较短 step schedule、频繁真机评测来约束。

但它不应再被表述为“比 restart 更符合 RECAP”。它的意义是省算力和快速诊断；restart 的意义才是固定初始化、累计数据和控制 drift。

### 2.3 两条线应怎样定义

| 实验 | 初始化 | 数据 | 研究问题 |
| --- | --- | --- | --- |
| `HG-restart`（RECAP-style 主线） | 官方 AHA RoboCOIN base | `D0 + H1` 累计数据 | 固定预训练初始化时，新一轮累计数据能否补充原模型 |
| `HG-cont`（工程对照） | `P0=step_007984.pt` | 同一 `D0 + H1` | 从上一轮继续是否能以更低计算成本快速修复错误 |

如果你的当前目标就是走 π*0.6/RECAP 风格并形成论文主线，算力只能先跑一个时，应先跑 `HG-restart`。如果当前只想用最小成本快速验证 correction 是否有效，则可先跑 `HG-cont`。理想情况下两者都保留，否则无法区分 restart 抑制 drift 的收益与 continuation 的优化便利。

---

## 3. 为什么不能“把目录堆起来就训练”

### 3.1 三种“直接混合”其实不同

“把数据堆在一起”至少可能表示：

1. 每个 episode 等概率；
2. 每个 frame / anchor 等概率；
3. 每个数据目录先等概率，再在目录内等概率。

在当前数据中，原始 demo 是 73 条完整/分阶段轨迹，HIL 是 97 个较短、集中在瓶颈附近的 correction segment。这三种实现会产生明显不同的有效分布。训练报告必须写清楚 sampling unit。

### 3.2 随机采样不会“浪费”没有抽到的数据

带放回随机采样不是删除数据。它在固定优化步数内控制各数据源的期望权重，并允许小数据源被重复学习。真正的问题是“一个所谓 epoch 到底覆盖了多少不同样本”。

如果一个池有 `N` 个样本，独立抽 `kN` 次后，某个样本仍从未被抽到的概率近似为：

```text
exp(-k)
```

约为：

| 抽样量 | 单个样本未出现概率 |
| --- | ---: |
| `1N` | 36.8% |
| `3N` | 5.0% |
| `5N` | 0.7% |

对小数据更稳的实现是 **source-stratified shuffled cycles**：每个 source 各自 shuffle，按 batch 配额循环取；一个池耗尽后再 reshuffle。这样每轮能保证覆盖，又能通过循环次数实现 oversampling，不需要复制文件，也不丢数据。

### 3.3 当前建议的比例

当前配置是每 GPU batch 4、3 GPU。可以在每张卡本地构成：

- 主实验：`2 demo + 2 correction`，即 50:50；
- 比例消融：`3 demo + 1 correction`，即 75:25；
- naive baseline：按有效 anchor 自然合并，不做 source balance。

50:50 并不是宇宙通用常数。它来自 IWR 对 bottleneck intervention 的重加权思想，也适合当前 correction 高度集中在抓偏、孔口对准和插入恢复的特点。75:25 更保守，通常更不容易损伤抓取/搬运。论文应报告两者，而不是事后只挑表现最好的一条不说明搜索过程。

### 3.4 epoch 在重加权训练中会变得含糊

一旦两个池以不同频率循环，“4 epoch”不再代表每条数据恰好看 4 次。建议改为 step-based 记录，并额外报告：

- optimizer steps；
- demo anchors consumed；
- correction anchors consumed；
- 两类有效 action tokens；
- 每个 source 的期望/实际重复次数；
- video loss 和 action loss 各自收到多少有效样本。

这样不同 sampler 的训练预算才可比较。

---

## 4. 当前数据究竟怎样进入 AHA-WAM loss

### 4.1 strict HG 的动作监督

对任意 40-step target，专家动作有效位应为：

```python
expert_valid = ~action_is_pad
expert_valid &= action_supervision_mask
expert_valid &= intervention
expert_valid &= human_action_valid
```

rich-v1/v2 的 autonomous 时间轴有缺口，不能把完整 raw episode 直接当作连续 video/action episode，也不能只补一列 mask 就允许 future query 跨过 HUMAN 边界。它们的连续 HUMAN 部分本身有效：既可以导出为 correction-only episode，也可以像当前实现一样用 zero-copy logical view 把每个连续 HUMAN 段视为独立 episode，并在段尾 clamp + pad。两者监督语义等价。

rich-v3 可以使用 raw，但必须遵循 [`SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md`](./SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md) 的规则。`SYSTEM_HOLD` 的 observation/state 是真实物理过程，action 是上一实际 target，但 `action_supervision_mask=false`，所以它不能进入直接 action loss。

### 4.2 当前源码已经实现什么

[`ahawam.py`](../../src/ahawam/models/wan22/ahawam.py) 的 `_compute_weighted_action_loss()` 生成 `[B,T]` 的逐 token MSE，并与 `action_is_pad` 相交。本轮 logical loader 在 HUMAN 边界处主动生成 pad mask，所以不必再让模型理解 collector 的 `control_source` 枚举；模型收到的所有非 pad correction action token 已经必然属于同一 HUMAN 段。video 侧另新增逐样本 `video_supervision_mask`，用于关闭短 correction 的直接 video loss。

本轮已经实现第二条等价路线：

- loader 只把 HUMAN anchor 暴露为训练 index；
- 每个连续 HUMAN 段成为 logical episode；
- future action/state/image 在 HUMAN→非 HUMAN 边界 clamp，并使用原有 pad mask；
- 因此 40-step action target 不可能跨进 POLICY/SYSTEM_HOLD；
- 不需要重新编码视频，效果等价于导出 correction-only 标准 episode。

不要在读取 raw parquet 时只看 anchor 当前行的 `intervention=true`，然后把未来 40 个 action 全部当专家。future chunk 可能跨过 HUMAN→PAUSE/POLICY 的边界。

### 4.3 video loss 的第一轮处理

本轮按实际数据长度采用双流训练：

```text
D_demo batch:
    video loss = on
    action loss = on

D_human correction, segment >= 49 frames:
    video loss = on
    action loss = HUMAN-valid only

D_human correction, segment < 49 frames:
    video loss = off
    action loss = HUMAN-valid only
```

原因不是 correction 视频没有价值，而是：

- 15 个 correction 段短于 49 帧的完整 future-video 覆盖需求；
- v1/v2 raw 的 autonomous gap 不可恢复；
- correction 从任务中途开始，缺少完整历史；
- 当前最先要检验的是动作纠正是否能解决抓偏与孔口误差。

其余 82 个长度达到 49 帧的 correction 段参与 video+action。长段尾部不足完整 horizon 的 anchor 不会被整条删除，而是沿用仓库的 latent-group padding mask；详见 [`AHAWAM_HIL_HUMAN_ONLY_TRAINING_IMPLEMENTATION_ZH.md`](./AHAWAM_HIL_HUMAN_ONLY_TRAINING_IMPLEMENTATION_ZH.md)。

### 4.4 normalization

`P0` 的动作输出语义绑定于：

```text
runs/so101_ahawam_plug_all/assets/dataset_stats.json
```

因此 continuation 与主消融统一使用这一份 stats。若继续训练时改成只根据 correction 重算均值方差，相同网络输出会被解释为不同关节目标，造成无意义的分布跳变。

训练前应额外检查 correction 中每个关节的 min/max/z-score 是否显著超出原始 demo。若只是覆盖了原数据中较少见的精密姿态，应保留旧 stats；若单位、校准或动作定义不一致，则应先停止训练排查，而不是靠重算 stats 掩盖问题。

---

## 5. π*0.6 / RECAP 原文到底怎样训练

### 5.1 它有两个不同的“预训练模型”

RECAP 的核心不是一个普通 VLA checkpoint，而是：

```text
V_pre   : 预训练的多任务 distributional value function
π_pre   : 已经学会接受 advantage positive/negative 条件的 VLA policy
```

论文的预训练数据是来自许多任务、许多机器人的 **数万小时 demonstration**。π0.6 的训练混合大体继承 π0.5，包括内部和外部 cross-embodiment robot data、家庭环境中的 mobile/non-mobile 数据、高层 subtask prediction 与 multimodal web data。论文没有公开足以复刻的完整任务清单、每项比例、全部原始数据或训练算力明细。

`V_pre` 在同一大规模机器人数据上学习 remaining-time return；value 模型使用 670M VLM backbone、201 个离散 value bins，并混入少量 multimodal web data 来缓解过拟合。然后 value 给 action 计算 advantage，策略把它二值化成文本条件：

```text
Advantage: positive
Advantage: negative
```

训练时 30% 概率丢弃这个条件，使模型同时学 conditional 与 unconditional action distribution。

### 5.2 下游第 0 轮

对任务 `l`：

1. 用该任务 demonstration 初始化累计数据集 `D_l`；
2. 从 `V_pre` 微调任务 value `V_l^0`；
3. 从 `π_pre` 微调任务 policy `π_l^0`；
4. demonstration 阶段把 indicator 固定为 positive，因此这一步在形式上就是 task SFT。

### 5.3 后续每一轮

第 `k` 轮：

1. 用上一轮 `π_l^(k-1)` 收 autonomous rollout，部分 rollout 允许人工 intervention；
2. 给每条 episode 标 success/failure，并保留时间长度；
3. 把新数据追加到累计 `D_l`，不丢旧轮数据；
4. **从 `V_pre` 重新微调** `V_l^k`，不是从 `V_l^(k-1)` 接着训；
5. **从 `π_pre` 重新微调** `π_l^k`，不是从 `π_l^(k-1)` 接着训。

论文明确说这种“每轮从 pretrained checkpoint 再微调、但数据累计”的方式用于减少多轮 drift；同时也承认持续从上一轮继续训练可能获得好结果，只是没有采用为主设置。

### 5.4 correction 与 autonomous data 怎样用

RECAP 并不是只训练 correction：

- correction action 强制设为 positive advantage；
- 其余动作由 value 的 n-step advantage 决定 positive/negative；
- autonomous、旧 policy、新 policy、demonstration 都可进入累计数据；
- 论文指出 intervention 本身有扰动，操作者也不能保证纠正质量或改善细微速度，所以 correction 主要修大错和探索障碍，速度与细节仍依赖 autonomous reward experience。

下游用 50-step lookahead 估 advantage。通常选阈值使约 40% rollout 数据为 positive；对于已经成功率高但速度慢的 laundry 设置，positive 比例收紧到约 10%。它不是简单地把“人类=正、机器人=负”。

### 5.5 原文规模与你当前规模的差异

公开的下游数据规模包括：

| 任务 | 迭代 | 每轮数据 |
| --- | ---: | --- |
| T-shirt/shorts laundry | 2 | 300 条纯 autonomous trajectory |
| box assembly | 2 | 600 条 autonomous + 360 条 intervention rollout |
| cafe | 1 | 414 条 autonomous + 429 条 correction episode |
| diverse laundry（附录汇总） | — | 450 条 evaluation + 287 条 correction |

这些不是“二三十条 demo 就完成 RECAP”。二三十条初始 demo 是 HIL-SERL 等 specialist RL 论文里常见的初始化规模；π*0.6 的关键底座则是数万小时 `V_pre/π_pre`，下游每轮也有数百条经验。

---

## 6. 官方 AHA RoboCOIN checkpoint 与 π_pre 是否类似

只能说“功能位置有一点像”，不能说“训练语义相同”。

### 6.1 相似点

- 都是下游真机任务微调前的通用 robot-data 初始化；
- 都希望通过跨任务/跨机器人数据降低新任务数据需求；
- 都保留了生成连续动作的预训练先验。

### 6.2 根本差异

| 项目 | AHA RoboCOIN base | RECAP `π_pre` |
| --- | --- | --- |
| 公开说明的规模 | RoboCOIN 选定子集上 45k steps | 数万小时、多任务、多机器人 |
| 主训练性质 | supervised world/action modeling | value relabel 后的 advantage-conditioned policy training |
| 独立 `V_pre` | 无 | 有，670M distributional value model |
| positive/negative 条件 | 无 | 有 |
| reward/outcome 语义 | 无 | success/failure + 每步时间代价 |
| web/value regularization | 没有对应 value 训练 | value 另有 multimodal web co-training |

RoboCOIN 完整数据集公开报告超过 18 万条 demonstration、15 种平台和 421 个任务，但 AHA README 只说明 checkpoint 用了“selected subset”训练 45k steps，没有公开该子集的精确 episode 数和 mixture。因此不能把完整 RoboCOIN 规模直接算成这个 checkpoint 实际看过的数据。

更准确的类比是：

```text
AHA-WAM-pretrained.pt  ≈ 普通 π0.6 的机器人监督预训练基底
而不是                ≈ RECAP 完成 offline-RL pretraining 后的 π_pre
```

---

## 7. 当前几十条数据能否训练 value function

### 7.1 “能拟合一个网络”不等于“能得到可用 critic”

RECAP 的稀疏 reward 是：

```text
terminal success : 0
terminal failure : -C_fail
other steps      : -1
```

所以成功轨迹也不是全都同一个标签：越接近完成，remaining-time return 越高。理论上，即使只有成功轨迹，也能训练一个“离成功还有多远”的 progress estimator。

但当前数据的问题是：

- 只有 35 条 episode，远小于一个从零训练视觉 value backbone 所需的覆盖；
- 全部最终成功，没有 terminal failure/timeout 对照；
- 每条都有人介入，接管前的坏状态最终也被救成 success；
- 初始条件几乎固定；
- 没有现成 `V_pre`；
- 同一失败外观可能由不同几何/接触原因造成，仅靠双 RGB 图像很难用几十条可靠区分。

因此当前数据可以训练一个 exploratory progress/takeover predictor，但不应直接作为 RECAP critic 给所有 robot action 打 advantage 标签。

### 7.2 为什么不能把接管前一段都粗暴标 negative

接管发生意味着操作者判断“继续这样可能出问题”，它是风险信号，不等于每个接管前 action 都有负 advantage：

- 人可能接管过早；
- policy 可能已经发出了仍在执行的 action chunk；
- 坏结果可能由更早的抓取姿态造成；
- 有些接管只是效率或风格判断；
- 接管前最后一帧的 robot action 与人类 correction 并不一定是同一 state 下可比较的反事实动作。

可以单独训练 intervention-risk model，或把 `time-to-intervention` 作为辅助标签；不要把它伪装成环境 return。

### 7.3 如果现在就想做小 value pilot

最保守的 pilot 是：

1. 冻结视觉 backbone，只训练小型 temporal value/progress head；
2. target 先用 success trajectory 的 normalized remaining time；
3. 按整条 episode 和初始条件划分 train/val，不能随机拆相邻 frame；
4. 独立报告它对 held-out trajectory 的 progress ranking、failure OOD 检测和 calibration；
5. 不让它在主策略训练中拥有决定性权重。

要进入真正 RECAP-like 主实验，建议再采：

- 无人接管 autonomous success；
- 无人接管 autonomous failure；
- timeout/stuck；
- 分层初始化下的 outcomes；
- 每条 episode 的成功、耗时、失败阶段与是否接管。

数量没有绝对门槛，但一个合理的起点是先获得至少几十条 success 和几十条 failure，而不是 35 条全成功；然后用 bootstrap 置信区间和按 initial-condition held-out 的泛化来决定 value 是否可信。

---

## 8. 立即可执行的训练矩阵

### 8.1 训练前先冻结评测

在新训练开始前，用 `P0` 做一次 autonomous-only 评测并保存视频。至少按以下 failure stage 标注：

```text
grasp miss
bad grasp / plug pose
approach error
coarse alignment error
fine alignment / hole search
contact but cannot insert
insert success
timeout / unsafe stop
```

同时记录成功率、插入成功率、总耗时、接管次数（评测时应为 0）以及插线板/插头初始化 bin。之后所有 checkpoint 使用同一评测协议。

### 8.2 最小主实验

| ID | 初始化 | sampler | loss | 推荐用途 |
| --- | --- | --- | --- | --- |
| `B0` | `P0` | 无新训练 | — | 采集 policy 基线 |
| `R1-natural` | 官方 AHA base | `D0+H1` 有效 anchor 自然累计 | demo full + correction action | 最直接的“在原始数据上加入 DAgger 数据重新训练” |
| `R1-25` | 官方 AHA base | 75% demo / 25% correction | demo video+action；correction HUMAN action | restart 的保守重加权版本 |
| `R1-50` | 官方 AHA base | 50% demo / 50% correction | 同上 | RECAP-style restart 主模型 |
| `C1-50` | `P0` | 与 `R1-50` 相同 | 同上 | continuation 工程对照 |
| `I1` | `P0` | IWR + demo replay | human/policy 分源 | 检验保留 robot behavior 是否减少漂移 |

### 8.3 第一轮超参数原则

对 `R1-natural/R1-25/R1-50`：

- 从官方 AHA RoboCOIN base 重新开始；
- 使用原 stats；
- `R1-natural` 第一版可以沿用原 `1e-5 / 4 epoch`，因为它就是最小改动的累计重训 baseline；
- 对 source-balanced 版本同时记录 optimizer steps、demo/correction consumption 和有效 token，避免“epoch”掩盖 oversampling；
- 至少按每 0.5 个 cumulative-data pass 保存 checkpoint；
- 用真机成功率选模型，不只看训练 action loss。

`R1-natural` 是你所说的“原始 checkpoint + 原始 02/04 数据 + DAgger 数据，从头再训一次”的最干净实现。`R1-25/R1-50` 则回答 correction 是否需要人为提高权重；不要一开始只有重加权版本而缺少自然累计 baseline。

对 `C1-50`：

- 从 `P0` 继续；
- 初始尝试 `2e-6 ~ 5e-6`，低于原始 `1e-5`；
- 采用较短 step schedule 并频繁保存；
- sampler、mask、stats 和 eval 与 `R1-50` 保持一致。

这里的 continuation 学习率只是针对当前全量微调风险给出的工程起点，不是 π*0.6 或 HG-DAgger 原论文规定值。

### 8.4 IWR 怎样加入当前数据

原始 IWR 把 intervention 与 non-intervention 分池，并按 50:50 抽样。当前若实现一个更适合 AHA 的 `IWR + demo replay`，可采用三类 loss：

```text
D_demo   : 专家全任务 replay，video + action
D_human  : HUMAN correction，action
D_policy : policy executed action，action regularization
```

第一版配额可设：

```text
25% D_demo + 50% D_human + 25% D_policy
```

另做论文更接近 IWR 的两池版本：`50% HUMAN / 50% POLICY`，再说明初始 demo 是否以及如何单独 replay。

重要限制：rich-v1/v2 缺少 autonomous 等待帧，不能直接构造可靠的连续 `D_policy` future window；因此：

- 所有版本都可贡献 correction-only `D_human`；
- 完整 IWR 的连续 policy window 第一版只使用 rich-v3；
- `SYSTEM_HOLD` 不属于 `D_policy` action target；
- 跨 deliberate pause、tick gap 或 dropped tick 的 window 必须拒绝。

---

## 9. 是否更推荐 π*0.6 之外的 RL

### 9.1 当前阶段的优先级

| 路线 | 当前适合度 | 判断 |
| --- | --- | --- |
| strict HG / IWR | **最高** | 已有数据直接可用，不需要 reward/value，先验证 correction 的价值 |
| task-specific insertion RL → distill | **高，但作为第二阶段** | 与精密插入高度匹配，计算和探索集中在末端 |
| RECAP-like AHA | 中期 | 需要 autonomous outcomes、value、advantage condition 和累计轮次训练 |
| WAM-RL joint video/action | 研究性强、近期风险高 | 架构最接近，但公开证据目前主要是仿真且训练很重 |
| 直接 full-WAM PPO/actor–critic | 低 | 真机样本效率、稳定性、算力和安全风险都不合适 |

### 9.2 HIL-SERL / RLDG 式末端 specialist

HIL-SERL 在 RAM、SSD、USB 等精密插入/装配任务中通常只用 20 条初始 demo，但随后还进行了约 18k–108k RL transitions，并使用小型视觉 actor–critic、10 Hz 控制、二值 reward classifier；USB 任务还使用 6D twist、gripper control 和力/力矩等 proprioception。它不是“20 条 demo 直接训出高成功率”。

RLDG 的思路对本任务尤其有价值：

```text
AHA-WAM 负责抓取、搬运、到孔口附近
        ↓
末端 specialist RL 负责最后精密对准/插入
        ↓
保留 specialist 的高质量成功 trajectory
        ↓
再蒸馏/微调回 AHA-WAM
```

RLDG 的实验也包含“human demo 负责 grasp/transport，RL data 负责 precision insertion”的多阶段任务。这比在 6B 级 WAM 上直接做在线 RL 更符合当前 SO101 的算力与安全条件。

但迁移不是零成本：SO101 当前是绝对关节目标、低成本舵机、无明确 F/T 反馈，而 HIL-SERL 使用更强的控制与传感配置。需要先加动作限幅、接触安全、自动 reset 或人工 reset 协议、可靠 success detector，并决定 specialist 的 action space 是 joint residual 还是 task-space delta。

### 9.3 WAM-RL 值得参考，但不应作为眼前主线

2026 年的 WAM-RL 与 AHA-WAM 的结构问题很接近：

- 用 successful online trajectory 做 video SFT；
- 用 KL 约束 world latent 不要偏离 frozen reference；
- 用 imagined future 与真实执行结果的一致性作为 actor reconstruction reward；
- 报告 actor-only 对短任务有效，长任务需 joint world/action update。

它给你的直接启发是：未来可以把 rich-v3 成功 HIL raw 用于受约束的 video adaptation，而不是永远冻结 world branch。

但其公开实验只在 LIBERO/RLBench 仿真，基于 Genie Envisioner-ACT，用 8 张 A800 训练约 8 小时。更关键的是，reconstruction reward 只奖励“执行与 world imagination 一致”；如果 world model 对孔位或接触结果预测错，actor 更忠实地实现错误 imagination 并不等于任务成功。因此它适合作为后续研究消融，不应替代当前的真实 success/intervention supervision。

---

## 10. 建议的论文叙事

当前数据足以形成一条清楚且不夸大的故事：

```text
RoboCOIN-pretrained AHA-WAM
    → 73-demo task SFT
    → 部署暴露 learner-induced grasp/alignment/insertion failures
    → 35 HIL rollouts / 97 correction segments
    → strict HG 与 source-balanced IWR
    → autonomous outcomes
    → value-guided 或 specialist-RL improvement
    → 可选 video/action co-adaptation
```

核心消融应回答：

1. 同样的新人工动作量，full offline demo 与 on-policy correction 哪个更有效；
2. correction-only 是否造成抓取/搬运遗忘，demo replay 是否修复；
3. natural concat、75:25、50:50 的差异；
4. 从 `P0` continuation 与从公开 AHA base cumulative restart 的差异；
5. action-only correction 与加入 rich-v3 video SFT 的差异；
6. 后续 autonomous reward/value 是否超越 HG/IWR，而不是只把同一人类数据换一个名字；
7. 提升发生在最终插入、抓取鲁棒性、恢复能力还是速度。

不要把所有 HIL episode 的 `outcome=success` 当成自主成功率。正式指标必须在无接管评测中计算，并把 human-assisted completion、intervention-free success 和完全失败分开。

---

## 11. 训练前阻塞项清单

- [ ] 给四批 HIL 数据增加只读 sidecar，绑定 `P0` SHA-256；
- [ ] 保存 `P0` 的固定 autonomous-only baseline；
- [ ] 确认 correction-only 导出已完成，或实现 future-40 source masks；
- [ ] 单元测试 HUMAN→非 HUMAN 边界不会污染 40-step target；
- [ ] 排除 SYSTEM_HOLD、无效 placeholder、padding 和跨 gap window；
- [ ] 为 demo/correction 分别建立 sampler 计数；
- [ ] continuation 固定使用原 SO101 stats；
- [ ] correction batch 能单独关闭 video loss，或有等价的 per-sample video mask；
- [ ] train/val/eval 按 episode 和初始化条件隔离，不能随机拆相邻 frame；
- [ ] 报告每种 source 的实际有效 action token 数；
- [ ] 真机每个 checkpoint 先做低速、限幅和安全检查。

---

## 12. 最终决策树

```text
现在是否已有可用 correction？
    是
    ├─ 要按 RECAP 的固定预训练初始化 + 累计数据语义推进
    │    ├─ AHA base + (02+04+HIL)，先跑 R1-natural
    │    └─ 再跑 R1-25 / R1-50 检验 correction 重加权
    │
    ├─ 只想最低成本快速验证 correction
    │    └─ P0 + 相同累计数据，跑 C1-50
    │
    ├─ strict HG 后抓取/搬运变差
    │    └─ 降低 correction 比例；再实现 IWR + demo replay
    │
    ├─ strict HG 后插入仍低成功
    │    ├─ 先采 autonomous success/failure/timeout
    │    └─ 末端 specialist RL + distillation 优先于 full-WAM online RL
    │
    └─ 想研究 world/video 是否也应适配
         └─ rich-v3 successful raw video SFT + frozen-reference KL，单独做消融
```

---

## 13. 主要证据来源

1. [π*0.6 / RECAP 论文](https://www.pi.website/download/pistar06.pdf)：Algorithm 1、value/reward、每轮从 `V_pre/π_pre` 重启、累计数据、advantage conditioning、数据规模与迭代数量。
2. [π0.6 model card](https://website.pi-asset.com/pi06star/PI06_model_card.pdf)：5B VLA、860M action expert、π0.5-like robot/web data mixture 与 Knowledge Insulation。
3. [Physical Intelligence π*0.6 项目页](https://www.pi.website/blog/pistar06)：从 demonstration、correction 到 autonomous experience 的官方流程概述。
4. [AHA-WAM 官方仓库](https://github.com/serene-sivy/AHA-WAM)：RoboCOIN selected subset、45k-step pretrained checkpoint 与下游初始化用法。
5. [RoboCOIN 论文](https://arxiv.org/abs/2511.17441)：完整数据集的 180k+ demonstrations、15 platforms、421 tasks；注意这不等于 AHA checkpoint 实际使用的精确子集。
6. [IWR 项目页](https://sites.google.com/stanford.edu/iwr)：intervention/non-intervention 分池、50:50 sampling 与迭代训练。
7. [HIL-SERL 论文](https://arxiv.org/abs/2410.21845)：20-demo 初始化、精密插入任务、真实 RL transitions、reward 与动作/传感配置。
8. [RLDG 论文](https://arxiv.org/abs/2412.09858)：用 task-specific RL 生成精密插入数据，再蒸馏给 generalist policy。
9. [WAM-RL 论文](https://arxiv.org/abs/2606.17906)：successful video SFT、latent KL、reconstruction reward 以及 actor-only/joint update 消融。

## 14. 与其他项目文档的关系

- 数据字段、SYSTEM_HOLD 与 future mask：[`SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md`](./SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md)
- 接管系统的实现背景：[`SO101_AHAWAM_HIL_DATA_COLLECTION_IMPLEMENTATION_HANDOFF_ZH.md`](./SO101_AHAWAM_HIL_DATA_COLLECTION_IMPLEMENTATION_HANDOFF_ZH.md)
- π*0.6、HG、IWR 的前期实验规划：[`PI06_RECAP_TO_SO101_PLUG_EXPERIMENT_PLAN_ZH.md`](../research/PI06_RECAP_TO_SO101_PLUG_EXPERIMENT_PLAN_ZH.md)
- AHA-WAM 数学与源码：[`AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md`](../architecture/AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md)
- 更广的 RL 与 video/action 联合优化：[`AHAWAM_RL_SCOPE_AND_VIDEO_ACTION_COTRAINING_REVIEW_ZH.md`](../research/AHAWAM_RL_SCOPE_AND_VIDEO_ACTION_COTRAINING_REVIEW_ZH.md)
