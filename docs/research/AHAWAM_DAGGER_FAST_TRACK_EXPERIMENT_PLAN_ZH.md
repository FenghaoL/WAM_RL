# AHA-WAM DAgger 快速实验清单

> 这是 [`AHAWAM_DAGGER_METHODS_AND_WAM_SPECIFIC_DESIGN_ZH.md`](./AHAWAM_DAGGER_METHODS_AND_WAM_SPECIFIC_DESIGN_ZH.md) 的执行摘要。完整文献证据、公式、评分和风险以主文档为准。

## 1. 当前判断

- correction-only 全量 continuation 失败不能证明 DAgger 无效，只能证明窄分布 HUMAN 段会让 6B 联合模型快速漂移。
- HALO-WA 的 HG-DAgger 是 human correction supervised imitation 对照，但论文未公开足够细的 sampler、更新对象和 loss 配方，不能当作唯一复现标准。
- 当前 AHA 普通 action 部署不显式生成未来视频；主贡献不应写成“未来 latent + 当前 latent 的 RL state”。
- 最值得做的是：结构化干预标签、frozen-base 小 adapter、flow uncertainty gate、OVCR-specific correction。

## 2. 数据标签

```text
demo / successful safe policy
    -> action anchor, gate=0

valid policy steps immediately before takeover
    -> risk/gate positive
    -> NEVER positive action BC

continuous HUMAN correction
    -> expert action flow target
    -> gate positive

release grace period / SYSTEM_HOLD / invalid reply
    -> mask
```

接管前窗口先比较 8、16、24 帧，不机械照抄论文；用录像估计操作者反应时间。

## 3. 最小主表

| ID | 训练 |
|---|---|
| B0 | task-SFT P0 |
| B1 | 已完成的 correction-only continuation |
| B2 | IWR 50:50 demo/correction，Video DiT frozen |
| B3 | SIRIUS weighted：demo/HUMAN/normal/pre-intv 四类 |
| B4 | frozen base + Action DiT LoRA 或 output residual + anchors |
| Ours | frozen base + intervention-gated OVCR adapter |

所有方法使用相同原始 demo、相同 HIL rollout、相同评测初始化和相近 trainable parameter count。

## 4. 实现顺序

### Step A：先验证数据方法

1. dataset 输出 `demo`、`human`、`safe_policy`、`pre_intv`、`masked`；
2. 统计每类 row/window 数和跨边界情况；
3. 冻结 Video DiT；
4. 跑 IWR 50:50；
5. 跑 SIRIUS：`pre_intv` 不进 action loss，只进 gate/risk loss。

### Step B：离线验证自动 gate

1. 固定 P0 和 video cache；
2. 在 action chunk 开头，对 P0 生成 action 重加噪；
3. 用训练时相同 scheduler/target 算 `U_flow`；
4. 绘制 safe、takeover 前、failure 三类分布；
5. 报 ROC-AUC、PR-AUC，以及在 95% safe quantile 下的 recall；
6. 只有分布能分开，再接真机自动提示/接管。

### Step C：小 adapter

先做 Action LoRA sanity baseline：

- frozen Video DiT；
- frozen base Action DiT；
- 只训少量 LoRA/gate；
- correction action loss + P0 distillation anchor；
- gate=0 时输出应数值接近 P0。

再做 OVCR adapter：

- 在 observation query 或 per-layer K/V delta 前加低秩 correction；
- 同样参数量、同样数据和学习率预算；
- Action DiT 保持冻结，让增益能归因于 WAM routing。

## 5. 下一轮采集必须补的字段

- HUMAN 期间继续生成并保存 `policy_shadow_action_chunk`；
- `policy_shadow_valid`、checkpoint hash；
- flow noise seed；
- planner cache generation/id/age；
- chunk request id 和 action index；
- `u_flow`、`u_route`、gate score；
- takeover reason 和 task phase；
- success/contact/insert 事件。

没有 shadow policy action 时，现有数据可训 expert adapter，但不能声称是严格 paired residual FlowCorrect。

## 6. Go / No-Go 标准

| 检查 | Go | No-Go 后动作 |
|---|---|---|
| IWR 是否保持抓取/搬运 | 不低于 P0，插入改善 | 降 correction 比例、加强 distillation |
| `U_flow` 是否预示 takeover | PR-AUC 明显高于类比例 | 不做自动 gate，改 learned intervention head |
| Action LoRA 是否优于 mixed SFT | 相同数据下更稳 | 检查 paired target、adapter 容量和 normalization |
| OVCR adapter 是否优于 Action LoRA | 多任务/多 seed 稳定提高 | 不强行写结构贡献；保留为负结果/消融 |
| 冻结 latent 是否缺毫米细节 | probe 不可分而 RGB 可分 | 才进入 selective video LoRA |

## 7. 评测记录

- 完整成功率和阶段成功率；
- 每方法每任务至少 30 次主评测；
- 人类控制秒数、接管次数、恢复耗时；
- gate precision/recall/false alarm；
- easy/旧初始化回归；
- jerk、延迟、参数量、显存；
- 5/10/20/全部 correction 的数据效率。

## 8. 当前一句话方案

**本周先跑 Video-frozen IWR/SIRIUS，离线验证 flow-loss gate；其后用 frozen-base Action LoRA 建立稳基线，再将同等容量 adapter 移到 OVCR，形成区别于 HALO-WA 外挂 actor–critic 的 WAM-specific intervention correction。**
