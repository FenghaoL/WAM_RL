# SO101 + AHA-WAM：截稿期最小采集方案与 rich-v4 完整路线图

> 更新日期：2026-09-02  
> 使用对象：运行 SO-101 follower、SO-101 leader、双相机和 AHA-WAM TCP client 的部署电脑上的 Codex/维护者。  
> 当前目标：不改 AHA 服务端和执行调度，继续使用已经验证的 rich-v3 采集主干，只补齐 IG-OVCR/HG/IWR 所需的最低字段与导出器；本文后半保留完整 rich-v4、paired residual 和实时执行路线，供截稿后使用。  
> 前置文档：[`SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md`](./SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md)、[`SO101_AHAWAM_HIL_DATA_COLLECTION_IMPLEMENTATION_HANDOFF_ZH.md`](./SO101_AHAWAM_HIL_DATA_COLLECTION_IMPLEMENTATION_HANDOFF_ZH.md)、[`AHAWAM_DAGGER_METHODS_AND_WAM_SPECIFIC_DESIGN_ZH.md`](../research/AHAWAM_DAGGER_METHODS_AND_WAM_SPECIFIC_DESIGN_ZH.md)。

---

## 0. 截稿期覆盖指令：部署端 Codex 先只执行本节

> **本节覆盖本文后面所有 rich-v4/prefetch/RTC 的工程建议。当前不要把第 3–11 节当作待办。**

截止目前，`20260829_155403` rich-v3 已经足以训练：

- 原始 HG-DAgger：初始 demos + HUMAN recovery；
- IWR-style：demos/HUMAN 两池或 robot/HUMAN 两池；
- frozen-base Action/OVCR adapter；
- 使用接管边界的 intervention gate。

因此当前选择是：**不实现完整版 rich-v4，不改 action FIFO，不做 prefetch，不改 AHA TCP server，不实现 online/offline shadow，不保存 flow seed/noise，不实现 RTC。** 保持 `executor_mode=SYNC_HOLD` 和 v3 的真实 `SYSTEM_HOLD` 时间轴。

不要另写一套 collector。只在现有 rich-v3 上做下面的 deadline-minimal patch；若需要新版本名，使用 `so101-ahawam-hil-rich-v3.1`，不要把缺少 full chunk ledger/shadow provenance 的数据误称为完整 rich-v4。

### 0.1 必须保留，当前已经基本具备

1. raw 始终保存整个逻辑 rollout：`POLICY + SYSTEM_HOLD + HUMAN`。训练时可以不用自主段，但采集时不要丢弃；
2. HUMAN 内部连续保存双相机、follower state、raw human proposal、安全过滤后的 actual executed action；
3. `action` 必须等于实际下发 target，`human_action` 保留为 clamp 前 proposal；
4. 保留 `control_source/phase/intervention/action_supervision_mask/takeover_segment_id`；
5. 保留 `pause_requested`、`human_control_started/stopped`、`policy_resume_requested` 的单调时钟；
6. FIFO 为空时继续记录真实 `SYSTEM_HOLD`，绝不能采用 rich-v2 的删时间做法；
7. takeover 时继续清 FIFO、递增 generation、废弃在途 reply；resume 时 hard reset；
8. 保留 reason、逻辑 episode、initial condition、success/failure/abort 和干预次数。

### 0.2 只补四个低成本缺口

1. 当前 manifest 的 `policy_checkpoint_sha256=null` 必须修复为真实 checkpoint SHA-256，并保存 checkpoint 路径/标识；
2. 把当前 `pause_requested` 明确作为 `takeover_requested` 语义，并附上 `trigger_frame_index` 与 `trigger_observation_monotonic_ns`，指向接管请求前最后一条已存 POLICY/SYSTEM_HOLD frame；
3. episode outcome 显式区分 `AUTONOMOUS_SUCCESS`、`SUCCESS_WITH_INTERVENTION`、`FAILURE`、`ABORT`；
4. exporter/validator 生成并检查下面四个视图，不改变 raw：

```text
D_demo       = 原始 02/04 demos
D_intv       = HUMAN 连续段；target=action/executed_action_arm
D_gate_pos   = takeover_requested 前 L 个有效 POLICY 帧；只训风险，不模仿其动作
D_gate_neg   = demos + 距离 takeover 足够远的安全 POLICY 帧
```

如果本地实现 `trigger_frame_index` 要改动较大，可以先由 exporter 用 monotonic timestamp 离线寻找最近前驱帧；不能因此阻塞采集。

### 0.3 截稿期训练契约

“只用人工干预段训练”必须精确解释为：

- **新增 HIL 动作监督只使用 `D_intv`**；
- 训练总体仍保留 `D_demo` 作为 anchor；
- `SYSTEM_HOLD` 和接管前坏 POLICY action 不进入 action/video BC；
- 第一版冻结 Video DiT，HUMAN 视频只作为 Action/OVCR observation，不计算 future-video loss；
- 冻结 base 时训练小 adapter；不要再次用 correction-only 对 6B 全模型更新；
- 若训练 learned gate，再使用 `D_gate_pos/D_gate_neg`；如果暂时没有 gate，这两类先保存但不阻塞 adapter。

原始 HG-DAgger 的数据集从 BC demonstrations 初始化，只在恢复过程中新增 HUMAN expert labels，并不是把初始 demos 删除后仅用 correction 重新训练。IWR 进一步保留 on-policy 样本做正则；SIRIUS 保留 demos、干预和正常 robot 类，并将 pre-intervention action 权重设为零。因此当前最小、稳妥的主实验是 `demo anchor + HUMAN correction + frozen base adapter`。

### 0.4 本轮明确不做

- 不修复 chunk 停顿；
- 不实现 eager/late prefetch；
- 不修改服务端 cache 或 ActionDiT sampler；
- 不做 RTC/training-time RTC；
- 不做 full request/chunk ledger、planner generation、seed/noise；
- 不做 shadow policy paired residual；
- 不补存 alignment/pause 视频；
- 不让这些事项阻塞新的 HIL 采集和 adapter 实验。

卡顿仍会改变策略到达的状态，并影响人类何时接管，所以它是实验限制，不是完全无影响。但只要动作/视频损失排除 POLICY 和 SYSTEM_HOLD，它不会被直接当作专家动作或连续 future-video target 学进去。最终所有方法必须使用同一 `SYNC_HOLD` executor 公平评测，并报告推理延迟、HOLD 比例和 executor mode。

### 0.5 部署端本轮验收

- v3 老数据仍可读；
- HUMAN 导出不跨 segment、gap 或 episode；
- HUMAN target 与实际 executed action 完全一致；
- takeover request 能映射到前一条真实存储帧；
- checkpoint hash 不为空；
- assisted/autonomous outcome 正确；
- raw 仍包含三种 source，而不是 correction-only；
- 未改服务端和动作执行时序；
- `git diff` 中没有 prefetch/RTC/cache/sampler 改动。

以下第 1–2 节用于解释现状；第 3–11 节是完整 rich-v4/实时执行的截稿后路线图，不属于当前交付。

---

## 1. 现有 rich-v3 的准确状态

### 1.1 v3 修复了时间轴，没修复机器人停顿

rich-v2 在 FIFO 空时保持机械臂但不记录相机/state/action，真实经过的推理时间被删除。机械臂期间仍可能因位置环收敛、惯性、重力或接触移动，所以相邻存储帧被错误压缩成 66.7 ms。

rich-v3 的处理是正确的：FIFO 空时仍以 15 Hz 保存 `SYSTEM_HOLD`，`action` 等于最后实际 follower target，`action_supervision_mask=false`。因此 v3 视频是真实的“执行—停住—继续”，不能再次剪掉等待帧。

### 1.2 最新 `20260829_155403` 的实测数字

| 项目 | 数值 |
|---|---:|
| episode / 总帧 | 18 / 13,870 |
| POLICY / HUMAN / SYSTEM_HOLD | 4,175 / 4,621 / 5,074 |
| action-supervised | 8,796 |
| accepted policy replies | 769 |
| request latency median / P90 / max | 652 / 736 / 1,587 ms |
| POLICY 连续段典型长度 | 8 帧 |
| HOLD 段 median / P90 / max | 10 / 11 / 24 帧 |
| POLICY safety-clamped | 490 / 4,175（11.7%） |
| HUMAN safety-clamped | 179 / 4,621（3.9%） |

第一条 policy action 相对 request observation 已中位约 678 ms；第 8 条约 1,152 ms。典型控制模式是执行 8 步（0.533 秒），再 HOLD 约 10 步（0.67 秒）。

v3 的 queue-empty 阶段已连续记录；但 `PAUSED → leader alignment → HUMAN` 和 `HUMAN → POLICY_WARMUP → AUTONOMOUS` 仍故意不存帧。validator 记录了 100 个不连续边界，绝大多数属于这种交接。因此任何训练 window 都不得跨这些 gap。

### 1.3 v3 已能做什么、还缺什么

现有逐帧字段足以训练：

- HUMAN action correction；
- demo/correction mixed SFT；
- pre-intervention gate/risk；
- SYSTEM_HOLD-aware 时序诊断。

不足之处：

- HUMAN 时 `policy_action_valid=false`，没有同状态 base/human action pair；
- 没保存被 takeover 清掉的完整 chunk 及未执行 slots；
- `policy_generation` 是客户端 stale-reply generation，不是 Video DiT planner generation；
- 没有 AHA 内部 `model_chunk_index=0..4`、planner prefill id/observation；
- 没有 action-flow 初始噪声，难以精确复现 FlowCorrect target；
- handover/warmup 不是完整物理时间轴；
- episode 只写 `success`，没有区分 autonomous success 与 human-assisted success。

---

## 2. IG-OVCR 需要的四类数据

### 2.1 HUMAN expert correction

目标为安全层之后真正发给 follower 的动作：

$$
(o_t,s_t,a_t^{H,exec},m_t^H).
$$

当前 clamp 行中 `human_action` 是未经安全过滤的 proposal，而 `action`/`executed_action_arm` 是实际目标。默认训练 target 应是 `action`；raw human proposal 仍须保存，用于分析操作者输入和 clamp。

用途：Action flow loss、OVCR adapter 梯度、gate positive、恢复和平滑度统计。

### 2.2 Base-preservation anchor

需要模型不应被改写的状态：

- 原始完整 demo；
- 真正无接管的 autonomous-success rollout；
- rescued rollout 中远离接管并经复核为正常的 POLICY 片段。

用途：gate negative、adapter delta=0 约束、与冻结 P0 的 action-flow distillation、旧技能回归测试。

`outcome=success` 但发生过 HUMAN 的 episode 只能叫 `SUCCESS_WITH_INTERVENTION`，不能把全部 POLICY 帧当成功正样本。

### 2.3 Intervention-boundary/risk

需要以 `takeover_requested` 为时间原点保存：

- 请求时间和原因；
- 接管前 K 个有效 POLICY actions；
- 当时双相机/state；
- request/chunk/planner metadata；
- observation/action age、queue depth、safety clamp。

用途：SIRIUS pre-intervention mask、APO-style undesirable label、learned gate、flow/route uncertainty ROC。

必须把风险原因分开：

```text
POLICY_ERROR
CONTACT_STUCK
QUEUE_STARVATION
STALE_OBSERVATION
SAFETY_CLAMP
OPERATOR_CAUTION
```

SYSTEM_HOLD 表示系统没有新动作，不是策略明确提出的坏动作；它不能与 pre-intervention POLICY 合并做 action preference。

### 2.4 Paired counterfactual/residual

严格 FlowCorrect/residual 需要：

$$
(o_t,a_t^{P_0},a_t^H),\quad \delta a_t=a_t^H-a_t^{P_0}.
$$

现有 v3 没有 HUMAN 时的有效 \(a^{P_0}\)。分三档实现：

| 档位 | 方案 | 建议 |
|---|---|---|
| L0 | 不生成 shadow；HUMAN 直接监督小 adapter | 现有数据立即可做 |
| L1 | 保存完整 request ledger、图像/state/seed，训练前离线重放 P0 | **rich-v4 首选** |
| L2 | HUMAN 期间独立 shadow-policy session 持续预测 | 有额外 GPU/独立 session 后再做 |

在线 shadow 绝不能共享 live AHA 的 `_inference_state`，否则会推进 `next_chunk_index` 和 planner state；基类或其他变体还可能推进 action-history K/V。必须独立进程/GPU/session；任何 shadow 超时都不得影响 HUMAN control。

---

## 3. rich-v4 总体架构

### 3.1 三线程解耦

```text
15 Hz controller
  read latest cameras/state
  arbitrate POLICY/HUMAN/HOLD
  safety filter -> send_action
  emit FrameRecord every real tick

single-flight inference worker
  snapshot observation
  communicate serially with stateful AHA session
  emit RequestRecord + full ChunkRecord
  never block controller

recorder/writer
  asynchronously writes parquet/PNG/events/chunks
  disk I/O cannot silently delay controller
```

同一个 live AHA session 只允许一个 in-flight request，不能并发发两个可能乱序推进 `next_chunk_index`、planner/cache state 的调用。

### 3.2 所有阶段都记录

将 `pause_frames_stored` 改为 true，以下阶段每个 15 Hz tick 都保存真实 observation/state/实际 follower target：

```text
AUTONOMOUS
POLICY_QUEUE_EMPTY
PAUSED_BEFORE_ALIGNMENT
LEADER_ALIGNMENT
PAUSED_LEADER_READY
HUMAN_CONTROL
PAUSED_AFTER_HUMAN
POLICY_RESET
POLICY_WARMUP
```

非主动命令阶段统一：

```text
control_source = SYSTEM_HOLD
action = last actual follower target
action_supervision_mask = false
policy_action_valid = false
human_action_valid = false
```

新增 `hold_reason` 区分 queue empty、alignment、reset、operator pause 等。这样训练端可以选择排除，raw 不再留下不可恢复的时间洞。

### 3.3 记录事实，算法标签由 exporter 派生

collector 不应固化 `safe_anchor/pre_intv_positive/flowcorrect_pair_valid`。它只记录谁提出、谁执行、何时执行、基于哪个观察、为什么接管。不同实验需要不同 K 和 horizon，训练 exporter 再派生标签。

---

## 4. rich-v4 schema

### 4.1 逐帧 parquet

保留 v3 字段，新增或明确：

| 字段 | 语义 |
|---|---|
| `contract_version` | metadata=`so101-ahawam-hil-rich-v4` |
| `phase` | 保留所有交接/等待子状态 |
| `hold_reason` | `NONE/QUEUE_EMPTY/ALIGNMENT/POLICY_RESET/OPERATOR_PAUSE/...` |
| `takeover_requested` / `release_requested` | 输入事件所在 tick；不等于 HUMAN 已开始/结束 |
| `intervention_reason_id` | 指向结构化原因字典 |
| `task_phase` | `GRASP/TRANSPORT/COARSE_ALIGN/FINE_ALIGN/CONTACT/INSERT/UNKNOWN` |
| `queue_depth_before/after` | 仲裁前后 FIFO 深度 |
| `inference_in_flight` | 当时是否等待 reply |
| `policy_obs_frame_index` | 当前 action 的 request observation |
| `policy_observation_age_ms` | action 发出时 observation 已多旧 |
| `model_chunk_index` | AHA 内部 0–4，不能用全局 chunk id代替 |
| `planner_generation_id` | Video DiT prefill/planner refresh id |
| `planner_observation_frame_index` / `planner_age_ms` | planner 来源与年龄 |
| `action_deadline_ns` / `action_sent_ns` | nominal deadline 与实际发送时刻 |
| `control_lateness_ms` | sent - deadline |
| `safety_clamp_norm` | proposal 与实际 action 差异 |
| `shadow_policy_action[6]` / `shadow_policy_action_valid` | 可选 paired base proposal |
| `shadow_policy_request_id` | 追溯 shadow ledger |

三种 action 的语义固定为：

```text
policy_action   server proposal，安全过滤前
human_action    leader-mapped proposal，安全过滤前
action          真正发送给 follower 的过滤后目标
```

任意零占位数组必须配 valid flag，不能通过数值是否为零判断有效性。

### 4.2 每次请求 ledger

新增 `recording_context/policy_requests.jsonl` 或 parquet，每个请求至少包含：

```text
request id / client generation / live-or-shadow session id
request observation frame + monotonic time
sent/reply monotonic time
round trip + server model latency
accepted/stale/cancelled/error + reason
queue depth at request
expected first execution time
model chunk index
planner generation + planner observation frame
checkpoint/config hash
inference steps / seed
chunk record id
```

request observation 若不恰好对应 recorder tick，保存独立 snapshot 或精确引用最近完整 frame；不能只用 wall clock 猜。

### 4.3 完整 action chunk ledger

每个 reply 到达后立即保存完整 `[8,6]` chunk，即使一个 slot 都没执行：

```text
chunk_record_id / request_id
raw_action_chunk_model / raw_action_chunk_arm
initial_noise_chunk（若 server 支持）/ flow seed
model_chunk_index / planner_generation_id
num_inference_steps
chunk_status=ACCEPTED/STALE/CANCELLED/ERROR
```

另存每个 slot：

```text
slot_status = EXECUTED / CLAMPED / SKIPPED_TAKEOVER /
              SKIPPED_STALE / SUPERSEDED / NEVER_QUEUED
executed_frame_index[8]
executed_action[8,6]
```

只从逐帧 POLICY 行反拼 chunk 会永久漏掉 takeover 后被清空的未执行动作。

### 4.4 WAM context 的最低可重建集合

不要每 tick 保存完整 30 层 K/V。保存：

```text
checkpoint/config/text-cache hash
live reset 边界
所有 request 的严格顺序与 observation/state
每次 planner prefill observation
model_chunk_index
seed 或 initial noise
前序完整 generated chunks
action_video_read_mode / num_history_frames
```

若服务端方便导出，可选保存小型诊断量：

```text
pooled OVCR query
delta-K/delta-V norm per layer
sampled flow self-loss
```

这些用于 gate/ROC，不替代 raw image/state 与可重放 ledger。

### 4.5 事件和 outcome

事件顺序至少包括：

```text
takeover_requested -> policy_invalidated -> queue_cleared
-> inflight_cancel_requested -> alignment_started/completed
-> human_started/stopped -> release_requested
-> remote_reset_started/completed -> warmup_started/completed
-> autonomous_resumed
```

pre-intervention 标签必须以 `takeover_requested` 为原点，而不是几秒后的 `human_started`。

episode outcome：

```text
SUCCESS_AUTONOMOUS
SUCCESS_WITH_INTERVENTION
FAILURE_AUTONOMOUS
FAILURE_AFTER_INTERVENTION
TIMEOUT
OPERATOR_ABORT
SAFETY_ABORT
```

每个 intervention segment 另记 `RECOVERED/NOT_RECOVERED/ABORTED`。

---

## 5. HUMAN 时的 shadow policy

### 5.1 当前阶段不必阻塞在 shadow 上

IG-OVCR 第一版可冻结 base，让小 adapter 在 HUMAN observation 上生成实际 expert target，再用 demo/safe states distill 到 P0。它不要求同一状态先存在 base action。

### 5.2 rich-v4 先保证离线重放

推荐 L1：

1. 保存上一次 live reset 到 takeover 的完整 request/chunk ledger；
2. HUMAN 时连续保存 observation/state；
3. 服务器训练前用冻结 P0 严格按 ledger 重放；
4. 将 `offline_shadow_action` 和 `reconstruction_status` 写入派生 dataset，不改 raw；
5. 随机抽取已知 POLICY chunk 比较重放误差，超阈值则该 session 的 residual pair invalid。

在线 L2 以后再做。shadow 只能写日志，永远不入 follower queue；HUMAN release 后 shadow 丢弃，live 仍 hard reset。

---

## 6. 停顿对不同训练的影响

| 用途 | 当前 v3 是否可用 | 处理 |
|---|---|---|
| HUMAN action adapter | 可以 | HUMAN 内部连续；配 demo anchor，冻结 base |
| mixed HG/IWR action | 可以 | HOLD action mask；rescued POLICY 不自动当正样本 |
| intervention gate | 可以 | queue starvation/stale observation 单独成类 |
| Video correction | 有条件 | 仅连续长 HUMAN 段；第一版冻结 Video DiT |
| full raw world loss | 分段可用 | queue wait 连续，handover gap 禁止跨越 |
| paired residual | 不足 | 需要离线/在线 shadow |
| smooth executor 最终数据 | 分布不一致 | 修复执行器后补一轮 on-policy HIL |

因此可以先用 HUMAN correction 做**小 adapter sanity test**，但不能只用 HUMAN 完成完整方法。gate 仍需要接管前 POLICY，保持能力仍需要 demo/safe anchor。不要再次做 correction-only 的 6B 全量微调。

rich-v2 缺失的等待图像无法事后恢复。它只适合连续 HUMAN 导出或按 tick gap 切段，不能通过删除等待时间修复 future-video/action 时序。

---

## 6.1 “按完整动作块取样”能否绕开停顿

结论是：**对定义严格的单 chunk 监督损失可以；对当前 AHA 的原生 40-step 联合训练、Video DiT、gate 和 RL transition 不可以一概而论。**

把一次请求严格表示为：

$$
X_i=(o_i^{req},s_i^{req},\hat A_i^{raw},A_i^{exec},\delta_i,E_i),
$$

其中 `request observation` 是发送请求时冻结的图像/状态，\(\hat A_i^{raw}\in\mathbb R^{8\times6}\) 是服务器完整返回，\(A_i^{exec}\) 是逐 slot clamp 后真正执行的部分，\(\delta_i\) 是推理延迟，\(E_i\) 记录每个 slot 是否执行、被接管跳过或 stale。

如果训练目标只是：

$$
\mathcal L_{chunk}=\ell(f_\theta(o_i^{req},s_i^{req}),A_i^*),
$$

并且专家 chunk \(A_i^*\) 与同一个 request observation 对齐，那么两个请求之间的 HOLD 不进入这个损失。此时用户的判断成立。但必须同时满足：

1. 直接读取 request snapshot 和服务器原始完整 chunk，不能从逐帧 POLICY 行“删掉 HOLD 后重新拼”；
2. 明确 target 是 expert/human、base proposal 还是 executed action；自主 policy 输出本身不是 DAgger expert label；
3. takeover 中未执行的 slots 不能伪装成 executed transition，但作为负 chunk/反事实 proposal 仍应保存；
4. raw proposal 和 safety-clamped execution 分开；
5. 单 chunk loader 不跨 request、handover、reset 或 planner refresh 边界。

### 6.1.1 当前 AHA 原生训练并不是独立 8-step 样本

当前配置是 `action_horizon=40`、`action_chunk_size=8`、`num_frames=49`。训练样本含 40 个连续动作，并为五个 chunk start 提取对齐图像/状态；ActionDiT 在一个训练样本中学习五块，Video DiT 同时学习连续 future video。

所以“把每次请求当独立 8-step 样本”需要一个专门的 adapter/chunk loss loader。它适合 frozen-base adapter、paired residual、set-supervised chunk loss；却不能不改代码就等价替代当前 40-step AHA full fine-tuning。

尤其不能把五个 8-step autonomous chunks 删除中间 HOLD 后拼成 40 步：这样相邻 action 和五个 chunk-start observations 在模型时钟上相隔 8 ticks，物理世界却可能相隔约 18 ticks，训练时间语义被压缩。若保留 HOLD，则它们是 executor-specific 的真实轨迹，必须显式 mask/condition，而不是假装普通策略动作。

### 6.1.2 不同方法是否仍受 HOLD 影响

| 方法/损失 | chunk 化后是否基本绕开 HOLD | 剩余问题 |
|---|---|---|
| HUMAN 8-step correction adapter | 是 | HUMAN 段必须连续；target 用实际 expert execution |
| base chunk distillation/anchor | 是 | request snapshot→raw proposal；本质是保能力，不是改进监督 |
| positive/negative chunk correction | 是 | 必须保存同状态 policy proposal 和 human desired set |
| 当前 AHA 40-step ActionDiT full loss | 否 | 需要五个物理连续 chunk-start obs 和 40-step 时间轴 |
| Video DiT loss | 否 | HOLD/惯性是真实 future；删除会制造跳变 |
| intervention gate | 部分 | queue starvation 可能与真实任务失败混淆，必须有 reason/phase |
| chunk-level offline RL/value | 否 | 这是 variable-duration semi-MDP；需 duration、reward、next obs |
| dynamics/world-model RL | 否 | next state 取决于 HOLD、clamp、接触和真实 elapsed time |

chunk-level RL 若把一次 request→执行结束当 macro transition，应保存：

$$
(o_i^{req}, A_i^{exec}, r_i, o_{i+1}, \Delta T_i, done_i),
$$

其中 \(\Delta T_i\) 不是固定 8 ticks，而包含可变推理等待和实际执行时长。若改用“第一条动作真正下发前”的 state 作为 RL state，behavior chunk 却是基于更早的 request observation 生成，行为策略概率又与该 state 不对齐。两种定义都可以研究，但不能混写成普通固定步长 MDP。

### 6.1.3 与已有工作的关系

- [IWR](https://arxiv.org/abs/2012.06733) 把 intervention/non-intervention 分开并等比例采样；自主部分主要起 regularization 作用，但它是逐时刻 BC，不专门解决慢生成式 chunk 的 compute HOLD。
- [SIRIUS](https://arxiv.org/abs/2211.08416) 以 human trust 重加权部署数据；它说明 autonomous 数据不必全丢，但当前系统必须把“人不信任动作”和“FIFO 空导致 HOLD”分开。
- [Set-Supervised Diffusion Policy](https://arxiv.org/abs/2606.01865) 明确使用 human correction 产生的 positive/negative action chunks，最直接支持“按块保存”的方向；它也反过来要求保存完整未执行 policy proposal，而不仅是执行过的逐帧 rows。
- [AutoIntervene](https://arxiv.org/abs/2608.07065) 针对 action-chunking policy 比较 visual/action support 并保留成功 rollout 的 intervention segment，说明 chunk proposal、task phase 和 correction segment 都是独立数据对象。
- [RACE](https://openreview.net/pdf?id=INsLvSCJ4z) 明确指出异步推理会使新 chunk 基于 outdated state 并产生 action alignment error；这不否定离线 chunk supervision，而是否定“chunk 化以后执行时序也自动无影响”。
- [PACE](https://arxiv.org/abs/2606.00537) 把 action chunk 看成执行一个 open-loop prefix 后重新观测，并发现最佳执行长度强烈依赖任务阶段；这与插入阶段不能盲目执行长块的判断一致。

经典 DAgger/IWR 主要面对较快的单步策略，所以很少把 GPU 等待建模为核心问题；这个问题在近年的 diffusion/flow/VLA action-chunk 执行工作中才被明确讨论。

---

## 6.2 截稿期采集版：rich-v4-data-lite

现在**不需要先修复停顿才开始采集**。更稳妥的是保持 v3 executor 不变、继续保存真实 `SYSTEM_HOLD`，只对数据契约做 append-only 升级。不要在十天窗口里一次实现本文所有 online shadow、RTC 和 replay 能力。

本轮必须增加的最小集合：

1. `request_ledger`：request id、冻结 observation 的 frame/image/state 引用与时间戳、send/reply、latency、accepted/stale、generation、model chunk index、planner refresh id；
2. `chunk_ledger`：reply 到达后立即保存完整 raw `[8,6]`，每个 slot 标 `EXECUTED/CLAMPED/SKIPPED_TAKEOVER/STALE`，并映射实际 frame id；
3. 边界事件：`takeover_requested`、`human_started/stopped`、release、结构化 reason、task phase；
4. 结果：`AUTONOMOUS_SUCCESS/SUCCESS_WITH_INTERVENTION/FAILURE/ABORTED`；
5. 继续保存逐帧真实相机/state、raw proposal、executed action、control source、HOLD reason 和 supervision mask；
6. run provenance：checkpoint/config/hash、15 Hz、chunk/horizon、相机和安全参数。

本轮可以推迟：online shadow session、精确 seed/noise replay、RTC、prefetch、新 contact sensor 和完整 value pipeline。字段可预留为 null。

采集前必须用 2 个 pilot episode 验证：完整 8 slots 可重建；中途 takeover 后剩余 slots 正确标 skipped；request snapshot 不引用 reply 时刻图片；HOLD 没被删除；HUMAN 连续段不跨 alignment gap。通过后再正式批量采。

如果来不及完成 data-lite，rich-v3 仍足够做 HUMAN correction adapter + demo anchor + 粗粒度 gate；但会永久失去未执行 policy proposal和可靠的 paired chunk，因此优先补 `request_ledger + full chunk_ledger`，其论文收益高于现在改 RTC。

---

## 7. RTC、预取和当前 AHA 的关系

### 7.1 RTC 的约束

[RTC](https://arxiv.org/abs/2506.07339) 在旧 chunk 执行期间生成新 chunk，把 reply 到达前必然会执行的动作作为 hard prefix，其余重叠动作以 soft guidance 做 flow/diffusion inpainting。

设预测 horizon 为 \(H\)，推理延迟为 \(d=\lfloor\delta/\Delta t\rfloor\) 个 ticks，两次 inference 启动间隔为 \(s\)。RTC 要求：

$$
d\le s\le H-d\quad\Rightarrow\quad H\ge2d.
$$

当前 TCP 每次实际返回 8 步：

```text
H_returned=8
control period=66.7 ms
median latency=652 ms -> d about 9-10
P90=736 ms -> d about 11
```

所以当前 8-step API 不满足 RTC。配置里的 40-step horizon 是五个依次生成的 8-step chunks，不是一次已经可编辑的 40-step output。

inference-time RTC 每个 flow step还需要 vector–Jacobian product。论文中 π0.5 的纯模型时间约从 76 ms 增到 97 ms；当前 AHA 约 650 ms，未经 profiling 直接加 RTC 可能让 queue starvation 更严重。

结论：AHA 是 flow policy，类型上兼容 RTC；但当前短 chunk、长 latency 和 stateful sequential cache 使 RTC 不是 drop-in。

### 7.2 当前 AHA 的跨 chunk state 不等于 action-history continuation

基类提供“把上一块 clean action 写入 `action_history_kv_cache`”的设施，但当前实际部署的 `AHAWAM` 叶类把 `_should_update_action_history()` 覆盖为 `False`。因此不能声称 chunk \(j+1\) 显式读取了 chunk \(j\) 的 action-history K/V。

同一个 40-step inference state 仍保存 Video DiT planner cache、初始 action latents 和 `next_chunk_index`；每个 8-step chunk 则重新使用本次 request 的 `chunk_obs_image/chunk_proprio` 构造 query，并改写供当前 ActionDiT 使用的 video K/V。也就是说，当前实现的关键闭环来源正是**每块的新图像和新状态**，而不是上一动作块的执行历史。

服务端收到的“当前帧”只在 request 发出时是当前的。简单改成“收到 chunk 后立刻请求下一块”，会让下一块基于当前 chunk 执行前的观测生成；模型既没有执行后的新图像，也没有显式 cached clean-action continuation 可以弥补。当前 `prefetch_actions: 0` 的 queue starvation 和 eager prefetch 的动态观测陈旧是两种不同问题，不存在无代价的替换。

### 7.2.1 同步等待和抢先预取的真实时间线

令推理延迟为 \(\delta=652\) ms，8-step chunk 执行时间为 \(T_c=8/15=533\) ms。

当前同步执行：

```text
t0              拍摄 O0，发送 request
t0 ... t0+652   机器人 HOLD；服务器基于 O0 推理
t0+652          chunk 到达并开始执行
```

第一条动作使用的图像在数值上已经旧了约 652 ms，但这段时间机械臂主要 HOLD，所以场景状态变化通常相对较小；代价是停顿。

立即预取：

```text
t0              chunk j 到达；拍摄 O0，发送 j+1；同时开始执行 j
t0 ... t0+533   机器人执行整个 j
t0+533...652    FIFO 空，短暂 HOLD
t0+652          j+1 到达并开始执行
```

此时图像同样旧约 652 ms，但期间机械臂已经执行了整个 chunk。当前 AHA 没有累计 action-history K/V；即使未来变体加入 planned continuation，它也仍不知道实际是否抓稳、是否碰孔、是否被 clamp。自由空间移动可能容忍这种 open-loop 延续；精密对孔和接触阶段不能默认容忍。

所以普通预取只解决 throughput 和边界停顿，不等于获得实时闭环。它必须记录并评估：

```text
observation_age_at_execution
robot_motion_since_request
planned_vs_executed_action_error
safety_clamp/contact/takeover rate
```

### 7.3 第一修复：低 flow-step + 单 in-flight prefetch

先用相同 observation/state 对 `num_inference_steps=10,8,6,5,4` 测：

- latency median/P90/P99；
- 与 10-step chunk 的 L1/L2；
- chunk 内速度、加速度、jerk；
- boundary jump、safety clamp；
- 固定 seed 与多 seed 稳定性。

然后才试验 inference worker 在 queue 仍有动作时启动下一请求。触发时刻不是固定“越早越好”，而是新鲜度—连续性的控制变量：

```python
if phase == AUTONOMOUS and not inference_in_flight and current_chunk_accepted:
    if queue_depth <= prefetch_trigger:
        obs = latest_complete_observation()
        submit_one_stateful_request(obs, generation, expected_model_chunk_index)
```

根据实测 P90，可计算为了在 chunk 边界前返回至少要剩余多少动作：

$$
N_{need}=\left\lceil\delta_{P90}/(1000/15)\right\rceil.
$$

只有 \(N_{need}\le8\) 才可能用当前 chunk 完全遮住下一次推理。

```text
prefetch_trigger = min(8, ceil(P90_latency / 66.7 ms) + safety_margin)
```

按当前 P90，结果会被截到 8，即为了连续性只能在新 chunk 入队后立刻拍图预取；这恰恰也是观测最陈旧的设置。它可把典型等待从约 10 ticks 降到约 2 ticks（652−533≈119 ms），但是否提高任务成功率必须实测，不能仅凭 HOLD 下降判定。若加速后 P90 显著小于 533 ms，应把 trigger 推迟到 chunk 尾部，在恰好来得及的前提下尽量使用更新的图像。

建议按任务阶段使用不同策略：

| 阶段 | 短期策略 |
|---|---|
| 抓取前、自由空间粗移动 | 可 A/B 测试 eager prefetch；限制速度并监测 clamp/takeover |
| 靠近插孔、尚未接触 | 优先等加速后采用 late prefetch；没有加速时同步 HOLD 可能更可靠 |
| 接触搜索和最终插入 | 默认使用最新观测的同步模式；除非已有低延迟 fast correction head/RTC-like 方法验证安全 |

不要仅凭图像或末端位置规则静默切换阶段。优先用任务状态机的显式 phase；每种 phase/executor 写入逐帧日志。

约束：

- live session 单 in-flight；
- generation/session/expected model chunk index 全匹配才入队；
- takeover 立即清 queue、递增 generation、取消/废弃 reply；
- resume remote reset；
- late reply 时真实 HOLD 并记录，绝不伪造动作；
- 接触阶段禁止用速度外推填空。

这里的“立即预取”不是并发生成两个 AHA chunk。live policy session 始终只有一个 GPU request 在途；只是让 GPU 生成 chunk \(j+1\) 与机器人执行 chunk \(j\) 在不同线程并行。其顺序正确性依赖同一 `_inference_state` 的 `next_chunk_index` 和 planner/cache state，而不是当前 AHA 中并不存在的累计 action-history K/V。

建议的线程职责：

```text
15 Hz controller: 只取 FIFO、做 safety clamp、下发、记录；绝不等待网络/GPU
inference worker: 单 in-flight TCP request；reply 验证后原子入 FIFO
recorder: 异步写盘；写盘拥塞显式报警，不阻塞控制周期
```

如果 FIFO 仍为空，controller 保持最后一个实际 position target 并记录 `SYSTEM_HOLD/QUEUE_STARVATION`。不要插值到尚未生成的目标，不要速度外推，也不要复制旧 chunk 冒充新策略动作。

若 5–6 step 的精度可接受且 P90 小于 533 ms，late prefetch 才可能同时改善停顿和观测新鲜度。否则依次考虑 step distillation/FastWAM、服务端 profiling/compile、快速 latest-observation correction head、重新训练更长可编辑 horizon，最后才是 RTC。降低 control Hz 会改变策略的物理时间尺度，不应未经重新验证直接采用。

### 7.4 training-time RTC

[Training-Time Action Conditioning](https://arxiv.org/abs/2512.05964) 随机模拟 delay，把 action prefix 设为 clean、对应 flow time 设为 1，只在 postfix 算 loss；推理时每步覆盖 prefix，不需 VJP。

AHA 已有 previous-chunk clean history，却没有“同一待生成 horizon 的逐 token clean prefix + noisy postfix”正式接口。仍需修改 per-token flow time、prefix loss mask、sampling overwrite 和 chunk causal mask。以后若输出扩展到至少 16–24 个可编辑动作，它比 inference-time VJP 更值得优先研究。

### 7.5 A2C2

[A2C2](https://arxiv.org/abs/2509.23224) 用小 correction head 每个控制 tick 读取最新 observation、base action、chunk position 和 base feature，输出 residual。它直接针对普通预取遗漏新观测的问题；但 FIFO 完全为空时没有 base action，不能单独解决停顿。

它适合作为 IG-OVCR 的外部 output-residual baseline，并与经过验证的 prefetch 组合；不能把两个绝对关节 chunk 直接线性平均后称为 RTC。还要注意：如果 IG-OVCR adapter 仍位于耗时的 ActionDiT 去噪循环内部，它看到的也只是 request 时刻的图像，并不自动成为“每 tick 最新观测纠正器”。要解决这一点，需要另一个真正轻量、可在控制频率运行的 head。

---

## 8. 状态机细节

### 8.1 AUTONOMOUS

```text
controller at 15 Hz consumes one slot
recorder saves actual action/state/image
queue <= trigger -> snapshot obs -> one request
reply -> first save full chunk -> validate -> enqueue or mark stale
```

### 8.2 Takeover

必须按顺序：

```text
timestamp takeover_requested immediately
record reason/task phase
increment generation
atomically clear queue and mark remaining slots SKIPPED_TAKEOVER
cancel/make in-flight stale
hold last actual follower target
record all alignment ticks
align leader
start HUMAN_CONTROL
```

不能等 GPU reply。`takeover_requested` 到 `human_started` 之间的数秒是 HOLD context，不是 expert action。

### 8.3 HUMAN

每 tick 保存 leader raw state、mapped human proposal、control scale、safety filter input/output/previous target、actual executed action、camera/state timestamps、segment/reason 和可选 shadow id。训练默认 target 是 actual executed action。

### 8.4 Release

```text
release_requested
stop HUMAN labels
hold and keep recording
discard shadow
invalidate live generation
remote hard reset
request from latest observation
wait for fresh validated chunk
resume AUTONOMOUS through existing safety limiter
```

旧 in-flight reply和 shadow state都不得变成 live action。

---

## 9. 四个派生训练视图

### `expert_correction_view`

- 可以带接管前 observation pre-roll；
- action loss 只覆盖连续 HUMAN + executed valid；
- HOLD/gap/padding 全 mask；
- 第一版冻结 Video DiT；
- 不要求 shadow。

### `anchor_view`

- demo + autonomous success + reviewed safe policy；
- adapter distill 到 P0、gate negative；
- 与 correction 显式 50:50 或 75:25 抽样。

### `intervention_gate_view`

- 以 `takeover_requested` 前 8/16/24 帧构造候选；
- valid POLICY 做 risk positive；
- SYSTEM_HOLD 做单独 starvation class；
- safe 且远离 takeover 做 negative；
- alignment/HUMAN/release grace 排除或单列。

### `paired_residual_view`

只保留 online-shadow valid 或 offline reconstruction passed 的样本，base/human action 必须对齐到同一真实时间和同一安全语义。禁止把接管前最后一个 policy action广播到整段 HUMAN。

---

## 10. Validator 与自动化测试

Validator 至少检查：

1. 每阶段 tick 单调且都有 frame；
2. parquet/fixed/wrist 帧数一致；
3. gap 有 dropped/event 原因；
4. `action` 等于实际安全过滤后目标；
5. HOLD action 等于上一实际 target 且 action mask=false；
6. raw proposal 与 valid flag 一致；
7. clamp norm 与 filter input/output 可追溯；
8. accepted request 对应唯一完整 chunk；
9. 每个 POLICY row 映射唯一 chunk slot；
10. stale/cancelled slot 绝不执行；
11. model chunk index 在 reset 内按 0–4 合法递进；
12. planner prefill observation 可追溯；
13. observation age/queue depth 一致；
14. takeover request ≤ HUMAN start ≤ HUMAN stop；
15. alignment/warmup 不标 HUMAN；
16. outcome 区分 autonomous/assisted；
17. shadow session 与 live 不同且永不执行；
18. replay 误差超阈值时 paired residual invalid。

Fake tests 至少覆盖：

- 20 ms inference：无 HOLD；
- 400 ms + 8-step：prefetch 后无 HOLD；
- 700 ms：出现真实 HOLD但时间不压缩；
- takeover 中 reply：stale且零执行；
- chunk 执行 3/8 时 takeover：后 5 slots 标 skipped；
- HUMAN clamp：raw/exec 都保留；
- release 后旧 generation 不可复活；
- alignment/warmup 每 tick 记录；
- writer 变慢不能静默掉控制帧；
- request ledger + seed 可重放；
- v3 仍可被兼容 reader 读取。

---

## 11. 交付与安全验收顺序

交付：rich-v4 schema、request/chunk ledger、single-flight prefetch worker、状态机扩展、四类 exporter、offline replay、validator、fake tests、无参数启动脚本、键位和急停说明。

验收顺序：

1. fake robot/policy；
2. follower 断电，真实相机/recorder；
3. 上电空载低速；
4. pause/alignment/HUMAN/release；
5. 10-step synchronous 兼容模式；
6. 低-step prefetch，对比 HOLD ratio、age、clamp、jerk；
7. 粗移动和抓取；
8. 最后才做插入。

每种执行器单独建立 run，manifest 明确：

```text
executor_mode=SYNC_HOLD/NATIVE_PREFETCH/RTC/TRAINING_TIME_RTC
num_inference_steps
prefetch_trigger
latency summary
```

不要无标记混合不同 executor domain 的数据。

---

## 12. 当前最终建议

现在可直接使用：

- v1/v2/v3 连续 HUMAN correction：frozen-base action/OVCR adapter；
- 02/04 demo：anchor；
- v3 接管前有效 POLICY：gate/risk；
- v3 SYSTEM_HOLD：执行器诊断，第一版不进 action/video loss。

下一轮必须补：完整 handover 时间轴、full chunk/slot ledger、request observation 映射、model chunk/planner id、seed/noise、autonomous/assisted outcome、结构化接管原因和离线 shadow 重放信息。

实时执行优先级：

```text
P0 保持 v3 的真实时间和 HOLD mask
P1 profile 10/8/6/5/4 flow steps
P2 先做可开关的 AHA-native single-flight prefetch；只在粗阶段 A/B 测试
P3 优先把延迟降到明显小于 8-step horizon，再使用 late prefetch
P4 若需接触阶段连续闭环，增加真正的 latest-observation fast correction head
P5 用验证后的 executor 补一轮 on-policy HIL
P6 更长可编辑 horizon / training-time RTC；最后才考虑 inference-time RTC VJP
```

一句话：**不要剪掉等待，也不要假设当前 AHA 有跨块 action-history continuation。v3 可立即做小 adapter 和 gate；普通预取只能用观测新鲜度换取较少 HOLD，在当前延迟下不适合未经验证直接用于最终插入。先加速，再尽可能晚地预取；接触阶段若要连续且闭环，需要真正读取最新观测的快速纠正头。**

## 13. 主要资料

- [RTC](https://arxiv.org/abs/2506.07339)
- [Training-Time Action Conditioning](https://arxiv.org/abs/2512.05964)
- [A2C2](https://arxiv.org/abs/2509.23224)
- [FlowCorrect](https://arxiv.org/abs/2602.22056)
- [Diff-DAgger](https://arxiv.org/abs/2410.14868)
- [SIRIUS](https://arxiv.org/abs/2211.08416)
- [Compliant Residual DAgger](https://arxiv.org/abs/2506.16685)
- [HALO-WA](https://arxiv.org/abs/2607.04265)
