# SO101 + AHA-WAM 同步 v3/data-lite 多用途 DAgger 采集交接

> 把本文交给机械臂控制电脑上的 Codex。
> 当前选择：不用 Flash student，不用 AHA async；用四轮 SFT policy 同步 rollout，并
> 完整保存 POLICY、SYSTEM_HOLD、HUMAN。HUMAN-only 是第一条训练 baseline，但不是
> raw 数据的唯一预定用途。

## 0. 决策结论

1. 现有 so101-ahawam-hil-rich-v3 可以立即继续采集，不必等待 RTC 或重写 executor。
2. v3 足够支持 HUMAN-only HG-DAgger、IWR 风格的 policy/human 分流、带真实 HOLD 的
   world/video 训练，以及较粗粒度的 intervention risk/value 分析。
3. 推理停顿不能被当成普通 policy action：SYSTEM_HOLD 是执行系统没有新决策时持续
   保持上一目标的真实 transition。
4. 若希望未来做精确的 chunk positive/negative、未执行 proposal、chunk-level offline
   RL 或时延对齐研究，纯 v3 信息不完全。建议在不改变 v3 parquet 语义的前提下增加
   data-lite request/chunk ledger。
5. data-lite 是推荐的小型 append-only 调整，不是完整 rich-v4，不应推迟正式采集。

## 1. 本次同步部署

trainer-state JSON 不是模型权重。服务器必须加载：

    runs/so101_ahawam_padapter_v2/full_zero2_offload_ahawam_base/
    formal-padapterv2-sft-ahawambase-bs4-acc1-4ep-3gpu-20260901-001743/
    model/checkpoints/weights/step_005328.pt

对应设置：

- stats：runs/so101_ahawam_padapter_v2/assets/dataset_stats.json；
- num_inference_steps=10；
- action_horizon=40；
- action_chunk_size=8；
- max_action_offset=7；
- num_history_frames=6；
- control rate=15 Hz；
- task：Remove the power adapter from the holder and plug it into the power strip.

服务器端配置和启动脚本已经改好：

    bash scripts/so101_pipeline/online_deploy/run_action_server.sh

机械臂端继续使用原先单连接 type=infer 协议。不要套用
scripts/so101_pipeline/flash_async_deploy 下的双连接说明。通过 SSH tunnel 时，本地仍连接
127.0.0.1:12000。

## 2. 推理停顿究竟能否进入数据

必须按训练用途分别回答。

### 2.1 HUMAN-only HG-DAgger

只监督连续 HUMAN correction。POLICY 和 SYSTEM_HOLD 不进入 action/video target。
这种用法几乎不受 HOLD 标签污染，但 HOLD 会改变人开始修正时所处的状态；这是实际
on-policy 数据分布，不需要从 raw 中删掉。

### 2.2 IWR 风格 BC

建立两个 sampler：

- D_I：连续 HUMAN expert；
- D_R：真实 POLICY 执行帧。

SYSTEM_HOLD 不属于 D_R，因为它没有新的 policy decision。IWR 可按独立比例采样 D_I
与 D_R；不能把所有 raw 行直接 concat 后统一 BC，否则长 HOLD 会被过度采样，并把
保持动作误解释成策略决策。

### 2.3 WAM/world/video 训练

可以保留 POLICY、SYSTEM_HOLD、HUMAN 的真实连续图像/state，因为 HOLD 期间机械臂仍
可能受惯性、重力和接触影响。这些是真实 dynamics，不应剪掉。

但训练窗口必须：

- 使用真实 executed action 作为 video conditioning；
- 对 direct action loss 分别使用 expert、policy、hold mask；
- 不跨 tick gap、dropped tick、pause、leader alignment、reset；
- 明确报告训练是否包含 HUMAN 导致的 dynamics。

action_supervision_mask=false 只能关闭 direct action loss，不能自动阻断共享 VideoDiT
参数收到 video loss 梯度；实验配置必须显式说明。

### 2.4 成功自主 rollout/self-training

可以从 outcome=AUTONOMOUS_SUCCESS 的 rollout 派生 policy-only 数据，但仍应：

- 排除 SYSTEM_HOLD 作为主动 action label；
- 保存 HOLD 作为真实视频 transition；
- 报告 success 筛选、intervention 数量和 HOLD 比例；
- 防止把有人类接管的最终成功轨迹误称为 autonomous success。

### 2.5 风险 gate 或粗粒度 value

v3 的 events、takeover_segment_id、policy frame 和最终 outcome 可构造粗标签：

- takeover_requested 之前若干有效 POLICY 帧作为风险候选；
- 距 takeover 足够远的 POLICY 帧作为相对安全候选；
- SYSTEM_HOLD 单列为 queue-starvation/system class。

不要把 HOLD 直接标成失败，也不要用 HUMAN 真正开始的时刻代替更早的 takeover request。

### 2.6 chunk-level offline RL 或成对偏好

v3 只保存逐帧已执行 action，接管后被清空的 policy chunk 尾部不可恢复；request 时刻
冻结的 image/state 也没有完整独立 ledger。因此它只能做近似分析，不足以严谨构造：

    request observation
    full proposed 8x6 chunk
    executed/skipped slot set
    elapsed duration
    reward and next request observation

如果这类方法仍是候选，就应增加下面的 data-lite。

## 3. v3 每种 control source 的固定语义

### POLICY

- action_supervision_mask=true；
- policy_action_valid=true；
- action 是真正经过安全层并下发的 target；
- policy_request_id、policy_chunk_id、action_index_in_chunk 有效；
- 可进入 policy regularization/self-training，但不是 expert label。

### HUMAN

- control_source=HUMAN；
- intervention=true；
- human_action_valid=true；
- action_supervision_mask=true；
- action 是裁剪后真正执行的 expert target；
- human_action 可保留原始 leader proposal，但训练 target 默认用 action。

### SYSTEM_HOLD

- action 是上一条实际 follower position target，不能写零或 measured state；
- action_supervision_mask=false；
- policy_action_valid=false；
- human_action_valid=false；
- 每个 15 Hz tick 仍保存图像、state 和实际 hold target；
- 用于真实时间轴、video dynamics 和系统诊断，不作为 policy/expert decision label。

## 4. 推荐增加的 rich-v3-compatible data-lite

不要改已有 parquet 字段含义。只在 recording_context 下增加两个 JSONL 或 parquet
ledger；如果本地已经有同类文件，就扩展而不是重复创建。

### 4.1 policy_requests

每次 infer 请求一行，至少保存：

- logical episode id；
- client policy generation；
- request id；
- 冻结 observation 对应的 frame id；
- fixed/wrist capture timestamp 或可追溯文件引用；
- request state_model；
- task；
- send/reply monotonic timestamp 和 RTT；
- accepted、stale、error；
- reply chunk id；
- checkpoint identity；
- executor mode=SYNC_HOLD。

关键要求：request observation 必须是发送时冻结的快照，不能事后拿 reply 到达时的图像
回填。

### 4.2 policy_chunks

reply 到达时立即保存整块，不等逐帧执行后再拼：

- generation、request id、chunk id；
- 完整 raw action_chunk，shape 8x6；
- server/model latency；
- 每个 slot 的 planned index；
- 后续更新的 EXECUTED、CLAMPED、SKIPPED_TAKEOVER、STALE 状态；
- executed frame id 和 scheduled/sent timestamp；
- 发生 takeover 时保留未执行 tail，不能删除。

### 4.3 边界事件与结果

继续或补齐：

- takeover_requested；
- human_control_started/stopped；
- release/resume/reset；
- request invalidated、queue cleared；
- intervention reason 和 task phase；
- AUTONOMOUS_SUCCESS、SUCCESS_WITH_INTERVENTION、FAILURE、ABORTED；
- episode duration、HOLD ticks、intervention count。

### 4.4 Provenance

旧 v3 manifest 的 policy_checkpoint_sha256 曾为 null。新 run 至少保存：

- step_005328.pt 的完整规范路径/名称和 SHA-256；
- deploy config SHA-256；
- num_inference_steps=10、horizon=40、chunk=8、fps=15；
- task、endpoint、collector commit；
- leader/follower calibration hash；
- contract_version=so101-ahawam-hil-rich-v3；
- data_extension=sync-request-chunk-ledger-v1。

如果控制电脑不能读取远端权重，允许从服务器配置显式传入 hash，但不能继续默认为 null。

## 5. data-lite 的实现优先级

若今天就要采集：

第一优先级，必须保持：

- v3 三种 control source 和所有 mask；
- FIFO 空时真实记录 SYSTEM_HOLD；
- takeover 清 FIFO、generation fencing、旧 reply 零执行；
- resume 时 server reset；
- 完整 HUMAN 连续段；
- rollout outcome。

第二优先级，建议采集前补：

- request ledger；
- reply 到达即保存完整 action chunk；
- 未执行 slot status；
- checkpoint provenance。

第三优先级，可以延期：

- online shadow policy；
- flow seed/noise；
- RTC/inpainting；
- prefetch/blending；
- contact sensor；
-完整逐步 reward。

如果第二优先级改动不能在短时间内通过 fake/pilot 测试，就先用原 v3 正式采；不要为未来
可能用到的方法破坏当前可靠 collector。

## 6. 训练侧必须显式选择数据 view

raw 数据只保存事实，不应在采集时提前删除某类帧。训练配置再选择：

| view | action target | video target | SYSTEM_HOLD |
|---|---|---|---|
| HG HUMAN-only | HUMAN action | 仅连续 HUMAN，可按段长开关 | 排除 |
| IWR intervention | HUMAN action | 可选 HUMAN | 排除 |
| IWR robot | POLICY executed action | 可选 POLICY | 排除 direct action |
| raw world | 分 source 单独 mask | POLICY+HOLD+HUMAN 或所选子集 | 保留真实 dynamics |
| successful policy | 成功 rollout 的 POLICY | POLICY+真实 HOLD | HOLD 非决策 |
| gate/value | 不做统一 BC | 依标签定义 | 单列系统状态 |

严禁把 raw v3 目录直接交给原始 AHA loader、让所有行无差别进入 action loss。标准 loader
若不知道 control_source/action_supervision_mask，就会错误模仿大量 HOLD 和失败 policy。

## 7. 给机械臂端 Codex 的具体任务

1. 保持当前单连接同步 infer 和 rich-v3 recorder。
2. 确认 endpoint、任务文字、10-step、8-action、15 Hz。
3. 不实现 Flash async、RTC 或 prefetch。
4. 审计现有 events 和 manifest 是否已满足第 4.3、4.4 节。
5. 若改动小，增加 request/chunk 两个 append-only ledger；若改动大，先不阻塞采集。
6. 录制 2 条 pilot，其中一条在 chunk 执行到 3/8 时 takeover。
7. 验证未执行 5 slots 被标 skipped、旧 reply 不执行、HOLD 不消失、HUMAN 连续。
8. 输出 validator、每类帧计数、request RTT、HOLD 比例和完整修改清单。

## 8. 交给 Codex 的相关文档

本轮以本文为主。还应阅读：

- docs/hil/SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md：v3 时间轴和 mask；
- docs/hil/AHAWAM_HIL_HUMAN_ONLY_TRAINING_IMPLEMENTATION_ZH.md：HUMAN baseline；
- docs/hil/SO101_AHAWAM_IG_OVCR_RICH_V4_COLLECTION_AND_REALTIME_EXECUTION_HANDOFF_ZH.md
  的第 6.2 节：data-lite 字段来源。

不要把 ROBOT_PC_CODEX_IMPLEMENTATION_TASK_ZH.md 当成本轮主文档；它针对已经暂停的
Flash async 双连接部署。
