# SO101 + AHA-WAM 同步 rich-v3 HUMAN-only DAgger 采集交接

> 可把本文直接交给机械臂控制电脑上的 Codex。
> 当前决定：暂停 Flash 蒸馏和异步部署；使用四轮 SFT policy 同步 rollout；保留真实推理
> 延迟；训练时只使用连续 HUMAN correction，再与原 demo 组合。

## 0. 结论

**继续使用现有 so101-ahawam-hil-rich-v3 collector 即可，不需要升级成 rich-v4。**

v3 已经具有 HUMAN-only 训练所需的核心信息：

- control_source 能区分 POLICY、HUMAN、SYSTEM_HOLD；
- intervention、human_action_valid、action_supervision_mask 能确认专家标签；
- takeover_segment_id 能切分每段连续人工修正；
- tick_index、dropped_tick 能阻止训练窗口跨越暂停或丢帧 gap；
- action 是真正下发给 follower 的安全裁剪后目标；
- fixed/wrist、state、action 和 15 Hz 时间轴都被保存；
- FIFO 空时 v3 会保存真实 SYSTEM_HOLD，没有把延迟时间删除。

推理延迟会改变 policy 到达的状态分布、使人更常接管，但这正是 on-policy DAgger 数据的
来源。只要训练 view 严格限定为连续 HUMAN 段，SYSTEM_HOLD 和 autonomous policy
动作就不会成为 video/action target。

## 1. 本次必须部署的模型

用户提到的文件：

    runs/so101_ahawam_padapter_v2/full_zero2_offload_ahawam_base/
    formal-padapterv2-sft-ahawambase-bs4-acc1-4ep-3gpu-20260901-001743/
    model/checkpoints/step_005328_trainer_state.json

只是训练进度元数据，**不是模型 checkpoint**。真正部署文件是：

    runs/so101_ahawam_padapter_v2/full_zero2_offload_ahawam_base/
    formal-padapterv2-sft-ahawambase-bs4-acc1-4ep-3gpu-20260901-001743/
    model/checkpoints/weights/step_005328.pt

服务器端已经把以下文件改为该权重、对应 stats、10-step 同步推理和正确任务文字：

- scripts/so101_pipeline/online_deploy/settings.yaml
- scripts/so101_pipeline/online_deploy/run_action_server.sh

服务器启动命令：

    bash scripts/so101_pipeline/online_deploy/run_action_server.sh

等待 Listening on 127.0.0.1:12000 后再启动本地 collector。服务使用 pickle，机械臂
电脑应通过 SSH tunnel 连接，不要将端口暴露到公网：

    ssh -N -L 12000:127.0.0.1:12000 USER@GPU_SERVER

本地 collector 仍连接 127.0.0.1:12000。

## 2. 控制电脑不要改成 Flash async 协议

本轮继续使用原先稳定的单 TCP connection、type=infer 同步接口：

    最新 fixed/wrist + 当前 6D state
        -> 同步 infer request
        -> 等待 8x6 action chunk
        -> 15 Hz 逐条执行
        -> FIFO 空时 SYSTEM_HOLD
        -> 用最新 observation 请求下一块

不要执行以下改动：

- 不要建立 Flash 文档要求的 image/action 双连接；
- 不要发送 type=image 或 type=action_request；
- 不要把 num_inference_steps 改成 2；
- 不要增加 RTC、prefetch、interpolation 或删除延迟帧；
- 不要重写已有 HIL 状态机和 recorder。

如果本地已经开始实现 Flash async，请保留在独立 mode 或 branch 中；当前正式采集配置
必须明确选择原来的同步模式，避免两套协议混用。

## 3. v3 控制和记录语义必须保持不变

### 3.1 POLICY

- server 返回完整 chunk 后才进入 FIFO；
- 每个 15 Hz tick 最多执行一个 action slot；
- 每 tick 都检查 takeover/ESTOP，chunk 不拥有控制权；
- 保存 policy proposal、实际执行 action、request/chunk/slot id 和 action age。

### 3.2 SYSTEM_HOLD

当 request 仍在推理且 FIFO 已空时：

- follower 继续保持最后一个实际发送的绝对 position target；
- 每个 15 Hz tick 仍保存 fixed/wrist、state 和一行 parquet；
- control_source=SYSTEM_HOLD；
- action 和 executed_action_arm 等于最后实际 follower target，不能写零；
- policy_action_valid=false、human_action_valid=false；
- action_supervision_mask=false；
- 不得删除、压缩或补插值伪造这段时间。

这些 HOLD 行用于真实时间轴和系统诊断，不作为 HUMAN-only action/video 监督。

### 3.3 HUMAN

人类 takeover 后：

- 立即清 action FIFO；
- policy_generation 加一，在途旧 reply 到达后必须丢弃；
- HUMAN 是 follower 的唯一写入者；
- control_source=HUMAN、intervention=true；
- human_action_valid=true、action_supervision_mask=true；
- action 必须是安全层裁剪后真正执行的 target，而不只是 leader 原始 proposal；
- 人工段内继续以 15 Hz 连续保存图像、state、action；
- 每次独立接管使用稳定且唯一的 takeover_segment_id。

resume policy 时必须清旧队列、作废旧 generation、remote hard reset，并从人类结束后的
最新 observation 请求新 chunk。不能继续使用接管前 server state。

## 4. 本轮只训练 HUMAN correction 时的精确定义

训练侧只把同时满足以下条件的行当作专家标签：

    human_valid =
        control_source == HUMAN
        and intervention
        and human_action_valid
        and action_supervision_mask
        and not dropped_tick

target 使用标准 action，而不是未裁剪的 human_action。连续段边界至少包括：

- raw episode 改变；
- takeover_segment_id 改变；
- frame_index 或 tick_index 不连续；
- dropped_tick=true；
- HUMAN valid 条件变为 false。

每个连续 HUMAN 段被视为一个新的逻辑训练 episode。不得让 video history、future video
或 action horizon 跨过 POLICY、SYSTEM_HOLD、leader alignment、pause 或 reset。

当前训练端已经实现这套 logical view：

- HUMAN 段长度大于等于 49：video + action supervision；
- HUMAN 段长度小于 49：action supervision，关闭直接 video loss；
- 段尾 future 不足时使用仓库已有 clamp/pad mask；
- POLICY 和 SYSTEM_HOLD 不进入公开训练索引。

对应实现与说明：

- scripts/so101_pipeline/hil_human_only_dataset.py
- docs/hil/AHAWAM_HIL_HUMAN_ONLY_TRAINING_IMPLEMENTATION_ZH.md
- docs/hil/SYSTEM_HOLD_TRAINING_HANDOFF_ZH.md

## 5. 是否需要 rich-v4-data-lite

对本轮的 demo + HUMAN correction BC/adapter，**不需要**。rich-v3 已足够稳定地构造
专家训练 view。

只有计划立即研究以下方法时，request ledger 和 full chunk ledger 才变成必要输入：

- 同一状态下 policy negative chunk 与 human positive chunk 的成对偏好学习；
- 未执行 policy proposal 的反事实监督；
- chunk-level offline RL 或 value；
- 精确研究 proposal age、slot execution 和异步 action alignment。

如果来得及，可以 append-only 保存完整 policy chunk 和每个 slot 的
EXECUTED、CLAMPED、SKIPPED_TAKEOVER、STALE，但不能为了它推迟当前采集，也不能改变
v3 逐帧数据语义。完全不加也不妨碍 HUMAN-only 主实验。

## 6. 唯一强烈建议补的小改动：provenance

旧 v3 manifest 中出现过：

    "policy_checkpoint_sha256": null

新 run 至少必须非空保存：

- checkpoint 的规范名称或绝对路径；
- checkpoint SHA-256；
- deploy config SHA-256；
- task instruction；
- server endpoint；
- num_inference_steps=10；
- chunk_size=8、horizon=40、fps=15；
- collector git commit、leader/follower calibration hash；
- contract_version=so101-ahawam-hil-rich-v3。

这只是 metadata/provenance 修复，不需要改 parquet schema。如果本地无法访问远端权重，
把服务器提供的 checkpoint SHA-256 作为显式 collector config 参数保存，不要继续留
null。

## 7. 采集前验收

正式批量采集前先录 2 条 pilot rollout，并验证：

1. run context 指向 step_005328.pt，而不是 trainer-state JSON 或旧 DAgger checkpoint；
2. policy task 是完整插头任务文字；
3. server 返回 8x6 finite action，动作/state 坐标没有重复转换；
4. FIFO 空时继续产生连续 SYSTEM_HOLD 行，mask 为 false；
5. takeover 在一个控制 tick 内阻止剩余 policy slot；
6. takeover 期间的旧 reply 为 stale 且零执行；
7. HUMAN 行四个条件全部成立，标准 action 等于实际安全执行 target；
8. HUMAN 段内 fixed/wrist/parquet 帧数一致且 tick 连续；
9. resume 发生 remote reset，旧 generation 不复活；
10. validator 无 error；gap 只能作为切段边界，不能被训练窗口跨越。

通过后即可继续采集，不需要等待异步、RTC 或 rich-v4 完成。

## 8. 给本地 Codex 的最终实施范围

本地 Codex 应先审计当前运行配置，然后只做：

1. 将 policy endpoint 和配置切回单连接同步 infer；
2. 确认任务文字、端口、10-step、8-action、15 Hz；
3. 保持现有 rich-v3 recorder 与 HIL generation fencing；
4. 为 run context/manifest 增加非空 checkpoint identity 和 SHA；
5. 录制并验证 2 条 pilot；
6. 报告改动文件、启动命令、validator 输出和 pilot 的 HUMAN/HOLD/POLICY 计数。

不要在本轮实现 Flash async、RTC、prefetch 或全量 rich-v4。
