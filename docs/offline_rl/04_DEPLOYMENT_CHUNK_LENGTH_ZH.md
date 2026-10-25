# 当前部署的生成长度、执行长度与 RL 动作块

核查日期：2026-10-24。配置为 `scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_action_dit_only.yaml`，仅做静态代码和已有日志审计，没有启动机械臂。

## 1. 当前配置不是“生成 40、执行前 8”

| 参数/数量 | 当前值 | 实际意义 |
| --- | ---: | --- |
| action_horizon | 40 | 一次 video/planner prefill 对应的动作布局跨度 |
| action_chunk_size | 8 | 每次 action phase 真正去噪、输出的动作数 |
| num_chunks | 5 | 40/8；同一 planner 周期内可处理 5 个动作请求 |
| serving.max_return_actions | 8 | 服务端返回上限；当前不会从模型返回的 8 中再裁掉尾部 |
| 常规客户端块长 | 8 | 已有 collector 配置/逐槽日志支持；接管、暂停或终止可以中断 |
| max_action_offset | 7 | 训练/模型支持的起点 offset 范围，不是“每次丢掉前 7 步” |

`AHAWAMPolicy.get_action` 每次获取最新观察、本体状态，调用一次 `_infer_action_chunk`，返回当前 8 步。达到第 5 块后，下次调用才刷新 planner。`infer_action_chunk` 从 horizon 张量切出 `[8*i:8*(i+1)]`，只对该片段运行动作去噪，并不是先把 40 步全部生成再取一小段。

可定位的代码：[配置](../../scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_action_dit_only.yaml)、[部署 wrapper](../../deploy/server/ahawam_policy.py)、[动作块实现](../../src/ahawam/models/wan22/ahawam_chunk_base.py)、[服务端裁剪](../../deploy/server/wam_policy_server.py)。

正常时序：

```text
planner prefill
  当前观察 -> action phase 0 -> 返回/执行 8 步
  新观察   -> action phase 1 -> 返回/执行 8 步
  新观察   -> action phase 2 -> 返回/执行 8 步
  新观察   -> action phase 3 -> 返回/执行 8 步
  新观察   -> action phase 4 -> 返回/执行 8 步
下一次请求刷新 planner
```

“同一 planner 周期”不等于五块共享同一张最新观测；每次 action 请求仍有最新观测驱动 OVCR。

## 2. 执行日志能证明到哪里

本仓库 `ahawam_tcp_client.py` 是通信封装和单次请求测试入口，不包含 Windows SO101 的完整机械臂执行循环，不能仅凭它证明每次都执行完。

已检查指定旧批次 `20260911_183057/recording_context/policy_chunks.jsonl`：194 个返回均为 `(8,6)`；其中属于 7 条已保存 episode 的有 155 块。按逐槽最终状态，将 `EXECUTED` 和 `CLAMPED` 都计为实际下发：149 块的 8 个槽位全部下发，6 块部分执行；剩余槽位有 10 个 `SKIPPED_TAKEOVER`、2 个 `STALE`。CLAMPED 是下发了经安全限幅的目标，不等于原始建议值原样执行。

这证明旧 collector 的常规模式是完整执行 8 步，而非固定每块只执行 4 步。例外来自接管/停止/成功结束等控制事件。

**该批次 checkpoint 是旧 SFT `step_006572`，不是当前 action-DiT-only `step_003285`。** 两者 chunk/horizon 协议相同，但旧日志不能冒充当前 checkpoint 的实际执行记录。对当前配置能确定的是：模型输出 8、服务端返回 8；若继续使用相同 collector 配置，则常规执行也是 8。

## 3. RLT 的参考长度和 RL 决策长度可以不同

RLT 用 `pi(A | z, proprio, A_ref)` 生成外部动作块，基础模型参考长度 H 与 RL 输出/执行长度 C 是不同概念。其 rollout 描述明确使用参考的 `A_ref[1:C]` 生成 C 步动作。参考是条件，不要求新动作与参考逐值相同，也不要求基础模型完整输出必须全部执行。[RLT §IV–V](https://arxiv.org/html/2604.23073v1)

本项目首版选择 C=8 的理由是已有参考块、执行协议、离线块边界三者自然一致；不是因为 YAML 中 action_horizon=40 就必须让 RL 输出 40 步。C=8 在 15 Hz 下约 0.533 秒有效动作时间；SYNC_HOLD 下还要加推理/通信等待。

更短 C 让策略更频繁闭环，但增加请求成本和等待；更长 C 可加快稀疏价值传播、保持连续行为，但延长开环时间、增加动作维度。应按实际耗时、对齐需求、基础模型接口和离线数据支持一起选。

## 4. 如果生成与执行不一致，RL 如何处理

若基础模型一次返回 H=8，计划只执行前 K=4：首版最清楚的定义是 actor 输出 4 步、reference 用前 4 步、critic 评价实际 4 步，next state 对齐执行完这 4 步后的观察。若把等待附在该动作上，reward/discount 还必须包含等待期间的时间与事件。

actor 也可以读完整 8 步 reference 作为额外上下文，但未执行的后 4 步不能被记录成执行事实。若采用“完整 planned command 是环境 action，执行器只用前缀”的另一种定义，也必须显式建模这种执行映射，并保证 Q 不错误依赖无效尾部。

特别注意本地 AHA wrapper 每次 action 请求都会让 `next_chunk_index += 1`。单独把 `max_return_actions` 改成 4，不会同步修改内部 8 步块推进、planner refresh 或数据合同；不能把它当成已经完成的 C=4 部署支持。需要一并明确是否取消尾部、何时重新 prefill/对齐，以及真实观测历史如何推进。

接管造成的随机中断与“固定只执行前 K 步”的常规 receding-horizon 协议不同。中途接管时仅保存实际执行前缀，不将人的后续成功归给未执行的 policy 尾部。当前离线原型对此的取舍见 [时序文档 §7](01_DATA_AND_TEMPORAL_CHOICES_ZH.md)。

目前建议保持参考/actor/常规有效执行长度均为 8，把是否附带 hold 作为另一项时序决策；未来比较 C=4 时，把执行器和内部缓存推进一起作为实验变量。
