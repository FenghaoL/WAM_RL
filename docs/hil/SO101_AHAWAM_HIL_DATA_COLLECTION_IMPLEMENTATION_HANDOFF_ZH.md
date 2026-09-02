# SO101 + AHA-WAM 在线接管数据采集：方法兼容性、代码复用判断与部署端实现交接

> 检索与本地代码审计日期：2026-08-27  
> AHA-WAM 工作区提交：`6af7e85`（工作区另有未提交实验配置，本文没有覆盖它们）  
> 使用对象：负责 SO101 follower、SO101 leader、双相机与 AHA-WAM 远程策略部署的本地电脑  
> 目标：先实现安全、可追溯、尽量与算法无关的在线接管采集系统；随后同一份原始数据可导出 HG-DAgger、IWR、HIL-SERL/RL、RLDG 或 RECAP-like 所需训练视图。

配套阅读：

- [从 π*0.6 / RECAP 到 SO101 插头插入的实验方案](../research/PI06_RECAP_TO_SO101_PLUG_EXPERIMENT_PLAN_ZH.md)
- [AHA-WAM 架构：从数学到代码](../architecture/AHAWAM_ARCHITECTURE_FROM_MATH_TO_CODE_ZH.md)
- [AHA-WAM RL 范围与 video/action 联合优化重审](../research/AHAWAM_RL_SCOPE_AND_VIDEO_ACTION_COTRAINING_REVIEW_ZH.md)

---

## 0. 最重要的结论

### 0.1 HIL-SERL、IWR、RLDG 的采集都等同于 HG-DAgger 吗

**不完全等同，但可以共用同一套“富标注”采集基础设施。**

它们共享的交互骨架是：

```text
模型自主运行
   ↓
人观察到即将失败、已经偏离或需要恢复
   ↓
实时暂停/覆盖模型控制
   ↓
人连续遥操作恢复并纠正
   ↓
交还当前策略，继续同一条 rollout
```

不同点主要不在“按钮怎样按”，而在采集后怎样使用数据：

| 方法 | 自主段 | 人类接管段 | reward/outcome | 核心训练方式 |
|---|---|---|---|---|
| 原始 HG-DAgger | 原始日志可留存，但不作为 expert target 聚合 | 作为 expert labels 聚合 | 非必需 | 初始 demo + intervention-only 模仿学习 |
| IWR | 保留为 robot dataset | 保留为 intervention dataset | 非必需 | 两类数据各占约 50% batch，抑制策略漂移 |
| HIL-SERL | 所有真实执行 transition 进入 RL replay | 接管 transition 还会进入 intervention/demo replay | 必需 | off-policy RL；人类数据帮助探索与 critic/actor 更新 |
| RLDG specialist 阶段 | 按 HIL-SERL 收集 | 按 HIL-SERL 接管 | 必需 | 先训练窄任务 RL specialist |
| RLDG distillation 阶段 | 主要保存收敛 specialist 的高质量自主 rollout | 不以接管段为主体 | 通常按成功筛选 | 用 RL 生成轨迹微调 VLA/generalist |
| RECAP-like | 完整保留 | 完整保留并特殊标记 | 必需 | 学 value/advantage，再训练 advantage-conditioned policy |

所以，回答“用它们的代码采出来能不能做 HG-DAgger”时必须加一个条件：

> 只有当 collector 没有过早丢弃自主段、没有丢失 intervention mask、没有把 policy proposal 与 human/executed action 混为一谈、并保留正确的 episode 边界时，原始数据才真正具有方法兼容性。

### 0.2 对当前项目的代码选择

建议采用：

> **复用 LeRobot v0.6 的 SO101 硬件/相机/数据集 API 与新 DAgger 状态机，保留当前可工作的 AHA-WAM TCP client，自己写一层很薄的仲裁、异步 action-chunk 和富标注 recorder 胶水。**

具体判断：

- **不建议 clone 原版 `rail-berkeley/hil-serl` 后硬改。**它的真机栈围绕 Franka、ROS/Flask、SpaceMouse、末端增量动作、JAX actor/learner 和 Gym 环境展开；对 SO101 关节位置控制与 AHA-WAM 远程 action chunk 来说耦合过深。
- **不建议从零重写 SO101 驱动、相机、校准和 leader/follower 交接。**LeRobot 已经提供这些能力。
- **不建议直接修改 pip 的 `site-packages` 或长期跟随 LeRobot `main`。**应固定一个经过真机 smoke test 的 tag/commit，并把本项目改动放在自己的目录。
- **不建议第一天就强行把 AHA-WAM 包成完整 LeRobot `PreTrainedPolicy`。**这是可选的长期工程化路径；当前更安全直接的是在现有部署 loop 中导入 LeRobot 硬件类与交接辅助函数。
- **不要直接使用旧的 LeRobot `rl.gym_manipulator` HIL-SERL 路径。**截至 2026-08-27，SO101 leader 缺少该路径要求的 `get_teleop_events()` 的问题仍有公开 issue/PR；LeRobot v0.6 新增的 `lerobot-rollout --strategy.type=dagger` 是另一条新路径，已经显式支持 `so_leader`。

| 候选代码 | 能否原样用于当前 AHA/SO101 | 应复用什么 | 主要局限 |
|---|---|---|---|
| 原版 HIL-SERL | 否 | intervention/replay 语义、RL 训练参考 | Franka + SpaceMouse + EE delta + JAX/ROS 栈 |
| IWR | 否 | robot/human 分区和 50/50 sampler | 没有面向当前 SO101/AHA 的通用 collector |
| RLDG | 否 | specialist→自主 rollout→VLA distillation 流程 | 官方仍让用户调用 HIL-SERL；最终数据采集说明不完整 |
| LeRobot 旧 `gym_manipulator` | 否 | 可读其 env/processor 设计 | SO101 leader 支持仍有 open issue/PR |
| LeRobot v0.6 新 DAgger | **部分，可以作为主基底** | SO101 硬件、相机、handover、事件、状态机、dataset API | 默认 action/schema、episode rotation 和 inference engine 不适配 AHA |
| 当前本地 AHA controller | **必须保留并扩展** | 已验证的 TCP、坐标变换、action prefetch/执行 | 尚缺 HIL arbiter、generation fencing 和富标注 recorder |

### 0.3 对 AHA-WAM 来说并非完全“与模型无关”

SO101 leader 的读取、按钮、相机和数据写盘大体与模型无关，但**控制切换并不与 AHA-WAM 无关**。当前 AHA-WAM 有四个必须专门处理的性质：

1. 策略一次返回 **8 个绝对关节动作**，不是一次一个动作；
2. 控制频率为 **15 Hz**，8 步约覆盖 **0.533 秒**；
3. TCP inference 是阻塞式的，接管不能等待 GPU 请求结束；
4. 服务端内部保存 video/action 推理状态，人在中间执行了不同动作后，不能继续沿用旧状态和旧 action queue。

因此接管瞬间必须在**部署电脑**上清空动作队列并让所有在途结果失效；交还模型前必须对远程 AHA policy 做 reset，并从最新观测重新请求动作。

---

## 1. 可直接发送给部署端 Codex 的任务说明

下面这一节可以连同本文全文直接交给部署端 Codex。部署端的实际目录、LeRobot 版本和已有 controller 可能与本工作区不同，因此要求它先审计再落代码，不能照猜测路径覆盖现有实现。

> 你正在 SO101 机械臂部署电脑上实现 AHA-WAM 在线人类接管数据采集系统。请先完整阅读本文，然后检查本机已有的 follower/leader/camera recorder、AHA-WAM TCP client、动作队列、LeRobot 版本、校准文件和数据格式。不要先启动上电运动，也不要直接覆盖现有可工作的 recorder。
>
> 目标不是实现 HG-DAgger 训练算法，而是实现一个安全且方法无关的 collector：AHA-WAM 自主运行，人按键或脚踏后应在一个 15 Hz 控制 tick 内停止继续执行任何排队的模型动作；SO101 leader 与 follower 平滑对齐后由人控制；交还模型时清空旧队列、使在途 inference 结果失效、调用 AHA server reset，并从最新观测重新 warm-start。每条逻辑 rollout 都要完整保存自主段、暂停/交接事件、接管段、交还后段与最终 outcome。
>
> 优先复用固定版本 LeRobot 的 `SOFollower`、`SOLeader`、相机、校准、`LeRobotDataset`、`teleop_smooth_move_to`/相关 handover 工具和 v0.6 `DAggerStrategy` 的状态机思想；不要 clone 并改造原版 Franka HIL-SERL。保留当前 AHA TCP client 和已有异步推理实现，只新增 `PolicyExecutor + ControlArbiter + HILStateMachine + RichRecorder` 胶水层。若本机已有同等组件，应小步扩展，而不是平行重写第二套控制器。
>
> 必须记录 `policy_action`、`human_action`、`executed_action`、各自 valid mask、`control_source`、`intervention`、逻辑 episode id、takeover segment id、policy generation/request/chunk/step id、单调时钟、双相机、实际 follower state、任务指令、checkpoint/config/calibration hash、终局 success/failure/timeout/abort 和 intervention reason。标准 `action` 必须始终等于当时真正发给 follower 的动作，并转换到当前 AHA 训练数据使用的 model/G05 坐标；同时另存 raw arm 坐标。
>
> 不要把 LeRobot v0.6 连续 DAgger 的“按视频文件大小自动轮转 episode”直接当任务 episode。当前插头任务必须有显式的逻辑 rollout 边界与 outcome；物理视频分片只能是存储细节。
>
> 实现顺序必须是：静态审计与版本冻结 → fake policy/fake robot 单元测试 → follower 不上电 dry-run → leader 单独对齐测试 → 限速/限幅空载测试 → 无物体的接管/交还测试 → 最后才是真实插头。每一阶段输出测试结果和仍未确认的假设。任何时刻硬件急停都必须独立于 Python 按键与网络。

部署端 Codex 最终应交付：

1. 本机审计报告与版本/路径清单；
2. 可运行 collector、配置文件和启动脚本；
3. fake robot/policy 测试；
4. 一条 30–60 秒 dry-run 数据及 schema validator 输出；
5. 队列失效、在途 reply 丢弃、remote reset 和 handover latency 的日志证据；
6. 数据导出/转换脚本或明确的后续转换接口；
7. 启动前 checklist、操作键位和紧急停止说明。

---

## 2. 文献和代码真正做了什么

### 2.1 原始 HG-DAgger

[HG-DAgger 原文](https://arxiv.org/abs/1810.02890)不是“先让模型失败一次，下一条轨迹提前一秒复现并接管”。它的定义是 live human gate：

- novice policy 真正在线控制；
- 人持续观察当前 rollout；
- 人认为系统离开安全/可恢复区域时立即接管；
- 人连续控制，直到回到适合交还的状态；
- 只把 human intervention/recovery labels 聚合到 expert dataset；
- 更新 policy 后重复采集。

它原始实验不是 SO101 操作，因此不能提供可直接使用的 leader/follower 驱动，但它定义了“什么数据才叫 HG-DAgger”。

### 2.2 IWR

[IWR 论文](https://arxiv.org/abs/2012.06733)和[项目页](https://sites.google.com/stanford.edu/iwr)使用与 HG 很接近的在线接管过程。关键差别在训练数据消费：

- 一条 trajectory 中同时有 robot 与 human samples；
- HG baseline 丢弃 robot samples，只学 intervention；
- IWR 把它们分为 $D_R$ 和 $D_I$；
- 每个训练 batch 对两类近似等比例采样。

因此，只要完整保存 mixed trajectory 和每帧来源，同一批采集可以公平构造 HG 与 IWR 视图。若 collector 默认只保存纠正窗口，IWR 所需的 robot regularization 数据就已经不可恢复。

### 2.3 原版 HIL-SERL

[HIL-SERL 论文](https://arxiv.org/abs/2410.21845)、[官方仓库](https://github.com/rail-berkeley/hil-serl)和[Franka walkthrough](https://github.com/rail-berkeley/hil-serl/blob/main/docs/franka_walkthrough.md)公开了真机 RL 的 actor/learner、replay buffer、reward classifier 和 SpaceMouse intervention 流程。

官方 [`examples/train_rlpd.py`](https://github.com/rail-berkeley/hil-serl/blob/main/examples/train_rlpd.py)的核心语义是：

```python
if "intervene_action" in info:
    actions = info.pop("intervene_action")

transition["actions"] = actions
data_store.insert(transition)

if already_intervened:
    intvn_data_store.insert(transition)
```

含义是：

- 人接管时，transition 的 `actions` 被替换为真正执行的人类 action；
- 所有 transition 进入主 RL buffer；
- intervention transition 额外进入 intervention/demo buffer。

这足以用于 HIL-SERL，也能导出 HG 的 human labels。不过它默认不会保留“同一时刻本来由策略提出、但未执行的 counterfactual policy action”。本项目为了研究复用性，应额外保存 `policy_action` 与 valid flag。

HIL-SERL 的官方建议也说明接管服务于 RL 探索：初期可以先让随机/不成熟策略探索若干步，再把物体引导到插入口附近，让 RL 在有价值区域练习。它与 HG 的“只把接管动作当 expert BC labels”不是同一个学习目标。

### 2.4 RLDG

[RLDG 论文](https://arxiv.org/abs/2412.09858)和[官方复现实验说明](https://generalist-distillation.github.io/RLDG/)分两段：

1. 用标准 HIL-SERL recipe、20 条初始 demo 和人类 intervention 训练窄任务 specialist；
2. specialist 收敛后自主 rollout，生成高质量轨迹，再微调 OpenVLA/Octo 等 generalist。

Connector insertion 的官方说明甚至要求为 USB-A、VGA、Ethernet 分别训练 specialist。其项目页的“Collecting Data”仍标为 coming soon，所以不能声称 RLDG 发布了一个比 LeRobot 更完整的通用 SO101 collector。

结论：

- RLDG 前半段采集可以借鉴 HIL-SERL；
- RLDG 最终给 VLA 的 distillation dataset 不是 HG intervention dataset；
- 当前 collector 应保留失败、接管和完整 outcome，不能在 raw 层直接只留成功；
- 将来训练出 RL specialist 后，再派生 `successful_specialist_rollouts` 视图。

### 2.5 LeRobot 中容易混淆的两条路径

#### 旧路径：`rl.gym_manipulator` / HIL-SERL

旧文档曾把 SO101 leader 写成可用设备，但公开 [issue #2952](https://github.com/huggingface/lerobot/issues/2952)显示 `SOLeader` 没有该路径要求的 `get_teleop_events()`；截至审计日，[PR #3086](https://github.com/huggingface/lerobot/pull/3086)仍是 open。该 PR 还需要补 joint key 校验、跳过 IK、当前姿态 neutral action 等逻辑，说明这并非只差一个按键函数。

因此不要把旧 HIL-SERL CLI 当成当前 SO101 的可靠 drop-in。

#### 新路径：v0.6 `lerobot-rollout --strategy.type=dagger`

LeRobot v0.6 发布了新的 [`DAggerStrategy`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/rollout/strategies/dagger.py)和[HIL collection 文档](https://github.com/huggingface/lerobot/blob/v0.6.0/docs/source/hil_data_collection.mdx)。它与旧 `gym_manipulator` 是不同实现，明确列出 `so_leader`：

```text
AUTONOMOUS --pause--> PAUSED --correction--> CORRECTING
     ^                                            |
     +----------- resume <--- PAUSED <------------+
```

它值得直接复用的能力包括：

- keyboard 或 foot pedal 事件；
- policy pause/resume；
- SO101 leader 的 `feedback_features`、`enable_torque()`、`disable_torque()` 与 `send_feedback()`；
- 自主到暂停时，把有电机的 leader 平滑移动到 follower 当前位置；
- 暂停期间 hold follower；
- 交还时 reset inference engine/interpolator；
- 每帧 `intervention=True/False`；
- 相机、硬件 processor 和 LeRobotDataset 写盘。

它的两个默认 recording mode 都不完全满足本项目：

| v0.6 mode | 行为 | 对本项目的不足 |
|---|---|---|
| `record_autonomous=False`（默认） | 只保存 correction；每次 correction 是一个 episode | 丢失 IWR/RECAP 所需自主段与真实逻辑 rollout 上下文 |
| `record_autonomous=True` | 自主与接管都保存；按目标视频文件大小/时间自动轮转 episode | 存储分片不等于插头任务的 success/failure 逻辑 episode |

此外，v0.6 默认一帧只保存一个 `action`：自主时是 policy action，接管时是 human action，再加一个 intervention flag。它没有完整保存三种 action 和在途 chunk 元数据。因此应复用实现结构，并扩展 recorder，而不是完全原样运行。

---

## 3. 当前 AHA-WAM/SO101 部署事实

### 3.1 现有远程接口

当前 robot-side client 是 [`ahawam_tcp_client.py`](../../scripts/so101_pipeline/online_deploy/ahawam_tcp_client.py)：

- 采用 length-prefixed pickle TCP；
- `health()`、`reset(instruction)`、`infer()`、`video_preview()`、`infer_with_video()`；
- 输入 6D state 和 fixed/wrist RGB；
- server 返回 action chunk；
- pickle 只能放在可信私网或 SSH tunnel 中，绝不能直接暴露到不可信网络。

SO101 follower 与 model/G05 坐标变换是：

$$
q_{model}=s\odot q_{arm}+b,
$$

$$
q_{arm}=(q_{model}-b)\odot s,
$$

其中：

```python
s = [1, -1, 1, 1, 1, 1]
b = [0, 90, 90, 0, 0, 0]
```

部署端实现必须调用同一份经过测试的函数，不能在 recorder、policy adapter 和 executor 中各复制一份“看起来一样”的常量。

### 3.2 当前动作时间尺度

[`settings.yaml`](../../scripts/so101_pipeline/online_deploy/settings.yaml)目前配置：

- `action_horizon: 40`；
- `action_chunk_size: 8`；
- server 最多返回 8 个动作；
- `input_control_hz: 15.0`。

所以一个 action chunk 的执行时间为：

$$
T_{chunk}=\frac{8}{15}\approx0.533\text{ s}.
$$

这就是为什么不能只在 chunk 边界检查接管按钮。仲裁器必须每个 15 Hz tick 都检查控制来源，policy chunk 只是本地 FIFO 的数据来源，不拥有机器人控制权。

### 3.3 AHA 服务端是有状态的

[`AHAWAMPolicy.get_action()`](../../deploy/server/ahawam_policy.py)读取 `next_chunk_index`；每次 action 推理会沿用预填充 video 状态。达到配置的 chunk 数后才触发新的 planner/video refresh。

[`_hard_reset()`](../../deploy/server/ahawam_policy.py)会清理：

- model history；
- `_inference_state`；
- episode prefill 状态；
- video preview/history bookkeeping。

若人在 chunk 中途接管，服务端并不知道原计划的剩余动作没有执行，也不知道人执行了另一条轨迹。因此交还后必须 hard reset，再由最新 observation 重建状态。

### 3.4 当前工作区没有完整 robot-side collector

本工作区能看到 TCP client、GPU server 和部署配置，但没有发现实际部署电脑上的：

- follower 的 15 Hz 主控制 loop；
- action FIFO/prefetch 实现；
- camera recorder；
- leader 接管；
- `policy_action/human_action/executed_action/intervention` 字段。

所以本文只能规定接口语义。部署端 Codex 必须先找到本机真正调用 `robot.send_action()` 的唯一位置，再在那里加入仲裁；不能在旁边新增第二个同时发命令的线程。

### 3.5 数据格式版本不能假定兼容

现有 AHA 数据是 vendored LeRobot `v2.1`，15 Hz，字段为：

- `action: [6]`；
- `observation.state: [6]`；
- `observation.images.fixed`；
- `observation.images.wrist`；
- timestamp/index/task 元数据。

AHA reader 也在本仓库内固定了 `CODEBASE_VERSION = "v2.1"`。LeRobot v0.6 的当前 dataset writer/目录布局不能被假定为可由这个 reader 直接读取。

因此采集系统要区分：

```text
raw collection dataset（部署端当前可靠 writer）
           ↓ validator / converter
AHA training view（现有 v2.1 + model/G05 action）
           ↓ method-specific manifest/mask
HG / IWR / RECAP / RL view
```

不能在采完几十条真机轨迹后才第一次验证转换。

### 3.6 开始采集前必须修正任务指令混杂

现有训练数据 task 是：

```text
Remove the power adapter from the holder and plug it into the power strip.
```

但当前 [`run_action_server.sh`](../../scripts/so101_pipeline/online_deploy/run_action_server.sh)中的默认 `INSTRUCTION` 仍是：

```text
Pick up the white block.
```

client 的 `reset(instruction)` 可以覆盖 server 初值，但正式 collector 不应依赖“之后也许覆盖”。启动 preflight 必须打印并检查：

- server startup instruction；
- client reset instruction；
- recorder task；
- text embedding cache 命中的 exact prompt。

四者必须指向同一插头任务。否则采集得到的是带语言条件混杂的数据。

---

## 4. 推荐的目标架构

```text
                         ┌────────────────────────────┐
fixed / wrist cameras ──>│ ObservationSampler @15 Hz │
follower actual state ──>│ monotonic timestamp       │
                         └─────────────┬──────────────┘
                                       │ obs_t
                         ┌─────────────▼──────────────┐
                         │       ControlArbiter       │
leader action ──────────>│ AUTO / PAUSED / HUMAN     │
                         │ sole owner of send_action  │
                         └───────┬───────────┬────────┘
                                 │           │
                     executed_t │           │ observation snapshot
                                 │           ▼
                                 │   ┌──────────────────────┐
                                 │   │ AHA PolicyExecutor   │
                                 │   │ TCP + chunk FIFO     │
                                 │   │ generation fencing   │
                                 │   │ reset + prefetch     │
                                 │   └──────────┬───────────┘
                                 │              │ policy proposal
                                 ▼              ▼
                         ┌────────────────────────────┐
                         │       RichRecorder         │
                         │ obs / 3 actions / sources  │
                         │ events / outcome / hashes  │
                         └─────────────┬──────────────┘
                                       │
                    ┌──────────────────┼──────────────────┐
                    ▼                  ▼                  ▼
              HG export          IWR export        RL/RECAP export
```

### 4.1 组件责任边界

#### `ObservationSampler`

- 每 tick 读取一次 follower 实际 state；
- 获取 fixed/wrist frame 和各自 capture timestamp；
- 生成统一 `obs_id`；
- 不决定 action，不写机器人。

#### `AHAWAMPolicyExecutor`

- 包装已有 `AHAWAMTCPClient`；
- 将 arm state 转为 model/G05；
- 后台请求 8-step chunk；
- 维护 chunk FIFO、request/chunk id 和 generation；
- 收到过期 reply 只记录 `discarded_stale_reply`，不得执行；
- 提供 `invalidate()` 与 `reset_then_prime(latest_obs)`。

#### `HILStateMachine`

- 消费 keyboard/pedal edge events；
- 只做合法状态迁移；
- 记录 pause、takeover、release、abort reason；
- 不直接写数据帧或 TCP。

#### `ControlArbiter`

- 是全进程唯一允许调用 `follower.send_action()` 的对象；
- AUTO 选 policy action；
- HUMAN 选 leader action；
- PAUSED/WARMING 发送当前 hold target；
- 做最终 joint limits、速度限制、finite check；
- 返回实际发送的 `executed_action` 给 recorder。

#### `RichRecorder`

- 保存每 tick 的 observation、候选 action、最终 action 和来源；
- 保存异步 event log 与 episode manifest；
- 先写临时/in-progress 标记，episode 正常结束后原子 finalize；
- crash 后能识别并恢复/隔离未完成 episode。

---

## 5. 接管状态机

### 5.1 推荐状态

```text
IDLE
  │ start episode
  ▼
AUTONOMOUS
  │ pause pressed
  ▼
PAUSING ── invalidate policy generation, flush FIFO, hold follower
  │ leader aligned to follower
  ▼
PAUSED
  │ correction pressed
  ▼
HUMAN_CONTROL
  │ correction pressed
  ▼
PAUSED_AFTER_HUMAN
  │ resume pressed
  ▼
POLICY_WARMUP ── remote reset + latest observation + fresh chunk
  │ fresh chunk accepted
  ▼
AUTONOMOUS
```

任何控制态都允许进入：

```text
SAFETY_STOP / ABORTED
```

但软件 `SAFETY_STOP` 不是机械急停的替代品。

### 5.2 为什么保留 pause 和 correction 两步

SO101 leader 在人未控制时未必与 follower 姿态一致。如果一按按钮就直接把 leader 当前关节值发给 follower，机械臂会跳变。

LeRobot v0.6 的做法更安全：

1. pause 立即禁止后续 policy action；
2. follower hold；
3. 给 leader 上适当 torque，并用 `send_feedback()` 平滑移动到 follower 当前姿态；
4. 对齐完成后进入 PAUSED；
5. operator 再确认 HUMAN_CONTROL，leader 解除 torque/可自由操纵。

“pause 两步”不会损害 HG 语义，因为第一步已经实时阻止错误模型动作；第二步只是安全交接。真正的硬件碰撞风险仍由独立急停处理。

这里的“一个 tick 内停止”精确指**软件不再发送新的旧 policy target**。已经送进电机控制器的最后一个位置目标及机械惯性不可能被 Python 撤回；进入 PAUSING 时应锁存一个经过限幅的 `hold_target_arm`，并依赖已验证的底层速度/力矩限制。它不能替代硬件急停或接触安全设计。

### 5.3 推荐输入设备

第一版可以沿用 v0.6 默认语义：

| 输入 | 建议功能 |
|---|---|
| `Space` 或踏板 1 | AUTO ↔ PAUSED；首先停止 policy |
| `Tab` 或踏板 2 | PAUSED ↔ HUMAN_CONTROL |
| `S` | 结束逻辑 episode，标记 success |
| `F` | 结束逻辑 episode，标记 failure |
| `T` | 结束逻辑 episode，标记 timeout |
| `R` | 放弃当前未 finalize episode，并显式记录 rerecord reason |
| `Esc` | 软件 abort；不替代物理急停 |

建议最终使用脚踏执行 pause/correction，让操作者双手留在 leader 上。必须使用 key-down edge 去抖；持续按键不能每 tick 来回切换。

### 5.4 intervention 的精确定义

```text
intervention = (control_source == HUMAN)
```

以下不能混为 intervention：

- PAUSED 中保持当前位置；
- leader 对齐过程；
- software safety clamp 改写动作；
- policy warmup 等待；
- 机器人 reset 到初始位置。

它们应有各自的 `control_source/phase/event`，默认从训练数据中排除。

---

## 6. AHA action queue 与在途 inference 的正确处理

### 6.1 接管时最危险的错误实现

错误例子：

```python
chunk = client.infer(...)["action_chunk"]
for action in chunk:
    robot.send_action(action)
```

若循环内部不检查接管事件，人按下按钮后最多还会执行剩余 7 步，也就是接近半秒的旧动作。对插孔附近已经足以碰撞、卡边或把插线板推走。

### 6.2 generation fencing

每次 pause、takeover、release、episode reset 都增加 `policy_generation`：

```python
class PolicyExecutor:
    generation = 0
    fifo = deque()
    inflight = None

    def invalidate(self, reason):
        self.generation += 1
        self.fifo.clear()
        self.need_remote_reset = True
        log_event("policy_invalidated", generation=self.generation, reason=reason)

    def start_request(self, obs):
        request_generation = self.generation
        request_id = next_id()
        self.inflight = submit_blocking_tcp_request(obs)
        self.inflight.meta = (request_generation, request_id)

    def accept_reply(self, reply):
        request_generation, request_id = self.inflight.meta
        if request_generation != self.generation:
            log_event("stale_reply_discarded", request_id=request_id)
            return
        self.fifo.extend(tag_each_step(reply["action_chunk"], request_id))
```

generation 的作用不是取消 GPU 计算，而是保证旧结果永远不会再次取得机器人控制权。

### 6.3 阻塞 TCP 请求不能作为接管阻塞条件

当前 client 的 `request()` 是同步阻塞调用。接管按钮必须由控制线程/事件线程处理，不能等待 policy thread 返回。

推荐：

- 一个专用 policy worker thread 独占 persistent TCP socket；
- 控制主线程只消费线程安全的 action FIFO；
- pause 时主线程立即清 FIFO 并切 HOLD/HUMAN；
- in-flight reply 返回后由 generation 检查丢弃；
- policy worker 串行发送 `reset` 和下一次 `infer`。

第一版不建议用“接管就强行关闭 socket”来模拟 cancel：这会增加 server/runtime 状态、重连和半包处理的不确定性。若以后实测单次 inference 长到影响交还，可以再设计明确的 server cancel/session protocol。

### 6.4 交还模型的顺序

必须是：

```text
停止 human action
    ↓
follower hold 当前实际姿态
    ↓
generation++，清空任何旧 FIFO
    ↓
等待/丢弃旧 in-flight reply
    ↓
client.reset(exact_task_instruction)
    ↓
用最新 fixed/wrist/state 发起 fresh infer
    ↓
校验 reply generation、shape、finite、limits
    ↓
只有 fresh chunk ready 后恢复 AUTONOMOUS
```

不要为了减少等待而在 fresh chunk 到达前恢复旧 queue。AHA 首次 prefill 的约数百毫秒里保持当前位置是可接受的；执行错误旧动作不可接受。

### 6.5 background prefetch

当前推理耗时与 8-step chunk 时长接近时，需要在 FIFO 尚未耗尽前预取下一 chunk。实现应配置：

- `low_watermark_steps`；
- `max_fifo_steps`；
- 每次只允许一个 in-flight request；
- reply 带 request/chunk/generation；
- FIFO 不得跨 intervention/release；
- 若 FIFO 耗尽而 fresh reply 未到，进入 HOLD，而不是重复最后一个可能仍在移动的目标。

第一版建议用已有 controller 当前已经验证过的 prefetch 时序，只加 fencing；不要在接管任务里同时大改平滑算法。

---

## 7. leader/follower 平滑交接

### 7.1 可复用的 LeRobot v0.6 能力

[`SOLeader`](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/teleoperators/so_leader/so_leader.py)已经提供：

- `action_features`；
- `feedback_features`；
- `get_action()`；
- `enable_torque()` / `disable_torque()`；
- `send_feedback()`。

新 DAgger strategy 使用 `teleop_smooth_move_to()` 把 leader 移到 follower 姿态。建议直接调用固定版本的 helper；若必须复制实现，要保留 Apache-2.0 attribution，并写版本来源。

### 7.2 必做校验

在 leader 能控制 follower 前，fail fast 检查：

1. 六个 motor key 集合完全一致；
2. key 顺序明确固定，不能依赖两个 dict “碰巧同序”；
3. 两端 normalization/calibration mode 一致；
4. follower 当前姿态映射到 leader feedback 后在 joint limits 内；
5. gripper range 与其他五关节不同，不能统一按角度处理；
6. leader 对齐最大速度/时间受限；
7. 对齐时 follower 必须 hold；
8. 操作者已知 leader 会主动移动，手不得阻挡。

### 7.3 人类 action 的语义

SO101 leader 输出的是关节位置 target。每个 tick 应经过同一 robot action processor、限位和必要的速度限制后，才成为：

```text
human_action_raw
      ↓ processor / safety
human_action_arm
      ↓ arm_to_model
human_action_model
      ↓ arbiter selection
executed_action
```

必须区分 raw human action 与安全处理后真正发送的 action。HG expert target 默认应使用真正执行且与观测对齐的 `executed_action_model`，不是未经限幅的 raw input。

---

## 8. 原始数据 schema

### 8.1 设计原则

1. `action` 永远表示从当前 observation 之后真正发给 follower 的 target；
2. 候选 action 与最终 executed action 分开；
3. 无效候选不能用一个合法的全零 action 冒充，要有 valid mask；
4. raw arm 坐标和 AHA model/G05 坐标都保留；
5. 逻辑 episode 与视频物理分片分离；
6. 原始数据不因失败、无人接管或动作质量差而删除；
7. 训练筛选通过 manifest/export 完成；
8. 使用 monotonic clock 判断先后和延迟，同时保留 wall clock 便于跨进程日志对齐。

### 8.2 每帧必要字段

下面是目标语义，不要求所有字段都放在同一 parquet；高频张量可放 LeRobotDataset，异步事件和字符串可放 JSONL sidecar。

| 字段 | 类型/shape | 说明 |
|---|---|---|
| `observation.state` | float32 `[6]` | AHA 训练使用的 model/G05 state |
| `observation.state_arm` | float32 `[6]` | follower API 读到的 arm 坐标 |
| `observation.images.fixed` | video/image | 原始固定相机 RGB |
| `observation.images.wrist` | video/image | 原始腕部相机 RGB |
| `action` | float32 `[6]` | **实际执行**的 model/G05 target；AHA canonical target |
| `executed_action_arm` | float32 `[6]` | 实际发送给 follower 的 arm target |
| `policy_action` | float32 `[6]` | 当前 tick 对应的 model/G05 policy proposal |
| `policy_action_valid` | bool `[1]` | HUMAN/PAUSED 时通常为 false，除非明确运行 counterfactual policy |
| `human_action` | float32 `[6]` | 处理后的 model/G05 human proposal |
| `human_action_valid` | bool `[1]` | 只有 HUMAN 时为 true |
| `control_source` | uint8 `[1]` | `0=policy,1=human,2=hold,3=safety,4=reset` |
| `phase` | uint8 `[1]` | 细分 AUTO/PAUSE/ALIGN/HUMAN/WARMUP 等 |
| `intervention` | bool `[1]` | 严格等于 source 是否为 human |
| `safety_clamped` | bool `[1]` | action 是否被最后安全层改写 |
| `policy_generation` | int64 `[1]` | queue fencing generation |
| `policy_request_id` | int64 `[1]` | 远程请求 id；无效时用 sentinel + valid |
| `policy_chunk_id` | int64 `[1]` | reply/chunk id |
| `action_index_in_chunk` | int16 `[1]` | 0–7 |
| `takeover_segment_id` | int32 `[1]` | 同一连续 human segment 共享 id；非 human 为 -1 |
| `logical_episode_id` | int64/string | 任务 rollout id，不能由视频分片代替 |
| `tick_index` | int64 `[1]` | 逻辑 episode 内连续 tick |
| `monotonic_ns` | int64 `[1]` | 决策 tick 单调时钟 |
| `wall_time_ns` | int64 `[1]` | 跨进程审计用 |
| `camera_fixed_age_ms` | float32 `[1]` | tick 与 fixed capture 的时间差 |
| `camera_wrist_age_ms` | float32 `[1]` | tick 与 wrist capture 的时间差 |
| `policy_action_age_ms` | float32 `[1]` | action proposal 从 observation/request 到执行的 age |
| `loop_dt_ms` | float32 `[1]` | 实际控制周期 |
| `dropped_tick` | bool `[1]` | deadline miss/丢帧诊断 |

没有有效 policy/human action 时可存零向量，但任何消费者必须先检查 valid mask；不要让这些零进入统计量或监督 loss。

### 8.3 event log

`events.jsonl` 至少记录：

```json
{
  "event_id": 42,
  "logical_episode_id": "20260827_run_0007",
  "monotonic_ns": 123456789,
  "event": "takeover_requested",
  "old_phase": "AUTONOMOUS",
  "new_phase": "PAUSING",
  "policy_generation": 9,
  "reason": "coarse_alignment_error"
}
```

事件类型建议包括：

- `episode_started`；
- `policy_request_started/replied/failed`；
- `policy_invalidated`；
- `stale_reply_discarded`；
- `pause_requested`；
- `leader_alignment_started/completed/failed`；
- `human_control_started/stopped`；
- `policy_remote_reset_started/completed`；
- `fresh_policy_ready`；
- `safety_clamp`；
- `deadline_miss`；
- `episode_finalized`；
- `episode_aborted`。

### 8.4 episode manifest

每条逻辑 rollout finalize 后写：

```json
{
  "logical_episode_id": "20260827_run_0007",
  "task": "Remove the power adapter from the holder and plug it into the power strip.",
  "outcome": "success",
  "termination": "operator_success_key",
  "initial_condition_bin": {
    "plug_position": "left",
    "plug_yaw": "medium",
    "power_strip_position": "center_jitter"
  },
  "num_ticks": 344,
  "num_intervention_segments": 2,
  "num_intervention_ticks": 61,
  "intervention_reasons": ["grasp_offset", "fine_alignment_error"],
  "policy_checkpoint_sha256": "...",
  "deploy_config_sha256": "...",
  "collector_git_commit": "...",
  "lerobot_commit": "...",
  "leader_calibration_sha256": "...",
  "follower_calibration_sha256": "...",
  "camera_config": {"fixed": "...", "wrist": "..."},
  "notes": ""
}
```

`outcome` 至少区分：

- `success`；
- `failure`；
- `timeout`；
- `safety_abort`；
- `operator_abort`；
- `system_error`；
- `rerecord_discarded`。

不要把所有非 success 都压成 failure；RECAP/value、系统可靠性和论文统计会需要这些区别。

### 8.5 暂停帧是否保存

推荐：

- 控制训练主表不保存长时间 PAUSED 重复帧；
- event log 保存 pause 起止和时长；
- 若需要诊断 leader alignment，可写单独 diagnostic stream；
- HOLD/WARMUP 超过阈值时写事件，不用制造大量重复 action target。

这样既不扭曲数据分布，也能完整审计接管延迟。

---

## 9. 从同一原始数据导出不同算法视图

### 9.1 HG-DAgger view

训练集由：

```text
原始人工 demonstrations 的全部有效帧
                  +
HIL rollout 中 intervention=True 的 human/executed action labels
```

组成。自主 policy action 可以留在 raw dataset，但不应被静默当作 expert positive target。

### 9.2 IWR view

构造两个 sampler/index：

```text
D_I = intervention=True 且 human_action_valid
D_R = control_source=policy 且 policy_action_valid
```

训练 batch 近似 50/50。`D_R` 的 target 是当时真正执行的 policy action，它的作用是 regularize 到上轮策略分布，不代表这些动作都成功。

### 9.3 HIL-SERL / off-policy RL view

每帧生成 transition：

```text
(obs_t, executed_action_t, reward_t, obs_{t+1}, done_t)
```

所有 transition 进主 replay；intervention transition 额外进入 intervention/demo replay。必须使用 `executed_action`，不能在接管期间误写 policy proposal。

当前 collector 暂时没有 reward classifier 也没关系，但必须保存 outcome、图像和正确 terminal，后续才能离线补 reward 或构建 classifier。

### 9.4 RLDG view

分清两个数据集：

- `specialist_training_raw`：上述 HIL/RL 原始 transitions；
- `specialist_distillation`：收敛 RL specialist 的自主 rollout，按论文协议决定成功筛选和任务变体。

不要把 human intervention frames 直接命名为 `RLDG data`。

### 9.5 RECAP-like view

需要：

- 完整逻辑 rollout；
- success/failure/timeout；
- intervention mask；
- autonomous 与 human 段；
- policy/checkpoint iteration；
- 后续 value/advantage label。

所以 RECAP 是本 schema 最严格的下游消费者。若 collector 能满足它，通常也能导出 HG/IWR；反过来只存 correction-only 则做不到。

---

## 10. AHA-WAM 的 40-step action target 与 intervention mask

### 10.1 为什么“把每段接管裁成一条 episode”并不完全解决问题

AHA 数据加载器从某个 anchor 取未来 40 个 action：

$$
A_t=[a_t,a_{t+1},\ldots,a_{t+39}].
$$

真实 mixed rollout 可能是：

```text
robot robot | human human human ... | robot robot ...
```

那么一个 40-step target 很容易跨控制来源边界。若直接把整个 action chunk 当 expert target，就会把自主错误动作混入 HG supervision；若把每段 human correction 裁成独立 episode，又会丢掉接管前视觉上下文，并产生大量 padding。

### 10.2 推荐的长期实现：per-step `action_supervision_mask`

dataset 对与 action 同样的 delta timestamps 读取未来 intervention/source，形成：

```text
action_supervision_mask: [40]
```

HG 时：

```python
mask = intervention_future & ~action_is_pad
```

IWR 时依据当前 sampler 和 future source 构造相应 mask/权重。

当前 [`_compute_weighted_action_loss()`](../../src/ahawam/models/wan22/ahawam.py)已经先得到 `[B, T]` 的逐时间步 MSE，但只支持 `action_is_pad`，尚未接收 expert/source mask。因此未来改动很局部：

```python
valid = (~action_is_pad) & action_supervision_mask
loss = (loss_token * scheduler_weight * valid).sum() / valid.sum().clamp(min=1)
```

同时 sampler 应避免产生 `valid.sum()==0` 的 anchor；不能只靠 clamp 后让全空样本贡献零 loss 而占用大量 batch。

### 10.3 第一版不改训练代码时的保守导出

若想先快速验证：

- 保留完整 raw mixed rollout；
- 另外导出 correction-only episode；
- 只选择至少有足够连续 human steps 的窗口；
- episode 末尾 padding 由现有 `action_is_pad` 处理；
- 明确报告该视图丢失接管边界上下文，不能宣称它与 per-step masked HG 完全等价。

不要为了省事把自主段和人类段全都当正向 SFT action。

### 10.4 video loss 的额外选择

AHA-WAM 同时训练 future video。对 intervention rollout，需要在实验方案中明确：

- 仅更新 ActionDiT，video branch 冻结；
- 或完整 episode 更新 video，学习“真实执行后”的未来；
- 或只在 human/positive window 更新 video。

这不是 collector 现在必须决定的事，所以 raw 数据应保留完整视频与 source mask，把选择留给训练消融。

---

## 11. 推荐的本地代码组织

部署端应按本机项目调整路径，但建议职责类似：

```text
robot_deploy_repo/
├── pyproject.toml / uv.lock
├── third_party/
│   └── lerobot/                 # 固定 tag/commit，只读参考或 editable install
├── src/ahawam_hil/
│   ├── config.py
│   ├── events.py                # keyboard/pedal edge events
│   ├── state_machine.py
│   ├── policy_executor.py       # TCP, FIFO, generation fencing
│   ├── arbiter.py               # sole send_action owner
│   ├── handover.py              # 调用 LeRobot smooth helper
│   ├── transforms.py            # 唯一 arm<->model 实现
│   ├── recorder.py
│   ├── schema.py
│   ├── collector.py
│   └── validators.py
├── configs/
│   └── plug_hil.yaml
├── scripts/
│   ├── run_plug_hil.sh
│   ├── validate_hil_dataset.py
│   └── export_ahawam_v21.py
└── tests/
    ├── test_state_machine.py
    ├── test_generation_fencing.py
    ├── test_action_transform.py
    ├── test_recorder_schema.py
    └── test_fake_end_to_end.py
```

### 11.1 是否 clone LeRobot

推荐在部署端：

1. 先记录当前 `lerobot-info`、Python、OS、相机和 motor bus 版本；
2. 新建隔离环境，不要原地升级已工作的采集环境；
3. clone LeRobot 并 checkout 固定 tag/commit，或安装固定 tag；
4. 对 v0.6.0/v0.6.1 候选分别做 SO101 和 camera smoke test；
5. 在 lockfile/manifest 中记录最终 commit；
6. 本项目代码导入它，不直接改它；
7. 若确实要复制 `dagger.py` 片段，保留 Apache-2.0 header 和 upstream commit。

为什么 clone 固定版本比只 `pip install` 更适合当前阶段：

- 需要核对 handover 和 recorder 的具体实现；
- 便于对照 bug fix；
- 能固定复现实验；
- 仍可用 editable install，而不必在 upstream tree 内写项目逻辑。

### 11.2 是否实现 LeRobot policy plugin

LeRobot 官方支持 out-of-tree `lerobot_policy_*` 插件；policy 类实现 `reset()`、`select_action()`/`predict_action_chunk()` 即可接入统一 CLI。

对 AHA-WAM 可以做：

```text
lerobot_policy_ahawam_remote/
  RemoteAHAWAMPolicy(PreTrainedPolicy)
       ├── select_action(): 从本地 FIFO 取一步
       ├── predict_action_chunk(): 远程 TCP infer
       └── reset(): remote reset + clear FIFO
```

但第一版不把它列为硬要求，原因是：

- AHA 远程服务有独立配置和 stateful reset；
- 接管需要 generation fencing，不只是 policy `reset()`；
- 需要扩展 DAgger recorder schema 和逻辑 episode；
- 强行适配 CLI 可能比扩展现有可工作 controller 更慢。

当基础 collector 稳定后，再把 `AHAWAMPolicyExecutor` 包成 plugin，可获得更好的 LeRobot CLI 复用。

### 11.3 不要 clone HIL-SERL 的理由

可以阅读并引用其 replay/intervention 语义，但不把 repo 作为 SO101 collector 基底：

- robot infra 是 Franka；
- action 是末端增量/SpaceMouse，不是 SO101 绝对关节；
- actor/learner 与 JAX RL 训练强耦合；
- reward/env reset/task wrapper 都是任务专用；
- AHA-WAM 当前只需要采集闭环，不需要同时移植整套 RL learner。

---

## 12. 配置文件建议

```yaml
task:
  instruction: "Remove the power adapter from the holder and plug it into the power strip."
  fps: 15
  max_episode_seconds: 40

robot:
  follower_port: /dev/ttyACM_FOLLOWER
  follower_id: so101_follower
  leader_port: /dev/ttyACM_LEADER
  leader_id: so101_leader
  max_delta_per_tick: [5, 5, 5, 5, 8, 8]  # 占位；必须按现有安全配置确认

cameras:
  fixed: {...}
  wrist: {...}

policy:
  host: 127.0.0.1
  port: 12001
  chunk_size: 8
  control_hz: 15
  low_watermark_steps: 3
  request_timeout_s: 1800
  hold_when_empty: true

handover:
  leader_sync_duration_s: 1.0
  require_second_confirmation: true

recording:
  root: data/so101_ahawam_plug_hil_raw
  save_autonomous: true
  save_human: true
  save_pause_frames: false
  record_policy_human_executed: true
  finalize_with_outcome_required: true

safety:
  require_physical_estop_ack: true
  trusted_pickle_tunnel_only: true
```

上面的 joint delta 只是 schema 示例，不能直接作为 SO101 真机数值。部署端应复用当前已经验证的 limits，再把 resolved limits 写进 session manifest。

---

## 13. 主控制 loop 的参考伪代码

```python
while not shutdown:
    tick_start = monotonic_ns()
    events = event_source.consume_edges()
    state_machine.apply(events)

    obs = observation_sampler.read()

    if state_machine.just_left_autonomous:
        policy.invalidate("human_pause")
        arbiter.latch_hold_target(obs.state_arm)

    if state_machine.just_left_human_control:
        arbiter.latch_hold_target(obs.state_arm)

    if state_machine.phase == PAUSING:
        executed_arm = arbiter.send_latched_hold()
        handover.align_leader_to_follower(obs.state_arm)
        state_machine.mark_aligned()

    elif state_machine.phase == HUMAN_CONTROL:
        human_raw = leader.get_action()
        human_arm = human_processor(human_raw, obs)
        executed_arm = arbiter.send_human(human_arm, obs)

    elif state_machine.phase == POLICY_WARMUP:
        executed_arm = arbiter.send_latched_hold()
        policy.ensure_reset_then_prime(obs, exact_instruction)
        if policy.fresh_action_ready_for_current_generation():
            state_machine.mark_policy_ready()

    elif state_machine.phase == AUTONOMOUS:
        policy.maybe_prefetch(obs)
        proposal = policy.pop_next_action()
        if proposal is None:
            arbiter.latch_hold_target(obs.state_arm)
            executed_arm = arbiter.send_latched_hold()
        else:
            executed_arm = arbiter.send_policy(model_to_arm(proposal.action_model), obs)

    else:
        executed_arm = arbiter.send_latched_hold()

    recorder.add_tick(
        observation=obs,
        policy_action=policy.proposal_or_invalid(),
        human_action=human_or_invalid(),
        executed_action_arm=executed_arm,
        executed_action_model=arm_to_model(executed_arm),
        phase=state_machine.phase,
        source=arbiter.last_source,
        ids=policy.ids,
        timing=collect_timing(),
    )

    sleep_until_next_15hz_tick(tick_start)
```

实际实现中，`align_leader_to_follower()` 不能在 15 Hz 主线程里用长阻塞调用而导致事件失去响应。可以让状态机处于 PAUSING、主线程持续 hold，handover worker 做有超时的平滑对齐；abort/急停仍需即时可用。

---

## 14. 实现阶段

### Phase A：只读审计

输出：

- 本机 repo 路径和 git status；
- LeRobot/Python/FFmpeg/camera/motor 版本；
- 现有 `send_action` 唯一调用链；
- 当前 action FIFO 和 inference thread；
- current dataset writer 与 codebase version；
- leader/follower calibration 文件与 motor keys；
- AHA task/checkpoint/config/stats hash；
- 当前 arm/model transform 的 round-trip 测试。

### Phase B：fake end-to-end

用 fake policy 模拟：

- 固定 8-step chunks；
- reply 延迟 50/500/1500 ms；
- reply 乱序或失败；
- NaN/shape 错误；
- pause 恰好发生在每个 chunk index 0–7。

断言：

- pause 后下一 tick 不再发送旧 generation action；
- stale reply 永不进入 active FIFO；
- release 一定发生 remote reset；
- fresh policy ready 前 follower source 是 HOLD；
- recorder 的 `action` 总等于 fake follower 收到的 target。

### Phase C：硬件连接但不执行任务

- 相机与 follower observation-only；
- leader read-only；
- 检查 15 Hz deadline；
- 录一条不发动作的 dry-run；
- validator 检查帧数、timestamp 单调、图像可解码、字段 shape/finite。

### Phase D：平滑交接

- 机械臂处于安全开阔姿态；
- follower hold；
- 测 leader 对齐，不让 leader action 控 follower；
- 确认 joint key、方向和 gripper；
- 再以低速/小限幅进入 HUMAN_CONTROL；
- 人松手/快速移动/越界时 safety clamp 可见且被记录。

### Phase E：模型接管闭环

- 不放插头或使用安全空场景；
- AHA 运行、在 chunk 的每个 index 触发 pause；
- 记录 pause-to-suppression latency；
- 人移动少量姿态；
- release 后检查 stale reply、reset、fresh infer；
- 连续重复至少 20 次，无旧动作复活、无 leader/follower 跳变。

### Phase F：真实采集 pilot

先采 3–5 条，不直接采 30–40 条：

- 1 条无人接管自主失败/成功；
- 1 条单次抓取纠正；
- 1 条精对准/插入纠正；
- 1 条同 episode 多次 intervention；
- 1 条 timeout/abort。

立即把数据传回训练机，跑 v2.1 converter、AHA loader 和一个 40-step batch smoke test。通过后再扩大采集。

---

## 15. 测试与验收标准

### 15.1 安全/控制验收

- [ ] 物理急停独立可用；
- [ ] 全程序只有一个 `send_action()` owner；
- [ ] pause-to-policy-suppression 的 p99 不超过一个 15 Hz tick，或报告实测上界；
- [ ] 任意 chunk index 接管都不会执行剩余旧 action；
- [ ] in-flight stale reply 被日志证明丢弃；
- [ ] HUMAN→AUTO 必有 remote reset；
- [ ] fresh reply 前始终 HOLD；
- [ ] leader/follower key、方向、范围和 gripper 均通过逐关节测试；
- [ ] 网络断开、server error、FIFO 空、NaN、deadline miss 均 fail to HOLD/STOP；
- [ ] pickle 只走 localhost/SSH tunnel/可信隔离网。

### 15.2 数据验收

- [ ] 每帧 `action == arm_to_model(executed_action_arm)` 在容差内；
- [ ] `intervention == (control_source == HUMAN)`；
- [ ] policy/human valid mask 与 phase 一致；
- [ ] `policy_generation` 在每次 invalidate 后严格增加；
- [ ] stale request id 从未出现在后续 executed policy frame；
- [ ] timestamp 单调，控制频率分布可报告；
- [ ] fixed/wrist 视频帧数与 parquet/frame table 对齐；
- [ ] logical episode 有且只有一个明确 outcome；
- [ ] 无人接管 rollout 也保存；
- [ ] failure/timeout/abort 不被 raw collector 删除；
- [ ] session manifest 包含 checkpoint/config/code/calibration hash；
- [ ] AHA v2.1 export 可被当前 loader 实际取出 40-step batch；
- [ ] method-specific export 不改写 raw 数据。

### 15.3 论文/实验验收

- [ ] baseline checkpoint 在采集开始前冻结；
- [ ] 每条 episode 记录 policy iteration；
- [ ] intervention 原因使用固定 taxonomy；
- [ ] 报告 episode 数之外，也报告 intervention segments/ticks、robot ticks 和人工分钟；
- [ ] 成功率评测数据与训练/调参初始化分离；
- [ ] HG、IWR、RECAP-like 的差异来自训练视图/算法，不是各自重新采一批不可比数据。

---

## 16. 常见错误与处理

### 16.1 只保存 human correction

后果：可以做近似 HG，但不能公平做 IWR、RECAP、value 或 intervention rate 分析。

处理：raw 始终 `save_autonomous=true`；需要 correction-only 时离线导出。

### 16.2 一帧只有 `action + intervention`

后果：无法区分 policy proposal、leader input、安全层改写和实际 action；debug 时不知道机器人究竟执行了什么。

处理：三种 action + valid mask + source。

### 16.3 接管只换 action source，不清 queue

后果：旧 AHA chunk 在人类 action 之间或交还后复活。

处理：generation fencing、FIFO flush、唯一 arbiter。

### 16.4 人执行后不 reset AHA

后果：服务端内部 video/action state 假定未执行的 policy plan 已发生，后续 chunk 与真实世界不一致。

处理：每次 human→policy 必须 hard reset + fresh observation。

### 16.5 把视频自动分片当 episode

后果：success/failure terminal 错位，value/RECAP 无法使用，长 rollout 被误切成多个任务。

处理：`logical_episode_id` 与存储 shard id 分离。

### 16.6 直接让 leader 当前姿态控制 follower

后果：接管瞬间跳变。

处理：先 pause/hold，再把 actuated leader 平滑同步到 follower。

### 16.7 混用 arm 与 model/G05 坐标

后果：训练数据统计、server 输入和真机目标发生 90° 偏置/符号翻转。

处理：唯一 transform module；raw 与 canonical 双存；每条 session 做 round-trip 和已知姿态 test。

### 16.8 直接升级 LeRobot 并继续往旧目录追加数据

后果：dataset schema/metadata/video writer 混版，训练端 v2.1 reader 可能静默误读或直接失败。

处理：新环境、新 raw dataset、固定 commit、显式 converter；不得跨 writer version 原地 append。

### 16.9 为保持“数据干净”删除失败

后果：丢失 RL/value/RECAP 最有价值的数据，也使 intervention rate 与真实策略表现失真。

处理：raw 不删；用 quality/outcome manifest 控制各训练视图。

---

## 17. 部署端 Codex 完成后的回报模板

```text
1. 本机环境
   - repo / commit:
   - LeRobot version / commit:
   - Python / OS:
   - follower / leader / cameras:

2. 复用与新增
   - 直接复用的 LeRobot 类/函数:
   - 修改/新增文件:
   - 是否实现 policy plugin，若否原因:

3. 控制语义
   - sole send_action owner:
   - action FIFO:
   - generation fencing:
   - remote reset sequence:
   - leader handover:

4. 数据
   - raw writer/version:
   - features:
   - logical episode/outcome:
   - AHA v2.1 converter:

5. 测试结果
   - fake delay/stale reply:
   - pause suppression p50/p95/p99:
   - 20 次 takeover/release:
   - schema validator:
   - 40-step AHA loader smoke test:

6. 未解决风险
   - ...

7. 首次真机 pilot 的精确启动命令和操作键位
```

---

## 18. 当前建议的最小可用版本与后续扩展

### MVP 必须有

- 复用 LeRobot SO101 leader/follower/camera；
- AUTO/PAUSED/HUMAN/WARMUP 状态机；
- 唯一 action arbiter；
- action FIFO generation fencing；
- human→policy remote reset；
- 全轨迹与三 action recorder；
- logical outcome；
- v2.1 转换 smoke test；
- fake 和低风险真机测试。

### MVP 后再做

- foot pedal；
- LeRobot out-of-tree AHA policy plugin；
- server-side session/cancel token；
- counterfactual policy proposal during human control；
- 自动 intervention reason suggestion；
- per-step `action_supervision_mask` 训练支持；
- IWR balanced sampler；
- reward classifier/value/RECAP；
- RLDG specialist rollout exporter。

不要把后续功能塞进第一版，从而延迟最关键的安全队列和数据正确性验证。

---

## 19. 参考资料

1. [HG-DAgger: Interactive Imitation Learning with Human Experts](https://arxiv.org/abs/1810.02890)
2. [Intervention Weighted Regression / IWR 论文](https://arxiv.org/abs/2012.06733)
3. [IWR 官方项目页](https://sites.google.com/stanford.edu/iwr)
4. [HIL-SERL 论文](https://arxiv.org/abs/2410.21845)
5. [HIL-SERL 官方仓库](https://github.com/rail-berkeley/hil-serl)
6. [HIL-SERL `train_rlpd.py`](https://github.com/rail-berkeley/hil-serl/blob/main/examples/train_rlpd.py)
7. [HIL-SERL Franka walkthrough](https://github.com/rail-berkeley/hil-serl/blob/main/docs/franka_walkthrough.md)
8. [RLDG 论文](https://arxiv.org/abs/2412.09858)
9. [RLDG 官方复现实验说明](https://generalist-distillation.github.io/RLDG/)
10. [LeRobot v0.6 HIL data collection 文档](https://github.com/huggingface/lerobot/blob/v0.6.0/docs/source/hil_data_collection.mdx)
11. [LeRobot v0.6 DAggerStrategy 源码](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/rollout/strategies/dagger.py)
12. [LeRobot v0.6 SOLeader 源码](https://github.com/huggingface/lerobot/blob/v0.6.0/src/lerobot/teleoperators/so_leader/so_leader.py)
13. [LeRobot out-of-tree policy plugin 文档](https://huggingface.co/docs/lerobot/v0.6.0/bring_your_own_policies)
14. [LeRobot 旧 HIL-SERL SO101 issue #2952](https://github.com/huggingface/lerobot/issues/2952)
15. [LeRobot 旧 HIL-SERL SO101 PR #3086](https://github.com/huggingface/lerobot/pull/3086)
16. [LeRobot v0.6 release blog](https://github.com/huggingface/blog/blob/main/lerobot-release-v060.md)

---

## 20. 最终决策

当前最合适的工程路线不是在 HIL-SERL、IWR、RLDG 中“三选一套 collector”，而是：

```text
LeRobot v0.6 SO101 硬件与安全交接
                 +
现有 AHA-WAM TCP/action-chunk 部署
                 +
本项目专用的 queue fencing 与富标注 recorder
                 =
可同时支持 HG / IWR / RL / RLDG / RECAP 的原始数据基础设施
```

接口的主体确实不是神经网络训练代码，但 AHA-WAM 的 8-step chunk、阻塞请求和服务端历史状态让它绝不是“换个 action source 就完事”。只要部署端严格做到**即时清队列、旧 reply 失效、交还前 hard reset、三 action 分开、逻辑 episode/outcome 完整**，这套数据就不会被 HG-DAgger 锁死，也能承接后续 IWR、HIL-SERL-style RL、RLDG 和 RECAP-like 研究。
