# 插入实验中“底座是否移动”的逐项核查

> 核查日期：2026-09-07。这里的“底座”指母口、孔、插线板、装配板或锁，而不是
> 已抓取公头或机器人末端的初始位姿。结论只采用论文、附录、官方项目和公开代码；
> “论文没写”不推断为“固定”或“随机”。

## 1. 结论先行

相关工作并没有统一做法，至少存在三种互不等价的实验：

1. **母口固定，只随机公头/末端起点。** 适合研究固定工位上的局部对准、接触恢复和
   速度；不能单独证明策略能通过视觉定位移动后的母口。
2. **母口在 episode 之间移动。** 这才直接迫使策略处理目标位置变化；若训练和测试
   位置还分开，则能进一步研究空间泛化。
3. **母口固定，但抓取关系、接触误差或前序策略造成的状态变化。** 仍可能真正需要
   视觉/触觉，但证明的是相对位姿恢复，不是任意目标定位。

因此，“固定底座的成功率”本身既不能证明模型没用视觉，也不能证明模型用了视觉。
它只证明在该固定工位分布下任务可解。要证明视觉因果作用，必须增加目标位姿变化或
做图像反事实/消融。

## 2. 与当前工作最接近的方法

| 工作 | 被插底座是否变化 | 实际随机的变量 | 可以支持的结论 |
|---|---|---|---|
| HALO-WA | **论文未明确报告**；公开 charger 配置呈现为固定目标 | `target_ee_pose` 是单一常量；reset 时机器人末端 x/y 各 `±2 cm`，yaw `±0.35 rad` | 支持固定工位附近的在线局部修正；不能据此声称在移动插座上视觉泛化 |
| RLT / RL Token | **未说明底座是否移动** | critical-phase 使用“slightly randomized initial configurations”；全文未把它分解成母口、公头、抓取或末端随机 | 只能保守表述为关键阶段初态有小扰动，不能写成插座位置随机 |
| HIL-SERL USB | **固定目标位姿** | 官方代码把 `TARGET_POSE` 写成常量，`RESET_POSE` 围绕它随机；论文表为 x/y 2 cm、yaw 10° | 学固定主板工位的抓取—插入与恢复，不等于移动主板定位 |
| RLDG Connector Insertion | **母口固定** | 已抓公头在母口上方 `10×10 cm` 平面、约 5 cm 高处初始化 | 合理的 last-inch specialist；不证明母口空间泛化 |
| RLDG FMB Insertion | **装配板移动** | 板在 `35×35 cm` 区域、`±15°` 内随机 | 明确要求视觉找到装配板；评测复用同一组板位姿以公平配对 |
| ViSk | **移动且留出测试位置** | 插线板 `20×7 cm`、固定朝向；USB hub `20×15 cm`；10 个 held-out 配置 | 对“目标位置变化＋未见位置”证据很强，但依赖多相机与触觉，范围不能直接照搬 SO101 |
| EDRIAD | **移动** | 每回合用 pan-tilt 机构均匀采样母口姿态，开口位移约 2–4 cm | 论文直接比较固定/随机及有无视觉：固定任务无视觉也能学，随机任务无视觉不能可靠解决 |
| SHIELD | **多种设置** | NIST 任务做新位置/朝向零样本测试；视觉策略还在执行中手动移动板；HDMI 最终课程达到初位 `±6 cm` 且动态移动；钥匙初始误差课程扩大 | 强证据说明应把固定、小扰动、移动目标分成不同难度，不应混成一个成功率 |
| Safe Self-Supervised USB | 未报告移动 NIST 板；重点是**抓取位姿变化** | 45 个不同 USB 初始抓取位姿；50 条 BC demo 仅 1/45，而视触觉方案 45/45 | 即使母口固定，公头—夹爪关系变化也是真问题；但结论仍不是移动目标泛化 |
| Sparsh-skin plug insertion | **明确固定 socket** | 末端在 `5×5×2 cm` 体积内随机，约在 socket 上方 10 cm | 明确的固定母口、随机机器人起点范式 |
| ACT / ALOHA | **真实任务物体会移动** | 真实任务物体沿 15 cm 参考线随机；仿真在二维区域均匀随机 | 示教学习从第一批数据就加入受控位置变化是常见且可行的 |
| FISH key insertion | **锁移动** | 评测在相同的一组 10 个随机锁位置上比较各方法 | 对视觉策略的空间条件化更有说服力 |
| π*0.6 / RECAP | **不是连接器插入实验** | laundry 有可变初态，box/espresso 是固定工作站式长程任务；没有给出可类比的“插座位姿随机化” | 不能拿 RECAP 的任务结果回答本问题；应看 RLT 的插入实验，但 RLT 对母口变化也未交代清楚 |

主要一手依据：

- [HALO-WA 论文](https://arxiv.org/abs/2607.04265)只列任务、相机、训练时间和 35 次评测，未定义母口位姿分布；本地公开配置
  `~/workspace/HALO-WA/rlinf/examples/embodiment/config/other/realworld_charger_sac_cnn_async.yaml`
  与 `franka_env.py` 显示随机量作用于 reset 末端，而不是物理目标。
- [RLT 原文](https://www.pi.website/download/rlt.pdf)只写 critical-phase 初态“slightly randomized”，
  没有报告 charger/Ethernet 母口移动范围。
- [HIL-SERL 官方 USB 配置](https://github.com/rail-berkeley/hil-serl/blob/main/examples/experiments/usb_pickup_insertion/config.py)
  将 `TARGET_POSE` 固定，并随机 reset；[论文附录](https://hil-serl.github.io/static/hil-serl-paper.pdf)
  报告 2 cm/10° randomization，但没有说移动主板。
- [RLDG 论文附录](https://www.roboticsproceedings.org/rss21/p028.pdf)明确区分固定母口的
  Connector Insertion 和随机装配板的 FMB Insertion。
- [ViSk 论文](https://openreview.net/pdf?id=uRBsAYSIeO)明确给出插线板/USB hub 训练区域和
  10 个 held-out 配置。
- [EDRIAD](https://arxiv.org/abs/1810.01531)明确给出母口 2–4 cm 变化，并报告固定任务无视觉也可解、随机任务依赖视觉。
- [SHIELD](https://www.roboticsproceedings.org/rss17/p088.pdf)明确包含新位置/朝向、移动 NIST
  板、动态 HDMI 母口与随机钥匙误差课程。
- [ACT / ALOHA](https://tonyzhaozh.github.io/aloha/aloha.pdf)明确真实任务物体沿 15 cm
  参考线随机。
- [Safe Self-Supervised USB](https://arxiv.org/abs/2210.01340)明确评测 45 个抓取位姿。
- [Sparsh-skin](https://openreview.net/pdf/d17cdfeeb31c2db1c18a391f62b788f7f85f8b5c.pdf)
  明确写 fixed socket 与随机末端体积。
- [FISH](https://www.roboticsproceedings.org/rss19/p009.pdf)对 key insertion 使用随机放置的锁，
  并在同一组 10 个位置上配对评测。

## 3. 为什么固定底座确实会产生捷径

若底座世界位姿为常量 (p_T)，机器人总从近似相同的 (q_0) 出发，动作标签就可能近似为

\[
a_t \approx f(q_t,t),
\]

而不需要图像 (I_t)。关节状态本身告诉模型“现在走到固定轨迹的哪里”，Video DiT 也可
预测“固定背景＋常见机械臂路径”。即便网络内部存在视觉特征，也不代表输出动作因果依赖
目标图像。

不过，下面几种情况会让固定母口仍然不是纯轨迹回放：

- 圆柱/公头的初始位置变化，导致抓取后的物体—夹爪外参变化；
- 前序抓取误差和碰撞使到达 pre-insertion 的状态变化；
- wrist 图像能看见相对错位；
- 触觉或 F/T 能通过接触搜索孔；
- 策略闭环重规划，而不是仅按时间播放动作。

这些因素能支持“固定工位鲁棒性”，但仍不能支持“模型能视觉定位移动底座”。

## 4. 对现有 cylinder 数据的抽样审计

抽样检查了以下 5 个 session 的 fixed-camera episode 首帧：

- `20260904_014410`
- `20260904_020841`
- `20260904_023259`
- `20260904_024009_val`
- `20260904_024404`

并逐条查看 `014410` 与 `024404` 各前 4 条 episode。可见：

- 木质底座在同一 session 内也有明显平移和偏航变化；
- 独立圆柱的初始位置也变化；
- 机器人/夹爪初态存在一定变化；
- 目标始终是底座上同一个相对孔位，但该孔的世界坐标并非固定。

因此，这批 cylinder 数据**不是严格固定底座的最坏情形**。它比 HALO-WA charger 示例、
HIL-SERL USB 和 RLDG Connector 的固定母口设置更能抑制绝对世界坐标捷径。

限制是：目前这个判断来自视频抽样，不是完整位姿标注；若摆放是“人随手放”，位置可能
和 session、圆柱起点、光照或示教质量相关，模型仍可学习这些相关性。现有 val 也只有少量
episode，不能自动当作严格 held-out pose 测试。

## 5. 最低成本而有说服力的验证

### 5.1 最关键的反事实位置对

保持以下变量尽量相同：

- 同一个机器人 home/qpos；
- 同一个圆柱起点与朝向；
- 同一个相机、光照和背景；

只把底座分别放在“左”和“右”两个位置。正确的初始搬运动作横向方向应随底座反转。
每个位置至少重复 10 次，交错执行，不要先做完左再做右。

如果模型在两处都向同一世界坐标走，基本可判定固定轨迹/训练分布中心捷径；如果动作能
随目标方向改变，才有较强的视觉条件化证据。

### 5.2 位置分层，而不是随手随机后混在一起

对 SO101 第一版建议：

- 训练：中心附近 `3×3` 位姿网格，可先用 x/y 各 `±2 cm`，yaw 先固定或仅 `±5°`；
- ID 测试：同一范围内重新 jitter；
- 插值测试：训练网格之间从未采集的精确位置；
- OOD 测试：略超训练边界的位置，前提是机械臂可靠可达；
- 每个模型使用同一份物理位置清单，交错测试并报告每个位置桶成功率。

实际范围应先由遥操作可达性和缩放后图像位移决定，不要为了“随机化”一次扩大到整个桌面。

### 5.3 视觉消融应优先离线做

真机直接喂黑图或错配图可能产生危险动作。更安全的证据是：

1. 从左/右底座位置采集相同或非常接近的 qpos observation；
2. 固定 state，交换 fixed-camera 图像，比较模型预测 action chunk；
3. 再交换 wrist 图像，判断哪一路真正改变对准动作；
4. 报告横向动作符号、chunk 差异和真实目标方向的一致率。

还可训练一个 `no-fixed-camera` 或图像冻结的小型消融，但不要把“遮图后失败”单独当作
视觉定位证明，因为遮图本身也是严重分布外输入。

## 6. 对论文叙事的直接建议

不要写“随机摆放保证模型使用视觉”。更严谨的表述是：

> 训练时在 episode 间改变 receptacle 的平面位姿，以减少固定世界坐标捷径；评测采用
> 预先定义的 seen-jitter、held-out interpolation 和 counterfactual left/right pose，
> 并通过固定 proprioception 下的图像交换验证动作对目标图像的敏感性。

对主表至少报告：

- 固定名义位置成功率；
- 训练范围内新 jitter 成功率；
- held-out 位置成功率；
- 每位置桶失败类型；
- 反事实 action-direction accuracy；
- DAgger 接管率随位置桶的变化。

这样即使固定位置性能很高，也不会被误当成视觉泛化；同时也能公平保留 HALO-WA、
HIL-SERL、RLDG Connector 这类“固定工位局部 specialist”作为合理 baseline。
