# SO101 在线 RL 的推荐训练协议

## 1. 先根据实际数据决定启动方式

本次读取 power-adapter-v2 的70个 parquet及两个采集目录的 ledger，聚合结果保存在 [audit_snapshot.json](audit_snapshot.json)。

| 历史日志指标 | 结果 |
|---|---:|
| 总帧数 | 26,658 |
| POLICY / HUMAN / SYSTEM_HOLD | 9,775 / 5,498 / 11,385 |
| episode标签 | 66 SUCCESS_WITH_INTERVENTION；3 FAILURE；1 AUTONOMOUS_SUCCESS |
| safety_clamped帧 / dropped_tick帧 | 1,026 / 13 |
| 模型延迟，1835个请求 | p50 537.96ms；p95 621.87ms |
| 请求往返 | p50 630.54ms；p95 730.85ms |
| loop_dt | p50 67.78ms；p95 68.48ms |
| manifest termination | 70条均为空字符串 |

以上是 **旧 HIL 采集时的日志统计**，不是当前 cylinder 或 power-adapter 服务测速。一个请求可能涉及不同缓存阶段，不能把总延迟拆成未经测量的 Video/Action DiT 各自耗时。

这批数据同时说明两件事：已有不少接管成功的经验可以利用；自主探索的成功样本很少。67条最终成功不能作为当前策略成功率。SYSTEM_HOLD 占比约42.7%，也不能删掉后把时间前后拼成连续自主轨迹。

现有 schema 已有实际动作、来源、chunk/request标识、时钟、相机年龄和限幅标记；manifest 已有 outcome。因此不是“完全没有 RL 数据基础”，而是还要恢复可靠 next-state、终止时刻、状态历史和 reward。失败与接管后成功的区别必须保留。

## 2. 第一阶段：局部任务 + 输出端小策略

**推荐先学习最后的对准/插入，不同时重新学习抓取、搬运和插入。** 先用固定的、可重复但有小范围变化的孔口初始状态建立局部成功信号，再让基础 AHA 完成前段、从它实际到达的状态进入 RL 阶段。

基础 checkpoint 在这一实验阶段固定；没有必要从预训练重新做一轮大模型 RL，也不再用 video loss在线更新主干。局部任务成功后再评估完整任务，二者单独报告。

### 2.1 先选时序模式，不能让 learner 掩盖控制瓶颈

为减少第一轮接口变化，先实现 **C=8 的块级 residual**：基础模型生成8步参考，小 actor一次修改8×6维动作，得到真实执行结果后构建transition。所有块级方法使用相同采样器、频率和缓存规则。

但历史往返 p50 已超过8/15≈0.533秒的动作覆盖长度。因此 C=8 只能作为第一轮接口/算法基线；**若当前测速仍有大量空队列或插入时反应过迟，主攻方向应改为低延迟逐步 residual，而不是继续增加 learner UTD。**

逐步方案保留 AHA 的8步基础参考，以15Hz执行小 actor；每个 tick读取新 proprio 和双相机的轻量冻结视觉特征，并输入当前 reference slot、参考年龄、chunk位置，输出6维修正。重型 AHA异步更新参考，小 actor最好运行在机器人电脑附近。仅反复读取半秒前的 RL token不等于获得15Hz视觉反馈。

这个逐步方案是本项目的控制系统扩展，不是声称原 RLT 已经实现了它。若本地无足够算力，可先测试小 encoder/MLP的CPU耗时；若服务器计算仍需跨网等候，就不能承诺每66.7ms闭环。执行位置与实测p95延迟共同决定是否可用。

不得只把 `max_return_actions` 改成1/2就宣称实现高频反馈：模型chunk index、offset、video刷新协议和废弃参考动作都必须一致。当前15Hz/8步训练条件不自动等价于更短块条件。

### 2.2 状态与表征

定义 x：

```text
x = [frozen_visual_or_OVCR_readout,
     current_proprio,
     measured/properly_estimated_velocity,
     reference_chunk_or_slot,
     chunk_index, reference_age, observation_age, task_phase]
```

模型当前state是6维。额外速度可由实际时间差估计，但应与原SFT输入分开，作为小策略的新输入，并正确处理暂停边界。不能把由未来帧算出的速度或未来视频token泄漏到当前actor。

先用冻结观测token的固定pooling做对照，再比较训练好的重建readout。已有 `RLTokenReadout` 可复用为原型，但最大64token、宽度256、teacher-forced重建的设计与RLT原文不完全相同；低重建loss并不保证保留了接触状态。在线阶段冻结readout，并记录版本。

供动作Q使用的x应包含actor看到的reference与时序信息。对于噪声Q，尤其需要能概括冻结生成器条件的历史特征；压缩后的x是否满足近似Markov性仍是建模假设，不能由“用了token”直接推出。

### 2.3 动作参数化

固定一组经本机限位与跟踪测试确定的每关节residual半径b，先不学习夹爪修正：

```text
u = tanh(μ_φ(x) + σ_φ(x) ε),  ε ~ N(0,I)
a_nominal = a_reference + b ⊙ u
a_executed = executor(a_nominal)
```

每个维度的b必须是已知物理坐标下的合法偏移，再映射到critic使用的归一化坐标。当前z-score下0.1表示0.1个数据标准差，不是0.1度，也不是动作范围的10%。

第一版actor最后一层均值输出零，使确定性策略起点为reference；初始随机幅度较小。先完整校验 `u=0` 与原策略输出一致，随后才采集带探索的动作。

critic评价的是**实际下发的目标动作**，proprio另外记录实际到达位置。不能拿未执行的policy提议作为HUMAN或限幅后的动作标签，也不能把舵机没到达目标误记为“执行了另一个目标命令”。

actor loss可以通过冻结critic对名义动作求梯度；执行器中离散拒绝/硬裁剪通常不可微。必须记录裁剪率，校准b使正常探索大部分不依赖裁剪；不能默默用直通梯度宣称是原SAC。

## 3. 明确的第一版更新规则

下面定义 **标准双Q SAC + 有界residual参数化**。网络规模、replay结构参考HIL-SERL，动作约束参考Policy Decorator；这是可审计的组合方案，不是它们某个官方实验的原样复现。[^1]

每条有效transition保存 (x,a_exec,R,x',terminated,discount)。令C_eff为真实执行步数，γ_step为每控制步折扣：

```text
R = Σ[j=0..C_eff-1] γ_step^j r[j]
Γ = γ_step^C_eff
u' ~ π_φ(. | x')
a' = a_ref(x') + b ⊙ u'

y = R + (1 - terminated) Γ [min_i Q_target_i(x',a') - α logπ_φ(u'|x')]
L_Q = mean_i E[(Q_i(x,a_exec) - stop_gradient(y))²]

u ~ π_φ(. | x)
a = a_ref(x) + b ⊙ u
L_actor = E[α logπ_φ(u|x) - min_i Q_i(x,a)]

L_logα = -E[logα · stop_gradient(logπ_φ(u|x) + target_entropy)]
Q_target ← (1-τ) Q_target + τ Q
```

这里熵定义在固定[-1,1]的u坐标中，log probability包括tanh Jacobian修正。不要一会使用关节动作单位，一会使用u单位，却保持同一target entropy；若改动作变量定义，必须连同密度变换一起说明。

target Q及target action构造不向actor/target网络传梯度；actor更新冻结Q参数但保留Q对action输入的导数。在线只有actor、两个Q和温度的optimizer，AHA及readout没有optimizer。

### 3.1 建议的起点参数

下表是 **本项目建议值，尚未在AHA实机验证**；公开原值见文档02。

| 参数 | 第一版建议 | 依据/修改原因 |
|---|---|---|
| actor / 每个Q | MLP 256×2，LayerNorm，FP32 | HIL-SERL小网络作为起点；避免直接搬3×2048 |
| actor LR | 1e-4 | Policy Decorator和UniSteer的量级 |
| critic LR / temperature LR | 3e-4 / 3e-4 | HIL-SERL/DSRL常用量级 |
| optimizer | Adam，betas=(0.9,0.999)，weight_decay=0 | 常规SAC设置，不沿用大模型SFT的AdamW |
| target τ | 0.005 | 多个公开实现一致 |
| batch | 256 | 与SFT全局12完全不同，不需强行一致 |
| replay容量 | online 100,000条有效transition；已核验demo单独保留 | 本项目内存起点；原始视频独立存盘 |
| replay组成 | 128已核验demo +128 online；两池有放回 | HIL-SERL/RLPD依据 |
| actor:critic更新频率 | 1:2 | 每两次critic梯度更新一次actor和温度 |
| 初始实际UTD_Q | 1，稳定后5；高UTD=20单独消融 | 小数据初期避免critic反复拟合极少经验 |
| α初始化 / target entropy | 0.01 / -d_u/2 | 参考HIL-SERL；熵坐标固定为u |
| 梯度裁剪 | 10 | 保守工程起点，须记录触发频率 |
| γ_step | 0.99，每15Hz控制步 | 本项目时间尺度选择，不照搬50Hz参数 |
| warmup | 至少1000条有效新transition，至少10条独立episode | 本项目门槛；采集基础动作/小扰动/接管，不做随机关节探索 |
| 模型发布 | 每50次learner更新最多发布一次，在控制边界原子切换 | 借鉴HIL-SERL，记录policy version |
| demo BC辅助loss | 第一轮关闭；只保留零residual初始化及demo critic replay | 避免混入额外BC系数；HUMAN-only BC作为单独对照 |

SAC actor可使用log-std边界[-5,2]、初值-3作为本项目起点，配合经标定的b；这是小范围启动的工程选择，不是RLT固定σ=0.002的复现。训练/评估时都保留相同动作坐标和边界；不要让温度自动调节悄悄改变物理residual上限。

d_u是当前真正学习的动作维数：锁定夹爪时，逐步为5，8步块为40；若开放全部6维则分别为6和48。不能用固定“-3”同时服务这些不同维度。

UTD_Q定义为累计critic梯度更新次数/新增真实transition数；一次同时更新双Q算一次critic更新。C=1和C=8的transition数不同，因此跨时序实验还要按真实秒数和执行tick计预算。暂停采集后也不无限消耗更新预算。

γ_step=0.99对应C=8时Γ≈0.9227，C=1时Γ=0.99，两者具有相同的物理折扣尺度。γ=0.97如果被同时当作逐步折扣和逐块折扣，会改变目标。持续时间不规则时可用Γ=exp(-Δt/T)，块内奖励也按实际时间折扣，但不能在删去等待后伪造时间。

### 3.2 为什么不把旧数据全部直接作为50% demo

[历史数据快照](audit_snapshot.json)显示已有最终outcome，但termination字段为空，parquet没有逐步reward。要从manifest和events确定成功时刻与episode结束，复位和最后保存帧不一定就是发生成功的瞬间。

HUMAN截段可先做BC；只有有连续观测、已知真实执行动作和明确next-state/reward边界的片段才进入TD。截段最后一帧不能因为“没有更多人工动作”就被标为任务终止，也不能无视中间暂停跳到另一状态。

可用于TD的历史transition需要固定新readout重算一致的观测历史。如果原request/cache上下文无法恢复，不能把其近似feature当精确在线feature；先限制为不依赖隐式cache的图像/proprio小策略，或重新采集种子数据。

逐步HUMAN动作最容易定义真实transition；将闭环人工轨迹打包成C步动作，是常见的时序抽象但不是严格等价于人在块起点一次承诺全部动作。对接触快速反馈场景，优先逐步TD；chunk实验必须标记这个近似并排除混合来源、跨gap和未执行尾部。

如当前只有少量可用完整demo，就先100%有效online replay，并把缺reward的纠正独立用于BC warmup。不能为满足“1:1”而造出不可验证的TD样本。

### 3.3 有放回采样并不浪费在线RL数据

SFT的epoch关注每条监督数据看几次；off-policy RL的target和当前策略持续变化，同一transition多次参与更新有意义。双buffer按固定batch配额采样，有放回是正常设计，不需要每轮把数据耗尽。

需要防的是一个长episode占据绝大多数batch、几段接管反复被过拟合、失败状态完全不被采到。记录episode覆盖、样本年龄、成功/失败比例、HUMAN占比；episode均匀采样作为后续消融，不在第一轮同时改变所有采样规则。

## 4. 奖励、在线rollout、随机性、人工接管

采用局部任务的可核验成功事件：第一次确认插入成功时+1并终止该局部episode，其他有效步0。失败/卡住/任务终止记录真实原因；超时是否bootstrap取决于是有限时域任务结束还是仅采集截断，后一种必须拥有真实final observation，不能用reset后的图像bootstrap。

可以先人工按键标成功，后续再训练成功分类器。不要把 `intervention=true` 当作+1，也不要把“孔口很近”自动当作成功。若将接管当负奖励，必须作为显式reward变体测试；否则可能学会减少求助而非学会插入。

在线RL必须收集改进策略的新rollout；只循环旧buffer是offline RL。早期保留人工接管有助于成功覆盖和设备恢复，人手不是每一步的训练目标。带接管的训练成功与无接管评估成功分开统计。

SAC本身有随机actor，通常不需再叠加另一套关节噪声。评估采用固定约定：residual均值，基础策略seed协议相同；噪声策略要分别报告均值噪声与随机采样效果，不能默认零噪声一定是最优基础模式。

块中途接管或限幅时，保留实际每tick动作；不得把人最终救成功的结果归给原先未执行的整块policy。块级版本先只训练完整、可解释的块和真实terminal；干预混合块保留原日志用于逐步RL/BC，不伪造动作标签。逐步版本可在控制源切换处定义真实的单步transition。

## 5. 第二条路线：噪声反演BC后接DSRL-SAC

先做无机器人执行的可控性/反演诊断，再决定是否投入实机预算。这条路线仍冻结AHA，不需要从base重新训练四轮。

### 5.1 当前块的噪声接口

在准备好当前观测条件之后，接收 `action_start_noise:[1,8,6]`，只替换当前chunk尚未积分的噪声。噪声actor可读取刚准备的观测/OVCR特征，再调用decoder；因此要拆开“prepare condition”和“decode”以避免先需要动作、后才能决定噪声的循环依赖。

如果噪声actor还要参考基础动作，就会额外多一次基础解码。第一版不加这个条件，公平记录F次数。必须保存实际注入的w，而非seed或actor均值；固定seed只固定随机序列，不是reward优化。

### 5.2 AHA离散采样器的正确反演

AHA scheduler用下降的σ网格，`delta=σ_next-σ_current<0`，每一步为：

```text
y = x + delta * v_θ(x, t, h)
```

逆向求x可以固定点迭代：

```text
x^(m+1) = y - delta * v_θ(x^(m), t, h)
```

从最终动作开始，按 **正向实际时间步的逆序** 恢复每个x。必须沿用原t、delta、sigma_shift、dtype、guidance和条件h，不能统一假设delta=1/10，也不能把训练中的线性加噪 `add_noise` 当作动作反演。[^2]

在固定点收敛条件不满足或低精度导致误差放大时，反演可能失败。先比较每步4/8/16次迭代，统计动作重建的平均、p95、逐关节最大误差、噪声范数及耗时。具体门槛来自本机动作标定，不能照搬FlowDAgger的0.001。

只使用全长有效当前动作块；不能给短HUMAN段补未来动作再声称找到精确latent。必要时用带mask的优化型反演，但这成为另一个方法，要单独计入通过decoder求梯度的离线成本。

### 5.3 如何进入训练

先用接受的 (x,w_h) 做小噪声策略BC，建议从batch256、LR1e-4、MLP256×3起步；学习率与网络有FlowDAgger参考，具体AHA输入/输出仍是迁移。在线DSRL-SAC另用Q_w评价真实注入w的回报，学习率可采用actor1e-4、Q3e-4、τ0.005、batch256、初期UTD_Q=1。[^3]

初期**仅把反演人工标签用于actor BC**；噪声critic的TD只使用真实经G解码并执行的w transition。这样不会把“解码动作与人工动作仍有误差”的结果当作精确latent transition。

反演误差在实际单位足够小，并通过同条件decode检查后，可另做UniSteer式变体：把人工transition与反演w加入噪声RL buffer，先监督更新再RL更新；明确这是近似MDP标签。它与纯BC、纯DSRL-SAC分别比较，不能默认这种近似永远无害。[^4]

若反演接受率很低或当前噪声无法覆盖所需纠正，就停止这条路线，继续输出端actor。OVCR latent不必成为噪声反演失败后的默认替代：它的反演与可控性证据更弱。

## 来源

[^1]: [HIL-SERL builder与loss](https://github.com/rail-berkeley/hil-serl/blob/c32939bccb65f3b8c43a9f9add3d322d4ab0264a/serl_launcher/serl_launcher/agents/continuous/sac.py)、[Policy Decorator](https://github.com/tongzhoumu/policy_decorator/blob/92ba9ba442587ae5989c286355ace2200a0537fb/online/pi_dec_diffusion_maniskill2.py)、[DSRL使用的SAC实现](https://github.com/ajwagen/stable-baselines3-dsrl/blob/10e5d311b5c42fe571ff3eeb72f622ca616a149d/stable_baselines3/sac/sac.py)。本文组合方案与这些来源的不同处已明确列出。
[^2]: 本地 [scheduler_continuous.py](../../../src/ahawam/models/wan22/schedulers/scheduler_continuous.py)；固定点方法参考 [FlowDAgger反演代码](https://github.com/microsoft/FlowDAgger/blob/d2e60b5bf5d39538783f1e73d460394b21eb0f28/flowdagger_pi05/flow_matching_inverter.py)。
[^3]: [FlowDAgger配置](https://github.com/microsoft/FlowDAgger/blob/d2e60b5bf5d39538783f1e73d460394b21eb0f28/flowdagger_pi05/train_flowdagger.py)、[DSRL官方代码](https://github.com/ajwagen/dsrl)。
[^4]: [UniSteer论文，算法1](https://arxiv.org/html/2605.10821v1)。这里更保守地先分离BC标签与TD数据，是本项目建议，非论文原算法。

本地统计来源见 [audit_snapshot.json](audit_snapshot.json)；外部参数完整对照见 [文档02](02_METHODS_AND_VERIFIED_DEFAULTS_ZH.md) 和 [来源清单](SOURCES_ZH.md)。
