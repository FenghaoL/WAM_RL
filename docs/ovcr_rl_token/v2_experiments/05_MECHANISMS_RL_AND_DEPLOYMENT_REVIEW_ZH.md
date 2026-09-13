# V32 / Q32 / T4 实现复核、RL 路线与部署审计

更新：2026-09-07。本文区分代码事实、论文方法与待验证推断。本次没有启动额外训练或切换在线服务。

## 1. 先确认现在运行的究竟是什么

V32 / Q32 / T4 和混合全量微调均已完成；全量任务 `p1_full_balanced_video` 退出码0，完成2664步。P2/P3 消融尚未运行。

`online_deploy/settings_dagger.yaml` 当前指向 **V32 step2664**，不是 T4，也不是新全量模型。stats 为正确的 power-adapter-v2 stats，路径和6维统计通过 preflight。

但当前 `online_deploy/run_action_server.sh` 硬编码：

```bash
DEPLOY_CONFIG="scripts/so101_pipeline/online_deploy/settings.yaml"
INSTRUCTION="Pick up the cylinder and insert it into the matching hole."
GPU_ID=2
PORT=12002
```

所以直接执行该脚本，不会读取 `settings_dagger.yaml`。不能由磁盘上有一份正确 YAML 推断实际进程加载了它。若其他入口显式传入了 `settings_dagger.yaml`，则以那个入口和实际进程为准；本次没有据此断言过去的所有真机测试都用错了模型。

要用此脚本测 power-adapter，应把其 config 指向 `settings_dagger.yaml`，并使用准确指令：

```
Remove the power adapter from the holder and plug it into the power strip.
```

客户端请求中的 instruction 会覆盖默认指令；客户端连接的隧道端口也必须对应12002。配置修改后需重启服务，已运行进程不会自动热加载 YAML。

## 2. 原生 OVCR 计算位置

代码：`src/ahawam/models/wan22/mot.py::LayerwiseChunkKVCacheEditor.build_layer_updated_cache`。

省略 batch/chunk/head 维度，设观测 queries 为 X，video cache 当前帧槽位为 K0/V0：

```
X: 32 × 4096
K0,V0: S × 3072             # 24 heads × 128
q = original_query_proj(X)
R = softmax(q K0ᵀ / √dh) V0
D = softmax(K0 Rᵀ / √dh) R  # 实际按 head 分别计算
ΔK0, ΔV0 = original_delta_MLP(D)
Kbase = K0 + gate · ΔK0
Vbase = V0 + gate · ΔV0
```

第一轮 attention 把原 video 特征读到32个观测 query；第二轮把读到的特征写回 S 个空间槽位。action DiT 使用这些 K/V，并保留自身已有的其他注意力路径。

当前 `current_only` 推理从缓存中切出 `:video_tokens_per_frame` 后编辑；不能描述成 adapter 任意改写整段预测未来视频。缓存可以带有模型历史/上下文，但这条接口操作的是当前帧槽位。

三种 adapter 都挂在该 editor 上，使用相同的训练与推理入口，不原地修改共享 cache，不增加 action/video 注意力序列长度。只在零基编号20–29层启用，各层参数独立。

## 3. V32：改写 value 内容

代码：`ovcr_adapter.py::ZeroResidual / OVCRAdapter.edit_cache`。

```
h = SiLU(Wdown LayerNorm(D) + bdown)   # 3072 → 32
ΔV = Wup h + bup                      # 32 → 3072
Knew = Kbase
Vnew = Vbase + ΔV
```

32 是瓶颈宽度，不是动作数量、query 数量或32个 RL token。可训练参数2,058,560；整个原模型、原 OVCR、观测编码器冻结，video loss 为0。

优点：直接纠正传给 action DiT 的内容；本层的 K 不变。注意只保证本层 adapter 不显式改 K，不能保证后续 action 隐状态与注意力权重完全不变。

限制与风险：

- 输入只有已解码的 D；若原路由已丢失腕部接触细节，V32不能凭空恢复信息。
- 只提供宽度32的特征残差；仅调整最后10层，无法重新训练低层视觉表征。
- 新残差加在原 gate 之后，**不乘原 gate**，scale=1 也不代表幅度有上界。既可能太弱，也可能干扰动作；需实际测 `||ΔV||/||Vbase||` 和动作偏移，而非凭名字判断稳定。

## 4. Q32：改变观测 queries 的路由

代码：`OVCRAdapter.edit_queries`。

```
Xnew = X + Wup SiLU(Wdown LayerNorm(X) + bdown) + bup
                         4096 → 32 → 4096
```

Xnew 再通过原 query projection、两轮 attention 和 delta MLP，因此间接改变 K/V。可训练参数2,744,640。

这里 Q 指 attention query，**不是 RL 的 Q-value/critic**。它作用于观测 query，不直接修改 action DiT 的 attention query 权重。

优点：原模型有可用视觉信息但路由不正确时，可能用少量参数重新定位相关内容。

限制：梯度需穿过冻结的 LayerNorm、attention、delta MLP 和可能较小的原 gate；softmax 饱和、原 query 投影的局部不敏感方向都可能削弱效果。这是机制推断，还没有通过真实 checkpoint 的 Jacobian 测量证实。代码测试与真实训练证明有梯度，不能据此认定路由改动足够大或足够正确。

## 5. T4：四个读出 token，再写回 K/V

代码：`TokenWriter.forward`。每层每个动作 chunk 独立执行：

```
F = concat(Pobs(X), Proute(R))        # (32+32) × 32
Z = softmax(E Fᵀ / √32) F           # E 为4×32可学习查询；Z 为4×32
A = softmax(Pspatial(D) Zᵀ / √32)   # S×4
[ΔK, ΔV] = Up(A Z)                  # S×(2×3072)
Knew = Kbase + ΔK
Vnew = Vbase + ΔV
```

可训练参数5,511,360。T4 的4指这里四个内部 token；它们不是4个动作，不是直接追加到 video/action Transformer 的四个序列位置，也没有独立的奖励目标。

一个具体容量限制：对固定输入、固定层和 chunk，输出的空间行是四个 token 经线性 up 后的凸组合（bias 可以并入四个原型）。所以这次 K/V 空间残差至多由四个原型组合而成；不意味着整个非线性模型只有四维，但意味着细粒度空间修正受到强约束。

其他风险：四个 token 可能收敛到相似内容；softmax 可能过于均匀或过尖；K/V 联合改写比单改 V 更容易改变路由；参数数目又多于 V/Q，所以首轮不能严格归因于 token 结构。

**它不是 RLT 复现。** 当前没有训练 T4 重建 VLA 表征，也没有 actor、critic、TD target、环境 reward 或 replay-based RL 更新。独立的 `RLTokenReadout` 文件是另一个重建读出原型，未接入这些已训练 checkpoint 的 RL 控制流程。

## 6. 三者共同的训练机制与可能瓶颈

所有 up weight/bias 零初始化，初始输出等于 P0。初始化第一步主要由 up 层得到非零梯度，down/读出分支会在 up 非零后逐渐获得梯度；这是零残差设计的代价，不是永久梯度断开。

目标：

```
L = L_action_flow(demo + HUMAN)
  + λ · MSE(v_student, v_frozen_P0) on valid demo actions
```

teacher/student 使用同一噪声与时间步；关闭 adapter 获得 P0 teacher。λ=1，anchor不作用于 HUMAN 标签。它约束的是 flow velocity，不是环境回报，也不保证闭环轨迹相同。rank32、最后10层、λ=1、lr1e-4 都是本地待验证选择，不是 RLT 论文给出的 AHA 最优参数。

目前效果不明显，优先调查以下原因，而不是直接断言参数不足：

1. **实际加载路径/指令不一致**：见第1节，是已确认的磁盘入口差异；实际测试是否受影响尚待运行记录核实。
2. **训练/部署历史不一致**：HUMAN逻辑分段会把接管前历史也截掉；训练多数短纠正窗口历史很少，部署中可能保留更长缓存。可以尝试保留同一原episode内真实的接管前观测作为 context，动作监督仍严格只来自 HUMAN；需重新处理不等间隔时间与 RoPE，不能直接拼接暂停帧。
3. **纠正数据量与完整恢复过程不足**：4,437训练帧只有约296秒；帧高度相关，很多短段只覆盖局部调整，未必包含重新稳定执行到成功的完整过程。增加数据次数比盲目重复帧更可能补足覆盖。
4. **offset 与 padding 减少有效目标**：短段在offset0–7后可能剩下很少甚至没有有效 action；1:1起点比例不等于有效动作 token 贡献1:1。
5. **表达能力/约束**：冻住全部原模型、只修最后10层、anchor、T4四原型瓶颈都可能限制纠正，但需去anchor/宽rank/中间层等消融证实。
6. **真实执行时序**：8动作在15Hz下只覆盖0.533秒；推理/网络若额外引入等待，实际闭环时间更长。模型shape正确不等于时序匹配。
7. **可见性/控制精度**：插入最后阶段若双相机看不到接触关系，或者标定/关节映射存在误差，单纯增加训练步数无法补足传感信息。

建议在同一记录观测序列上比较 P0 与每个adapter的动作差、K/V残差范数、T4 token相似度/attention熵，再做正常起点与困难起点的真机配对测试。没有测量之前，不把“adapter没学到”或“骨干一定有问题”写成结论。

## 7. 相关工作做的是什么

- **IWR**：对 intervention 与 non-intervention 等比例采样，并设置不平衡采样对照。它保留的 non-intervention 含自主数据，因此本项目 demo+HUMAN 只是借鉴。[原文](https://arxiv.org/html/2012.06733)
- **SIRIUS**：区分 demo、自主、干预和干预前数据，并消融来源权重；支持优先核查数据语义，而非把自主帧一律当专家动作。[原文](https://arxiv.org/html/2211.08416)
- **RLT**：先得到可用于小型RL网络的读出表征，再冻结大模型与读出，利用参考动作与在线经验优化小策略。[官方说明](https://www.pi.website/research/rlt)
- **RLT 算法细节**：critic 对动作chunk做TD学习，actor为以参考chunk为条件的Gaussian策略，带参考动作MSE约束和reference-action dropout；可加入人工干预，warmup先由VLA采集经验。[论文§IV](https://arxiv.org/html/2604.23073v1)
- **HIL-SERL**：视觉表征、RLPD式off-policy学习、demo/RL双buffer、人工干预和视觉成功分类器组成完整系统；不是只加一层adapter。[论文](https://arxiv.org/html/2410.21845v2)、[官方实现](https://github.com/rail-berkeley/hil-serl)
- **RECAP / π*0.6**：使用价值和advantage条件训练，汇总任务demo、自主经验及纠正，每轮从其pretrained policy/value重训以减小漂移。论文也明确保留“持续训练上轮模型可能有效”的可能性；四轮普通SFT并不等于RECAP。[论文§V](https://www.pi.website/download/pistar06.pdf)
- **IQL**：可以只用带回报/transition的离线数据学习Q/V，再提取advantage加权策略，说明RL并不在定义上强制在线rollout。但数据覆盖不足仍限制改进。[原文](https://arxiv.org/abs/2110.06169)

## 8. 对本项目更合适的 RL 起点

优先选一个能较可靠到达插座附近的 P0/full-SFT 策略，只把“对准并插入”定义为第一阶段RL子任务。先把失败阶段缩短、重置方式固定和成功标签做准，避免同时解决长任务奖励稀疏与残差探索。

推荐的初版接口是动作级 actor，而非马上把 actor 塞进 OVCR：

```
当前双相机+本体状态 → 冻结AHA → 观测/路由特征与参考chunk a_ref
                                  ↓
                         冻结的RL读出 z
                                  ↓
                小actor π(a | z, proprio, a_ref)
                                  ↓
                     执行动作，记录reward和next observation
                                  ↓
                   replay → Q/actor更新（大模型不反传）
```

可先用简单pooling读出做对照，再用带重建预训练的读出，防止把整套RL成败归结为token设计。当前 `RLTokenReadout` 只提供部分离线重建能力，尚缺真实特征导出、actor/critic、采集与训练闭环。

本地设计建议：actor先输出归一化6维关节目标的有界残差，与参考chunk组合；初期保持夹爪参考策略，集中探索对准动作。使用小网络、双Q/target网络、replay、参考动作正则以及HUMAN BC辅助目标。以下是方案而非已实现算法：

```
a = a_ref + bounded_residual(z, proprio, a_ref, noise)
L_actor = -Q(z, a) + β ||a-a_ref||² + λ_bc L_HUMAN_BC
```

若按8步chunk建模，next observation必须位于**实际执行完该chunk**以后。一般的可变长度目标为：

```
y = Σ(j=0…C-1) γ^j r[t+j] + γ^C (1-terminal) Q_target(x_next, a_next)
```

C是实际有效执行步数。若只执行了前3步就被接管/中止，不能仍按8步目标；未执行的未来命令不能当作发生过的动作。超时与真实终止分开，暂停/reset边界不能伪造普通transition。观测有历史/缓存时，RL状态需要包含足够的上下文/缓存阶段信息，或训练/采集统一reset规则，不能假定单个z严格满足Markov性质。

不优先做 OVCR-latent actor 的原因：一是通过冻结action DiT的多步去噪反传仍很贵；二是human关节动作没有天然对应的“正确OVCR latent”，无法直接当latent-Q的动作标签；三是critic容易利用未校准latent区域。动作级actor可直接使用实际执行的HUMAN动作做off-policy学习，接口更清楚。后续可以把RL改进蒸馏回AHA，或单独探索latent策略。

## 9. 在线 rollout、随机性和人类的分工

**在线rollout**：RLT/HIL-SERL式方案需要持续的新机器人交互，但不要求每帧更新网络。可以先“采集一批→训练→冻结评估→再采集”，再实现异步actor/learner。纯离线RL理论上可行，当前只截取的HUMAN片段不够提供可靠长期价值；首先需要审核原始轨迹的reward、实际动作、时间戳、next observation和终止定义。

**随机性**：RL的定义不是“有噪声”；确定性策略也能用RL学习。在线局部改进通常需要有控制的探索，RLT使用小固定方差Gaussian actor。建议从归一化动作残差的低频、时间相关小扰动开始，按关节范围约束，记录扰动后真正执行的动作；不要逐帧对所有关节加入大白噪声。评估时使用固定采样协议，避免把随机运气当进步。

当前seed42用于每次prefill生成动作初始噪声；相同seed重新初始化可复用噪声模板，但观测改变时动作仍改变。`seed: null`增加采样变化，不会自动产生reward优化、replay或探索策略。warmup/dropout/noise属于不同机制：reference dropout发生在actor训练输入上，不是在机器人上随机丢掉控制命令。[RLT论文](https://arxiv.org/html/2604.23073v1)

**人工接管**：不是所有RL算法的数学要求；实际SO101初期建议保留，用于恢复和提供困难状态的高质量动作。可以减少接管频率，但人类还可能负责成功判定和重置，这三种角色要分别统计。成功奖励、失败奖励和干预标签不能混为一谈；尤其不能把“没有接管”直接当成功。

最小buffer应保存：原始观测/本体状态或可复现特征、参考chunk、提议动作、实际下发且限幅后的动作、实际执行长度/时间、下一观测、reward、terminated/truncated、接管标记、策略版本和chunk/cache阶段。若包含动作残差，必须保留参考动作，才能在正确坐标中重建实际总动作。

## 10. 是否重做四轮 SFT

值得作为有明确假设的对照，但要先说明从哪里出发、四轮指什么。

| 实验 | 初始化 | 数据遍历 | 回答的问题 |
|---|---|---|---|
| 已完成full | P0 step5328 | 1:1，2664步 | 在已有任务能力上短程纠正 |
| 四轮继续训练 | 同一个P0重新开始optimizer | 自然混合4轮，约6808步 | 更多预算、较温和纠正权重是否有利 |
| 四轮重做任务SFT | `checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt` | 与上项相同自然混合4轮 | demo与纠正一开始共同学习是否减小路径依赖 |
| 四轮1:1 | P0或基座，必须分别命名 | 10656步 | 更大纠正曝光量，不能与自然4轮视作等算力 |

自然混合每轮20,421起点，全局batch12，约1702步；四轮每个HUMAN起点约4遍。1:1的四轮则是demo63,936/HUMAN63,936起点曝光，HUMAN约14.4遍，显著更易过拟合。不能只用“都是4epoch”比较。

我的优先选择是：**先验证实际部署与早期full checkpoint，再试自然混合四轮，并配一组相同预算的基座重训。** 用相同 split、stats、batch12、lr1e-5及视频mask，保存每轮和中间checkpoint。若要隔离“混合比例”的影响，再加等step的1:1对照；不要同时更换初始化、比例、训练步数后归因给一个因素。

从本地 P0 的原配置确认，它的初始化正是上述 AHA pretrained checkpoint，文件存在。基座到SO101有原有维度适配需求，应复用原P0的 `checkpoint_shape_adapt` 配置，不能直接照搬adapter实验的严格shape设置。从base重训也不是随机初始化。

RECAP的价值/advantage模块并不会因重新SFT四轮而出现；这里借鉴的是“每轮聚合数据后重新任务适配”的实验路径，而不是宣称复现π*0.6。

## 11. 已完成 full 的结果应该怎样读

以下来自修复验证mask后的同一小子集，每来源12个loss样本、3个动作推理样本：

| step | demo action loss | HUMAN action loss | demo action L1 | HUMAN action L1 |
|---:|---:|---:|---:|---:|
| 0 / P0 | 0.2431 | 0.0910 | 1.9253 | 1.2284 |
| 888 | 0.2040 | 0.0402 | 2.1124 | 0.8492 |
| 1776 | 0.2100 | 0.0471 | 1.9142 | 0.9114 |
| 2664 | 0.2322 | 0.0469 | 1.9438 | 0.8602 |

这表明纠正监督有被拟合，且更多step并非单调更好；不能据此宣布闭环成功率提高，也不能把这个小子集当上限。先比较888/1776/2664的配对真机表现。旧adapter验证是在mask修复前执行，不能直接与本表比较，应统一重新评估。

## 12. `settings_dagger.yaml` 参数结论

shape层面匹配：6D、horizon40、chunk8、maxoffset7、32queries、history6、current_only、RoPE stride4、video384×320、两相机processor、z-score、正确stats。10步推理适用于本轮未蒸馏的模型；`checkpoint_shape_adapt:false`用于严格加载已适配SO101权重是合理的。`mot_checkpoint_mixed_attn:false`只关掉推理不需要的梯度checkpoint，不代表切换算法。`text_cfg_scale:1.0`在AHA部分路径不是可随意调高的通用增强旋钮。

需要注意：

- 它目前仍是V32；若想试full，要把checkpoint指向 `p1_full_balanced_video/model/checkpoints/weights/step_00XXXX.pt`，stats不变。
- cylinder注释和preview目录是复制遗留，会误导阅读/归档；不是动作模型识别任务的依据。
- `input_control_hz:15`目前在该policy实现主要用于推导preview FPS，**不会让机器人自动维持15Hz**；真实频率由客户端控制与网络/推理延迟决定。
- 8动作、15Hz只有0.533秒执行窗口。记录观测时间、推理结束时间、首个/最后一个执行时间；如果依次等待推理再动作，暂停无法靠改这个YAML消除。
- 双相机key、裁剪/缩放、SO101关节坐标转换和reset必须与采集一致。仅把 `raw_shape` 写成正确数字，不会修复实际图像裁剪或相机位置变化。

本次只做代码/元数据审计，未发起机器人动作；当前脚本没有自动改为dagger，避免无意切换正在使用的cylinder任务。
