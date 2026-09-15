# 无需重训的推理优化：可做的工程与可主张的贡献

## 1. 第三贡献不必临时凑成加速算法

推荐第三项首先是**低成本真机上的恢复行为与部署条件分析**。若今天得到可复现的加速和闭环收益，可以增加系统贡献；仅修改步数或启用编译器通常属于实现优化，不足以单独构成强算法贡献。

尤其AHA-WAM原论文§3.5已经描述异步planner/executor、TensorRT/CUDA Graph及循环不变量优化；其Flash还涉及蒸馏。复用这些可以改善当前部署，但应注明继承关系，不能把“异步双分支”或通用图捕获写成本项目首创。[1](https://arxiv.org/html/2606.09811v1)

对当前工作，更有联动的研究问题是：**纠正能力在不同反馈延迟下如何变化，后训练收益是否能在有限控制计算预算下保留？** 这样速度分析与“接触偏差后的闭环恢复”相连，而不是一段与方法无关的infra故事。

## 2. 当前代码已经做了什么

本轮读取的vidext部署配置为history6、action horizon40、chunk8、10步动作求解、缓存文本embedding；推理走action-only服务。以下是代码路径事实，不代表已测得每段耗时。

| 路径 | 当前行为 | 优化时不能误判的地方 |
|---|---|---|
| `ahawam_policy.py:get_action` | horizon用尽后soft reset，并在需要时同步prefill视频 | 一个40步horizon含5个8步chunk；prefill请求与普通chunk不能混为相同延迟 |
| `ahawam.py:_prepare_inference_chunk_conditioning` | 当前图像/本体条件及OVCR更新在去噪循环外准备 | 不能声称首次把OVCR移出每个采样步；当前已经如此 |
| `ahawam_chunk_base.py:infer_action_chunk` | 当前仅去噪8步chunk，再更新状态 | 返回8步不代表每次生成40步后丢掉32步 |
| `ahawam.py:_prepare_inference_cross_attn_kv_cache` | 当前AHA实现返回None | 下层有缓存接口，但此路径未启用；是可检查的复用机会 |
| `_build_prefilled_action_pre_and_mask` | 每次去噪预测调用时构造pre-DiT输入/attention mask | 形状/位置不变的mask可能缓存；噪声依赖token、时间调制不可缓存为常量 |
| `wan_video_dit.py:flash_attention` | 名称为flash，实际compatibility路径调用PyTorch SDPA | 不能仅凭函数名认定用了某个FlashAttention kernel，也不必先装另一套库 |
| 文本缓存与preview | 文本encoder已可不加载；preview有独立配置 | 降preview步数不会加速未调用preview的动作服务 |

仓库还有async server/runtime，但当前启动脚本未自动启用该路径。发现实现入口不等于当前模型已在双GPU异步执行，更不等于同任务硬件资源下可复现原论文速度。

## 3. 今天最可行的三个层次

### A. 同一checkpoint的10/8/6步扫描：最快验证

仅改变 `num_inference_steps`，保持图像尺寸、history、chunk、缓存更新、噪声规则和部署资源不变。先离线比较同观测和初始噪声下的动作偏差，再在预先固定开发初始化测6步是否失稳；候选筛选后做独立正式复测。

不建议直接跳到1–2步并借用Flash结果：未蒸馏checkpoint没有少步性能保证。降低步数会改变数值积分，可能牺牲精密插入的动作质量；离线动作接近只是检查，不证明闭环成功不变。

如果端到端延迟写成

$$
T(N)=T_{\mathrm{fixed}}+N T_{\mathrm{step}},
$$

10到6步的理论加速是$(T_{fixed}+10T_{step})/(T_{fixed}+6T_{step})$，不是自动1.67倍。若视频prefill、VAE、网络等待占大头，动作步数减少的收益会更小。

论文定位：计算—性能折中实验。不是新求解器。若1/13、5/13、6/13的原主结果均为10步，改步数后的结果需另标，不覆盖原记录。

### B. 请求内不变量缓存与局部编译：最值得检查的工程

先考虑语言投影与每层cross-attention K/V、与噪声无关的mask/位置索引。代码已有 `_compute_cross_attn_kv_cache_for_tokens`、`MoT.compute_cross_attn_kv` 和下游cache参数，可作为起点。

但不能只把返回None改成某个cache：`skip_context_embedding`与obs/proprio融合路径必须同步核对，缓存的键至少覆盖模型权重、任务文本、device/dtype；若包含观测/本体信息则每次观测更新失效。原生OVCR的当前视觉路由也必须随新观测更新，不能跨请求复用而错过纠偏信号。

对循环内固定张量路径可试局部 `torch.compile`，再考虑CUDA Graph。图捕获要求稳定buffer、合适的形状与输入生命周期；当前history有效长度、chunk位置、字典状态变更及CPU同步会增加工作量。[2](https://docs.pytorch.org/docs/stable/generated/torch.compile.html) [3](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)

预期最有可能复用的是固定8步action chunk的热点，而非整个包含网络服务与可变history的Python对象。不要升级整套CUDA/PyTorch依赖来赶论文；先用现有环境和隔离入口。该项本轮未实现，因此没有缓存等价或加速结果。

需要的验证顺序：同checkpoint/同输入/同初始噪声，逐步flow与最终动作误差；覆盖chunk0–4、history冷启动到满、任务切换、episode reset；然后测暖机后p50/p95；最后检查真机闭环。浮点执行顺序改变时应报告误差阈值，不轻率称bitwise等价。

### C. 利用现有异步结构减少HOLD：潜在收益大，接线风险也大

同步服务在prefill边界可能产生长等待。复用已有双进程planner/action路径可能降低这些停顿，但需保证：K/V版本原子切换、观测时间戳与chunk索引一致、接管/重置时清理旧请求、已承诺动作不被重复执行。使用额外GPU还需计入总资源。

这是在AHA-WAM既有架构上的部署实现，不是新异步算法。应比较相同模型、10步求解下同步/异步的真实观测年龄、HOLD占比和恢复成功率，不能只比较一段网络forward时间。

## 4. 相关免重训方法为何不能直接当开关

| 方法 | 原工作机制 | 对AHA-WAM的判断 |
|---|---|---|
| RTC [4] | 推理时对已承诺动作前缀进行约束，生成与之连续的新chunk，掩盖推理延迟 | 主要改善执行时序；guidance可能增加单次计算。当前按8步chunk逐个生成，缺少可直接复用的完整未来尾部，需明确重叠动作和时间轴 |
| RTI-DP [5] | 使用此前预测作为后续去噪初值，减少重复求解 | 原方案并非自动适配当前flow时间表；重新加噪、tail补全和条件突变需处理，尤其纠错时旧计划可能恰好错误 |
| AHA-WAM原加速 [1] | 静态热点、缓存/图优化；另有需训练的Flash | 最适合向已有合作实现对齐，但需区分继承工程与本论文贡献 |
| τ₀-WM候选筛选 [6] | 自一致评分和动作条件世界模拟器辅助修正 | 当前没有对应已校准价值/模拟接口；多候选可能更慢，自一致也不能自动识别共同偏差 |

RTC和RTI-DP都提供“无需重训”的先例，但无需重训不等于无需改部署。当前不建议今天从零移植RTC并立即用于正式主表。先profile现有等待来源，才能知道应减少求解成本还是掩盖通信/prefill延迟。

## 5. 更有WAM联动的候选：按观测变化刷新planner

**候选，未实现也未验证。** 慢视频上下文负责稳定任务信息，最新观测经OVCR修正动作。可研究固定刷新频率是否在大偏差时产生过旧条件，以及是否在低变化阶段浪费prefill。

一种原型定义：从已有视觉query提取池化特征q，以最近prefill时的q_ref计算归一化差异

$$
d_t=\frac{\|q_t-q_{ref}\|_2}{\|q_{ref}\|_2+\epsilon},\qquad
\mathrm{refresh}_t=\mathbf1[d_t>\delta\ \lor\ \mathrm{age}_t>A_{max}].
$$

阈值由独立开发轨迹确定，可配滞回避免频繁切换。光照/腕部运动也会改变特征，静态图像不保证接触状态不变，故d不是成功概率或不确定性估计。当前chunk horizon与RoPE位置假定固定节奏，提前/推迟prefill必须同步重设计划索引和实际时间位置；不能简单删掉每五chunk的reset。

这比一般编译器更贴近WAM慢/快分支，但属于较高风险的新设计。需与固定刷新、相同平均prefill预算的对照比较，才能判断收益来自触发规则还是更多视频计算。已有AHA异步/OVCR解决相近问题，不能在没有更完整检索和实测前声称新颖。适合后续继续做，不建议作为今天必须拼出的第三点。

## 6. 不改模型也能完成的强实验：纠正策略×反馈延迟

对SFT和选定纠正模型，采用同一动作求解设置，设置原始延迟、增加100ms、增加200ms三个受控条件。数值先在开发阶段确认仍属可执行范围，再冻结。延迟实验可以从局部恢复起步，独立于完整任务主表。

RTC本身分析了注入延迟下的行为，支持将闭环延迟作为实验变量；但其机器人、频率和chunk长度不同，不能直接移用其结果或保证本设置同样安全有效。[4](https://arxiv.org/html/2506.07339v1)

要固定注入位置：请求返回后、首个新动作执行前，或观测发送前，只选一种并记录。action-only同步客户端可能把等待变成HOLD，异步客户端则可能继续旧动作；两者是不同实验，不能混称“延迟增加”。报告观测采集到动作执行的年龄、HOLD时长、阶段成功率与完成时间，而非仅服务端latency。

若后训练在延迟扰动下更稳，可作为明确范围内的恢复鲁棒性发现；如果只在原始时序下有收益，同样说明适用边界。外加延迟不会证明对机械精度漂移鲁棒。

## 7. 计时与公平比较协议

1. 固定checkpoint、任务文本、stats、图像尺寸、history、chunk、seed规则、GPU与服务部署模式。服务器同时训练会影响延迟，应单列共载与空闲结果。
2. 用真实已记录输入做至少5次暖机，随后固定输入序列测至少50–100次；分别覆盖prefill和非prefill请求。数量是本项目预算建议，不能以重复同一帧代替闭环评估。
3. 拆分图像处理/VAE、视频prefill、OVCR、动作去噪、传输/客户端等待。CUDA热点用events或恰当同步，端到端用客户端实际时间戳；不要把未同步CPU计时当GPU运行时间。
4. 报告p50/p95、prefill次数、峰值显存、冷启动/编译成本和实际使用GPU数。主速度数字应涵盖实际动作服务所需部分，明确是否包含网络。
5. 每个速度候选做独立成功率复测；少步带来的质量变化、异步带来的状态年龄变化与等价缓存优化分开呈现。

没有profile就无法合理承诺“今晚能快几倍”。当前文档只给出代码可行性和验证协议，没有进行GPU性能测量。

## 8. 第三贡献的分情形表述

| 截稿前证据 | 推荐第三贡献 |
|---|---|
| 两任务改善 + 局部恢复图 | 真实低成本机器人上的任务完成与偏差恢复评估 |
| 再有延迟扫描 | 分析纠正WAM在反馈时延下的恢复范围 |
| 精确缓存优化确有加速且成功率保持 | 可复现的部署实现与计算—性能分析，注明继承的优化思想 |
| 新planner触发策略有同预算对照 | 可讨论自适应视觉更新方法，但需重新定位相关工作 |
| 只有开compile或减少步数 | 实现细节/附录，不作为独立算法贡献 |

最稳妥的选择是先把已有纠正能力解释清楚。临时加速不应占掉第二任务和一个关键机制消融的全部测试时间。

## 代码与文献来源

- [ahawam_policy.py](../../../deploy/server/ahawam_policy.py)：`get_action`、prefill与chunk调度。
- [ahawam_chunk_base.py](../../../src/ahawam/models/wan22/ahawam_chunk_base.py)：`infer_action_chunk`和循环内预测。
- [ahawam.py](../../../src/ahawam/models/wan22/ahawam.py)：当前条件构造及cross-attention缓存关闭。
- [mot.py](../../../src/ahawam/models/wan22/mot.py)、[wan_video_dit.py](../../../src/ahawam/models/wan22/wan_video_dit.py)：K/V缓存接口、SDPA调用。
- [当前vidext部署配置](../../../scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_full_balanced_video_vidext.yaml)、[异步runtime](../../../deploy/server/async_runtime.py)：配置与备用路径，非性能实测。

1. Cai et al. [AHA-WAM](https://arxiv.org/html/2606.09811v1), arXiv 2026，§3.5。通用加速与异步架构均需承认先例。
2. PyTorch, [torch.compile](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)，官方当前文档；具体支持以服务器安装版本为准。
3. PyTorch, [Accelerating PyTorch with CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/), 2021。静态捕获与buffer语义。
4. [Real-Time Execution of Action Chunking Flow Policies](https://arxiv.org/html/2506.07339v1), 2025，§4、实验与附录A.3。RTC的时序目标、guidance开销和端到端计时区分。
5. Duan, Yin, Kragic, [Real-Time Iteration Scheme for Diffusion Policy](https://rti-dp.github.io/), IROS 2025。温启动去噪；项目页描述的结果主要为模拟验证。
6. Zhou et al. [τ₀-WM](https://arxiv.org/html/2606.01027v1), arXiv 2026，§VI。其动作条件模拟和候选评估不是当前AHA配置已有能力。
