# cylinder-v1：OVCR 纠正训练实验（2026-09-08）

把 power-adapter v2 上已经跑完的那套纠正训练协议，搬到 cylinder 插入任务：从 cylinder 任务 SFT `step_006572.pt` 出发做 V32 / Q32 / T4 / 全量四组 1:1 纠正训练，再从 AHA 预训练权重出发做一次自然混合四轮全量 SFT。本轮新增 wandb 在线记录，并把固定验证子集扩大到每来源 48 个 loss 样本 + 12 个动作推理样本。

- [数据、切分、训练预算与实验设置](01_PROTOCOL_ZH.md)
- [启动记录、验证结果与限制](02_EXECUTION_ZH.md)
- [关闭视频记忆（num_history_frames=0）的 SFT 与纠正训练](03_NOHIST_ZH.md)
- [无记忆 SFT 之上、加入 09-11 纠正片段的四组纠正训练](04_NOHIST_CLIPS_ZH.md)
- [原 SFT（history 6）上加入 09-11 纠正片段的三组 adapter 训练](05_CLIPS_HIST6_ZH.md)
- [只用 09-11 recover 短片段（裁掉前 0.5 秒）的三组 adapter 训练；含裁剪机制与历史帧实测](06_RECOVER_CLIPS_ZH.md)
- [训练管线逐层讲解：一个 sample 是什么、边界怎么处理、一个 step 发生了什么](07_TRAINING_PIPELINE_ZH.md)
- [HIL v2：历史可跨越接管边界的纠正训练，含历史间隔的选择依据](08_HILV2_HISTORY_ZH.md)
- [抓取对准后不合爪的诊断，以及纠正样本窗口改造（视频/动作延伸到接管之后）](09_GRASP_STALL_DIAGNOSIS_ZH.md)
- [ICRA 截止前：历史 KV 审计、局部恢复采集、最小消融与投稿分支](04_ICRA_RECOVERY_AND_MEMORY_PLAN_ZH.md)

数据事实（已冻结在 `runs/ovcr_adapter_cylinder_v1/data_manifest.json`）：demo 训练 19,708 帧 / 验证 661 帧（`20260904_024009_val`）；DAgger 80 条 rollout 中人类接管 3,067 帧、94 段，划分为训练 2,393 帧 / 77 段与验证 674 帧 / 17 段。自然混合下 HUMAN 只占 10.8%，比 padapter 轮的 21.7% 更少，因此 1:1 采样会把 HUMAN 起点重复约 8.2 遍，过拟合风险更高——这是本轮读结果时首先要看的地方。

上一轮的方法学讨论（RLT 与本实现的区别、三种注入位置的机制与容量限制、是否需要 critic）仍然适用，见 [../v2_experiments/05_MECHANISMS_RL_AND_DEPLOYMENT_REVIEW_ZH.md](../v2_experiments/05_MECHANISMS_RL_AND_DEPLOYMENT_REVIEW_ZH.md)。本轮属于**带 replay 的纠正模仿学习**，没有 reward、critic 或在线 RL 更新。

代码入口在 `scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/`（配置生成器与队列启动脚本），训练/队列本体复用 `../v2_experiments/` 中已经跑通的 `run.py`、`suite.py`、`launch.sh`。
