# cylinder-v1：无记忆 SFT 之上、加入 09-11 纠正片段的四组纠正训练

2026-09-11 按用户要求：取消原计划在无记忆 SFT 之后运行的 `p1_full_balanced_video_nohist`，改为在同一个无记忆 SFT 上依次运行四个实验，并把新录制的纠正片段并入 HUMAN 纠正数据。

## 1. 取消与排队方式

- **取消**：`runs/ovcr_adapter_cylinder_v1/configs/p1_full_balanced_video_nohist.yaml` 已移到 `configs_cancelled/`。正在运行的 `sft_nohist_pretrained_4ep` 不受影响；队列在 SFT 正常结束、记录为 completed 之后，读取下一个配置时报 `Unknown experiment` 并停止，因此旧队列最终状态会是 `stopped`——这是预期行为，不是 SFT 出错。
- **新队列**：tmux 会话 `ovcr_cylinder_nohist_clips` 先运行 `wait_for_queue.sh`，等 SFT 队列进程退出后检查三件事：SFT 记录为 completed 且退出码 0；`sft_nohist_pretrained_4ep/.../step_006572.pt` 存在；GPU 0/1/2 各有 ≥36GB 空闲。任一不满足就不启动（已用假目录测过成功、SFT 失败、队列被杀、权重缺失四种情况）。满足后按顺序运行：

```
p1_full_balanced_video_nohist_clips → p1_v32_balanced_nohist_clips
  → p1_q32_balanced_nohist_clips → p1_t4_balanced_nohist_clips
```

## 2. 数据：09-11 片段作为纠正数据，而不是 demo

`data/so101_ahawam_cylinder_insertion_v1/01_pick_up_and_insert_cylinder/20260911_*` 共 4 个 session、101 条、4,706 帧（长度 31–120，中位数 45）。它们是从偏离插入位姿开始的遥操作纠正，存储格式与普通 demo 相同（没有 HIL 的 `control_source` 等列），所以新增 `SO101CorrectionClipDataset`：**每条 episode 整体视为一个 HUMAN 片段**，`data_source=1`，episode 结尾按原有 padding 截断，与 HIL HUMAN 片段合并为同一个纠正来源（`ConcatCorrectionDataset`）。它们不在 demo 训练目录里（生成器会断言），也不计入 demo。

| 来源 | 训练帧 | 片段 | 完整 49 帧视频窗口起点 |
|---|---:|---:|---:|
| HIL HUMAN（冻结的训练切分，不变） | 2,393 | 77 | 271 |
| 09-11 纠正片段（全部用于训练） | 4,706 | 101 | 307 |
| **纠正来源合计** | **7,099** | 178 | 578 |
| demo（同样 4 个 session，不变） | 19,708 | 94 | — |

- 自然混合中纠正占比从 10.8% 升到 26.5%；1:1 采样下纠正起点从约 8.2 遍降到约 **2.78 遍**，过拟合风险明显降低。
- 视频损失规则与之前完全相同：`p1_full_balanced_video_nohist_clips` 中，纠正样本只有当前起点的 13 张采样帧全部落在片段内（即剩余 ≥49 帧）才计算视频损失，其余只算动作；三个 adapter 实验 `lambda_video=0`、原模型冻结，本来就没有视频损失。
- 验证集不变：demo 为 `20260904_024009_val`，HUMAN 为同样 13 条 held-out HIL rollout。09-11 片段没有留出验证部分——这是为了让曲线与之前所有 cylinder 实验可比；代价是没有专门衡量"对这类纠正片段的拟合"的指标。
- HIL 目录中新增的 `20260910_152524_val`（6 条 rollout，328 个 HUMAN 帧）**不参与训练**：HIL 切分直接复制自冻结 manifest，生成器还校验了原 HIL parquet 的 sha256 未变。它可用于之后额外评估。

## 3. 四个实验与之前的差异

每个配置都复制自对应的 history-6 cylinder 配置，只改：

| 项 | history-6 版本 | 本轮 |
|---|---|---|
| 初始化 | 原 cylinder SFT `step_006572` | 无记忆 SFT `sft_nohist_pretrained_4ep/step_006572` |
| `num_history_frames` | 6 | 0 |
| 纠正来源 | HIL HUMAN 2,393 帧 | HIL HUMAN + 09-11 片段 = 7,099 帧 |

其余不变：1:1 demo/纠正 batch、3285 步（demo 一遍）、全局 batch 12、全量 lr 1e-5 / adapter lr 1e-4、adapter 的 P0 anchor λ=1、每 1095 步存权重、每 219 步验证（48 loss + 12 动作样本/来源）。wandb：project `ahawam-so101-cylinder-hil-v1`，group `cylinder-v1-nohist-clips`。

注意：这里同时改变了"起点（无记忆 SFT）"和"纠正数据（加入片段）"两件事。要单独归因"加片段的作用"，需要把本轮与一个"无记忆 SFT + 仅 HIL HUMAN"的对照比较——那正是被取消的 `p1_full_balanced_video_nohist`；需要时可把它从 `configs_cancelled/` 移回并单独排队。

## 4. 命令与产物

```bash
# 生成（已执行；拒绝覆盖）
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/prepare_nohist_clips.py
# 真实数据检查：片段边界上的视频 mask、来源标签、混合 batch
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/check_clip_data.py \
  --config runs/ovcr_adapter_cylinder_v1_nohist_clips/configs/p1_full_balanced_video_nohist_clips.yaml
# 排队（已执行）
bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_nohist_clips_after_sft.sh
# 查看
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/ovcr_adapter_cylinder_v1_nohist_clips
tmux attach -t ovcr_cylinder_nohist_clips
```

manifest：`runs/ovcr_adapter_cylinder_v1_nohist_clips/data_manifest.json`（含每条片段长度与 parquet sha256）；输出：`runs/ovcr_adapter_cylinder_v1_nohist_clips/<实验名>/model/`。

部署这些权重时同样要设 `num_history_frames: 0`，并使用 cylinder stats 与原任务指令。

## 5. 运行记录（2026-09-11）

**无记忆 SFT 已成功完成**：`sft_nohist_pretrained_4ep` 于 08:30 跑满 6572 步，退出码 0，4 个权重齐全，日志无 traceback/OOM/NaN。同一固定子集（48 loss + 12 动作样本/来源）上的验证：

| step | demo action loss | demo action L1 | HUMAN action loss | HUMAN action L1 |
|---:|---:|---:|---:|---:|
| 0（预训练） | 1.7050 | 11.6237 | 1.4639 | 11.5316 |
| 1640（约 1 epoch） | 0.0990 | 1.9588 | 0.0612 | 1.5547 |
| 3280 | 0.1363 | 2.1549 | 0.0827 | 1.7390 |
| 4920 | 0.1540 | 2.0938 | 0.0909 | 1.6926 |
| 6572（最终） | 0.1628 | 2.1170 | 0.0902 | 1.6122 |
| 对照：原 history-6 SFT 最终 | 0.1694 | 2.2979 | 0.0795 | 1.5563 |

最终权重与原 SFT 相比：demo 略好，HUMAN 略差，差距都不大。teacher-forcing 动作损失在约 1 个 epoch 时最低、之后回升，而 10 步推理的动作 L1 基本持平；是否意味着应该选更早的 SFT 权重，需要真机或更大验证集判断。按用户要求，后续实验仍从最终 `step_006572.pt` 出发。

**第一次交接失败**：等待脚本在 SFT 结束后（约 08:31）检查 GPU 空闲显存时，`nvidia-smi` 暂时输出了以 `Failed` 开头的 NVML 错误而不是数字；脚本在 `set -u` 下把它当成未定义变量直接退出（`line 27: Failed: unbound variable`），四个实验没有启动，GPU 空闲约 6.5 小时。当时的测试没覆盖到这种暂时性错误。

**修复**：`wait_for_queue.sh` 与 `launch.sh` 一样先加载 `use-nvidia-580159.sh`，并把非数字读数视为"尚未就绪"、记录原文后每 20 秒重试（最多 30 分钟）；`launch.sh` 的显存检查也改为最多重试 10 次（每次间隔 30 秒），避免之后四个实验中任何一个在启动时因同样的暂时性错误整队停止。两处都用"前两次返回 NVML 错误、之后正常"的假 `nvidia-smi` 测过：均重试后通过，`launch.sh` 随后正常拉起三进程。

**重新启动**：15:09 重新运行 `launch_nohist_clips_after_sft.sh`，等待脚本立即通过，队列开始运行 `p1_full_balanced_video_nohist_clips`。第一个实验已确认正常训练：`DATA_LENGTHS [19708, 7099, 661, 674]`，step 0 验证与无记忆 SFT 最终值完全一致（说明严格加载成功），`history_len=0`，wandb run [ui5v5ro4](https://wandb.ai/fenghaoliu8-university-of-michigan/ahawam-so101-cylinder-hil-v1/runs/ui5v5ro4)。实测约 9 s/step：数据加载约 0.01 s、前向约 1 s，与无记忆 SFT 相同；反向 4.8–6.8 s，与 history-6 的 `p1_full_balanced_video` 相当，所以这是 1:1 纠正全量训练在本管线中的正常速度（此前按 SFT 速度估计偏乐观）。按 history-6 同名实验的实际用时推算：全量实验约 09-11 22:45 结束，三个 adapter 合计约 10.1 小时，四个实验预计 09-12 08:52 左右全部结束（估计值）。

## 6. 按用户要求停止（2026-09-11 15:27）

四个实验的队列已全部停止：`p1_full_balanced_video_nohist_clips` 停在约 94/3285 步，还没有保存任何权重；三个 adapter 实验没有启动。`runs/ovcr_adapter_cylinder_v1_nohist_clips/suite_status.json` 记为 `stopped`。上面给出的完成时间估计已作废。

停止时发现 `suite.py` 本身的 bug：训练进程收到 SIGTERM 后正常退出，但 `suite.py` 在信号处理函数里又调用了 `child.wait()`，而主线程被打断时正阻塞在同一个 `wait()` 里、持有 Popen 的内部锁，结果重入等锁死锁（进程停在 `futex_wait_queue_me`，子进程变成僵尸），状态文件一直显示 running。已手动清理（训练进程确认全部退出后对 `suite.py` 发 SIGKILL，并如实更新状态文件），并修复：信号处理函数现在只负责向实验进程组发 SIGTERM；由 `finally` 在锁释放后回收子进程（最多等 180 秒，超时再 SIGKILL），并把运行中的记录标为 `stopped`。用一个带长时间子进程的最小复现验证过：旧写法同样卡在锁上，新写法立即退出并回收子进程。所以文档里"向 `suite.pid` 发 SIGTERM 停止队列"的方法现在是有效的。

要恢复这四个实验：`p1_full_balanced_video_nohist_clips/` 目录已经存在（只有日志和配置快照，没有权重），队列拒绝覆盖已有输出目录，所以重新运行前需要先把这个目录移走或删掉。
