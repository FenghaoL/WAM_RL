# cylinder-v1：原 SFT（history 6）上加入 09-11 纠正片段的三组 adapter 训练

2026-09-11 按用户调整后的策略：不再用无记忆 SFT，改为回到原 cylinder 任务 SFT
`formal-cylinderv1-sft-ahawambase-bs4-acc1-4ep-3gpu-20260905-022135/step_006572.pt`，依次训练
V32 → Q32 → T4 三个 OVCR adapter，纠正数据为 DAgger 人类接管部分 + 09-11 新录制的纠正片段。

## 1. 与已完成的 history-6 V32/Q32/T4 的唯一区别：纠正数据多了片段

| 项 | history-6 `p1_*_balanced`（09-08 已完成） | 本轮 `p1_*_balanced_clips` |
|---|---|---|
| 起点 | 原 SFT `step_006572` | 同左 |
| `num_history_frames` | 6 | 6（与起点训练时一致；用户本轮未要求关闭记忆） |
| 纠正来源 | HIL HUMAN 2,393 帧 / 77 段 | HIL HUMAN 2,393 帧 + 09-11 片段 4,706 帧 / 101 条 = **7,099 帧** |
| 1:1 采样下纠正重复次数 | 约 8.2 遍 | 约 2.78 遍 |

其余全部相同：同样 4 个 demo session（19,708 帧），1:1 demo/纠正 batch，3285 步，全局 batch 12，lr 1e-4，P0 anchor λ=1，冻结原模型只训 adapter，无视频损失，每 1095 步存权重，每 219 步在同样的 demo val（`20260904_024009_val`）与 HUMAN val（13 条 held-out HIL rollout）固定子集上验证。因此本轮与 09-08 的三组是**单因素对照**（只多了片段数据），step 0 的验证值应与那三组完全相同。

09-11 片段按整条 episode 视为一个 HUMAN 片段（`data_source=1`），不计入 demo；其历史帧只能来自同一条片段内部，这与 HIL HUMAN 片段按接管边界截断历史的处理一致。

## 2. 启动前检查

- 数据：`DATA_LENGTHS [19708, 7099, 661, 674]`，与冻结 manifest（`runs/ovcr_adapter_cylinder_v1_clips/data_manifest.json`）一致；HIL parquet 的 sha256 与原切分时相同；片段 session 仍是同样 4 个。
- 真实片段边界与混合 batch：来源标签、视频 mask、episode 结尾 padding 正确；demo / HIL / 片段样本（含 `video_history` 等历史字段）能组成同一个 batch。记录在 `clip_data_check_p1_v32_balanced_clips.json`。
- 首次尝试启动时 GPU 2/3 被用户的部署服务占用，训练脚本要求 3 张各 ≥36GB 空闲的 GPU，因此等用户停掉 GPU 2 上的服务后才启动。

## 3. 命令与产物

```bash
# 生成（已执行；拒绝覆盖）
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/prepare_clips.py --preset hist6
# 启动（已执行）
RUN_ROOT=runs/ovcr_adapter_cylinder_v1_clips SESSION=ovcr_cylinder_clips GPU_IDS=0,1,2 \
NAMES="p1_v32_balanced_clips p1_q32_balanced_clips p1_t4_balanced_clips" \
bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_suite.sh
# 查看 / 停止
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/ovcr_adapter_cylinder_v1_clips
kill -TERM "$(cat runs/ovcr_adapter_cylinder_v1_clips/suite.pid)"
```

wandb：project `ahawam-so101-cylinder-hil-v1`，group `cylinder-v1-clips`。输出：`runs/ovcr_adapter_cylinder_v1_clips/<实验名>/model/`。

部署这些 adapter 权重时使用 `num_history_frames: 6`（与训练一致）、cylinder stats 和原任务指令。
