# power adapter HIL v3：p1_full_balanced_video 与视频延伸

2026-09-14。按用户优先级只跑两个全量纠正实验，先 `_vidext`，再普通版。训练配方与 cylinder
`p1_full_balanced_video_hilv2` 一致，便于两个任务横向对照。

## 数据与起点

| 项 | 值 |
|---|---|
| 起点 | power adapter 任务 SFT `formal-padapterv2-sft-...-20260901-001743/checkpoints/weights/step_005328.pt`（history 6）。用户给出的 `step_005328_trainer_state.json` 只是训练状态记录，权重是同目录 `weights/step_005328.pt` |
| demo | 原 SFT 的 3 个训练 session，15,984 帧；验证 `20260831_225142_val`，1,735 帧；stats `runs/so101_ahawam_padapter_v2/assets/dataset_stats.json` |
| DAgger | `data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v3`：4 个 session、91 条 rollout、36,411 帧；人类接管 5,329 帧 / 137 段；SYSTEM_HOLD 约 48%；结果标签 85 成功（有接管）/ 5 自主成功 / 1 失败 |
| 划分 | 数据没有 `_val` session，按 sha256(`20260914:session:episode`) 在每个 session 内留出约 20% 含接管的 rollout：训练 67 条 / 4,289 帧 / 110 段，验证 18 条 / 1,040 帧 / 27 段 |
| 任务指令 | `Remove the power adapter from the holder and plug it into the power strip.` |

与 cylinder hil_v2 相比，接管段明显更长（段长中位 28–60 帧，42 段 ≥49 帧），即使不延伸视频，也有相当比例的纠正样本本身就能拿到完整视频窗口。

## 两个实验

| 顺序 | 实验 | 纠正样本的未来视频窗口 |
|---|---|---|
| 1 | `p1_full_balanced_video_hilv3_vidext` | 延伸到接管之后：只走执行帧（跳过 SYSTEM_HOLD），只在成功 rollout 上；动作标签仍只用人类段 |
| 2 | `p1_full_balanced_video_hilv3` | 截止在接管段末尾（原规则） |

共同设置：全量微调（VAE 冻结）、demo:纠正 = 1:1、2664 步（demo 过一遍）、全局 batch 12、lr 1e-5、视频 + 动作损失、
历史 6 帧可跨出接管段、每步回退 48 个执行帧、每 178 步验证（16 loss + 4 动作推理 batch/卡）、每 888 步存权重、wandb
project `ahawam-so101-padapter-hil-v3`，group `padapter-hil-v3-full`。

## 命令

```bash
PYTHONPATH=src:. python scripts/so101_pipeline/ovcr_adapter/padapter_experiments/prepare_hil_v3.py
RUN_ROOT=runs/ovcr_adapter_padapter_hil_v3 SESSION=ovcr_padapter_hilv3 GPU_IDS=0,1,2 \
NAMES="p1_full_balanced_video_hilv3_vidext p1_full_balanced_video_hilv3" \
bash scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/launch_suite.sh
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/ovcr_adapter_padapter_hil_v3
kill -TERM "$(cat runs/ovcr_adapter_padapter_hil_v3/suite.pid)"      # 停止
```

部署这些权重时：`num_history_frames: 6`、power adapter v2 stats、上面的任务指令。

## 追加实验：人类段当 demo、demo:纠正 = 2:1（2026-09-14 排队）

run root `runs/ovcr_adapter_padapter_hil_v3_segdemo_2to1`，生成器
`scripts/so101_pipeline/ovcr_adapter/hil_experiments/prepare_segdemo_2to1.py`。数据清单是上面 run 的
`data_manifest.json` 原样复制（`data_manifest_padapter_hil_v3.json`，配置里记录 sha256），划分、验证集完全相同。

| 实验 | 与 `p1_full_balanced_video_hilv3` 的唯一差别 |
|---|---|
| `p1_full_balanced_video_hilv3_segdemo` | 训练纠正样本把每个人类段当独立 demo：历史只取段内帧，段内不够就按 demo 规则 padding（不取接管前的执行帧）；未来视频只取段内，4 帧 latent 步全 padding 的位置不算视频 loss；不做延伸。`correction_history_crosses_segment: false`、`correction_history_skip_system_hold: false`、`correction_future_extension: none` |
| `p1_full_balanced_video_hilv3_demo2to1` | 采样 `balanced_2_1`：每步全局 12 样本 = 8 demo + 4 纠正（见 cylinder 10 文档第 3 节）。历史、视频规则同普通版 |

两者都保持 2664 步、验证集规则不变（验证纠正样本仍跨界取历史，与本 run 其他实验可比）。
data-only 长度 `[15984, 4289, 1735, 1040]`。2:1 采样器在真实数据上复核：1998 步/采样周期，三种每卡组成各 666 步；
2664 步内 demo 约过 1.33 遍、纠正约 2.48 遍（1:1 普通版为 1.00 / 3.73）。

队列：tmux `ovcr_segdemo_chain`，接在本 run 普通版之后，顺序为
普通版结束 → cylinder `full_context_truncated`（见 cylinder 10 文档第 6 节）→ `_segdemo` → `_demo2to1`。
每一步都用 `wait_for_queue.sh` 确认上一个实验 completed、exit 0 且最终权重存在，任何失败都会中止后续。

```bash
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/ovcr_adapter_padapter_hil_v3_segdemo_2to1
tmux attach -t ovcr_segdemo_chain
tmux kill-session -t ovcr_segdemo_chain      # 还在等待阶段时取消整条链；已开始训练则对对应 run root 的 suite.pid 发 TERM
```

## 续训到与 cylinder 相同的 DAgger 遍数（2026-09-16）

run root `runs/ovcr_adapter_padapter_hil_v3_cont`，生成器
`scripts/so101_pipeline/ovcr_adapter/padapter_experiments/prepare_hil_v3_continue.py`。

**多跑多少步**：以"每条 DAgger 训练样本被抽到的平均次数"对齐 cylinder hil_v2 的三个 full 权重
（普通版、vidext、full_context_truncated，均 3285 步）。cylinder：3285 步 × 每步 6 个纠正样本 ÷ 3460 = 5.70 遍；
power adapter 要达到同样遍数需 ⌈3285 × 4289 / 3460⌉ = **4073 步**，即在已完成的 2664 步之后再跑 **1409 步**
（续训后 5.698 遍）。步数由脚本从两边 manifest 和配置实时算出，不是手填。

**怎么接上**：用训练器自带的只含权重的续训（`resume: .../step_002664.pt`，`init_checkpoint: null`，`max_steps: 4073`），
效果等同于"原训练本来计划跑 4073 步，现在停在第 2664 步继续"：

| 状态 | 处理 |
|---|---|
| step 计数 | 从文件名解析为 2664，wandb/validation 的 step 轴接着往后 |
| LR 调度 | 按 4073 步重建（5% warmup + cosine，最低 1% 峰值），连续 step 到 2664：续训起点 LR = **3.0e-6**，1409 步后衰减到 1e-7。已用 torch 模拟核对，原 2664 步调度在 step 2600 = 1.156e-7，与原训练日志一致 |
| sampler | demo 一个 epoch 正好 2664 步，续训从 epoch 1 开头开始，随机种子 seed+1——即原训练若继续跑时第二个 epoch 本来会用的排列，不会重放第一个 epoch 的顺序 |
| AdamW 动量 | **无法恢复**：原权重文件只存了权重（`state/` 为空，`optimizer_state_saved: false`）。betas=(0.9, 0.95)，一阶/二阶动量的记忆长度约 10/20 步，续训开始后几十步内即回到稳态；代价约等于起始 LR 下十几步的额外噪声更新 |
| 其余 | batch 组成（每步 6 demo + 6 纠正）、数据、seed、eval 每 178 步、save 每 888 步（3552 和最终 4073）、验证集均与原训练相同 |

与原配置逐键比对，只改了 `output_dir`、`max_steps`、`init_checkpoint`/`resume`、`resume_global_batch_size`、wandb 名称/分组
（`padapter-hil-v3-cont`）和 `experiment.*` 元信息。续训开始前的第一次验证（step 2664）应与原训练最终验证一致，可据此确认权重正确加载。

副作用：demo 在 4073 步内约被看 1.53 遍（cylinder 为 1.0 遍）。

队列：tmux `ovcr_padapter_cont`，GPU 0/1/2，顺序为 普通版 → segdemo → vidext，每个约 4 小时。
部署 yaml：`padapter_v3_p1_full_balanced_video{,_segdemo,_vidext}_cont4073.yaml`，端口 12121（full）、12123（segdemo）、12122（vidext）。

```bash
python scripts/so101_pipeline/ovcr_adapter/v2_experiments/status.py runs/ovcr_adapter_padapter_hil_v3_cont
kill -TERM "$(cat runs/ovcr_adapter_padapter_hil_v3_cont/suite.pid)"      # 停止
```
