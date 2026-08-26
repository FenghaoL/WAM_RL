# NW2 checkpoints 冻结 Video DiT continuation

从原 NW2 全量微调 run 的 `step_001988`、`step_002982`、`step_003976`
权重分别开始新训练。Video DiT 冻结；Action DiT、极小的 action branch embedding，
以及模型通过 `get_additional_trainable_modules()` 声明的 proprio/视觉条件模块参与
训练。逐层 Chunk-KV editor 仍参与前向计算，但它本身是十亿级模块，所以保持冻结。

`step_*_trainer_state.json` 只有训练进度元数据，原 run 没有保存 optimizer state。
因此每个 continuation 实际加载同名 `weights/step_*.pt`，并新建 AdamW 和学习率
调度器。每个输出目录中的 `manifest.json` 同时记录 JSON 与 PT 来源。

单任务入口：

```bash
GPU_IDS=1 NUM_GPUS=1 SOURCE_STEP=001988 NUM_EPOCHS=2 BATCH_SIZE=4 \
  bash scripts/so101_pipeline/freeze_video_from_nw2_ckpts/run_one.sh
```

正式三任务并行入口（默认单卡 batch 6、2 epoch）：

```bash
bash scripts/so101_pipeline/freeze_video_from_nw2_ckpts/launch_all.sh
bash scripts/so101_pipeline/freeze_video_from_nw2_ckpts/status.sh
```

GPU 占位与释放：

```bash
bash scripts/so101_pipeline/freeze_video_from_nw2_ckpts/reserve_gpus.sh
bash scripts/so101_pipeline/freeze_video_from_nw2_ckpts/release_gpus.sh
```

输出按 source checkpoint 放在原 run 下面：

```text
.../formal-nw2-...-20260818-232749/freeze_video_continuations/
  from_step_001988/<run>/
  from_step_002982/<run>/
  from_step_003976/<run>/
```
