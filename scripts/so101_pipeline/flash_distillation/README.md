# SO101 AHA-WAM-Flash distillation

This directory adapts the repository's released `ahawam_ode` implementation to
the exact 6-D power-adapter-v2 checkpoint at step 005328. It does not treat the
small trainer-state JSON as model weights.

The original launcher uses physical A100 GPUs 0 and 3, DeepSpeed ZeRO-2 with CPU
optimizer offload, micro-batch 1, gradient accumulation 8, five epochs, and the
repository's official 16-step teacher/captured ODE states. The four-GPU restart
uses GPUs 0,1,2,3 and gradient accumulation 4, preserving the effective batch of
16 and every model/distillation hyperparameter. It is a fresh run from the SFT
teacher, not a resume from the two-GPU student.

Both launchers save a deployable student checkpoint after every epoch. The
requested online student sampler is 2-step; the released generic YAML defaults
to 1-step, so this choice is explicit here.

Commands:

```bash
bash scripts/so101_pipeline/flash_distillation/start_training.sh
bash scripts/so101_pipeline/flash_distillation/start_training_4gpu.sh
bash scripts/so101_pipeline/flash_distillation/status_training.sh
bash scripts/so101_pipeline/flash_distillation/stop_training.sh
```

The latest run pointer is
`runs/so101_ahawam_padapter_v2/latest_flash_run.txt`. Checkpoints are under
`<run>/model/checkpoints/weights/step_*.pt`. The async deploy resolver accepts an
explicit `FLASH_CHECKPOINT=/path/to/step_N.pt` or otherwise selects the newest
checkpoint from that pointer.
