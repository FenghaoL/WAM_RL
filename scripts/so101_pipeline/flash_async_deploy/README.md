# Flash + async deployment

The completed SO101 Flash checkpoint currently selected by the no-argument
launcher is:

```text
runs/so101_ahawam_padapter_v2/flash_distillation/
formal-padapterv2-flash-ode2-from-step005328-4xa100-20260903-124708/
model/checkpoints/weights/step_004995.pt
```

`resolve_flash_checkpoint.py` reads `latest_flash_run.txt`, chooses its highest
step, and now verifies the embedded ODE metadata before creating
`settings.resolved.yaml`. It will reject an ordinary SFT checkpoint rather than
silently running that checkpoint with two oversized denoising steps.

After the distillation run has saved at least one epoch checkpoint:

```bash
bash scripts/so101_pipeline/flash_async_deploy/run_action_server_async_flash.sh
```

The resolver selects the highest `step_*.pt` in the latest Flash run. Override
it only with another distilled student:

```bash
FLASH_CHECKPOINT=/absolute/path/to/step_N.pt \
  bash scripts/so101_pipeline/flash_async_deploy/run_action_server_async_flash.sh
```

In a second terminal, before connecting hardware:

```bash
bash scripts/so101_pipeline/flash_async_deploy/smoke_async_protocol.sh
```

The server is deliberately bound to localhost because its pickle TCP protocol
is not safe on an untrusted network. Use an SSH tunnel from the robot computer.
Read `ROBOT_CLIENT_HANDOFF_ZH.md` before changing the controller.

For a self-contained task specification that can be pasted directly into the
Codex session on the robot computer, use
`ROBOT_PC_CODEX_IMPLEMENTATION_TASK_ZH.md`.

This is a single-GPU, two-TCP-connection deployment. The GPU event loop gives
action requests priority and refreshes the clean current-image Video-DiT K/V
state whenever there is room; the Action DiT uses the latest published K/V plus
the image/state carried by the current action request. It is not RTC and it does
not generate a new action chunk concurrently with execution on the robot side.
