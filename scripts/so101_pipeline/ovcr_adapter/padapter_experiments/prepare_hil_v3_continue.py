"""Continue the finished power-adapter HIL v3 full runs to cylinder's DAgger exposure.

Criterion: every HUMAN (DAgger) training sample is drawn as often as in the cylinder
hil_v2 full runs.  Cylinder draws 6 correction samples per step (1:1, global batch 12)
for 3285 steps over 3460 samples = 5.70 passes; the power adapter reaches the same
count at ceil(3285 * 6 * 4289 / (3460 * 6)) = 4073 steps, i.e. 1409 steps beyond 2664.

Continuation uses the trainer's weight-only resume (``resume: .../step_002664.pt``):
the run behaves as if it had been planned for the target step count and is now at
step 2664 -- LR scheduler advanced to that step (5% warmup + cosine to 1% of peak),
sampler on epoch 1 (the permutation the original run would have drawn next), same
batch composition, seed, eval and save cadence.  AdamW moments cannot be restored
(the source runs saved weights only); with betas (0.9, 0.95) they refill within tens
of steps.  Queue order: plain, segdemo, vidext.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

ROOT = Path(__file__).resolve().parents[4]
RUN_ROOT_RELATIVE = "runs/ovcr_adapter_padapter_hil_v3_cont"
RUN_ROOT = ROOT / RUN_ROOT_RELATIVE
SOURCE_MANIFEST = "runs/ovcr_adapter_padapter_hil_v3/data_manifest.json"
WANDB_GROUP = "padapter-hil-v3-cont"
# the latest cylinder full runs whose DAgger exposure is the target
CYLINDER_CONFIGS = [
    "runs/ovcr_adapter_cylinder_hil_v2/configs/p1_full_balanced_video_hilv2.yaml",
    "runs/ovcr_adapter_cylinder_hil_v2_full/configs/p1_full_balanced_video_hilv2_vidext.yaml",
    "runs/icra2027_cylinder_controls/configs/full_context_truncated.yaml",
]
# (source run root, experiment), in queue order
SOURCES = [
    ("runs/ovcr_adapter_padapter_hil_v3", "p1_full_balanced_video_hilv3"),
    ("runs/ovcr_adapter_padapter_hil_v3_segdemo_2to1", "p1_full_balanced_video_hilv3_segdemo"),
    ("runs/ovcr_adapter_padapter_hil_v3", "p1_full_balanced_video_hilv3_vidext"),
]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exposure(cfg) -> tuple[int, int, int]:
    """(steps, correction samples per step, HUMAN training samples) of a 1:1 run."""
    if cfg.data.train.sampling_strategy != "balanced_2_2":
        raise ValueError(f"{cfg.experiment.name}: expected balanced_2_2 sampling")
    manifest = json.loads((ROOT / cfg.experiment.run_root /
                           cfg.experiment.get("manifest_file", "data_manifest.json")).read_text())
    per_step = int(cfg.batch_size) * int(cfg.experiment.expected_world_size) // 2
    return int(cfg.max_steps), per_step, int(manifest["train"]["human_frames"])


def lr_at(step: int, total: int, lr: float) -> float:
    """LR the trainer's scheduler gives after `step` steps of a `total`-step run."""
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([param], lr=lr)
    warmup = int(total * 0.05)
    scheduler = SequentialLR(optimizer, [
        LinearLR(optimizer, start_factor=1.0 / warmup, end_factor=1.0, total_iters=warmup),
        CosineAnnealingLR(optimizer, T_max=max(total - warmup, 1), eta_min=lr * 0.01),
    ], milestones=[warmup])
    for _ in range(step):
        scheduler.step()
    return float(optimizer.param_groups[0]["lr"])


def main():
    cylinder = {exposure(OmegaConf.load(ROOT / path)) for path in CYLINDER_CONFIGS}
    if len(cylinder) != 1:
        raise ValueError(f"Cylinder runs disagree on DAgger exposure: {cylinder}")
    cyl_steps, cyl_per_step, cyl_human = cylinder.pop()

    (RUN_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    manifest_path = RUN_ROOT / "data_manifest.json"
    if manifest_path.exists() and sha256(manifest_path) != sha256(ROOT / SOURCE_MANIFEST):
        raise ValueError(f"{manifest_path} differs from {SOURCE_MANIFEST}")
    shutil.copyfile(ROOT / SOURCE_MANIFEST, manifest_path)
    manifest_hash = sha256(manifest_path)

    plan = []
    for source_root, name in SOURCES:
        cfg = OmegaConf.load(ROOT / source_root / "configs" / f"{name}.yaml")
        if cfg.experiment.manifest_sha256 != manifest_hash:
            raise ValueError(f"{name}: trained on a different data manifest")
        status = json.loads((ROOT / source_root / "suite_status.json").read_text())
        record = [r for r in status["experiments"] if r["name"] == name][-1]
        if record["state"] != "completed" or record.get("exit_code") != 0:
            raise ValueError(f"{name}: source run did not complete: {record}")
        source_steps, per_step, human = exposure(cfg)
        checkpoint = f"{source_root}/{name}/model/checkpoints/weights/step_{source_steps:06d}.pt"
        if not (ROOT / checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        target = math.ceil(cyl_steps * cyl_per_step * human / (cyl_human * per_step))
        new_name = f"{name}_cont{target}"

        cfg.output_dir = f"{RUN_ROOT_RELATIVE}/{new_name}/model"
        cfg.max_steps = target
        cfg.init_checkpoint = None
        cfg.resume = checkpoint
        cfg.resume_global_batch_size = int(cfg.batch_size) * int(cfg.experiment.expected_world_size)
        cfg.wandb.name = new_name
        cfg.wandb.group = WANDB_GROUP
        cfg.experiment.name = new_name
        cfg.experiment.run_root = RUN_ROOT_RELATIVE
        cfg.experiment.manifest_file = "data_manifest.json"
        cfg.experiment.reference_run = f"{source_root}/{name}"
        continuation = dict(
            source_checkpoint=checkpoint,
            source_steps=source_steps,
            target_steps=target,
            criterion="HUMAN sample passes matched to cylinder hil_v2 full runs",
            cylinder_passes=round(cyl_steps * cyl_per_step / cyl_human, 4),
            target_passes=round(target * per_step / human, 4),
            lr_at_resume=lr_at(source_steps, target, float(cfg.learning_rate)),
            optimizer_state="not restored (weights-only source checkpoint)",
        )
        cfg.experiment.continuation = continuation
        OmegaConf.save(cfg, RUN_ROOT / "configs" / f"{new_name}.yaml")
        plan.append(dict(name=new_name, **continuation))
        print(json.dumps(plan[-1]))
    (RUN_ROOT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
