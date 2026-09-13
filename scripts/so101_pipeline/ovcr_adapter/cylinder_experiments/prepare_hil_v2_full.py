"""Three single-change variants of ``p1_full_balanced_video_hilv2``, queued in order.

1. ``vidext``   -- for HUMAN correction samples the future *video* window continues
   past the takeover over the rollout's later executing frames (SYSTEM_HOLD
   skipped), only in successful rollouts. Action/state labels stay on the span.
2. ``actext``   -- as 1, and actions/states follow the same timeline, so the
   policy's gripper closure after a grasp-alignment takeover is supervised.
3. ``demo2to1`` -- no extension; every 12-sample global batch holds 8 demo and 4
   correction samples instead of 6 and 6.

No start-frame filter: removing windows with at most two moving frame steps (any
joint changing by at least one encoder tick) would drop only ~5% genuinely static
windows; the rest would be the last 1-3 frames of each span. Same data, split,
validation, 3285 steps, lr, video rule and history setting as the base run, so
each variant differs from it in exactly one respect.
"""
from __future__ import annotations

import copy
import json

from omegaconf import OmegaConf

from scripts.so101_pipeline.ovcr_adapter.cylinder_experiments.prepare import ROOT
from scripts.so101_pipeline.ovcr_adapter.cylinder_experiments.prepare_hil_v2 import (
    RUN_ROOT as BASE_ROOT, RUN_ROOT_RELATIVE as BASE_RELATIVE, sha256)

RUN_ROOT_RELATIVE = "runs/ovcr_adapter_cylinder_hil_v2_full"
RUN_ROOT = ROOT / RUN_ROOT_RELATIVE
BASE_CONFIG = "p1_full_balanced_video_hilv2"
WANDB_GROUP = "cylinder-hil-v2-full"
EXPERIMENTS = [
    ("vidext", dict(correction_future_extension="video", correction_extension_success_only=True)),
    ("actext", dict(correction_future_extension="video_and_actions", correction_extension_success_only=True)),
    ("demo2to1", dict(sampling_strategy="balanced_2_1")),
]


def main():
    if (RUN_ROOT / "data_manifest.json").exists():
        raise FileExistsError(f"Refusing to replace {RUN_ROOT / 'data_manifest.json'}")
    base_manifest_path = BASE_ROOT / "data_manifest.json"
    manifest = copy.deepcopy(json.loads(base_manifest_path.read_text()))
    manifest["task"] = "cylinder_hil_v2_full_variants"
    manifest["parent_manifest"] = str(base_manifest_path.relative_to(ROOT))
    manifest["parent_manifest_sha256"] = sha256(base_manifest_path)
    manifest["variants"] = {name: change for name, change in EXPERIMENTS}
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_hash = sha256(RUN_ROOT / "data_manifest.json")
    (RUN_ROOT / "configs").mkdir(exist_ok=True)
    summary = {}
    for tag, change in EXPERIMENTS:
        name = f"{BASE_CONFIG}_{tag}"
        cfg = OmegaConf.load(BASE_ROOT / "configs" / f"{BASE_CONFIG}.yaml")
        for key, value in change.items():
            cfg.data.train[key] = value
        cfg.output_dir = f"{RUN_ROOT_RELATIVE}/{name}/model"
        cfg.wandb.name = name
        cfg.wandb.group = WANDB_GROUP
        cfg.experiment.name = name
        cfg.experiment.run_root = RUN_ROOT_RELATIVE
        cfg.experiment.manifest_sha256 = manifest_hash
        cfg.experiment.reference_run = f"{BASE_RELATIVE}/{BASE_CONFIG}"
        cfg.experiment.variant = dict(change)
        OmegaConf.save(cfg, RUN_ROOT / "configs" / f"{name}.yaml", resolve=True)
        summary[name] = dict(change, sampling=cfg.data.train.sampling_strategy,
                             extension=cfg.data.train.get("correction_future_extension", "none"),
                             max_steps=cfg.max_steps, lr=cfg.learning_rate, lambda_video=cfg.model.loss.lambda_video)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
