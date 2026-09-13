"""cylinder HIL v2 correction runs whose history may reach before the takeover.

Start: the original cylinder task SFT (history 6).  Correction data: the HUMAN
takeover spans of ``data/ahawam_hil_raw/so101_ahawam_cylinder_hil_v2`` only.

What is new versus every earlier round: the current frame, the supervised action
window and the predicted video stay inside the HUMAN span, but the six video
history frames may come from before the takeover -- the policy-driven part of
the same rollout, which is what the model actually sees at deployment.  History
still stops at the start of the raw rollout.

History spacing: the main runs keep the SFT spacing of 48 frames.  A single
extra V32 run uses 40 frames, because at deployment each prefill advances the
video RoPE position by exactly ``action_horizon // video_rope_frame_stride``
= 40 // 4 = 10 units (ahawam.py), i.e. a fixed 40-frame spacing regardless of
how long the chunk really took.  Scaling the spacing by the measured
demo/rollout speed ratio is deliberately NOT done: deployment never encodes the
real elapsed time, and the real gap between prefills is highly variable
(median 31.5 frames, p90 258), so no single scaled value matches it.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from omegaconf import OmegaConf

from scripts.so101_pipeline.ovcr_adapter.cylinder_experiments.prepare import (
    BASE, P0, ROOT, RUN_ROOT as PARENT_ROOT, RUN_ROOT_RELATIVE as PARENT_RELATIVE)
from scripts.so101_pipeline.ovcr_adapter.v2_experiments.prepare import audit

RUN_ROOT_RELATIVE = "runs/ovcr_adapter_cylinder_hil_v2"
RUN_ROOT = ROOT / RUN_ROOT_RELATIVE
HIL = "data/ahawam_hil_raw/so101_ahawam_cylinder_hil_v2"
SPLIT_TAG = "20260912"
WANDB_GROUP = "cylinder-hil-v2-history"
SUFFIX = "_hilv2"
HISTORY_STRIDE = 48          # same number of frames as the demo SFT
SFT_PARITY_STRIDE = 72       # executing frames that cover as much task progress as one SFT step
# (base configuration, suffix, history stride, count executing frames only)
#
# Main runs step back 48 *executing* frames: SYSTEM_HOLD is idle time, so 48 raw
# frames of a policy-driven rollout cover only ~38% of the task progress that 48
# demo frames do, while 48 executing frames cover ~65% and keep the RoPE geometry
# the SFT produced (history exactly 12 position units apart).  The two extra V32
# runs change only this setting: raw 48 is the naive default, and skip-hold 72
# reaches ~97% coverage but leaves 35% of samples without any history at all.
EXPERIMENTS = [
    ("p1_v32_balanced", SUFFIX, HISTORY_STRIDE, True),
    ("p1_q32_balanced", SUFFIX, HISTORY_STRIDE, True),
    ("p1_t4_balanced", SUFFIX, HISTORY_STRIDE, True),
    ("p1_full_balanced_video", SUFFIX, HISTORY_STRIDE, True),
    ("p1_v32_balanced", f"{SUFFIX}_raw{HISTORY_STRIDE}", HISTORY_STRIDE, False),
    ("p1_v32_balanced", f"{SUFFIX}_skip{SFT_PARITY_STRIDE}", SFT_PARITY_STRIDE, True),
]


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_by_val_sessions(report: dict) -> dict:
    """Sessions whose directory ends in `_val` are validation; the rest train.

    The recording side already marks its held-out session that way, which is a
    cleaner split than hashing episodes inside a session: no rollout ends up
    half in training and half in validation.
    """
    selection = {"train": {}, "val": {}}
    for record in report["episodes"]:
        root = record["root"]
        split = "val" if root.rsplit("/", 1)[-1].endswith("_val") else "train"
        record["split"] = split
        selection["train"].setdefault(root, [])
        selection["val"].setdefault(root, [])
        if record["human_frames"] > 0:
            selection[split][root].append(record["episode"])
    for split in ("train", "val"):
        for root in selection[split]:
            selection[split][root] = sorted(selection[split][root])
        chosen = [r for r in report["episodes"] if r["split"] == split]
        report[split] = dict(episodes=sum(r["human_frames"] > 0 for r in chosen),
                             human_frames=sum(r["human_frames"] for r in chosen),
                             human_segments=sum(len(r["segment_lengths"]) for r in chosen))
    # Only sessions that actually contribute episodes stay in the dataset lists.
    selection["train"] = {k: v for k, v in selection["train"].items() if v}
    selection["val"] = {k: v for k, v in selection["val"].items() if v}
    report["correction_episode_selection"] = selection
    report["split_rule"] = "directories ending in _val are validation-only"
    return report


def main():
    if (RUN_ROOT / "data_manifest.json").exists():
        raise FileExistsError(f"Refusing to replace {RUN_ROOT / 'data_manifest.json'}")
    cfg0 = OmegaConf.load(ROOT / BASE / "config.yaml")
    report = audit(HIL, SPLIT_TAG)
    report = split_by_val_sessions(report)
    p0 = ROOT / P0
    report["task"] = "cylinder_hil_v2_history"
    report["checkpoint"] = dict(path=P0, size=p0.stat().st_size)
    report["norm_stats"] = dict(
        path=cfg0.data.train.pretrained_norm_stats,
        sha256=sha256(ROOT / cfg0.data.train.pretrained_norm_stats))
    report["demo_train_dirs"] = list(cfg0.data.train.dataset_dirs)
    report["demo_val_dirs"] = list(cfg0.data.val.dataset_dirs)
    report["demo_train_frames"] = sum(
        json.loads(line)["length"] for directory in cfg0.data.train.dataset_dirs
        for line in (ROOT / directory / "meta/episodes.jsonl").read_text().splitlines() if line.strip())
    report["demo_val_frames"] = sum(
        json.loads(line)["length"] for directory in cfg0.data.val.dataset_dirs
        for line in (ROOT / directory / "meta/episodes.jsonl").read_text().splitlines() if line.strip())
    report["natural_human_fraction_train"] = report["train"]["human_frames"] / (
        report["train"]["human_frames"] + report["demo_train_frames"])
    report["history"] = dict(
        crosses_segment=True, bounded_by="raw rollout start",
        main_stride_frames=HISTORY_STRIDE, main_counts_executing_frames_only=True,
        ablations=["raw48 (no skipping)", f"skip-hold {SFT_PARITY_STRIDE} (SFT stage parity)"],
        system_hold_control_source=2, deployment_position_increment_frames=40,
        note="current frame, actions and predicted video stay inside the HUMAN span")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "data_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest_hash = sha256(RUN_ROOT / "data_manifest.json")

    configurations = RUN_ROOT / "configs"
    configurations.mkdir(exist_ok=True)
    summary = []
    for base, suffix, stride, skip_hold in EXPERIMENTS:
        name = base + suffix
        cfg = OmegaConf.load(PARENT_ROOT / "configs" / f"{base}.yaml")
        cfg.init_checkpoint = P0
        cfg.model.num_history_frames = 6
        cfg.data.train.correction_dataset_dirs = list(report["correction_episode_selection"]["train"])
        cfg.data.train.correction_episode_selection = report["correction_episode_selection"]["train"]
        cfg.data.train.correction_clip_dataset_dirs = []
        cfg.data.train.correction_history_crosses_segment = True
        cfg.data.train.correction_history_frame_stride = stride
        cfg.data.train.correction_history_skip_system_hold = skip_hold
        cfg.correction_val.dataset_dirs = list(report["correction_episode_selection"]["val"])
        cfg.correction_val.episode_selection = report["correction_episode_selection"]["val"]
        cfg.correction_val.history_crosses_segment = True
        cfg.correction_val.history_frame_stride = stride
        cfg.correction_val.history_skip_system_hold = skip_hold
        cfg.output_dir = f"{RUN_ROOT_RELATIVE}/{name}/model"
        cfg.wandb.name = name
        cfg.wandb.group = WANDB_GROUP
        cfg.experiment.name = name
        cfg.experiment.run_root = RUN_ROOT_RELATIVE
        cfg.experiment.manifest_sha256 = manifest_hash
        cfg.experiment.initialization = P0
        cfg.experiment.num_history_frames = 6
        cfg.experiment.history_crosses_segment = True
        cfg.experiment.history_frame_stride = stride
        cfg.experiment.history_skip_system_hold = skip_hold
        cfg.experiment.correction_sources = [f"{HIL} HUMAN train split"]
        cfg.experiment.reference_run = f"{PARENT_RELATIVE}/{base}"
        OmegaConf.save(cfg, configurations / f"{name}.yaml", resolve=True)
        summary.append(dict(name=name, scope=cfg.experiment.training_scope, history_stride=stride,
                            skip_system_hold=skip_hold,
                            lr=cfg.learning_rate, lambda_video=cfg.model.loss.lambda_video,
                            anchor=cfg.model.ovcr_anchor_weight, max_steps=cfg.max_steps,
                            save_every=cfg.save_every))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"episodes", "correction_episode_selection"}}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
