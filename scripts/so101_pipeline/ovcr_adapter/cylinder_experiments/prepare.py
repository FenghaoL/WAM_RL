"""Freeze the cylinder-v1 HIL split and generate the five three-GPU experiments.

Same protocol as ``runs/ovcr_adapter_padapter_v2`` applied to the cylinder task:
three OVCR residual adapters (V32/Q32/T4) and one full fine-tune, all starting
from the cylinder task SFT policy ``step_006572.pt`` with a 1:1 demo/HUMAN mix,
plus one four-epoch natural-mixture full SFT restarted from the AHA pretrained
checkpoint.  Differences from the padapter round: Weights & Biases logging is
enabled, and the fixed validation subsets are larger so the curves are readable.

The demo validation set is whatever ``data.val`` of the task SFT already used,
i.e. the held-out session ``20260904_024009_val``; HUMAN validation is a
held-out fifth of the HIL episodes of every collection session.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from scripts.so101_pipeline.ovcr_adapter.v2_experiments.prepare import audit

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
ADAPTER_DEFINITIONS = HERE.parent
RUN_ROOT = ROOT / "runs/ovcr_adapter_cylinder_v1"
RUN_ROOT_RELATIVE = "runs/ovcr_adapter_cylinder_v1"
BASE = ("runs/so101_ahawam_cylinder_v1/full_zero2_offload_ahawam_base/"
        "formal-cylinderv1-sft-ahawambase-bs4-acc1-4ep-3gpu-20260905-022135/model")
P0 = f"{BASE}/checkpoints/weights/step_006572.pt"
PRETRAINED = "checkpoints/AHA-WAM-Pretrained/AHA-WAM-pretrained.pt"
HIL = "data/ahawam_hil_raw/so101_ahawam_cylinder_hil_v1"
SPLIT_TAG = "20260908"
WANDB_PROJECT = "ahawam-so101-cylinder-hil-v1"
WANDB_GROUP = "cylinder-v1-ovcr-correction"

# name, training scope, adapter definition, sampling, anchor, initialization, priority
EXPERIMENTS = [
    ("p1_v32_balanced", "adapter", "value", "balanced_2_2", 1.0, "p0", 1),
    ("p1_q32_balanced", "adapter", "query", "balanced_2_2", 1.0, "p0", 1),
    ("p1_t4_balanced", "adapter", "token_kv", "balanced_2_2", 1.0, "p0", 1),
    ("p1_full_balanced_video", "full", None, "balanced_2_2", 0.0, "p0", 1),
    ("b_full_pretrained_natural_4ep", "full", None, "natural", 0.0, "pretrained", 2),
]


def dataset_frames(dataset_dirs) -> int:
    return sum(json.loads(line)["length"]
               for directory in dataset_dirs
               for line in (ROOT / directory / "meta/episodes.jsonl").read_text().splitlines()
               if line.strip())


def build_report(cfg0) -> dict:
    report = audit(HIL, SPLIT_TAG)
    p0 = ROOT / P0
    pretrained = ROOT / PRETRAINED
    for path in (p0, pretrained):
        if not path.is_file():
            raise FileNotFoundError(path)
    report["task"] = "cylinder_v1"
    report["checkpoint"] = dict(path=P0, size=p0.stat().st_size, mtime_ns=p0.stat().st_mtime_ns)
    report["pretrained_checkpoint"] = dict(path=PRETRAINED, size=pretrained.stat().st_size)
    report["norm_stats"] = dict(
        path=cfg0.data.train.pretrained_norm_stats,
        sha256=hashlib.sha256((ROOT / cfg0.data.train.pretrained_norm_stats).read_bytes()).hexdigest())
    report["demo_train_dirs"] = list(cfg0.data.train.dataset_dirs)
    report["demo_val_dirs"] = list(cfg0.data.val.dataset_dirs)
    report["demo_train_frames"] = dataset_frames(cfg0.data.train.dataset_dirs)
    report["demo_val_frames"] = dataset_frames(cfg0.data.val.dataset_dirs)
    report["natural_human_fraction_train"] = (
        report["train"]["human_frames"] / (report["train"]["human_frames"] + report["demo_train_frames"]))
    return report


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if (RUN_ROOT / "suite.pid").exists():
        raise RuntimeError("A suite PID file exists: do not regenerate configurations of an active suite")
    manifest_path = RUN_ROOT / "data_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to replace the frozen split: {manifest_path}")
    cfg0 = OmegaConf.load(ROOT / BASE / "config.yaml")
    report = build_report(cfg0)
    manifest_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    configurations = RUN_ROOT / "configs"
    configurations.mkdir(exist_ok=True)
    # One complete demo pass under the 1:1 sampler: 2 demo samples per rank, three ranks.
    balanced_steps = math.ceil(report["demo_train_frames"] / 6)
    natural_epoch_steps = math.ceil(
        (report["demo_train_frames"] + report["train"]["human_frames"]) / 12)
    summary = []
    for name, scope, adapter, sampling, anchor, initialization, priority in EXPERIMENTS:
        cfg = copy.deepcopy(cfg0)
        cfg.data.train._target_ = "scripts.so101_pipeline.hil_human_only_dataset.SO101DemoHumanCorrectionDataset"
        cfg.data.train.demo_dataset_dirs = list(cfg0.data.train.dataset_dirs)
        del cfg.data.train.dataset_dirs
        cfg.data.train.correction_dataset_dirs = list(report["correction_episode_selection"]["train"])
        cfg.data.train.correction_episode_selection = report["correction_episode_selection"]["train"]
        cfg.data.train.sampling_strategy = sampling
        cfg.data.train.video_supervision_min_segment_frames = 49
        correction_val = copy.deepcopy(cfg0.data.val)
        correction_val._target_ = "scripts.so101_pipeline.hil_human_only_dataset.SO101HumanCorrectionRobotVideoDataset"
        correction_val.dataset_dirs = list(report["correction_episode_selection"]["val"])
        correction_val.episode_selection = report["correction_episode_selection"]["val"]
        cfg.correction_val = correction_val
        cfg.init_checkpoint = P0 if initialization == "p0" else PRETRAINED
        cfg.resume = None
        cfg.base_init_checkpoint = None
        # Strict load for the already SO101-adapted policy; the pretrained restart
        # keeps the shape adaptation the original task SFT used (14D -> 6D).
        cfg.model.checkpoint_shape_adapt = initialization == "pretrained"
        cfg.model.ovcr_adapter = (None if adapter is None
                                  else OmegaConf.load(ADAPTER_DEFINITIONS / f"{adapter}.yaml"))
        cfg.model.ovcr_anchor_weight = anchor
        cfg.model.loss.lambda_video = 0.0 if scope == "adapter" else 1.0
        cfg.freeze_video_dit = scope == "adapter"
        cfg.learning_rate = 1e-4 if scope == "adapter" else 1e-5
        if scope == "full":
            # Only fully covered 49-frame HUMAN windows carry video supervision.
            cfg.data.train.video_supervision_require_full_window = True
            cfg.correction_val.video_supervision_require_full_window = True
        cfg.history_aware_batching = False
        cfg.batch_size = 4
        cfg.gradient_accumulation_steps = 1
        cfg.num_workers = 2
        if sampling == "natural":
            steps_per_epoch, epochs = natural_epoch_steps, 4
            cfg.max_steps = epochs * steps_per_epoch
            cfg.num_epochs = epochs
            cfg.save_every = steps_per_epoch
            cfg.eval_every = math.ceil(steps_per_epoch / 4)
        else:
            steps_per_epoch, epochs = balanced_steps, 1
            cfg.max_steps = balanced_steps
            cfg.num_epochs = 20  # max_steps takes precedence in the stock trainer
            cfg.save_every = math.ceil(balanced_steps / 3)
            cfg.eval_every = math.ceil(balanced_steps / 15)
        # 48 fixed loss samples and 12 action-inference samples per source and eval.
        cfg.eval_num_val_batches = 16
        cfg.eval_num_action_batches = 4
        cfg.save_at_end = True
        cfg.zero2_save_training_state = False
        cfg.wandb.enabled = True
        cfg.wandb.project = WANDB_PROJECT
        cfg.wandb.group = WANDB_GROUP
        cfg.wandb.name = name
        cfg.wandb.mode = "online"
        cfg.output_dir = f"{RUN_ROOT_RELATIVE}/{name}/model"
        cfg.experiment = dict(
            name=name, priority=priority, task="cylinder_v1", run_root=RUN_ROOT_RELATIVE,
            training_scope=scope, adapter=adapter, sampling=sampling, anchor_weight=anchor,
            initialization=P0 if initialization == "p0" else PRETRAINED,
            steps_per_epoch=steps_per_epoch, planned_epochs=epochs,
            manifest_sha256=manifest_hash, expected_world_size=3, global_batch=12)
        OmegaConf.save(cfg, configurations / f"{name}.yaml", resolve=True)
        summary.append(dict(name=name, scope=scope, adapter=adapter, sampling=sampling,
                            anchor=anchor, lr=cfg.learning_rate, init=cfg.init_checkpoint,
                            max_steps=cfg.max_steps, save_every=cfg.save_every,
                            eval_every=cfg.eval_every))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"episodes", "correction_episode_selection"}}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
