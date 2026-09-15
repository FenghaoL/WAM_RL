"""Power-adapter HIL v3: p1_full_balanced_video with and without video extension.

Start: the power-adapter task SFT ``step_005328.pt`` (history 6).  Correction data:
the HUMAN takeover spans of ``data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v3``.
Recipe mirrors the latest cylinder ``p1_full_balanced_video_hilv2`` run:

* full fine-tune, 1:1 demo/correction batches, video + action loss, lr 1e-5;
* the six history frames may reach before the takeover start, stepping back 48
  executing frames at a time (SYSTEM_HOLD skipped); current frame, actions and
  predicted video stay inside the HUMAN span;
* ``_vidext`` additionally lets the future *video* window continue past the span
  over later executing frames of successful rollouts; action labels stay human-only.

v3 has no ``*_val`` session, so about 20% of each session's takeover rollouts are
held out by hashing, as in the earlier power-adapter rounds.  Demo train/val,
normalization stats and task text are those of the SFT run.  Step budget: one
demo pass under 1:1 (ceil(demo frames / 6)).
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from scripts.so101_pipeline.ovcr_adapter.v2_experiments.prepare import audit

ROOT = Path(__file__).resolve().parents[4]
RUN_ROOT_RELATIVE = "runs/ovcr_adapter_padapter_hil_v3"
RUN_ROOT = ROOT / RUN_ROOT_RELATIVE
SFT_MODEL = ("runs/so101_ahawam_padapter_v2/full_zero2_offload_ahawam_base/"
             "formal-padapterv2-sft-ahawambase-bs4-acc1-4ep-3gpu-20260901-001743/model")
P0 = f"{SFT_MODEL}/checkpoints/weights/step_005328.pt"
BASE_CONFIG = "runs/ovcr_adapter_padapter_v2/configs/p1_full_balanced_video.yaml"
HIL = "data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v3"
SPLIT_TAG = "20260914"
TASK = "Remove the power adapter from the holder and plug it into the power strip."
WANDB_PROJECT = "ahawam-so101-padapter-hil-v3"
WANDB_GROUP = "padapter-hil-v3-full"
HISTORY_STRIDE = 48
# queue order: video extension first, then the plain run
EXPERIMENTS = [
    ("p1_full_balanced_video_hilv3_vidext",
     dict(correction_future_extension="video", correction_extension_success_only=True)),
    ("p1_full_balanced_video_hilv3", {}),
]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frames(dirs) -> int:
    return sum(json.loads(line)["length"] for d in dirs
               for line in (ROOT / d / "meta/episodes.jsonl").read_text().splitlines() if line.strip())


def main():
    if (RUN_ROOT / "data_manifest.json").exists():
        raise FileExistsError(f"Refusing to replace {RUN_ROOT / 'data_manifest.json'}")
    if not (ROOT / P0).is_file():
        raise FileNotFoundError(P0)
    for session in sorted((ROOT / HIL).iterdir()):
        tasks = [json.loads(l)["task"] for l in (session / "meta/tasks.jsonl").read_text().splitlines() if l.strip()]
        if tasks != [TASK]:
            raise ValueError(f"{session}: unexpected task text {tasks}")
    sft = OmegaConf.load(ROOT / SFT_MODEL / "config.yaml")
    report = audit(HIL, SPLIT_TAG)
    report["task"] = "padapter_hil_v3"
    report["checkpoint"] = dict(path=P0, size=(ROOT / P0).stat().st_size)
    report["norm_stats"] = dict(path=sft.data.train.pretrained_norm_stats,
                                sha256=sha256(ROOT / sft.data.train.pretrained_norm_stats))
    report["demo_train_dirs"] = list(sft.data.train.dataset_dirs)
    report["demo_val_dirs"] = list(sft.data.val.dataset_dirs)
    report["demo_train_frames"] = frames(sft.data.train.dataset_dirs)
    report["demo_val_frames"] = frames(sft.data.val.dataset_dirs)
    report["natural_human_fraction_train"] = report["train"]["human_frames"] / (
        report["train"]["human_frames"] + report["demo_train_frames"])
    report["history"] = dict(crosses_segment=True, stride_executing_frames=HISTORY_STRIDE, skip_system_hold=True)
    report["split_rule"] = f"per session, ~20% of takeover rollouts held out by sha256({SPLIT_TAG}:session:episode)"
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "data_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest_hash = sha256(RUN_ROOT / "data_manifest.json")

    steps = math.ceil(report["demo_train_frames"] / 6)
    (RUN_ROOT / "configs").mkdir(exist_ok=True)
    summary = []
    for name, change in EXPERIMENTS:
        cfg = OmegaConf.load(ROOT / BASE_CONFIG)
        if cfg.experiment.get("training_scope") != "full":
            raise ValueError(f"{BASE_CONFIG} is not a full fine-tune configuration")
        cfg.init_checkpoint = P0
        cfg.model.num_history_frames = 6
        cfg.data.train.demo_dataset_dirs = list(sft.data.train.dataset_dirs)
        cfg.data.train.correction_dataset_dirs = list(report["correction_episode_selection"]["train"])
        cfg.data.train.correction_episode_selection = report["correction_episode_selection"]["train"]
        cfg.data.train.correction_history_crosses_segment = True
        cfg.data.train.correction_history_frame_stride = HISTORY_STRIDE
        cfg.data.train.correction_history_skip_system_hold = True
        for key, value in change.items():
            cfg.data.train[key] = value
        cfg.correction_val.dataset_dirs = list(report["correction_episode_selection"]["val"])
        cfg.correction_val.episode_selection = report["correction_episode_selection"]["val"]
        cfg.correction_val.history_crosses_segment = True
        cfg.correction_val.history_frame_stride = HISTORY_STRIDE
        cfg.correction_val.history_skip_system_hold = True
        cfg.max_steps = steps
        cfg.save_every = math.ceil(steps / 3)
        cfg.eval_every = math.ceil(steps / 15)
        cfg.eval_num_val_batches = 16
        cfg.eval_num_action_batches = 4
        cfg.wandb.enabled = True
        cfg.wandb.project = WANDB_PROJECT
        cfg.wandb.group = WANDB_GROUP
        cfg.wandb.name = name
        cfg.wandb.mode = "online"
        cfg.output_dir = f"{RUN_ROOT_RELATIVE}/{name}/model"
        cfg.experiment = dict(
            name=name, task="padapter_hil_v3", run_root=RUN_ROOT_RELATIVE, training_scope="full",
            initialization=P0, num_history_frames=6, history_crosses_segment=True,
            history_frame_stride=HISTORY_STRIDE, history_skip_system_hold=True, variant=dict(change),
            manifest_sha256=manifest_hash, expected_world_size=3, global_batch=12)
        OmegaConf.save(cfg, RUN_ROOT / "configs" / f"{name}.yaml", resolve=True)
        summary.append(dict(name=name, extension=cfg.data.train.get("correction_future_extension", "none"),
                            max_steps=cfg.max_steps, save_every=cfg.save_every, eval_every=cfg.eval_every,
                            lr=cfg.learning_rate, lambda_video=cfg.model.loss.lambda_video,
                            sampling=cfg.data.train.sampling_strategy))
    print(json.dumps({k: report[k] for k in ("train", "val", "demo_train_frames", "demo_val_frames",
                                            "natural_human_fraction_train", "human_segments",
                                            "segments_shorter_than_49")}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
