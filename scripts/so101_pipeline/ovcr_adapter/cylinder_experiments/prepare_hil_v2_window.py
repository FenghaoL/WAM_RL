"""Two correction-window variants of ``p1_full_balanced_video_hilv2``.

Diagnosis that motivates them (hil_v2 train, docs 09):
* in grasp-alignment takeovers the operator never closes the gripper (0/62 spans)
  and hands control back after holding still, so 32% of those start frames
  supervise an entirely static "hold, gripper open" action window;
* in the recorded successful rollouts the policy closes the gripper a median of
  7 executing frames after the handback, but windows stopped at the span end.

Both variants, for HUMAN correction samples only:
* the future video window continues past the span over the later executing
  frames of the rollout, skipping SYSTEM_HOLD, only for successful rollouts;
* start frames whose supervised action window is static (every joint, gripper
  included, varies < 1 deg over it) are not used as the current frame.

Variant A keeps actions/states on the HUMAN span (labels stay human-only).
Variant B lets actions/states follow the extended timeline too, so the policy's
gripper closure in successful rollouts becomes supervised; this alone cuts static
start frames from 21.8% to 2.3%.  The HUMAN validation set is unchanged.
"""
from __future__ import annotations

import copy
import json

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf

from scripts.so101_pipeline.hil_human_only_dataset import (
    SYSTEM_HOLD_CONTROL_SOURCE, _FutureWindowRemapper, _human_segments_from_columns, _to_numpy,
    static_start_mask)
from scripts.so101_pipeline.ovcr_adapter.cylinder_experiments.prepare import ROOT
from scripts.so101_pipeline.ovcr_adapter.cylinder_experiments.prepare_hil_v2 import (
    RUN_ROOT as BASE_ROOT, RUN_ROOT_RELATIVE as BASE_RELATIVE, sha256)

RUN_ROOT_RELATIVE = "runs/ovcr_adapter_cylinder_hil_v2_window"
RUN_ROOT = ROOT / RUN_ROOT_RELATIVE
BASE_CONFIG = "p1_full_balanced_video_hilv2"
STATIC_RANGE_DEG = 1.0
ACTION_WINDOW = 48
WANDB_GROUP = "cylinder-hil-v2-window"
VARIANTS = [("winA", "video"), ("winB", "video_and_actions")]
CONTROL = ["episode_index", "frame_index", "tick_index", "control_source", "intervention",
           "human_action_valid", "dropped_tick", "action_supervision_mask"]


def count_start_frames(selection: dict, mode: str) -> tuple[int, int]:
    """Kept/excluded HUMAN start frames with the dataset's own rule and timeline."""
    kept = excluded = 0
    for root, episodes in selection.items():
        episodes = {int(e) for e in episodes}
        labels = {int(json.loads(line)["logical_episode_index"]): json.loads(line).get("outcome", "")
                  for line in (ROOT / root / "recording_context/episodes_manifest.jsonl").read_text().splitlines()
                  if line.strip()}
        for path in sorted((ROOT / root / "data").glob("chunk-*/*.parquet")):
            table = pq.read_table(path)
            columns = {n: _to_numpy(table[n]) for n in CONTROL if n in table.column_names}
            if int(columns["episode_index"][0]) not in episodes:
                continue
            actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
            n = len(actions)
            segments = _human_segments_from_columns(columns)
            segment_end = np.full(n, -1, dtype=np.int64)
            for start, end in segments:
                segment_end[start:end] = end
            success = labels.get(int(_to_numpy(table["logical_episode_index"])[0])) == "success"
            executing = np.flatnonzero(columns["control_source"].astype(np.int64) != SYSTEM_HOLD_CONTROL_SOURCE)
            remapper = _FutureWindowRemapper(
                segment_end=segment_end, rollout_start=np.zeros(n, dtype=np.int64),
                executing_by_rollout={0: executing.astype(np.int64)},
                extendable_rollouts={0} if success else set(),
                extend_actions=mode == "video_and_actions")
            for start, end in segments:
                mask = static_start_mask(actions, start, end, range_deg=STATIC_RANGE_DEG,
                                         window=ACTION_WINDOW, remapper=remapper)
                kept += int((~mask).sum())
                excluded += int(mask.sum())
    return kept, excluded


def main():
    base_manifest_path = BASE_ROOT / "data_manifest.json"
    base_manifest = json.loads(base_manifest_path.read_text())
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    (RUN_ROOT / "configs").mkdir(exist_ok=True)
    summary = {}
    for tag, mode in VARIANTS:
        name = f"{BASE_CONFIG}_{tag}"
        manifest_file = f"data_manifest_{tag}.json"
        if (RUN_ROOT / manifest_file).exists():
            raise FileExistsError(f"Refusing to replace {RUN_ROOT / manifest_file}")
        kept, excluded = count_start_frames(base_manifest["correction_episode_selection"]["train"], mode)
        manifest = copy.deepcopy(base_manifest)
        manifest["task"] = f"cylinder_hil_v2_window_{tag}"
        manifest["parent_manifest"] = str(base_manifest_path.relative_to(ROOT))
        manifest["parent_manifest_sha256"] = sha256(base_manifest_path)
        manifest["train"] = dict(base_manifest["train"], human_frames=kept,
                                 human_frames_before_static_filter=base_manifest["train"]["human_frames"],
                                 excluded_static_start_frames=excluded)
        manifest["window"] = dict(
            future_extension=mode, skip_system_hold=True, extension_success_only=True,
            static_range_deg=STATIC_RANGE_DEG, action_window=ACTION_WINDOW,
            static_definition="supervised action window (timeline actually trained on) varies < range on every joint",
            validation="HUMAN validation identical to the base run",
            note="train.human_frames counts HUMAN start frames kept after the static filter")
        (RUN_ROOT / manifest_file).write_text(json.dumps(manifest, indent=2) + "\n")
        cfg = OmegaConf.load(BASE_ROOT / "configs" / f"{BASE_CONFIG}.yaml")
        cfg.data.train.correction_exclude_static_starts = True
        cfg.data.train.correction_static_range_deg = STATIC_RANGE_DEG
        cfg.data.train.correction_future_extension = mode
        cfg.data.train.correction_extension_success_only = True
        cfg.output_dir = f"{RUN_ROOT_RELATIVE}/{name}/model"
        cfg.wandb.name = name
        cfg.wandb.group = WANDB_GROUP
        cfg.experiment.name = name
        cfg.experiment.run_root = RUN_ROOT_RELATIVE
        cfg.experiment.manifest_file = manifest_file
        cfg.experiment.manifest_sha256 = sha256(RUN_ROOT / manifest_file)
        cfg.experiment.reference_run = f"{BASE_RELATIVE}/{BASE_CONFIG}"
        cfg.experiment.future_extension = mode
        cfg.experiment.exclude_static_starts = True
        OmegaConf.save(cfg, RUN_ROOT / "configs" / f"{name}.yaml", resolve=True)
        summary[name] = dict(mode=mode, start_frames_kept=kept, excluded=excluded,
                             excluded_pct=round(100 * excluded / (kept + excluded), 1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
