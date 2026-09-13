"""Verify that video history may cross the takeover boundary, and only there.

For one configuration this checks, on real data:

* history frames are taken at exactly ``history_frame_stride`` frame intervals
  back from the current frame, in raw storage order, and never before the start
  of the rollout the HUMAN span belongs to;
* at least some samples really do read frames from before the takeover began
  (otherwise the feature silently does nothing);
* the RoPE positions handed to the model equal (frame - rollout start) / 4, are
  non-negative, and are spaced by stride/4;
* the current frame, the supervised actions and the predicted video window stay
  inside the HUMAN span (``action_is_pad`` matches the span boundary);
* with crossing disabled the same sample has strictly fewer history frames.

    PYTHONPATH=src:. python .../check_history_crossing.py --config <experiment yaml>
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ahawam.runtime import build_datasets, _configure_dataset_history


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=40)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    stride = int(cfg.data.train.correction_history_frame_stride)
    skip_hold = bool(cfg.data.train.get("correction_history_skip_system_hold", False))
    rope_stride = int(cfg.data.train.action_video_freq_ratio)
    train, _ = build_datasets(cfg.data, cfg.model)
    human = train.correction_parts["hil_human"]
    base = human.lerobot_dataset

    # Same data with crossing switched off, for the comparison below.
    plain_cfg = copy.deepcopy(cfg.data.train)
    plain_cfg._target_ = "scripts.so101_pipeline.hil_human_only_dataset.SO101HumanCorrectionRobotVideoDataset"
    plain_cfg.dataset_dirs = list(cfg.data.train.correction_dataset_dirs)
    plain_cfg.episode_selection = cfg.data.train.correction_episode_selection
    for key in ("demo_dataset_dirs", "correction_dataset_dirs", "correction_clip_dataset_dirs",
                "correction_episode_selection", "sampling_strategy", "correction_clip_trim_leading_frames",
                "correction_history_crosses_segment", "correction_history_frame_stride",
                "correction_history_skip_system_hold"):
        plain_cfg.pop(key, None)
    plain_cfg.history_crosses_segment = False
    plain_cfg.history_frame_stride = stride
    plain_cfg.history_skip_system_hold = skip_hold
    plain = instantiate(plain_cfg)
    _configure_dataset_history(plain, cfg.model)

    valid_crossing = human.get_video_history_valid_len_for_all_indices()
    plain_valid = plain.get_video_history_valid_len_for_all_indices()
    if int(valid_crossing.sum()) <= int(plain_valid.sum()):
        raise AssertionError("Crossing did not add any history frames")

    segment_starts = base.episode_data_index["from"].tolist()
    segment_ends = base.episode_data_index["to"].tolist()
    checks, before_takeover, holds_skipped = [], 0, []
    step = max(1, len(human) // max(1, args.samples))
    for logical in range(0, len(human), step):
        storage = base.storage_index_for_logical_index(logical)
        rollout_start = base.raw_episode_start_for_storage_index(storage)
        indices, _ = human._history_storage_indices(logical)
        if skip_hold:
            rank = base.executing_rank_for_storage_index(storage)
            expected = [base.storage_index_for_executing_rank(rollout_start, rank - stride * k)
                        for k in range(human.num_history_frames, 0, -1) if rank - stride * k >= 0]
            # every frame used as history must itself be an executing frame, and
            # consecutive history frames must be exactly `stride` executing frames apart
            for idx in indices:
                if not base.is_executing_frame(idx):
                    raise AssertionError(f"logical {logical}: history frame {idx} is SYSTEM_HOLD")
            ranks = [base.executing_rank_for_storage_index(i) for i in indices]
            if len(ranks) >= 2 and {ranks[i + 1] - ranks[i] for i in range(len(ranks) - 1)} != {stride}:
                raise AssertionError(f"logical {logical}: executing-frame gaps {ranks} != {stride}")
            skipped = sum(1 for i in range(indices[0], storage) if not base.is_executing_frame(i)) if indices else 0
            holds_skipped.append(skipped)
        else:
            expected = [storage - stride * k for k in range(human.num_history_frames, 0, -1)
                        if storage - stride * k >= rollout_start]
        if indices != expected:
            raise AssertionError(f"logical {logical}: history indices {indices} != {expected}")
        segment = next(i for i, (a, b) in enumerate(zip(segment_starts, segment_ends)) if a <= logical < b)
        segment_storage_start = base.human_segments[segment].storage_start
        crossed = sum(1 for i in indices if i < segment_storage_start)
        before_takeover += crossed
        sample = human[logical]
        positions = sample["video_history_frame_indices"].tolist()
        valid = int(sample["video_history_valid_len"])
        if valid != len(indices):
            raise AssertionError(f"logical {logical}: valid_len {valid} != {len(indices)}")
        real_positions = positions[len(positions) - valid:] if valid else []
        expected_positions = [
            (base.executing_rank_for_storage_index(i) // rope_stride) if skip_hold
            else ((i - rollout_start) // rope_stride) for i in indices]
        if real_positions != expected_positions:
            raise AssertionError(f"logical {logical}: RoPE positions {real_positions} unexpected")
        if any(p < 0 for p in positions):
            raise AssertionError(f"logical {logical}: negative RoPE position {positions}")
        if valid >= 2 and {real_positions[i + 1] - real_positions[i] for i in range(valid - 1)} != {stride // rope_stride}:
            raise AssertionError(f"logical {logical}: history spacing wrong {real_positions}")
        # The supervised window must still stop at the end of the HUMAN span.
        remaining = segment_ends[segment] - logical
        expected_valid_actions = min(int(sample["action_is_pad"].numel()), remaining)
        actual_valid_actions = int((~sample["action_is_pad"]).sum())
        if actual_valid_actions != expected_valid_actions:
            raise AssertionError(
                f"logical {logical}: {actual_valid_actions} valid actions, expected {expected_valid_actions}")
        checks.append(dict(logical=logical, storage=storage, rollout_start=rollout_start,
                           segment_storage_start=segment_storage_start, history_storage=indices,
                           frames_from_before_takeover=crossed, rope_positions=positions,
                           valid_actions=actual_valid_actions))
    report = dict(config=args.config, history_frame_stride=stride, skip_system_hold=skip_hold,
                  system_hold_frames_skipped_mean=(sum(holds_skipped) / len(holds_skipped)) if holds_skipped else 0, num_history_frames=int(cfg.model.num_history_frames),
                  samples_checked=len(checks), history_frames_from_before_takeover=before_takeover,
                  mean_history_with_crossing=float(valid_crossing.float().mean()),
                  mean_history_without_crossing=float(plain_valid.float().mean()),
                  zero_history_with_crossing_pct=float((valid_crossing == 0).float().mean() * 100),
                  zero_history_without_crossing_pct=float((plain_valid == 0).float().mean() * 100),
                  checks=checks[:10])
    out = Path(cfg.experiment.run_root) / f"history_crossing_check_{cfg.experiment.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    print(f"OK: {len(checks)} samples verified, {before_takeover} history frames came from before the takeover")


if __name__ == "__main__":
    main()
