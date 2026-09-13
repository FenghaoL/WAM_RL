"""Verify the correction-window variants on real data, independently of the remapper.

The expected timeline of every checked start frame is rebuilt here from raw
parquet control columns and the recording_context outcome labels, then compared
with what the training dataset returns:

* each checked future image equals the current-frame image of a plain
  (no remapping) dataset at the expected storage frame, and the padding flags
  match the timeline length;
* each checked action equals the plain dataset's action at the expected frame;
  in variant A actions stop at the HUMAN span end, in variant B they follow the
  extended timeline;
* no frame beyond the span is SYSTEM_HOLD, and failed rollouts are never extended;
* retained start frames have a moving supervised action window and excluded ones a
  static one (< range on every joint over the timeline actually trained on), and
  the number of excluded start frames matches the variant manifest;
* the video supervision mask is true exactly when all 13 sampled images are real.

    PYTHONPATH=src:. python .../check_future_window.py --config <variant yaml> --samples 10
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ahawam.runtime import build_datasets
from scripts.so101_pipeline.hil_human_only_dataset import SYSTEM_HOLD_CONTROL_SOURCE, _to_numpy


def plain_twin(cfg):
    raw = copy.deepcopy(cfg.data.train)
    raw._target_ = "scripts.so101_pipeline.so101_dataset.SO101RobotVideoDataset"
    raw.dataset_dirs = list(cfg.data.train.correction_dataset_dirs)
    for key in [k for k in list(raw.keys()) if k.startswith("correction_") or k in (
            "demo_dataset_dirs", "sampling_strategy", "video_supervision_min_segment_frames",
            "video_supervision_require_full_window")]:
        raw.pop(key, None)
    dataset = instantiate(raw)
    dataset.configure_video_history_memory(num_history_frames=0)
    return dataset


def global_arrays(plain):
    """control_source, rollout start/end and rollout outcome per storage index."""
    cs, r_start, r_end, outcome, actions = [], [], [], {}, []
    offset = 0
    for dataset in plain.lerobot_dataset.multi_dataset._datasets:
        table = dataset.hf_dataset.data
        cs.append(_to_numpy(table.column("control_source")).astype(np.int64))
        actions.append(np.asarray(table.column("action").to_pylist(), dtype=np.float64))
        logical = _to_numpy(table.column("logical_episode_index")).astype(np.int64)
        root = Path(str(getattr(dataset, "root", None) or dataset.meta.root))
        labels = {int(json.loads(l)["logical_episode_index"]): json.loads(l).get("outcome", "")
                  for l in (root / "recording_context/episodes_manifest.jsonl").read_text().splitlines() if l.strip()}
        starts = dataset.episode_data_index["from"].tolist(); ends = dataset.episode_data_index["to"].tolist()
        rs = np.zeros(len(logical), np.int64); re = np.zeros(len(logical), np.int64)
        for s, e in zip(starts, ends):
            rs[s:e] = s + offset; re[s:e] = e + offset
            outcome[int(s + offset)] = labels.get(int(logical[s]), "")
        r_start.append(rs); r_end.append(re)
        offset += len(logical)
    return np.concatenate(cs), np.concatenate(r_start), np.concatenate(r_end), outcome, np.concatenate(actions)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    mode = str(cfg.data.train.correction_future_extension)
    static_range = float(cfg.data.train.get("correction_static_range_deg", 1.0))
    exclude_static = bool(cfg.data.train.get("correction_exclude_static_starts", False))
    train, _ = build_datasets(cfg.data, cfg.model)
    human = train.correction_parts["hil_human"]
    base = human.lerobot_dataset
    plain = plain_twin(cfg)
    cs, r_start, r_end, outcome, actions = global_arrays(plain)

    manifest = json.loads((Path(cfg.experiment.run_root) / cfg.experiment.get("manifest_file", "data_manifest.json")).read_text())
    expected_excluded = manifest["train"].get("excluded_static_start_frames", 0)
    if base.excluded_static_starts != expected_excluded:
        raise AssertionError(f"excluded starts {base.excluded_static_starts} != manifest {expected_excluded}")

    def timeline(t, end):
        human_line = list(range(t, end))
        allowed = outcome[int(r_start[t])] == "success"
        after = [j for j in range(end, int(r_end[t])) if cs[j] != SYSTEM_HOLD_CONTROL_SOURCE] if allowed else []
        return human_line, human_line + after, allowed

    for segment_id, segment in enumerate(base.human_segments):
        kept = set(int(v) for v in base._storage_indices[base._segment_ids == segment_id].tolist())
        for t in range(segment.storage_start, segment.storage_end):
            human_line, extended, _ = timeline(t, segment.storage_end)
            line = extended if mode == "video_and_actions" else human_line
            w = actions[line[:48]]
            static = bool((w.max(axis=0) - w.min(axis=0)).max() < static_range)
            if exclude_static and static == (t in kept):
                raise AssertionError(f"segment {segment_id} start {t}: static={static} but kept={t in kept}")
            if not exclude_static and t not in kept:
                raise AssertionError(f"segment {segment_id} start {t} dropped although no filter is configured")

    rng = np.random.default_rng(0)
    picks = sorted(rng.choice(len(human), size=min(args.samples, len(human)), replace=False).tolist())
    image_checks = action_checks = extended_samples = failed_rollout_samples = 0
    for logical in picks:
        s = base.storage_index_for_logical_index(logical)
        segment = base.human_segments[int(base._segment_ids[logical])]
        b = segment.storage_end
        human_line, extended, allowed = timeline(s, b)
        if any(cs[j] == SYSTEM_HOLD_CONTROL_SOURCE for j in extended[len(human_line):]):
            raise AssertionError("SYSTEM_HOLD frame inside the extended timeline")
        extended_samples += len(extended) > len(human_line); failed_rollout_samples += (not allowed)
        np.random.seed(123)
        sample = human[logical]
        # images: every 4th frame of the extended timeline
        for k in (0, 3, 6, 9, 12):
            pos = 4 * k
            expected_pad = pos >= len(extended)
            target = extended[pos] if not expected_pad else extended[-1]
            if bool(sample["image_is_pad"][k]) != expected_pad:
                raise AssertionError(f"logical {logical}: image {k} pad flag")
            if not torch.equal(sample["video"][:, k], plain[target]["video"][:, 0]):
                raise AssertionError(f"logical {logical}: image {k} != frame {target}")
            image_checks += 1
        # actions: extended timeline in variant B, HUMAN span in variant A
        line = extended if mode == "video_and_actions" else human_line
        for pos in sorted(p for p in {0, len(human_line) - 1, len(human_line), len(human_line) + 3, 47} if 0 <= p < 48):
            expected_pad = pos >= len(line)
            target = line[pos] if not expected_pad else line[-1]
            if bool(sample["action_is_pad"][pos]) != expected_pad:
                raise AssertionError(f"logical {logical}: action {pos} pad flag ({mode})")
            if not torch.equal(sample["action"][pos], plain[target]["action"][0]):
                raise AssertionError(f"logical {logical}: action {pos} != frame {target}")
            action_checks += 1
        full_window = not bool(sample["image_is_pad"].any())
        if bool(sample["video_supervision_mask"]) != full_window:
            raise AssertionError(f"logical {logical}: video supervision mask")

    report = dict(config=args.config, mode=mode, start_frames=len(human),
                  excluded_static_starts=int(base.excluded_static_starts),
                  rollouts_extendable=f"{base.extension_rollouts_allowed}/{base.extension_rollouts_total}",
                  samples_checked=len(picks), samples_with_extension=extended_samples,
                  samples_from_failed_rollouts=failed_rollout_samples,
                  image_comparisons=image_checks, action_comparisons=action_checks)
    out = Path(cfg.experiment.run_root) / f"future_window_check_{cfg.experiment.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print("OK: extended windows match the raw recording frame by frame")


if __name__ == "__main__":
    main()
