"""Prove that trimming clip heads keeps actions, proprio and video frames aligned.

Builds the trimmed correction-clip dataset of one configuration plus an untrimmed
twin of the same sessions, then compares the samples that must be identical:
logical frame ``i`` of a trimmed episode is stored frame ``i + trim`` of the
untrimmed one.  Every tensor is compared exactly, including the sampled video
window and the padding masks, and a negative control confirms the trim actually
moved the episode start.  Nothing on disk is read-modified; the trim is a
load-time logical view.

    PYTHONPATH=src:. python .../check_trim_alignment.py --config <experiment yaml>
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ahawam.runtime import build_datasets, _configure_dataset_history

COMPARED = ("action", "proprio", "video", "chunk_obs_images", "chunk_obs_images_no_offset",
            "action_offset", "action_is_pad", "image_is_pad", "proprio_is_pad", "data_source")


def sample_at(dataset, index: int, seed: int):
    """Fetch one sample under a fixed RNG.

    ``action_offset`` is drawn per __getitem__ with np.random, and the chunk
    observation images are selected with it, so two datasets must draw the same
    offset before their samples can be compared.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    return dataset[int(index)]


def untrimmed_twin(cfg):
    """The same clip sessions, same processor, trim disabled."""
    raw = copy.deepcopy(cfg.data.train)
    raw._target_ = "scripts.so101_pipeline.hil_human_only_dataset.SO101CorrectionClipDataset"
    raw.dataset_dirs = list(cfg.data.train.correction_clip_dataset_dirs)
    for key in ("demo_dataset_dirs", "correction_dataset_dirs", "correction_clip_dataset_dirs",
                "correction_episode_selection", "sampling_strategy", "correction_clip_trim_leading_frames"):
        raw.pop(key, None)
    raw.trim_leading_frames = 0
    dataset = instantiate(raw)
    _configure_dataset_history(dataset, cfg.model)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=8, help="episodes to compare (0 = all)")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    trim = int(cfg.data.train.correction_clip_trim_leading_frames)
    train, _ = build_datasets(cfg.data, cfg.model)
    trimmed = train.correction_parts["correction_clips"]
    raw = untrimmed_twin(cfg)

    trimmed_index, raw_index = trimmed.lerobot_dataset.episode_data_index, raw.lerobot_dataset.episode_data_index
    t_from, t_to = trimmed_index["from"].tolist(), trimmed_index["to"].tolist()
    r_from, r_to = raw_index["from"].tolist(), raw_index["to"].tolist()
    survivors = [i for i, (a, b) in enumerate(zip(r_from, r_to)) if b - a > trim]
    if len(survivors) != len(t_from):
        raise AssertionError(f"{len(t_from)} trimmed episodes vs {len(survivors)} surviving raw episodes")
    if len(raw) - len(trimmed) != trim * len(survivors) + sum(
            r_to[i] - r_from[i] for i in range(len(r_from)) if i not in survivors):
        raise AssertionError("Frame accounting does not match the requested trim")

    checks, episodes = [], survivors if args.episodes == 0 else survivors[:args.episodes]
    for position, raw_episode in enumerate(episodes):
        length = t_to[position] - t_from[position]
        for offset in sorted({0, 1, length // 2, length - 1}):
            seed = 9000 + 17 * raw_episode + offset
            a = sample_at(trimmed, t_from[position] + offset, seed)
            b = sample_at(raw, r_from[raw_episode] + trim + offset, seed)
            mismatched = [k for k in COMPARED if not torch.equal(a[k], b[k])]
            if mismatched:
                raise AssertionError(f"episode {raw_episode} offset {offset}: {mismatched} differ")
            checks.append(dict(episode=raw_episode, trimmed_length=length, offset=offset,
                               padded_actions=int(a["action_is_pad"].sum()),
                               padded_images=int(a["image_is_pad"].sum()),
                               video_supervised=bool(a["video_supervision_mask"])))
        # Negative control: the trimmed episode must NOT start at the raw start.
        if trim > 0 and torch.equal(sample_at(trimmed, t_from[position], 7)["action"],
                                    sample_at(raw, r_from[raw_episode], 7)["action"]):
            raise AssertionError(f"episode {raw_episode}: trim did not move the episode start")

    report = dict(config=args.config, trim_leading_frames=trim,
                  fps_hint="0.5 s at 15 fps = 7.5 frames, rounded up to 8",
                  raw_frames=len(raw), trimmed_frames=len(trimmed),
                  raw_episodes=len(r_from), trimmed_episodes=len(t_from),
                  dropped_episodes=int(getattr(trimmed.lerobot_dataset, "dropped_episodes", 0)),
                  compared_tensors=list(COMPARED), episodes_checked=len(episodes), checks=checks)
    out = Path(cfg.experiment.run_root) / f"trim_alignment_check_{cfg.experiment.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "checks"}, indent=2))
    print(f"OK: {len(checks)} paired samples identical after trimming; report -> {out}")


if __name__ == "__main__":
    main()
