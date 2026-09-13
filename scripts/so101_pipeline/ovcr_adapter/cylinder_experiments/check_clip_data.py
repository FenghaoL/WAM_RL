"""Real-data check of the combined correction source of one configuration.

Verifies source labels and the video rule on actual clip boundaries, that the
mixture places clips after the HIL HUMAN spans, and that demo, HIL and clip
samples collate into one batch.  Writes <run_root>/clip_data_check_<name>.json.
"""
import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import default_collate

from ahawam.runtime import build_datasets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    cfg = OmegaConf.load(p.parse_args().config)
    train, _ = build_datasets(cfg.data, cfg.model)
    hil = train.correction_parts.get("hil_human")
    clips = train.correction_parts["correction_clips"]
    hil_length = len(hil) if hil is not None else 0
    assert train.correction_length == hil_length + len(clips)
    full_window = bool(cfg.data.train.get("video_supervision_require_full_window", False))
    base = clips.lerobot_dataset
    starts, ends = base.episode_data_index["from"].tolist(), base.episode_data_index["to"].tolist()
    lengths = [b - a for a, b in zip(starts, ends)]
    checks = []
    long_ep = max(range(len(lengths)), key=lambda i: lengths[i])
    short_ep = min(range(len(lengths)), key=lambda i: lengths[i])
    for ep in (long_ep, short_ep):
        n, s0 = lengths[ep], starts[ep]
        for offset in sorted({0, max(0, n - 49), max(0, n - 48), n - 1}):
            sample = clips[s0 + offset]
            expected = n >= 49 and (not full_window or n - offset >= 49)
            actual = bool(sample["video_supervision_mask"])
            assert actual == expected, (n, offset, actual, expected)
            assert int(sample["data_source"]) == 1
            checks.append(dict(episode_length=n, offset=offset, remaining=n - offset,
                               video_supervised=actual, padded_images=int(sample["image_is_pad"].sum()),
                               padded_actions=int(sample["action_is_pad"].sum())))
    # Global mixture index of a clip frame = demo + HIL HUMAN + clip-local index.
    via_mixture = train[train.demo_length + hil_length + starts[long_ep]]
    direct = clips[starts[long_ep]]
    assert torch.equal(via_mixture["action"], direct["action"]) and int(via_mixture["data_source"]) == 1
    batch = default_collate([train[0], train[train.demo_length], via_mixture])
    assert batch["data_source"].tolist() == [0, 1, 1]
    report = dict(config=str(p.parse_args().config), full_window_rule=full_window, demo_length=train.demo_length,
                  hil_human_length=hil_length, clip_length=len(clips), clip_episodes=len(lengths),
                  clip_episode_lengths=lengths,
                  boundary_checks=checks, mixed_batch_keys=sorted(batch))
    out = Path(cfg.experiment.run_root) / f"clip_data_check_{cfg.experiment.name}.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "mixed_batch_keys"}, indent=2))


if __name__ == "__main__":
    main()
