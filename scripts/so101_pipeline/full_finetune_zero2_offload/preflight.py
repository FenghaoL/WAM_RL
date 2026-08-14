#!/usr/bin/env python3
"""Fast metadata and prepared-asset checks for the formal SO101 run."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from ahawam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from scripts.so101_pipeline.config_utils import load_training_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True)
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    cfg = load_training_config(args.settings)

    total_episodes = 0
    total_frames = 0
    prompts = set()
    for raw_dir in cfg.data.train.dataset_dirs:
        root = Path(str(raw_dir))
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        if info["codebase_version"] != "v2.1" or int(info["fps"]) != 15:
            raise ValueError(f"Unexpected dataset metadata in {root}: {info}")
        for key in ("action", "observation.state"):
            if info["features"][key]["shape"] != [6]:
                raise ValueError(f"{root}: {key} is not 6-D")
        episodes = [
            json.loads(line)
            for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()
            if line.strip()
        ]
        total_episodes += len(episodes)
        total_frames += sum(int(row["length"]) for row in episodes)
        for line in (root / "meta" / "tasks.jsonl").read_text().splitlines():
            if line.strip():
                prompts.add(DEFAULT_PROMPT.format(task=json.loads(line)["task"]))
        for episode in episodes:
            ep = int(episode["episode_index"])
            chunk = ep // int(info["chunks_size"])
            for camera in ("observation.images.fixed", "observation.images.wrist"):
                path = root / "videos" / f"chunk-{chunk:03d}" / camera / f"episode_{ep:06d}.mp4"
                if not path.is_file():
                    raise FileNotFoundError(path)

    print(
        f"datasets={len(cfg.data.train.dataset_dirs)} episodes={total_episodes} "
        f"frames={total_frames} unique_prompts={len(prompts)}"
    )
    if args.metadata_only:
        return

    base_checkpoint = Path(str(cfg.base_init_checkpoint))
    checkpoint = Path(str(cfg.init_checkpoint))
    stats = Path(str(cfg.data.train.pretrained_norm_stats))
    cache = Path(str(cfg.data.train.text_embedding_cache_dir))
    if not base_checkpoint.is_file():
        raise FileNotFoundError(base_checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not stats.is_file():
        raise FileNotFoundError(stats)
    payload = json.loads(stats.read_text(encoding="utf-8"))
    for group in ("action", "state"):
        if len(payload[group]["default"]["global_mean"]) != 6:
            raise ValueError(f"{stats}: {group}.default is not 6-D")
    for prompt in prompts:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        path = cache / f"{digest}.t5_len128.wan22ti2v5b.pt"
        if not path.is_file():
            raise FileNotFoundError(path)
        encoded = torch.load(path, map_location="cpu", weights_only=True)
        if "context" not in encoded or "mask" not in encoded:
            raise ValueError(f"Malformed text cache: {path}")
    print(
        f"base_checkpoint={base_checkpoint} overlay_checkpoint={checkpoint} "
        f"stats={stats} text_cache={cache} status=ready"
    )


if __name__ == "__main__":
    main()
