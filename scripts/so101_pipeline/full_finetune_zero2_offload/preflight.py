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

    dataset_dirs = [("train", d) for d in cfg.data.train.dataset_dirs]
    if cfg.data.get("val") is not None:
        dataset_dirs += [("val", d) for d in cfg.data.val.dataset_dirs]

    total_episodes = 0
    total_frames = 0
    split_counts: dict[str, list[int]] = {}
    prompts = set()
    for split, raw_dir in dataset_dirs:
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
        counts = split_counts.setdefault(split, [0, 0])
        counts[0] += len(episodes)
        counts[1] += sum(int(row["length"]) for row in episodes)
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

    detail = " ".join(
        f"{split}_episodes={c[0]} {split}_frames={c[1]}"
        for split, c in sorted(split_counts.items())
    )
    print(
        f"datasets={len(dataset_dirs)} episodes={total_episodes} "
        f"frames={total_frames} unique_prompts={len(prompts)} {detail}"
    )
    if cfg.data.get("val") is not None:
        train_dirs = {str(d) for d in cfg.data.train.dataset_dirs}
        overlap = train_dirs & {str(d) for d in cfg.data.val.dataset_dirs}
        if overlap:
            raise ValueError(
                f"val dataset_dirs overlap the training set: {sorted(overlap)}. "
                "Validation metrics would be measured on seen data."
            )
        if float(cfg.data.val.val_set_proportion) > 1e-6:
            raise ValueError(
                "data.val.val_set_proportion must be 0 so the held-out dirs are used "
                f"in full; got {cfg.data.val.val_set_proportion}."
            )
    if args.metadata_only:
        return

    raw_base = cfg.get("base_init_checkpoint")
    base_checkpoint = Path(str(raw_base)) if raw_base else None
    checkpoint = Path(str(cfg.init_checkpoint))
    stats = Path(str(cfg.data.train.pretrained_norm_stats))
    cache = Path(str(cfg.data.train.text_embedding_cache_dir))
    if base_checkpoint is not None and not base_checkpoint.is_file():
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
        f"base_checkpoint={base_checkpoint or '<none, single-stage init>'} "
        f"init_checkpoint={checkpoint} "
        f"stats={stats} text_cache={cache} status=ready"
    )


if __name__ == "__main__":
    main()
