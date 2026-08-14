#!/usr/bin/env python3
"""Validate converted metadata, tensors and real video decoding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ahawam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/so101_g05_square_v1_ahawam_v21")
    parser.add_argument("--decode-episodes", default="0,1,62,124")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    episodes = [json.loads(line) for line in (root / "meta" / "episodes.jsonl").read_text().splitlines()]
    tasks = [json.loads(line) for line in (root / "meta" / "tasks.jsonl").read_text().splitlines()]

    if info["codebase_version"] != "v2.1":
        raise ValueError(f"Expected v2.1, got {info['codebase_version']}")
    if len(episodes) != int(info["total_episodes"]):
        raise ValueError("Episode metadata count does not match info.json")
    if sum(int(item["length"]) for item in episodes) != int(info["total_frames"]):
        raise ValueError("Episode lengths do not sum to total_frames")
    if len(tasks) != int(info["total_tasks"]):
        raise ValueError("Task count does not match info.json")

    expected_index = 0
    for episode in episodes:
        ep = int(episode["episode_index"])
        chunk = ep // int(info["chunks_size"])
        parquet_path = root / "data" / f"chunk-{chunk:03d}" / f"episode_{ep:06d}.parquet"
        table = pq.read_table(parquet_path)
        if table.num_rows != int(episode["length"]):
            raise ValueError(f"Length mismatch in {parquet_path}")
        frame_index = np.asarray(table["frame_index"])
        index = np.asarray(table["index"])
        if not np.array_equal(frame_index, np.arange(table.num_rows)):
            raise ValueError(f"Bad frame_index in {parquet_path}")
        if not np.array_equal(index, np.arange(expected_index, expected_index + table.num_rows)):
            raise ValueError(f"Bad global index in {parquet_path}")
        expected_index += table.num_rows
        for video_key in ("observation.images.fixed", "observation.images.wrist"):
            video_path = root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{ep:06d}.mp4"
            if not video_path.is_file():
                raise FileNotFoundError(video_path)

    decode_episodes = [int(value) for value in args.decode_episodes.split(",") if value.strip()]
    dataset = LeRobotDataset(
        repo_id=root.name,
        root=root,
        episodes=decode_episodes,
        video_backend="pyav",
    )
    offset = 0
    decoded = []
    for selected_index, ep in enumerate(decode_episodes):
        sample_index = offset + min(3, int(episodes[ep]["length"]) - 1)
        sample = dataset[sample_index]
        shapes = {
            key: list(sample[key].shape)
            for key in ("observation.images.fixed", "observation.images.wrist")
        }
        decoded.append({"episode": ep, "task": sample["task"], "image_shapes": shapes})
        offset += int(episodes[ep]["length"])

    print(json.dumps({"info": info, "decoded": decoded}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
