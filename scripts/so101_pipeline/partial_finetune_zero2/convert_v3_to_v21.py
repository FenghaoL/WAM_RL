#!/usr/bin/env python3
"""Build a LeRobot v2.1 view of the SO101 square-task v3 datasets.

The source videos are not copied or transcoded. Each output episode gets a
symlink to its source v3 video, while its parquet timestamps are shifted to the
absolute position of that episode inside the shared v3 video file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


VIDEO_KEYS = (
    "observation.images.fixed",
    "observation.images.wrist",
)
STAT_FEATURES = ("action", "observation.state")
STAT_NAMES = ("min", "max", "mean", "std", "count")


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, ensure_ascii=True, indent=2)
        file.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=True) + "\n")


def discover_sources(source_root: Path) -> list[Path]:
    sources = sorted(source_root.glob("*/*_g05_model_frame"))
    if not sources:
        raise FileNotFoundError(
            f"No '*/*_g05_model_frame' datasets found below {source_root}"
        )
    return sources


def load_episode_rows(source: Path) -> list[dict]:
    paths = sorted((source / "meta" / "episodes").rglob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"Missing v3 episode metadata below {source}")
    return pa.concat_tables([pq.read_table(path) for path in paths]).to_pylist()


def load_data_table(source: Path, episode_row: dict) -> pa.Table:
    chunk = int(episode_row["data/chunk_index"])
    file_index = int(episode_row["data/file_index"])
    path = source / "data" / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.parquet"
    if not path.is_file():
        raise FileNotFoundError(path)
    return pq.read_table(path)


def set_column(table: pa.Table, name: str, values: pa.Array) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise KeyError(f"Required column '{name}' is missing")
    return table.set_column(index, name, values)


def episode_stats(row: dict) -> dict:
    stats: dict[str, dict] = {}
    for feature in STAT_FEATURES:
        stats[feature] = {
            name: row[f"stats/{feature}/{name}"] for name in STAT_NAMES
        }
    return stats


def aggregate_stats(stats_rows: list[dict]) -> dict:
    result: dict[str, dict] = {}
    for feature in STAT_FEATURES:
        means = np.stack([np.asarray(row[feature]["mean"], dtype=np.float64) for row in stats_rows])
        stds = np.stack([np.asarray(row[feature]["std"], dtype=np.float64) for row in stats_rows])
        counts = np.asarray(
            [float(np.asarray(row[feature]["count"]).reshape(-1)[0]) for row in stats_rows],
            dtype=np.float64,
        )
        total = float(counts.sum())
        mean = (means * counts[:, None]).sum(axis=0) / total
        variance = ((stds**2 + (means - mean) ** 2) * counts[:, None]).sum(axis=0) / total
        result[feature] = {
            "min": np.stack([np.asarray(row[feature]["min"]) for row in stats_rows]).min(axis=0).tolist(),
            "max": np.stack([np.asarray(row[feature]["max"]) for row in stats_rows]).max(axis=0).tolist(),
            "mean": mean.tolist(),
            "std": np.sqrt(variance).tolist(),
            "count": [int(total)],
        }
    return result


def source_video_path(source: Path, episode_row: dict, video_key: str) -> Path:
    chunk = int(episode_row[f"videos/{video_key}/chunk_index"])
    file_index = int(episode_row[f"videos/{video_key}/file_index"])
    path = source / "videos" / video_key / f"chunk-{chunk:03d}" / f"file-{file_index:03d}.mp4"
    if not path.is_file():
        raise FileNotFoundError(path)
    return path.resolve()


def make_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(target, start=link.parent.resolve())
    link.symlink_to(relative_target)


def convert(source_root: Path, output_root: Path) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {output_root}. Remove it deliberately or choose another path."
        )
    output_root.mkdir(parents=True, exist_ok=True)

    sources = discover_sources(source_root)
    first_info = read_json(sources[0] / "meta" / "info.json")
    expected_features = first_info["features"]
    expected_fps = int(first_info["fps"])
    tasks: list[str] = []
    task_to_index: dict[str, int] = {}
    episodes: list[dict] = []
    episodes_stats: list[dict] = []
    plain_stats: list[dict] = []
    manifest: list[dict] = []
    global_episode = 0
    global_frame = 0

    for source in sources:
        info = read_json(source / "meta" / "info.json")
        if int(info["fps"]) != expected_fps:
            raise ValueError(f"FPS mismatch in {source}: {info['fps']} != {expected_fps}")
        if info["features"] != expected_features:
            raise ValueError(f"Feature schema mismatch in {source}")

        data_tables: dict[tuple[int, int], pa.Table] = {}
        for row in load_episode_rows(source):
            source_episode = int(row["episode_index"])
            task_names = list(row["tasks"])
            if len(task_names) != 1:
                raise ValueError(
                    f"Expected one task for {source} episode {source_episode}, got {task_names}"
                )
            task = task_names[0]
            if task not in task_to_index:
                task_to_index[task] = len(tasks)
                tasks.append(task)
            task_index = task_to_index[task]

            data_key = (int(row["data/chunk_index"]), int(row["data/file_index"]))
            if data_key not in data_tables:
                data_tables[data_key] = load_data_table(source, row)
            start = int(row["dataset_from_index"])
            end = int(row["dataset_to_index"])
            length = int(row["length"])
            if end - start != length:
                raise ValueError(
                    f"Bad episode slice in {source}: [{start}, {end}) != length {length}"
                )
            table = data_tables[data_key].slice(start, length)

            video_starts = [
                float(row[f"videos/{key}/from_timestamp"]) for key in VIDEO_KEYS
            ]
            if max(video_starts) - min(video_starts) > 1e-6:
                raise ValueError(
                    f"Camera start timestamps differ in {source} episode {source_episode}: {video_starts}"
                )
            absolute_timestamps = np.array(
                table["timestamp"].to_numpy(), dtype=np.float32, copy=True
            )
            absolute_timestamps += np.float32(video_starts[0])
            table = set_column(table, "timestamp", pa.array(absolute_timestamps, type=pa.float32()))
            table = set_column(
                table,
                "frame_index",
                pa.array(np.arange(length, dtype=np.int64), type=pa.int64()),
            )
            table = set_column(
                table,
                "episode_index",
                pa.array(np.full(length, global_episode, dtype=np.int64), type=pa.int64()),
            )
            table = set_column(
                table,
                "index",
                pa.array(np.arange(global_frame, global_frame + length, dtype=np.int64), type=pa.int64()),
            )
            table = set_column(
                table,
                "task_index",
                pa.array(np.full(length, task_index, dtype=np.int64), type=pa.int64()),
            )

            chunk = global_episode // 1000
            parquet_path = output_root / "data" / f"chunk-{chunk:03d}" / f"episode_{global_episode:06d}.parquet"
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            # datasets<4 cannot parse the newer Hugging Face "List" metadata
            # embedded by LeRobot v3. The Arrow column types themselves are
            # sufficient, so omit only that optional schema metadata.
            pq.write_table(
                table.replace_schema_metadata(None),
                parquet_path,
                compression="zstd",
            )

            for video_key in VIDEO_KEYS:
                target = source_video_path(source, row, video_key)
                link = output_root / "videos" / f"chunk-{chunk:03d}" / video_key / f"episode_{global_episode:06d}.mp4"
                make_symlink(link, target)

            stats = episode_stats(row)
            plain_stats.append(stats)
            episodes_stats.append({"episode_index": global_episode, "stats": stats})
            episodes.append(
                {
                    "episode_index": global_episode,
                    "tasks": [task],
                    "length": length,
                }
            )
            manifest.append(
                {
                    "episode_index": global_episode,
                    "source": str(source.resolve()),
                    "source_episode_index": source_episode,
                    "task": task,
                    "length": length,
                    "video_from_timestamp": video_starts[0],
                }
            )
            global_episode += 1
            global_frame += length

    info = dict(first_info)
    info.update(
        {
            "codebase_version": "v2.1",
            "total_episodes": global_episode,
            "total_frames": global_frame,
            "total_tasks": len(tasks),
            "total_videos": global_episode * len(VIDEO_KEYS),
            "total_chunks": (global_episode + 999) // 1000,
            "chunks_size": 1000,
            "splits": {"train": f"0:{global_episode}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        }
    )
    write_json(output_root / "meta" / "info.json", info)
    write_json(output_root / "meta" / "stats.json", aggregate_stats(plain_stats))
    write_json(output_root / "meta" / "conversion_manifest.json", {"episodes": manifest})
    write_jsonl(
        output_root / "meta" / "tasks.jsonl",
        [{"task_index": index, "task": task} for index, task in enumerate(tasks)],
    )
    write_jsonl(output_root / "meta" / "episodes.jsonl", episodes)
    write_jsonl(output_root / "meta" / "episodes_stats.jsonl", episodes_stats)

    print(f"sources={len(sources)}")
    print(f"episodes={global_episode}")
    print(f"frames={global_frame}")
    print(f"tasks={len(tasks)}")
    print(f"output={output_root.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="data/so101_g05_square_v1")
    parser.add_argument("--output", default="data/so101_g05_square_v1_ahawam_v21")
    args = parser.parse_args()
    convert(Path(args.source).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    main()
