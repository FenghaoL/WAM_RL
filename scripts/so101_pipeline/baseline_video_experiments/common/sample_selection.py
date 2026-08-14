#!/usr/bin/env python3
"""Pick a diverse, reproducible set of SO101 evaluation windows.

Every baseline in this directory scores the same windows, so the numbers can be
compared directly. The selection is written to a manifest and reused; it is
never recomputed per run.

Geometry of one window (from the SO101 training config):

    num_frames             = 49 control frames  (LeRobot rows, 15 Hz)
    action_video_freq_ratio= 4                  -> 13 video frames at 3.75 Hz
    clip span              = 48 control frames  = 3.2 s
    latent frames          = 4                  (VAE temporal factor 4)
    history stride         = 48 control frames  = one whole clip back, RoPE 12
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _spread(items: list[Any]) -> list[Any]:
    """Reorder a list so consecutive picks are spread across its whole range."""
    if len(items) <= 2:
        return list(items)
    order = np.linspace(0, len(items) - 1, num=len(items)).round().astype(int)
    seen: set[int] = set()
    result = []
    step = max(1, len(items) // 4)
    for start in range(step):
        for position in range(start, len(items), step):
            index = int(order[position])
            if index not in seen:
                seen.add(index)
                result.append(items[index])
    for index, item in enumerate(items):
        if index not in seen:
            result.append(item)
    return result


def build_candidates(dataset: Any) -> dict[str, list[dict[str, Any]]]:
    """Enumerate every padding-free window that has at least one history frame."""
    starts = dataset.lerobot_dataset.episode_data_index["from"].tolist()
    ends = dataset.lerobot_dataset.episode_data_index["to"].tolist()
    horizon = int(dataset.num_frames - 1) * int(dataset.global_sample_stride)
    num_history = int(dataset.num_history_frames)

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode_slot, (start, end) in enumerate(zip(starts, ends)):
        length = int(end) - int(start)
        latest_relative = length - int(dataset.num_frames)
        if latest_relative < horizon:
            continue
        max_k = latest_relative // horizon
        for k in range(1, max_k + 1):
            relative = k * horizon
            sample_index = int(start) + relative
            # `dataset_dirs` may list several recording sessions, and the global
            # index runs across their concatenation. Looking the row up in
            # `_datasets[0]` would mislabel every sample past the first session,
            # so resolve which sub-dataset owns this index first.
            raw, local_index = dataset._resolve_lerobot_dataset_for_global_index(
                sample_index
            )
            session = next(
                i for i, d in enumerate(dataset.lerobot_dataset.multi_dataset._datasets)
                if d is raw
            )
            raw_item = raw.hf_dataset[local_index]
            task = str(raw.meta.tasks[int(raw_item["task_index"].item())])
            candidates[task].append(
                {
                    "sample_index": sample_index,
                    "episode_slot": episode_slot,
                    "session": session,
                    "source_episode": int(raw_item["episode_index"].item()),
                    "source_frame": int(raw_item["frame_index"].item()),
                    "episode_length": length,
                    "task": task,
                    "history_steps_back": k,
                    "history_frames": min(k, num_history),
                }
            )
    return dict(candidates)


def _order_within_task(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Interleave history depths, and spread episodes inside each depth."""
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_depth[int(item["history_frames"])].append(item)
    depths = sorted(by_depth, reverse=True)
    lanes = [_spread(sorted(by_depth[d], key=lambda x: x["source_episode"])) for d in depths]

    ordered: list[dict[str, Any]] = []
    position = 0
    while any(position < len(lane) for lane in lanes):
        for lane in lanes:
            if position < len(lane):
                ordered.append(lane[position])
        position += 1
    return ordered


def select_so101_samples(dataset: Any, count: int) -> list[dict[str, Any]]:
    """Round-robin across tasks, then across history depth and episode."""
    if count <= 0:
        raise ValueError(f"`count` must be positive, got {count}")
    candidates = build_candidates(dataset)
    if not candidates:
        raise ValueError("No padding-free SO101 window has a usable history frame.")

    lanes = {task: _order_within_task(items) for task, items in candidates.items()}
    tasks = sorted(lanes)
    selected: list[dict[str, Any]] = []
    position = 0
    while len(selected) < count:
        progressed = False
        for task in tasks:
            if position < len(lanes[task]) and len(selected) < count:
                selected.append(lanes[task][position])
                progressed = True
        if not progressed:
            break
        position += 1
    if len(selected) < count:
        raise ValueError(
            f"Only {len(selected)} SO101 windows are available, requested {count}."
        )
    return selected


def safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def sample_tag(order: int, selection: dict[str, Any]) -> str:
    """`episode_index` restarts at 0 in every recording session, so the tag also
    carries the session index to stay unique across a multi-directory dataset."""
    task = safe_name(str(selection["task"]))[:32]
    session = int(selection.get("session", 0))
    return (
        f"{order:02d}_s{session}_ep{int(selection['source_episode']):03d}"
        f"_f{int(selection['source_frame']):04d}_{task}"
    )


def save_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if "samples" not in manifest:
        raise ValueError(f"Manifest {path} has no `samples` list.")
    return manifest
