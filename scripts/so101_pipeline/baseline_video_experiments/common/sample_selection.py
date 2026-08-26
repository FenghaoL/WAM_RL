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


def build_candidates(
    dataset: Any,
    *,
    min_history_steps: int = 1,
    window_stride: int | None = None,
    include_episode_end: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Enumerate every padding-free window.

    `min_history_steps=1` (the default) keeps only windows that have at least one
    history frame behind them. Pass `0` to also admit the window that starts at
    frame 0 of an episode -- the robot's actual initial state, which by definition
    has no history. That is a real deployment condition: the online policy's
    `_history_kv_entries` is empty for the first planner tick of every episode.
    Admitting k=0 also rescues short episodes (>= num_frames instead of
    >= num_frames + horizon), which would otherwise be skipped entirely.

    `window_stride` (in control frames) sets how densely windows are placed
    inside an episode; it defaults to one whole clip (48), which is what the
    history spacing uses. A smaller stride yields more windows per episode.
    History is unaffected: history frames always sit at `sample_idx - 48*j`
    whatever the window offset, so a window at any offset still gets
    `offset // 48` valid history frames.

    `include_episode_end` additionally emits the very last window an episode
    admits (offset `length - num_frames`), which covers its final 3.2 s. The
    stride grid usually misses it, and that tail is where the task actually
    completes.
    """
    starts = dataset.lerobot_dataset.episode_data_index["from"].tolist()
    ends = dataset.lerobot_dataset.episode_data_index["to"].tolist()
    horizon = int(dataset.num_frames - 1) * int(dataset.global_sample_stride)
    num_history = int(dataset.num_history_frames)
    stride = int(window_stride) if window_stride else horizon
    if stride <= 0:
        raise ValueError(f"`window_stride` must be positive, got {window_stride}")

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode_slot, (start, end) in enumerate(zip(starts, ends)):
        length = int(end) - int(start)
        latest_relative = length - int(dataset.num_frames)
        if latest_relative < 0:
            continue
        if latest_relative // horizon < min_history_steps:
            continue
        offsets = list(range(0, latest_relative + 1, stride))
        if include_episode_end and latest_relative not in offsets:
            offsets.append(latest_relative)
        for relative in offsets:
            k = relative // horizon
            if k < int(min_history_steps):
                continue
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
                    "relative": relative,
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


def select_so101_samples(
    dataset: Any,
    count: int,
    *,
    per_episode_start: bool = False,
    per_episode_end: bool = False,
    window_stride: int | None = None,
    min_history_steps: int | None = None,
) -> list[dict[str, Any]]:
    """Round-robin across tasks, then across history depth and episode.

    With `per_episode_start=True` the selection first reserves one window at
    frame 0 of *every* episode, so no episode's initial state is left unscored,
    then fills the remaining budget with the usual diverse round-robin. `count`
    is raised to fit the reserved windows if it was set too low.

    `min_history_steps` controls the fill pool and defaults to 1, or to 0 when
    `per_episode_start` is on -- asking for episode starts means opting into
    history-free windows, so the rest of the pool should admit them too.
    """
    if count <= 0:
        raise ValueError(f"`count` must be positive, got {count}")
    if min_history_steps is None:
        min_history_steps = 0 if per_episode_start else 1
    num_frames = int(dataset.num_frames)

    selected: list[dict[str, Any]] = []
    seen: set[int] = set()
    if per_episode_start or per_episode_end:
        edges = build_candidates(
            dataset,
            min_history_steps=0,
            window_stride=window_stride,
            include_episode_end=per_episode_end,
        )
        reserved = []
        for items in edges.values():
            for x in items:
                is_start = x["relative"] == 0
                is_end = x["relative"] == int(x["episode_length"]) - num_frames
                if (per_episode_start and is_start) or (per_episode_end and is_end):
                    reserved.append(x)
        for item in sorted(
            reserved, key=lambda x: (x["task"], x["session"], x["source_episode"], x["relative"])
        ):
            if int(item["sample_index"]) in seen:
                continue
            selected.append(item)
            seen.add(int(item["sample_index"]))
        if selected and count < len(selected):
            print(
                f"[select] count={count} 装不下 {len(selected)} 个 episode 首/尾窗口，"
                f"自动提高到 {len(selected)}",
                flush=True,
            )
            count = len(selected)

    candidates = build_candidates(
        dataset, min_history_steps=min_history_steps, window_stride=window_stride
    )
    if not candidates and not selected:
        raise ValueError("No padding-free SO101 window is available.")

    lanes = {task: _order_within_task(items) for task, items in candidates.items()}
    tasks = sorted(lanes)
    position = 0
    while len(selected) < count:
        progressed = False
        for task in tasks:
            if position < len(lanes[task]) and len(selected) < count:
                item = lanes[task][position]
                progressed = True
                if int(item["sample_index"]) not in seen:
                    selected.append(item)
                    seen.add(int(item["sample_index"]))
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
