"""Human-only logical views for SO101 AHA-WAM HIL fine-tuning.

The raw HIL recordings intentionally retain POLICY and SYSTEM_HOLD rows for
auditability.  This module never exposes those rows as training samples.  Each
contiguous HUMAN run is treated as an independent logical episode; future
queries are clamped at that run's boundary and marked as padding before image
augmentation or action/state normalization.
"""

from __future__ import annotations

import copy
import json
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch

from ahawam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from scripts.so101_pipeline.so101_dataset import SO101RobotVideoDataset


HUMAN_CONTROL_SOURCE = 1
SYSTEM_HOLD_CONTROL_SOURCE = 2  # the arm holds the previous target while waiting for the policy


@dataclass(frozen=True)
class HumanSegment:
    storage_start: int
    storage_end: int

    @property
    def length(self) -> int:
        return self.storage_end - self.storage_start


def _to_numpy(column: Any, *, dtype=None) -> np.ndarray:
    if hasattr(column, "combine_chunks"):
        column = column.combine_chunks()
    values = column.to_numpy(zero_copy_only=False)
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    return values


def _human_segments_from_columns(columns: dict[str, np.ndarray]) -> list[tuple[int, int]]:
    required = {
        "episode_index",
        "frame_index",
        "tick_index",
        "control_source",
        "intervention",
        "human_action_valid",
        "dropped_tick",
    }
    missing = required - set(columns)
    if missing:
        raise ValueError(f"HIL data is missing required columns: {sorted(missing)}")

    episode = columns["episode_index"].astype(np.int64, copy=False)
    frame = columns["frame_index"].astype(np.int64, copy=False)
    tick = columns["tick_index"].astype(np.int64, copy=False)
    dropped = columns["dropped_tick"].astype(bool, copy=False)
    human = (
        (columns["control_source"].astype(np.int64, copy=False) == HUMAN_CONTROL_SOURCE)
        & columns["intervention"].astype(bool, copy=False)
        & columns["human_action_valid"].astype(bool, copy=False)
    )
    if "action_supervision_mask" in columns:
        supervision = columns["action_supervision_mask"].astype(bool, copy=False)
        invalid_human = human & ~supervision
        if bool(invalid_human.any()):
            raise ValueError(
                f"Found {int(invalid_human.sum())} HUMAN rows with "
                "action_supervision_mask=false."
            )

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, is_human in enumerate(human.tolist()):
        contiguous = (
            idx > 0
            and episode[idx] == episode[idx - 1]
            and frame[idx] == frame[idx - 1] + 1
            and tick[idx] == tick[idx - 1] + 1
            and not dropped[idx]
        )
        if is_human and (start is None or not contiguous):
            if start is not None:
                segments.append((start, idx))
            start = idx
        elif not is_human and start is not None:
            segments.append((start, idx))
            start = None
    if start is not None:
        segments.append((start, len(human)))
    return segments


def static_start_mask(actions: np.ndarray, start: int, end: int, *, range_deg: float,
                      window: int, remapper: "_FutureWindowRemapper | None" = None) -> np.ndarray:
    """True for start frames of ``[start, end)`` whose supervised action window is static.

    The window is the action timeline a sample starting there is trained on -- the
    rest of the HUMAN span, or the extended timeline when actions are extended --
    cut to ``window`` steps. It is static when every joint, gripper included, varies
    by less than ``range_deg`` over it: such a sample only teaches holding the pose.
    Per-frame velocity is deliberately not used: human fine alignment is slow
    (~0.05 deg/frame) and would be mistaken for a pause.
    """
    mask = np.zeros(max(int(end) - int(start), 0), dtype=bool)
    for k, t in enumerate(range(int(start), int(end))):
        line = remapper.action_line(t) if remapper is not None else np.arange(t, int(end))
        w = actions[np.asarray(line[:int(window)], dtype=np.int64)]
        mask[k] = bool((w.max(axis=0) - w.min(axis=0)).max() < float(range_deg))
    return mask

def _rollout_outcomes(dataset, table, episode_starts) -> dict[int, str]:
    """Map each local episode start to its rollout outcome from recording_context."""
    root = Path(str(getattr(dataset, "root", None) or dataset.meta.root))
    manifest = root / "recording_context" / "episodes_manifest.jsonl"
    if not manifest.is_file() or "logical_episode_index" not in table.column_names:
        return {}
    by_logical = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            by_logical[int(record["logical_episode_index"])] = str(record.get("outcome", ""))
    logical = _to_numpy(table.column("logical_episode_index")).astype(np.int64)
    return {int(s): by_logical.get(int(logical[int(s)]), "") for s in episode_starts}


class _FutureWindowRemapper:
    """Query timeline for HUMAN start frames of one LeRobot dataset.

    From a start frame the timeline is the rest of its HUMAN span followed, for an
    extendable rollout, by every later executing (non-SYSTEM_HOLD) frame of that
    rollout. Image queries follow it; action/state queries follow it only when
    ``extend_actions``, otherwise they stop at the end of the HUMAN span. Queries
    past the timeline repeat its last frame and are marked as padding, exactly like
    LeRobot's own episode-boundary handling.
    """

    def __init__(self, *, segment_end, rollout_start, executing_by_rollout,
                 extendable_rollouts, extend_actions: bool):
        self.segment_end = segment_end
        self.rollout_start = rollout_start
        self.executing_by_rollout = executing_by_rollout
        self.extendable_rollouts = extendable_rollouts
        self.extend_actions = bool(extend_actions)

    def timelines(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        end = int(self.segment_end[idx])
        human = np.arange(int(idx), end, dtype=np.int64)
        rollout = int(self.rollout_start[idx])
        if rollout in self.extendable_rollouts:
            executing = self.executing_by_rollout[rollout]
            return human, np.concatenate([human, executing[executing >= end]])
        return human, human

    def action_line(self, idx: int) -> np.ndarray:
        human, extended = self.timelines(int(idx))
        return extended if self.extend_actions else human

    def __call__(self, idx, delta_indices):
        idx = int(idx)
        if idx < 0 or idx >= len(self.segment_end) or int(self.segment_end[idx]) < 0:
            return None
        human, extended = self.timelines(idx)
        query, padding = {}, {}
        for key, deltas in delta_indices.items():
            line = extended if (key.startswith("observation.images") or self.extend_actions) else human
            positions = [int(d) for d in deltas]
            if any(position < 0 for position in positions):
                raise ValueError("Future-window remapping supports only non-negative query offsets.")
            query[key] = [int(line[pos]) if pos < len(line) else int(line[-1]) for pos in positions]
            padding[f"{key}_is_pad"] = torch.BoolTensor([pos >= len(line) for pos in positions])
        return query, padding


def summarize_human_dataset_dirs(dataset_dirs: Iterable[str | Path]) -> dict[str, Any]:
    """Read only parquet control columns and report the logical HUMAN view."""
    all_lengths: list[int] = []
    total_raw_frames = 0
    total_raw_episodes = 0
    for raw_root in dataset_dirs:
        root = Path(str(raw_root))
        parquet_paths = sorted((root / "data").glob("chunk-*/*.parquet"))
        if not parquet_paths:
            raise FileNotFoundError(f"No episode parquet files found under {root / 'data'}")
        for parquet_path in parquet_paths:
            schema_names = set(pq.read_schema(parquet_path).names)
            names = [
                "episode_index",
                "frame_index",
                "tick_index",
                "control_source",
                "intervention",
                "human_action_valid",
                "dropped_tick",
            ]
            if "action_supervision_mask" in schema_names:
                names.append("action_supervision_mask")
            table = pq.read_table(parquet_path, columns=names)
            columns = {name: _to_numpy(table[name]) for name in names}
            lengths = [end - start for start, end in _human_segments_from_columns(columns)]
            all_lengths.extend(lengths)
            total_raw_frames += table.num_rows
            total_raw_episodes += 1
    return {
        "raw_episodes": total_raw_episodes,
        "raw_frames": total_raw_frames,
        "human_segments": len(all_lengths),
        "human_frames": sum(all_lengths),
        "segment_lengths": all_lengths,
    }


class HumanCorrectionBaseLerobotDataset(BaseLerobotDataset):
    """A zero-copy logical dataset containing only contiguous HUMAN spans."""

    def __init__(self, *args, video_supervision_min_segment_frames: int = 49,
                 episode_selection=None, exclude_static_starts: bool = False,
                 static_range_deg: float = 1.0,
                 future_extension: str = "none", extension_success_only: bool = True,
                 **kwargs):
        self.video_supervision_min_segment_frames = int(
            video_supervision_min_segment_frames
        )
        if self.video_supervision_min_segment_frames <= 0:
            raise ValueError("video_supervision_min_segment_frames must be positive")
        if future_extension not in {"none", "video", "video_and_actions"}:
            raise ValueError(f"future_extension must be none, video or video_and_actions, got {future_extension!r}")
        self.exclude_static_starts = bool(exclude_static_starts)
        self.static_range_deg = float(static_range_deg)
        self.future_extension = future_extension
        self.extension_success_only = bool(extension_success_only)
        self.extension_rollouts_total = 0
        self.extension_rollouts_allowed = 0
        super().__init__(*args, **kwargs)

        segments: list[HumanSegment] = []
        segment_static_masks: list[np.ndarray] = []
        executing_chunks: list[np.ndarray] = []
        dataset_storage_offset = 0
        for dataset in self.multi_dataset._datasets:
            table = dataset.hf_dataset.data
            available = set(table.column_names)
            names = [
                "episode_index",
                "frame_index",
                "tick_index",
                "control_source",
                "intervention",
                "human_action_valid",
                "dropped_tick",
            ]
            if "action_supervision_mask" in available:
                names.append("action_supervision_mask")
            columns = {name: _to_numpy(table.column(name)) for name in names}
            actions = (
                np.asarray(table.column("action").to_pylist(), dtype=np.float64)
                if self.exclude_static_starts else None
            )
            segment_end_local = np.full(int(dataset.num_frames), -1, dtype=np.int64)
            local_segments: list[tuple[int, int]] = []
            executing_chunks.append(
                columns["control_source"].astype(np.int64) != SYSTEM_HOLD_CONTROL_SOURCE
            )
            for local_start, local_end in _human_segments_from_columns(columns):
                if episode_selection is not None:
                    allowed = episode_selection.get(str(dataset.repo_id))
                    if allowed is None:
                        raise ValueError(f"Missing episode selection for {dataset.repo_id}")
                    if int(columns["episode_index"][local_start]) not in allowed:
                        continue
                segments.append(
                    HumanSegment(
                        storage_start=dataset_storage_offset + local_start,
                        storage_end=dataset_storage_offset + local_end,
                    )
                )
                local_segments.append((local_start, local_end))
                segment_end_local[local_start:local_end] = local_end
            remapper = None
            if self.future_extension != "none":
                remapper = self._build_future_remapper(dataset, table, columns, segment_end_local)
                dataset.query_index_remapper = remapper
            for local_start, local_end in local_segments:
                # Judged on the same action timeline the queries use, so extended
                # windows (e.g. a later gripper closure) count as motion.
                segment_static_masks.append(
                    static_start_mask(actions, local_start, local_end,
                                      range_deg=self.static_range_deg,
                                      window=int(self.action_size), remapper=remapper)
                    if self.exclude_static_starts
                    else np.zeros(local_end - local_start, dtype=bool)
                )
            dataset_storage_offset += int(dataset.num_frames)

        if not segments:
            raise ValueError("No valid HUMAN correction segments were found.")
        self.human_segments = segments
        storage_indices: list[int] = []
        logical_starts: list[int] = []
        logical_ends: list[int] = []
        segment_ids: list[int] = []
        logical_cursor = 0
        self.excluded_static_starts = 0
        for segment_id, segment in enumerate(segments):
            # Start frames with a static supervised window are dropped; the segment
            # boundaries used for windows are unchanged, so other starts still see
            # those frames.
            keep = [segment.storage_start + k for k in range(segment.length)
                    if not segment_static_masks[segment_id][k]]
            self.excluded_static_starts += segment.length - len(keep)
            logical_starts.append(logical_cursor)
            storage_indices.extend(keep)
            segment_ids.extend([segment_id] * len(keep))
            logical_cursor += len(keep)
            logical_ends.append(logical_cursor)
        if not storage_indices:
            raise ValueError("No HUMAN start frames remain after excluding static starts.")
        # The raw recording boundaries stay available: history may look back past the
        # start of a HUMAN span, but never past the start of the rollout it came from.
        raw_starts = self.episode_data_index["from"].clone()
        raw_ends = self.episode_data_index["to"].clone()
        self.raw_episode_data_index = {"from": raw_starts, "to": raw_ends}
        self._raw_episode_start_per_index = torch.repeat_interleave(
            raw_starts, (raw_ends - raw_starts)
        )
        # Rank of every frame among the *executing* frames of its rollout, so history
        # can step back a fixed number of executing frames and skip SYSTEM_HOLD.
        executing = np.concatenate(executing_chunks) if executing_chunks else np.zeros(0, dtype=bool)
        self._is_executing = torch.from_numpy(np.ascontiguousarray(executing))
        self._executing_rank_per_index = torch.full((int(self._is_executing.numel()),), -1, dtype=torch.long)
        self._executing_positions: dict[int, torch.Tensor] = {}
        for episode_start, episode_end in zip(raw_starts.tolist(), raw_ends.tolist()):
            mask = self._is_executing[episode_start:episode_end]
            self._executing_rank_per_index[episode_start:episode_end] = torch.cumsum(mask.long(), dim=0) - 1
            self._executing_positions[int(episode_start)] = (
                torch.nonzero(mask, as_tuple=False).flatten() + int(episode_start)
            )
        self._storage_indices = torch.tensor(storage_indices, dtype=torch.long)
        self._segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        self.episode_data_index = {
            "from": torch.tensor(logical_starts, dtype=torch.long),
            "to": torch.tensor(logical_ends, dtype=torch.long),
        }

    def __len__(self) -> int:
        return int(self._storage_indices.numel())

    def _map_sample_index(self, sample_idx: int) -> int:
        return int(self._storage_indices[int(sample_idx)].item())

    def storage_index_for_logical_index(self, sample_idx: int) -> int:
        return self._map_sample_index(sample_idx)

    def _build_future_remapper(self, dataset, table, columns, segment_end_local):
        starts = [int(v) for v in dataset.episode_data_index["from"].tolist()]
        ends = [int(v) for v in dataset.episode_data_index["to"].tolist()]
        executing = columns["control_source"].astype(np.int64) != SYSTEM_HOLD_CONTROL_SOURCE
        rollout_start = np.zeros(int(dataset.num_frames), dtype=np.int64)
        executing_by_rollout: dict[int, np.ndarray] = {}
        for episode_start, episode_end in zip(starts, ends):
            rollout_start[episode_start:episode_end] = episode_start
            executing_by_rollout[episode_start] = (
                np.flatnonzero(executing[episode_start:episode_end]) + episode_start
            ).astype(np.int64)
        outcomes = _rollout_outcomes(dataset, table, starts)
        extendable = {
            start for start in starts
            if not self.extension_success_only or outcomes.get(start) == "success"
        }
        self.extension_rollouts_total += len(starts)
        self.extension_rollouts_allowed += len(extendable)
        return _FutureWindowRemapper(
            segment_end=segment_end_local, rollout_start=rollout_start,
            executing_by_rollout=executing_by_rollout, extendable_rollouts=extendable,
            extend_actions=self.future_extension == "video_and_actions",
        )

    def raw_episode_start_for_storage_index(self, storage_idx: int) -> int:
        return int(self._raw_episode_start_per_index[int(storage_idx)].item())

    def executing_rank_for_storage_index(self, storage_idx: int) -> int:
        """How many executing (non-SYSTEM_HOLD) frames precede this one in its rollout."""
        return int(self._executing_rank_per_index[int(storage_idx)].item())

    def storage_index_for_executing_rank(self, rollout_start: int, rank: int) -> int:
        return int(self._executing_positions[int(rollout_start)][int(rank)].item())

    def is_executing_frame(self, storage_idx: int) -> bool:
        return bool(self._is_executing[int(storage_idx)].item())

    @staticmethod
    def _clamp_sequence_to_human_boundary(
        value: torch.Tensor,
        is_pad: torch.Tensor,
        offsets: list[int],
        *,
        remaining_frames: int,
        stride: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = torch.tensor(
            [offset * stride < remaining_frames for offset in offsets],
            dtype=torch.bool,
            device=value.device,
        )
        if int(valid.numel()) != int(value.shape[0]):
            raise ValueError(
                f"Sequence/query mismatch: {value.shape[0]} values for {len(offsets)} offsets."
            )
        last_valid = int(valid.nonzero(as_tuple=False)[-1].item())
        value = value.clone()
        value[~valid] = value[last_valid].clone()
        is_pad = torch.as_tensor(is_pad, dtype=torch.bool).clone()
        is_pad |= ~valid.to(device=is_pad.device)
        return value, is_pad

    def _get_additional_data(self, sample, lerobot_sample):
        logical_idx = int(sample["idx"])
        segment_id = int(self._segment_ids[logical_idx].item())
        segment = self.human_segments[segment_id]
        storage_idx = self._map_sample_index(logical_idx)
        remaining = segment.storage_end - storage_idx
        stride = int(self.global_sample_stride)
        if self.future_extension != "none":
            # The query remapper already built every window along its timeline and
            # marked what lies past it as padding; clamping here would cut the
            # extension back to the end of the HUMAN span. Video eligibility is then
            # decided by the full-window rule on image_is_pad.
            sample["video_supervision_mask"] = torch.tensor(True, dtype=torch.bool)
            sample["data_source"] = torch.tensor(1, dtype=torch.long)
            return sample

        action_offsets = list(range(self.action_size))
        for key in sample["action"]:
            sample["action"][key], sample["action_is_pad"] = (
                self._clamp_sequence_to_human_boundary(
                    sample["action"][key],
                    sample["action_is_pad"],
                    action_offsets,
                    remaining_frames=remaining,
                    stride=stride,
                )
            )

        state_offsets = list(range(self.obs_size))
        for key in sample["state"]:
            sample["state"][key], sample["state_is_pad"] = (
                self._clamp_sequence_to_human_boundary(
                    sample["state"][key],
                    sample["state_is_pad"],
                    state_offsets,
                    remaining_frames=remaining,
                    stride=stride,
                )
            )

        image_offsets = (
            self.image_sample_indices
            if self.image_sample_indices is not None
            else state_offsets
        )
        for key in sample["images"]:
            sample["images"][key], sample["image_is_pad"] = (
                self._clamp_sequence_to_human_boundary(
                    sample["images"][key],
                    sample["image_is_pad"],
                    image_offsets,
                    remaining_frames=remaining,
                    stride=stride,
                )
            )

        sample["video_supervision_mask"] = torch.tensor(
            segment.length >= self.video_supervision_min_segment_frames,
            dtype=torch.bool,
        )
        sample["data_source"] = torch.tensor(1, dtype=torch.long)
        return sample


class SO101HumanCorrectionRobotVideoDataset(SO101RobotVideoDataset):
    """HUMAN spans as logical episodes.

    ``history_crosses_segment`` lets the video history reach back before the
    takeover began -- into the policy-driven part of the same rollout, which is
    what the model sees at deployment -- while the current frame, the action
    window and the predicted video stay inside the HUMAN span.  History is still
    clamped at the start of the raw rollout.  ``history_frame_stride`` overrides
    the spacing between history frames (default: one full prediction horizon,
    ``(num_frames - 1) * global_sample_stride``).
    """

    def __init__(self, *args, video_supervision_min_segment_frames: int = 49,
                 episode_selection=None, video_supervision_require_full_window=False,
                 history_crosses_segment: bool = False,
                 history_frame_stride: int | None = None,
                 history_skip_system_hold: bool = False,
                 exclude_static_starts: bool = False, static_range_deg: float = 1.0,
                 future_extension: str = "none",
                 extension_success_only: bool = True, **kwargs):
        self.video_supervision_require_full_window = bool(video_supervision_require_full_window)
        self.episode_selection = episode_selection
        self.video_supervision_min_segment_frames = int(
            video_supervision_min_segment_frames
        )
        self.history_crosses_segment = bool(history_crosses_segment)
        # Count the stride in executing frames only: the arm is idle during
        # SYSTEM_HOLD, so raw frames would make the memory cover far less task
        # progress in policy-driven rollouts than it does in the demos the
        # backbone was trained on.
        self.history_skip_system_hold = bool(history_skip_system_hold)
        self.history_frame_stride_override = (
            None if history_frame_stride is None else int(history_frame_stride)
        )
        if self.history_frame_stride_override is not None and self.history_frame_stride_override <= 0:
            raise ValueError("history_frame_stride must be positive")
        self._storage_frame_cache: "OrderedDict[int, torch.Tensor]" = OrderedDict()
        self._resolve_raw_storage = False
        self.exclude_static_starts = bool(exclude_static_starts)
        self.static_range_deg = float(static_range_deg)
        self.future_extension = str(future_extension)
        self.extension_success_only = bool(extension_success_only)
        if self.exclude_static_starts and not self.history_crosses_segment:
            # Dropped start frames leave gaps inside a span; only the storage-based
            # history/RoPE path is defined for that.
            raise ValueError("exclude_static_starts requires history_crosses_segment=true")
        if self.future_extension != "none" and not self.video_supervision_require_full_window:
            raise ValueError("future_extension requires video_supervision_require_full_window=true")
        super().__init__(*args, **kwargs)

    @property
    def history_frame_stride(self) -> int:
        if self.history_frame_stride_override is not None:
            return self.history_frame_stride_override
        return int(self.num_frames - 1) * int(self.global_sample_stride)

    def _history_storage_indices(self, sample_idx: int) -> tuple[list[int], int]:
        """History storage indices for one logical index, plus the rollout start."""
        storage_idx = self.lerobot_dataset.storage_index_for_logical_index(int(sample_idx))
        raw_start = self.lerobot_dataset.raw_episode_start_for_storage_index(storage_idx)
        stride = self.history_frame_stride
        if self.history_skip_system_hold:
            rank = self.lerobot_dataset.executing_rank_for_storage_index(storage_idx)
            indices = [
                self.lerobot_dataset.storage_index_for_executing_rank(raw_start, rank - stride * step)
                for step in range(self.num_history_frames, 0, -1)
                if rank - stride * step >= 0
            ]
            return indices, raw_start
        indices = [
            storage_idx - stride * step
            for step in range(self.num_history_frames, 0, -1)
            if storage_idx - stride * step >= raw_start
        ]
        return indices, raw_start

    def _rope_position_for_storage_index(self, storage_idx: int, rollout_start: int) -> int:
        """Temporal position of a frame, in the same units the main clip uses.

        With SYSTEM_HOLD skipped the position counts executing frames, so history
        steps are exactly `stride / action_video_freq_ratio` apart -- the same
        geometry the demo SFT produced -- instead of being compressed by idle time.
        """
        rope_stride = int(self.action_video_freq_ratio)
        if self.history_skip_system_hold:
            return self.lerobot_dataset.executing_rank_for_storage_index(storage_idx) // rope_stride
        return (int(storage_idx) - int(rollout_start)) // rope_stride

    def _processed_frame_by_storage_index(self, storage_idx: int) -> torch.Tensor:
        key = int(storage_idx)
        cached = self._storage_frame_cache.get(key)
        if cached is not None:
            self._storage_frame_cache.move_to_end(key)
            return cached.clone()
        self._resolve_raw_storage = True  # read the frame by storage index, not logical
        try:
            pixel_values = self._get_single_frame_pixel_values(key)
        finally:
            self._resolve_raw_storage = False
        frame = self._process_sampled_video_tensor(pixel_values)[:, 0].contiguous()
        if self._history_frame_cache_maxsize > 0:
            self._storage_frame_cache[key] = frame
            if len(self._storage_frame_cache) > self._history_frame_cache_maxsize:
                self._storage_frame_cache.popitem(last=False)
        return frame.clone()

    def _build_video_history_frames(self, sample_idx, current_frame=None):
        if self.num_history_frames <= 0 or not self.history_crosses_segment:
            return super()._build_video_history_frames(sample_idx, current_frame)
        indices, raw_start = self._history_storage_indices(sample_idx)
        frames = [self._processed_frame_by_storage_index(idx) for idx in indices]
        positions = [self._rope_position_for_storage_index(idx, raw_start) for idx in indices]
        valid_len = len(frames)
        if frames:
            pad_frame = frames[0].new_zeros(frames[0].shape)
        else:
            reference = current_frame if current_frame is not None else self._get_processed_single_frame(sample_idx)
            pad_frame = reference.new_zeros(reference.shape)
        for _ in range(self.num_history_frames - valid_len):
            frames.insert(0, pad_frame)
            positions.insert(0, 0)
        return (
            torch.stack(frames, dim=1),
            torch.tensor(valid_len, dtype=torch.long),
            torch.tensor(positions, dtype=torch.long),
        )

    def _get_video_rope_frame_index(self, sample_idx: int) -> int:
        # With crossing enabled, history and current clip must share one origin:
        # the raw rollout start, so history positions stay non-negative and the
        # spacing between them is physically correct.
        if not self.history_crosses_segment or self.num_history_frames <= 0:
            return super()._get_video_rope_frame_index(sample_idx)
        storage_idx = self.lerobot_dataset.storage_index_for_logical_index(int(sample_idx))
        raw_start = self.lerobot_dataset.raw_episode_start_for_storage_index(storage_idx)
        return self._rope_position_for_storage_index(storage_idx, raw_start)

    def get_video_history_valid_len_for_index(self, sample_idx: int) -> int:
        if not self.history_crosses_segment or self.num_history_frames <= 0:
            return super().get_video_history_valid_len_for_index(sample_idx)
        return len(self._history_storage_indices(sample_idx)[0])

    def get_video_history_valid_len_for_all_indices(self) -> torch.Tensor:
        if not self.history_crosses_segment or self.num_history_frames <= 0:
            return super().get_video_history_valid_len_for_all_indices()
        storage = self.lerobot_dataset._storage_indices
        stride = self.history_frame_stride
        steps = torch.arange(1, self.num_history_frames + 1).unsqueeze(0)
        if self.history_skip_system_hold:
            rank = self.lerobot_dataset._executing_rank_per_index[storage]
            available = (rank.unsqueeze(1) - stride * steps) >= 0
        else:
            raw_start = self.lerobot_dataset._raw_episode_start_per_index[storage]
            available = (storage.unsqueeze(1) - stride * steps) >= raw_start.unsqueeze(1)
        return available.sum(dim=1).to(torch.int16)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if self.video_supervision_require_full_window:
            sample["video_supervision_mask"] = (
                sample["video_supervision_mask"].bool()
                & ~sample["image_is_pad"].bool().any()
            )
        return sample

    def _build_lerobot_dataset(self, **kwargs) -> BaseLerobotDataset:
        return HumanCorrectionBaseLerobotDataset(
            **kwargs,
            episode_selection=self.episode_selection,
            video_supervision_min_segment_frames=(
                self.video_supervision_min_segment_frames
            ),
            exclude_static_starts=self.exclude_static_starts,
            static_range_deg=self.static_range_deg,
            future_extension=self.future_extension,
            extension_success_only=self.extension_success_only,
        )

    def _resolve_lerobot_dataset_for_global_index(self, sample_idx: int):
        if self._resolve_raw_storage:
            return super()._resolve_lerobot_dataset_for_global_index(sample_idx)
        storage_idx = self.lerobot_dataset.storage_index_for_logical_index(sample_idx)
        return super()._resolve_lerobot_dataset_for_global_index(storage_idx)


class CorrectionClipBaseLerobotDataset(BaseLerobotDataset):
    """Ordinary episodes in which every frame is a human correction.

    Short teleoperated recoveries recorded from perturbed insertion poses carry no
    HIL control columns.  Each episode is one HUMAN segment: the base loader
    already clamps and pads future queries at the episode end, so only the source
    label and the segment-length video rule are added here.

    ``trim_leading_frames`` drops the idle head of every episode the same way the
    HIL view drops non-HUMAN rows: no stored frame is modified or re-encoded, the
    logical episode simply starts later.  Action, state and image queries are all
    resolved from the same shifted storage index, so they cannot fall out of sync;
    history and the episode-start boundary move with it.  Episodes with no frame
    left after the trim are dropped.
    """

    def __init__(self, *args, video_supervision_min_segment_frames: int = 49,
                 trim_leading_frames: int = 0, **kwargs):
        self.video_supervision_min_segment_frames = int(
            video_supervision_min_segment_frames
        )
        if self.video_supervision_min_segment_frames <= 0:
            raise ValueError("video_supervision_min_segment_frames must be positive")
        self.trim_leading_frames = int(trim_leading_frames)
        if self.trim_leading_frames < 0:
            raise ValueError("trim_leading_frames must be >= 0")
        super().__init__(*args, **kwargs)
        storage, lengths, logical_from, logical_to, dropped = self.logical_view(
            self.episode_data_index["from"], self.episode_data_index["to"],
            self.trim_leading_frames,
        )
        if not storage:
            raise ValueError("No correction clip frames remain after trimming.")
        self._storage_indices = torch.tensor(storage, dtype=torch.long)
        self._episode_length_per_index = torch.tensor(lengths, dtype=torch.long)
        self.episode_data_index = {
            "from": torch.tensor(logical_from, dtype=torch.long),
            "to": torch.tensor(logical_to, dtype=torch.long),
        }
        self.dropped_episodes = dropped

    @staticmethod
    def logical_view(starts, ends, trim_leading_frames: int):
        """Map trimmed episodes onto storage indices.

        Returns the storage index of every logical frame, the trimmed length of the
        episode each logical frame belongs to, the logical episode boundaries, and
        how many episodes were too short to survive the trim.
        """
        starts = torch.as_tensor(starts, dtype=torch.long).cpu().tolist()
        ends = torch.as_tensor(ends, dtype=torch.long).cpu().tolist()
        trim = int(trim_leading_frames)
        if trim < 0:
            raise ValueError("trim_leading_frames must be >= 0")
        storage: list[int] = []
        lengths: list[int] = []
        logical_from: list[int] = []
        logical_to: list[int] = []
        cursor = 0
        dropped = 0
        for episode_start, episode_end in zip(starts, ends):
            first = int(episode_start) + trim
            if first >= int(episode_end):
                dropped += 1
                continue
            length = int(episode_end) - first
            storage.extend(range(first, int(episode_end)))
            lengths.extend([length] * length)
            logical_from.append(cursor)
            cursor += length
            logical_to.append(cursor)
        return storage, lengths, logical_from, logical_to, dropped

    def __len__(self) -> int:
        return int(self._storage_indices.numel())

    def _map_sample_index(self, sample_idx: int) -> int:
        return int(self._storage_indices[int(sample_idx)].item())

    def storage_index_for_logical_index(self, sample_idx: int) -> int:
        return self._map_sample_index(sample_idx)

    def _get_additional_data(self, sample, lerobot_sample):
        length = int(self._episode_length_per_index[int(sample["idx"])].item())
        sample["video_supervision_mask"] = torch.tensor(
            length >= self.video_supervision_min_segment_frames, dtype=torch.bool
        )
        sample["data_source"] = torch.tensor(1, dtype=torch.long)
        return sample


class SO101CorrectionClipDataset(SO101RobotVideoDataset):
    """Whole-episode correction recordings, treated exactly like HIL HUMAN spans."""

    def __init__(self, *args, video_supervision_min_segment_frames: int = 49,
                 video_supervision_require_full_window=False,
                 trim_leading_frames: int = 0, **kwargs):
        self.video_supervision_min_segment_frames = int(
            video_supervision_min_segment_frames
        )
        self.video_supervision_require_full_window = bool(video_supervision_require_full_window)
        self.trim_leading_frames = int(trim_leading_frames)
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        if self.video_supervision_require_full_window:
            sample["video_supervision_mask"] = (
                sample["video_supervision_mask"].bool()
                & ~sample["image_is_pad"].bool().any()
            )
        return sample

    def _build_lerobot_dataset(self, **kwargs) -> BaseLerobotDataset:
        return CorrectionClipBaseLerobotDataset(
            **kwargs,
            video_supervision_min_segment_frames=(
                self.video_supervision_min_segment_frames
            ),
            trim_leading_frames=self.trim_leading_frames,
        )

    def _resolve_lerobot_dataset_for_global_index(self, sample_idx: int):
        storage_idx = self.lerobot_dataset.storage_index_for_logical_index(sample_idx)
        return super()._resolve_lerobot_dataset_for_global_index(storage_idx)


class ConcatCorrectionDataset(torch.utils.data.Dataset):
    """Several correction views exposed as the single correction source."""

    def __init__(self, parts):
        self.parts = list(parts)
        if not self.parts:
            raise ValueError("At least one correction dataset is required.")
        self.offsets = np.cumsum([0] + [len(part) for part in self.parts]).tolist()

    def __len__(self) -> int:
        return int(self.offsets[-1])

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        part = int(np.searchsorted(self.offsets, idx, side="right")) - 1
        return self.parts[part][idx - int(self.offsets[part])]

    def configure_video_history_memory(self, *, num_history_frames: int) -> None:
        for part in self.parts:
            part.configure_video_history_memory(num_history_frames=num_history_frames)

    def get_video_history_valid_len_for_all_indices(self) -> torch.Tensor:
        return torch.cat(
            [torch.as_tensor(part.get_video_history_valid_len_for_all_indices())
             for part in self.parts]
        )


class SO101DemoHumanCorrectionDataset(torch.utils.data.Dataset):
    """Concatenate demos with a HUMAN-only correction logical view.

    ``sampling_strategy`` is consumed by the trainer's sampler.  It does not
    alter the natural dataset index space, so validation and direct indexing
    remain unsurprising.
    """

    def __init__(
        self,
        *,
        demo_dataset_dirs,
        correction_dataset_dirs=None,
        sampling_strategy: str = "natural",
        correction_episode_selection=None,
        correction_clip_dataset_dirs=None,
        correction_clip_trim_leading_frames: int = 0,
        correction_history_crosses_segment: bool = False,
        correction_history_frame_stride: int | None = None,
        correction_history_skip_system_hold: bool = False,
        correction_exclude_static_starts: bool = False,
        correction_static_range_deg: float = 1.0,
        correction_future_extension: str = "none",
        correction_extension_success_only: bool = True,
        video_supervision_require_full_window=False,
        video_supervision_min_segment_frames: int = 49,
        processor=None,
        **dataset_kwargs,
    ):
        if sampling_strategy not in {"natural", "balanced_2_2", "balanced_2_1", "balanced_3_1", "human_only"}:
            raise ValueError(
                "sampling_strategy must be natural, balanced_2_2, balanced_2_1, balanced_3_1 or human_only; got "
                f"{sampling_strategy!r}."
            )
        self.sampling_strategy = sampling_strategy
        # The repository base Hydra task contains `dataset_dirs`; OmegaConf
        # merges rather than deletes that key when this composite target is
        # selected.  The explicit source lists above are authoritative.
        dataset_kwargs.pop("dataset_dirs", None)
        self.demo_dataset = SO101RobotVideoDataset(
            dataset_dirs=demo_dataset_dirs,
            processor=copy.deepcopy(processor),
            **dataset_kwargs,
        )
        # Either correction source may be used alone; both join the same HUMAN
        # source, so the demo/correction sampler ratio counts them together.
        self.correction_parts = {}
        if correction_dataset_dirs:
            self.correction_parts["hil_human"] = SO101HumanCorrectionRobotVideoDataset(
                dataset_dirs=correction_dataset_dirs,
                processor=copy.deepcopy(processor),
                episode_selection=correction_episode_selection,
                video_supervision_require_full_window=video_supervision_require_full_window,
                video_supervision_min_segment_frames=video_supervision_min_segment_frames,
                history_crosses_segment=correction_history_crosses_segment,
                history_frame_stride=correction_history_frame_stride,
                history_skip_system_hold=correction_history_skip_system_hold,
                exclude_static_starts=correction_exclude_static_starts,
                static_range_deg=correction_static_range_deg,
                future_extension=correction_future_extension,
                extension_success_only=correction_extension_success_only,
                **dataset_kwargs,
            )
        if correction_clip_dataset_dirs:
            self.correction_parts["correction_clips"] = SO101CorrectionClipDataset(
                dataset_dirs=correction_clip_dataset_dirs,
                processor=copy.deepcopy(processor),
                video_supervision_require_full_window=video_supervision_require_full_window,
                video_supervision_min_segment_frames=video_supervision_min_segment_frames,
                trim_leading_frames=correction_clip_trim_leading_frames,
                **dataset_kwargs,
            )
        if not self.correction_parts:
            raise ValueError("At least one correction source is required.")
        self.correction_dataset = (
            next(iter(self.correction_parts.values())) if len(self.correction_parts) == 1
            else ConcatCorrectionDataset(list(self.correction_parts.values()))
        )
        self.demo_length = len(self.demo_dataset)
        self.correction_length = len(self.correction_dataset)
        if self.demo_length == 0 or self.correction_length == 0:
            raise ValueError(
                f"Both sources must be non-empty, got demo={self.demo_length}, "
                f"correction={self.correction_length}."
            )

    def __len__(self) -> int:
        return self.demo_length + self.correction_length

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)
        if idx < self.demo_length:
            sample = self.demo_dataset[idx]
            sample["video_supervision_mask"] = torch.tensor(True)
            sample["data_source"] = torch.tensor(0, dtype=torch.long)
            return sample
        return self.correction_dataset[idx - self.demo_length]

    def configure_video_history_memory(self, *, num_history_frames: int) -> None:
        self.demo_dataset.configure_video_history_memory(
            num_history_frames=num_history_frames
        )
        self.correction_dataset.configure_video_history_memory(
            num_history_frames=num_history_frames
        )

    def get_video_history_valid_len_for_all_indices(self) -> torch.Tensor:
        return torch.cat(
            [
                self.demo_dataset.get_video_history_valid_len_for_all_indices(),
                self.correction_dataset.get_video_history_valid_len_for_all_indices(),
            ]
        )

    def balanced_epoch_num_samples(self, *, batch_size: int, num_processes: int) -> int:
        if int(batch_size) != 4:
            raise ValueError("balanced_2_2 requires per-GPU batch_size=4.")
        global_source_samples = 2 * max(int(num_processes), 1)
        global_steps = max(
            1,
            (max(self.demo_length, self.correction_length) + global_source_samples - 1)
            // global_source_samples,
        )
        return global_steps * max(int(num_processes), 1) * int(batch_size)
