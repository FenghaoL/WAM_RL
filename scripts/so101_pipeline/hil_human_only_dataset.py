"""Human-only logical views for SO101 AHA-WAM HIL fine-tuning.

The raw HIL recordings intentionally retain POLICY and SYSTEM_HOLD rows for
auditability.  This module never exposes those rows as training samples.  Each
contiguous HUMAN run is treated as an independent logical episode; future
queries are clamped at that run's boundary and marked as padding before image
augmentation or action/state normalization.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch

from ahawam.datasets.lerobot.base_lerobot_dataset import BaseLerobotDataset
from scripts.so101_pipeline.so101_dataset import SO101RobotVideoDataset


HUMAN_CONTROL_SOURCE = 1


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

    def __init__(self, *args, video_supervision_min_segment_frames: int = 49, **kwargs):
        self.video_supervision_min_segment_frames = int(
            video_supervision_min_segment_frames
        )
        if self.video_supervision_min_segment_frames <= 0:
            raise ValueError("video_supervision_min_segment_frames must be positive")
        super().__init__(*args, **kwargs)

        segments: list[HumanSegment] = []
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
            for local_start, local_end in _human_segments_from_columns(columns):
                segments.append(
                    HumanSegment(
                        storage_start=dataset_storage_offset + local_start,
                        storage_end=dataset_storage_offset + local_end,
                    )
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
        for segment_id, segment in enumerate(segments):
            logical_starts.append(logical_cursor)
            storage_indices.extend(range(segment.storage_start, segment.storage_end))
            segment_ids.extend([segment_id] * segment.length)
            logical_cursor += segment.length
            logical_ends.append(logical_cursor)
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
    def __init__(self, *args, video_supervision_min_segment_frames: int = 49, **kwargs):
        self.video_supervision_min_segment_frames = int(
            video_supervision_min_segment_frames
        )
        super().__init__(*args, **kwargs)

    def _build_lerobot_dataset(self, **kwargs) -> BaseLerobotDataset:
        return HumanCorrectionBaseLerobotDataset(
            **kwargs,
            video_supervision_min_segment_frames=(
                self.video_supervision_min_segment_frames
            ),
        )

    def _resolve_lerobot_dataset_for_global_index(self, sample_idx: int):
        storage_idx = self.lerobot_dataset.storage_index_for_logical_index(sample_idx)
        return super()._resolve_lerobot_dataset_for_global_index(storage_idx)


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
        correction_dataset_dirs,
        sampling_strategy: str = "natural",
        video_supervision_min_segment_frames: int = 49,
        processor=None,
        **dataset_kwargs,
    ):
        if sampling_strategy not in {"natural", "balanced_2_2"}:
            raise ValueError(
                "sampling_strategy must be 'natural' or 'balanced_2_2', got "
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
        self.correction_dataset = SO101HumanCorrectionRobotVideoDataset(
            dataset_dirs=correction_dataset_dirs,
            processor=copy.deepcopy(processor),
            video_supervision_min_segment_frames=video_supervision_min_segment_frames,
            **dataset_kwargs,
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
