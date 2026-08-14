"""ZeRO-3 sampler that keeps every distributed micro-batch on one history branch."""

from __future__ import annotations

from typing import Iterator

import torch

from ahawam.utils.samplers import HistoryAwareResumableEpochSampler


class ZeRO3HistoryAlignedSampler(HistoryAwareResumableEpochSampler):
    """Pad each history bucket to complete distributed micro-batches.

    Accelerate shards consecutive ``batch_size * num_processes`` indices across
    ranks. An undersized bucket tail must therefore not touch the next bucket:
    that would make ranks execute different history-dependent MoT paths. The
    tail is padded by repeating indices from the same bucket. At most one
    distributed micro-batch minus one sample is repeated per history length.
    """

    priority_history_len: int | None = None

    @property
    def distributed_micro_batch_size(self) -> int:
        return max(self.batch_size * self.num_processes, 1)

    def _padded_sample_count(self) -> int:
        chunk_size = self.distributed_micro_batch_size
        if self.valid_history_lens.numel() == 0:
            return 0
        counts = torch.bincount(self.valid_history_lens)
        padded_counts = torch.div(
            counts + chunk_size - 1,
            chunk_size,
            rounding_mode="floor",
        ) * chunk_size
        return int(padded_counts.sum().item())

    def __len__(self) -> int:
        sample_count = self._padded_sample_count()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            skipped = (
                self.resume_batch_offset
                * self.batch_size
                * self.num_processes
            )
            sample_count = max(sample_count - skipped, 0)
        return sample_count

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch + self.epoch_offset)
        permutation = torch.randperm(len(self.dataset), generator=generator)
        valid_lens = self.valid_history_lens
        max_len = int(valid_lens.max().item()) if valid_lens.numel() else 0
        chunk_size = self.distributed_micro_batch_size

        chunks: list[tuple[int, list[int]]] = []
        for history_len in range(max_len + 1):
            bucket = permutation[valid_lens[permutation] == history_len].tolist()
            if not bucket:
                continue
            for start in range(0, len(bucket), chunk_size):
                chunk = bucket[start : start + chunk_size]
                if len(chunk) < chunk_size:
                    missing = chunk_size - len(chunk)
                    repeats = (bucket * ((missing + len(bucket) - 1) // len(bucket)))[
                        :missing
                    ]
                    chunk.extend(repeats)
                chunks.append((history_len, chunk))

        if chunks:
            chunk_order = torch.randperm(len(chunks), generator=generator).tolist()
            if self.priority_history_len is not None:
                priority = int(self.priority_history_len)
                chunk_order.sort(
                    key=lambda chunk_idx: chunks[int(chunk_idx)][0] != priority
                )
            ordered = [
                sample_idx
                for chunk_idx in chunk_order
                for sample_idx in chunks[int(chunk_idx)][1]
            ]
        else:
            ordered = []

        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = (
                self.resume_batch_offset
                * self.batch_size
                * self.num_processes
            )
            ordered = ordered[sample_offset:]
        return iter(ordered)
