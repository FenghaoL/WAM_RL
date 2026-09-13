from typing import Iterator, Sized

import torch
from torch.utils.data import Sampler


class ResumableEpochSampler(Sampler[int]):
    def __init__(self, dataset: Sized, seed: int, batch_size: int, num_processes: int):
        self.dataset = dataset
        self.seed = int(seed)
        self.batch_size = int(batch_size)
        self.num_processes = int(num_processes)
        self.epoch = 0
        self.epoch_offset = 0
        self.resume_batch_offset = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def set_epoch_offset(self, epoch_offset: int):
        self.epoch_offset = int(epoch_offset)

    def set_resume_batch_offset(self, batch_in_epoch: int):
        self.resume_batch_offset = int(batch_in_epoch)

    def clear_resume_batch_offset(self):
        self.resume_batch_offset = 0

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = self.resume_batch_offset * self.batch_size * self.num_processes
            indices = indices[sample_offset:]
        return iter(indices)

    def __len__(self) -> int:
        return len(self.dataset)


class HistoryAwareResumableEpochSampler(ResumableEpochSampler):
    """Shuffle history-length-homogeneous batches across the whole epoch."""

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
    ):
        super().__init__(
            dataset=dataset,
            seed=seed,
            batch_size=batch_size,
            num_processes=num_processes,
        )
        get_valid_lens = getattr(
            dataset, "get_video_history_valid_len_for_all_indices", None
        )
        if get_valid_lens is None:
            raise TypeError(
                "`HistoryAwareResumableEpochSampler` requires dataset method "
                "`get_video_history_valid_len_for_all_indices()`."
            )
        valid_lens = get_valid_lens()
        if not isinstance(valid_lens, torch.Tensor):
            valid_lens = torch.as_tensor(valid_lens)
        if valid_lens.ndim != 1 or int(valid_lens.shape[0]) != len(dataset):
            raise ValueError(
                "`get_video_history_valid_len_for_all_indices()` must return "
                f"[len(dataset)], got shape {tuple(valid_lens.shape)}."
            )
        self.valid_history_lens = valid_lens.to(device="cpu", dtype=torch.long)

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        perm = torch.randperm(len(self.dataset), generator=g)
        valid_lens = self.valid_history_lens
        max_len = int(valid_lens.max().item()) if valid_lens.numel() else 0
        chunk_size = max(self.batch_size * self.num_processes, 1)

        chunks: list[list[int]] = []
        for bucket_id in range(max_len + 1):
            bucket_indices = perm[valid_lens[perm] == int(bucket_id)]
            for start in range(0, int(bucket_indices.numel()), chunk_size):
                chunk = bucket_indices[start : start + chunk_size].tolist()
                if chunk:
                    chunks.append(chunk)

        if chunks:
            chunk_order = torch.randperm(len(chunks), generator=g).tolist()
            ordered = [
                sample_idx
                for chunk_idx in chunk_order
                for sample_idx in chunks[int(chunk_idx)]
            ]
        else:
            ordered = []

        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = (
                self.resume_batch_offset * self.batch_size * self.num_processes
            )
            ordered = ordered[sample_offset:]
        return iter(ordered)


class BalancedTwoSourceEpochSampler(ResumableEpochSampler):
    """Build every local batch from two demo and two correction samples.

    The longer source is covered once per epoch (up to global-batch rounding);
    the shorter source cycles through independently shuffled permutations.
    Keeping the 2:2 grouping in sampler order also preserves the ratio after
    Accelerate shards whole DataLoader batches across ranks.
    """

    def __init__(
        self,
        dataset: Sized,
        seed: int,
        batch_size: int,
        num_processes: int,
    ):
        super().__init__(dataset, seed, batch_size, num_processes)
        if self.batch_size != 4:
            raise ValueError(
                "BalancedTwoSourceEpochSampler requires batch_size=4 for 2:2 batches."
            )
        for name in ("demo_length", "correction_length", "balanced_epoch_num_samples"):
            if not hasattr(dataset, name):
                raise TypeError(f"Balanced two-source dataset is missing `{name}`.")
        self.demo_length = int(dataset.demo_length)
        self.correction_length = int(dataset.correction_length)
        self.epoch_num_samples = int(
            dataset.balanced_epoch_num_samples(
                batch_size=self.batch_size,
                num_processes=self.num_processes,
            )
        )
        if self.epoch_num_samples % self.batch_size != 0:
            raise ValueError("Balanced epoch size must be divisible by batch_size.")

    @staticmethod
    def _draw_shuffled(
        *, length: int, count: int, generator: torch.Generator, index_offset: int
    ) -> list[int]:
        out: list[int] = []
        while len(out) < count:
            take = min(length, count - len(out))
            out.extend(
                int(index_offset + idx)
                for idx in torch.randperm(length, generator=generator)[:take].tolist()
            )
        return out

    def __iter__(self) -> Iterator[int]:
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed + self.epoch + self.epoch_offset)
        num_batches = self.epoch_num_samples // self.batch_size
        per_source = num_batches * 2
        demos = self._draw_shuffled(
            length=self.demo_length,
            count=per_source,
            generator=g,
            index_offset=0,
        )
        corrections = self._draw_shuffled(
            length=self.correction_length,
            count=per_source,
            generator=g,
            index_offset=self.demo_length,
        )
        ordered: list[int] = []
        for batch_idx in range(num_batches):
            batch = [
                demos[2 * batch_idx],
                demos[2 * batch_idx + 1],
                corrections[2 * batch_idx],
                corrections[2 * batch_idx + 1],
            ]
            within_batch = torch.randperm(4, generator=g).tolist()
            ordered.extend(batch[int(pos)] for pos in within_batch)
        if self.epoch == 0 and self.resume_batch_offset > 0:
            sample_offset = (
                self.resume_batch_offset * self.batch_size * self.num_processes
            )
            ordered = ordered[sample_offset:]
        return iter(ordered)

    def __len__(self) -> int:
        return self.epoch_num_samples


class RatioTwoSourceEpochSampler(ResumableEpochSampler):
    """Exact source ratio per local batch, shuffled without replacement per cycle.

    This builds a global index stream; Accelerate shards whole batches. Every
    active source is fully covered before its shuffled stream cycles. Rounding
    to a global batch may repeat a small tail. No weighted replacement sampling.
    """
    def __init__(self, dataset, seed, batch_size, num_processes):
        super().__init__(dataset, seed, batch_size, num_processes)
        ratios = {"balanced_2_2": (1, 1), "balanced_3_1": (3, 1), "human_only": (0, 1)}
        demo, human = ratios[dataset.sampling_strategy]
        denominator = demo + human
        if batch_size % denominator:
            raise ValueError(f"batch_size={batch_size} must be divisible by ratio denominator {denominator}")
        self.counts = (batch_size * demo // denominator, batch_size * human // denominator)
        self.lengths = (int(dataset.demo_length), int(dataset.correction_length))
        if any(length <= 0 for length, count in zip(self.lengths, self.counts) if count):
            raise ValueError("Active source must not be empty")
        global_steps = max((length + count * num_processes - 1) // (count * num_processes)
                           for length, count in zip(self.lengths, self.counts) if count)
        self.num_batches = global_steps * num_processes

    def __len__(self):
        return self.num_batches * self.batch_size

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch + self.epoch_offset)
        streams = [BalancedTwoSourceEpochSampler._draw_shuffled(
            length=length, count=count * self.num_batches, generator=g, index_offset=offset)
            for length, count, offset in zip(self.lengths, self.counts, (0, self.lengths[0]))]
        indices = []
        for i in range(self.num_batches):
            batch = streams[0][i*self.counts[0]:(i+1)*self.counts[0]] + streams[1][i*self.counts[1]:(i+1)*self.counts[1]]
            indices.extend(batch[j] for j in torch.randperm(self.batch_size, generator=g).tolist())
        if self.epoch == 0 and self.resume_batch_offset:
            indices = indices[self.resume_batch_offset * self.batch_size * self.num_processes:]
        return iter(indices)


class GlobalRatioTwoSourceEpochSampler(ResumableEpochSampler):
    """Exact demo:correction ratio per *global* batch, e.g. 2:1 with 4 samples per rank.

    A 2:1 ratio needs 8 demo + 4 correction samples in a 12-sample global batch,
    which no 4-sample local batch can express. Each step therefore gives one rank
    one more correction sample than the others -- rotating which rank -- so every
    global batch has the exact ratio and every local batch keeps the same size
    (gradients stay uniformly weighted across ranks). Both sources are shuffled
    permutations drawn without replacement, cycling when exhausted. Accelerate
    hands whole local batches to ranks in order, as for the other samplers.
    """

    RATIOS = {"balanced_2_1": (2, 1)}

    def __init__(self, dataset, seed, batch_size, num_processes):
        super().__init__(dataset, seed, batch_size, num_processes)
        demo, correction = self.RATIOS[dataset.sampling_strategy]
        global_batch = self.batch_size * self.num_processes
        if global_batch % (demo + correction):
            raise ValueError(
                f"global batch {global_batch} must be divisible by the ratio total {demo + correction}")
        self.correction_per_step = global_batch * correction // (demo + correction)
        self.demo_per_step = global_batch - self.correction_per_step
        self.lengths = (int(dataset.demo_length), int(dataset.correction_length))
        if min(self.lengths) <= 0:
            raise ValueError("Both sources must be non-empty")
        self.num_steps = max(-(-self.lengths[0] // self.demo_per_step),
                             -(-self.lengths[1] // self.correction_per_step))

    def __len__(self):
        return self.num_steps * self.batch_size * self.num_processes

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch + self.epoch_offset)
        demos = BalancedTwoSourceEpochSampler._draw_shuffled(
            length=self.lengths[0], count=self.demo_per_step * self.num_steps, generator=g, index_offset=0)
        corrections = BalancedTwoSourceEpochSampler._draw_shuffled(
            length=self.lengths[1], count=self.correction_per_step * self.num_steps, generator=g,
            index_offset=self.lengths[0])
        base, extra = divmod(self.correction_per_step, self.num_processes)
        indices, d, c = [], 0, 0
        for step in range(self.num_steps):
            for rank in range(self.num_processes):
                n_correction = base + (1 if (rank - step) % self.num_processes < extra else 0)
                n_demo = self.batch_size - n_correction
                batch = demos[d:d + n_demo] + corrections[c:c + n_correction]
                d += n_demo
                c += n_correction
                indices.extend(batch[j] for j in torch.randperm(self.batch_size, generator=g).tolist())
        if self.epoch == 0 and self.resume_batch_offset:
            indices = indices[self.resume_batch_offset * self.batch_size * self.num_processes:]
        return iter(indices)
