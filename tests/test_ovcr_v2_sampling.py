"""Coverage, distributed batch composition and resumability of v2 ablations."""
import unittest
from ahawam.utils.samplers import RatioTwoSourceEpochSampler


class Sources:
    demo_length = 72
    correction_length = 13
    def __init__(self, strategy): self.sampling_strategy = strategy
    def __len__(self): return self.demo_length + self.correction_length


class SamplingTests(unittest.TestCase):
    def test_distributed_ratio_coverage_and_seed(self):
        for strategy, demo_count in (("balanced_2_2", 2), ("balanced_3_1", 3), ("human_only", 0)):
            sampler = RatioTwoSourceEpochSampler(Sources(strategy), 42, 4, 3)
            indices = list(sampler)
            self.assertEqual(indices, list(sampler))
            self.assertEqual(len(indices) % 12, 0)
            for rank in range(3):
                for start in range(rank*4, len(indices), 12):
                    self.assertEqual(sum(i < 72 for i in indices[start:start+4]), demo_count)
            if demo_count:
                self.assertEqual(set(i for i in indices if i < 72), set(range(72)))
            self.assertEqual(set(i for i in indices if i >= 72), set(range(72,85)))
            # Within-batch shuffling can move a cycle boundary, so test the raw
            # source draw separately below. Resuming skips complete global batches.
            sampler.set_resume_batch_offset(1)
            self.assertEqual(list(sampler), indices[12:])
            sampler.clear_resume_batch_offset()
            sampler.set_epoch(1)
            self.assertNotEqual(indices, list(sampler))

    def test_no_replacement_within_each_source_cycle(self):
        import torch
        from ahawam.utils.samplers import BalancedTwoSourceEpochSampler
        stream = BalancedTwoSourceEpochSampler._draw_shuffled(length=13, count=40, generator=torch.Generator().manual_seed(7), index_offset=72)
        for start in (0,13,26):
            self.assertEqual(set(stream[start:start+13]), set(range(72,85)))

    def test_invalid_batch_rejected(self):
        with self.assertRaises(ValueError):
            RatioTwoSourceEpochSampler(Sources("balanced_3_1"), 42, 2, 3)


class GlobalRatioTests(unittest.TestCase):
    def test_two_to_one_per_global_step_rotates_the_extra_correction(self):
        from ahawam.utils.samplers import GlobalRatioTwoSourceEpochSampler
        sources = Sources("balanced_2_1")              # 72 demo, 13 correction
        sampler = GlobalRatioTwoSourceEpochSampler(sources, 42, 4, 3)
        indices = list(sampler)
        self.assertEqual(indices, list(sampler))       # deterministic for a fixed seed/epoch
        self.assertEqual(len(indices), len(sampler))
        extra_rank = []
        for step in range(len(indices) // 12):
            chunk = indices[12 * step:12 * (step + 1)]
            self.assertEqual(sum(i >= sources.demo_length for i in chunk), 4)   # 8 demo : 4 correction
            per_rank = [sum(i >= sources.demo_length for i in chunk[4 * r:4 * (r + 1)]) for r in range(3)]
            self.assertEqual(sorted(per_rank), [1, 1, 2])
            extra_rank.append(per_rank.index(2))
        self.assertEqual(set(extra_rank[:3]), {0, 1, 2})
        demo_draws = [i for i in indices if i < sources.demo_length][:72]      # 9 steps x 8
        self.assertEqual(sorted(demo_draws), list(range(72)))                   # no replacement within a cycle


if __name__ == '__main__': unittest.main()
