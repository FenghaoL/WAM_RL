"""The combined HUMAN correction source and whole-episode clip length lookup."""
import unittest
import torch
from scripts.so101_pipeline.hil_human_only_dataset import (
    ConcatCorrectionDataset, CorrectionClipBaseLerobotDataset)


class Part:
    def __init__(self, tag, n):
        self.tag, self.n, self.history = tag, n, None
    def __len__(self):
        return self.n
    def __getitem__(self, i):
        if not 0 <= i < self.n:
            raise IndexError(i)
        return (self.tag, i)
    def configure_video_history_memory(self, *, num_history_frames):
        self.history = num_history_frames
    def get_video_history_valid_len_for_all_indices(self):
        return torch.full((self.n,), self.history)


class CorrectionUnionTests(unittest.TestCase):
    def test_indexing_bounds_and_history_forwarding(self):
        hil, clips = Part("hil", 3), Part("clip", 2)
        union = ConcatCorrectionDataset([hil, clips])
        self.assertEqual(len(union), 5)
        self.assertEqual([union[i] for i in range(5)],
                         [("hil", 0), ("hil", 1), ("hil", 2), ("clip", 0), ("clip", 1)])
        for bad in (-1, 5):
            with self.assertRaises(IndexError):
                union[bad]
        union.configure_video_history_memory(num_history_frames=0)
        self.assertEqual((hil.history, clips.history), (0, 0))
        self.assertEqual(union.get_video_history_valid_len_for_all_indices().tolist(), [0] * 5)

    def test_logical_view_without_trim_is_the_identity(self):
        storage, lengths, starts, ends, dropped = CorrectionClipBaseLerobotDataset.logical_view(
            torch.tensor([0, 3, 5]), torch.tensor([3, 5, 9]), 0)
        self.assertEqual(storage, list(range(9)))
        self.assertEqual(lengths, [3, 3, 3, 2, 2, 4, 4, 4, 4])
        self.assertEqual((starts, ends, dropped), ([0, 3, 5], [3, 5, 9], 0))

    def test_logical_view_trims_heads_and_drops_short_episodes(self):
        storage, lengths, starts, ends, dropped = CorrectionClipBaseLerobotDataset.logical_view(
            torch.tensor([0, 3, 5]), torch.tensor([3, 5, 9]), 2)
        # episode 0 keeps frame 2, episode 1 (2 frames) disappears, episode 2 keeps 7..8
        self.assertEqual(storage, [2, 7, 8])
        self.assertEqual(lengths, [1, 2, 2])
        self.assertEqual((starts, ends, dropped), ([0, 1], [1, 3], 1))

    def test_logical_view_rejects_negative_trim(self):
        with self.assertRaises(ValueError):
            CorrectionClipBaseLerobotDataset.logical_view(torch.tensor([0]), torch.tensor([3]), -1)



class WindowRuleTests(unittest.TestCase):
    def test_static_start_mask_follows_the_supervised_timeline(self):
        import numpy as np
        from scripts.so101_pipeline.hil_human_only_dataset import _FutureWindowRemapper, static_start_mask
        actions = np.zeros((10, 6))
        actions[1:, 0] = 3.0                   # span 0..4: moves between frames 0 and 1, then holds
        actions[6, 5] = 8.0                     # after the span a later executing frame closes the gripper
        span_only = static_start_mask(actions, 0, 4, range_deg=1.0, window=48)
        self.assertEqual(span_only.astype(int).tolist(), [0, 1, 1, 1])      # from frame 1 on: nothing moves
        remapper = _FutureWindowRemapper(
            segment_end=np.array([4, 4, 4, 4, -1, -1, -1, -1, -1, -1]), rollout_start=np.zeros(10, np.int64),
            executing_by_rollout={0: np.array([0, 1, 2, 3, 6, 7])}, extendable_rollouts={0}, extend_actions=True)
        extended = static_start_mask(actions, 0, 4, range_deg=1.0, window=48, remapper=remapper)
        self.assertEqual(extended.astype(int).tolist(), [0, 0, 0, 0])       # the closure makes every window move
        video_only = _FutureWindowRemapper(
            segment_end=remapper.segment_end, rollout_start=remapper.rollout_start,
            executing_by_rollout=remapper.executing_by_rollout, extendable_rollouts={0}, extend_actions=False)
        self.assertEqual(static_start_mask(actions, 0, 4, range_deg=1.0, window=48, remapper=video_only)
                         .astype(int).tolist(), [0, 1, 1, 1])               # variant A: labels stay on the span

    def test_remapper_extends_images_over_executing_frames_only(self):
        import numpy as np
        from scripts.so101_pipeline.hil_human_only_dataset import _FutureWindowRemapper
        common = dict(segment_end=np.array([3, 3, 3, -1, -1, -1, -1]), rollout_start=np.zeros(7, np.int64),
                      executing_by_rollout={0: np.array([0, 1, 2, 4, 6])})
        deltas = {"observation.images.fixed": [0, 1, 2, 3], "action": [0, 1, 2, 3]}
        video_only = _FutureWindowRemapper(**common, extendable_rollouts={0}, extend_actions=False)
        q, pad = video_only(1, deltas)
        self.assertEqual(q["observation.images.fixed"], [1, 2, 4, 6])       # holds 3 and 5 skipped
        self.assertEqual(pad["observation.images.fixed_is_pad"].tolist(), [False] * 4)
        self.assertEqual(q["action"], [1, 2, 2, 2])                          # actions stop at the span end
        self.assertEqual(pad["action_is_pad"].tolist(), [False, False, True, True])
        both = _FutureWindowRemapper(**common, extendable_rollouts={0}, extend_actions=True)
        self.assertEqual(both(1, deltas)[0]["action"], [1, 2, 4, 6])
        failed = _FutureWindowRemapper(**common, extendable_rollouts=set(), extend_actions=True)
        q, pad = failed(1, deltas)
        self.assertEqual(q["observation.images.fixed"], [1, 2, 2, 2])       # failed rollout: no extension
        self.assertEqual(pad["observation.images.fixed_is_pad"].tolist(), [False, False, True, True])
        self.assertIsNone(failed(4, deltas))                                # non-HUMAN frame: default LeRobot


if __name__ == "__main__":
    unittest.main()
