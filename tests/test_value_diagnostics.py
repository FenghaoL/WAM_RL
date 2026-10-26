"""Keep diagnostic probes distinct from training transitions and actual outcomes."""
import unittest

import numpy as np
import pandas as pd

from ahawam.offline_rl.value_diagnostics import diagnostic_windows, episode_group, recorded_returns


def rows(sources, gap=None):
    n = len(sources)
    ticks = np.arange(n)
    if gap is not None:
        ticks[gap:] += 1
    return pd.DataFrame(dict(control_source=sources, tick_index=ticks, dropped_tick=False))


class DiagnosticTests(unittest.TestCase):
    def test_hold_removed_endpoint_image_required_and_gap_invalidates(self):
        t = rows([0, 1, 2, 2, 0, 0, 0])
        windows, why = diagnostic_windows(t, 3)
        self.assertEqual(windows[0], [0, 1, 4])
        self.assertEqual(why[2], "hold")
        self.assertNotIn(5, windows)
        broken, why = diagnostic_windows(rows([0, 1, 2, 2, 0, 0, 0], gap=3), 3)
        self.assertNotIn(0, broken)
        self.assertEqual(why[0], "tick_gap_or_drop")

    def test_missing_success_never_becomes_zero_return_or_failure(self):
        t = rows([0] * 8)
        windows, _ = diagnostic_windows(t, 3)
        episode = dict(status="pending_success_annotation", first_success=None, root="x_val")
        self.assertTrue(np.isnan(recorded_returns(t, episode, windows)).all())
        self.assertEqual(episode_group(episode, set()), ("excluded_early_stop/val_source", "success_early_stop_no_success_frame"))

    def test_discount_matches_incoming_success_command_and_post_goal_na(self):
        t = rows([0, 0, 2, 2, 0, 0, 0, 0])
        windows, _ = diagnostic_windows(t, 3)
        episode = dict(status="ready", first_success=3)
        values = recorded_returns(t, episode, windows, gamma=.9)
        self.assertAlmostEqual(values[0], .9)  # effective success image 2; command 1 rewarded
        self.assertEqual(values[1], 1.)
        self.assertTrue(np.isnan(values[4]))


if __name__ == "__main__":
    unittest.main()
