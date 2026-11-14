"""Episode storage order is the task clock; collector ticks include pauses.

The cylinder HIL collector explicitly omits pause frames. A jump in tick_index
does not imply an episode boundary or missing task actions. strict_ticks is
retained only to reproduce the original offline experiments and cache.
"""
from __future__ import annotations

import numpy as np


TIMELINE_MODES = ("episode_frames", "strict_ticks")


def validate_timeline_mode(mode):
    if mode not in TIMELINE_MODES:
        raise ValueError(f"Unknown timeline_mode: {mode}")


def history_start(table, frame, mode="episode_frames"):
    validate_timeline_mode(mode)
    if not 0 <= frame < len(table):
        raise IndexError(frame)
    if mode == "episode_frames":
        return 0
    gaps = np.flatnonzero(np.diff(table.tick_index.to_numpy()[:frame + 1]) != 1)
    return int(gaps[-1]) + 1 if len(gaps) else 0


def span_is_continuous(table, start, end, mode="episode_frames"):
    """Inclusive observation span within one already-validated episode table.

    dropped_tick on a saved row alone is not evidence a row is absent.
    Both it and tick jumps remain audit metadata under episode_frames.
    """
    validate_timeline_mode(mode)
    if not 0 <= start <= end < len(table):
        return False
    if mode == "episode_frames":
        return True
    ticks = table.tick_index.to_numpy()[start:end + 1]
    return bool((np.diff(ticks) == 1).all() and not table.dropped_tick.iloc[start:end + 1].any())
