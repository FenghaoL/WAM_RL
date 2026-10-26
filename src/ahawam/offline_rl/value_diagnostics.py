"""Evaluation-only action windows; never add rewards or modify training splits."""
from __future__ import annotations

import numpy as np

from .data import effective_success_frame


def diagnostic_windows(table, chunk_size=8):
    """Natural active chunks at all raw frames, including excluded/post-goal probes.

    Q is undefined here for HOLD, incomplete tails, missing ticks or dropped rows.
    Every included command needs its immediately following recorded observation.
    """
    source = table.control_source.to_numpy()
    active = np.flatnonzero(source[:-1] != 2)
    ticks = table.tick_index.to_numpy()
    dropped = table.dropped_tick.to_numpy(dtype=bool)
    windows = {}
    reasons = ["hold" if s == 2 else "incomplete_chunk" for s in source]
    for pos, start in enumerate(active):
        indices = active[pos:pos + chunk_size]
        if len(indices) != chunk_size:
            continue
        end = int(indices[-1]) + 1
        if np.any(np.diff(ticks[start:end+1]) != 1) or dropped[start:end+1].any():
            reasons[start] = "tick_gap_or_drop"
            continue
        windows[int(start)] = indices.tolist()
        reasons[start] = "valid_chunk"
    return windows, reasons


def recorded_returns(table, episode, windows, gamma=.997):
    """Discounted recorded-behavior return, NOT a ground-truth learned-policy Q.

    Unobserved success (early stops), post-terminal observations and unobserved
    commands across gaps have no known positive return reference.
    """
    values = np.full(len(table), np.nan)
    if episode["status"] != "ready":
        return values
    success = effective_success_frame(table, episode["first_success"])
    if success is None:
        for start in windows:
            values[start] = 0.
        return values
    source = table.control_source.to_numpy()
    ticks = table.tick_index.to_numpy()
    dropped = table.dropped_tick.to_numpy(dtype=bool)
    for start in windows:
        if start >= success or source[success-1] == 2:
            continue
        if np.any(np.diff(ticks[start:success+1]) != 1) or dropped[start:success+1].any():
            continue
        remaining = int((source[start:success] != 2).sum())
        if remaining:
            values[start] = gamma ** (remaining-1)
    return values


def episode_group(episode, validation_uids):
    if episode["status"] != "ready":
        partition = "val_source" if episode["root"].endswith("_val") else "train_source"
        return f"excluded_early_stop/{partition}", "success_early_stop_no_success_frame"
    split = "val" if episode["episode_uid"] in validation_uids else "train"
    return f"{split}/{episode['outcome']}", episode["outcome"]
