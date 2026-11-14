"""Build executed-action transitions from SO101 frame annotations.

An observation at row k precedes action[k]. A success observation at k
therefore rewards transition k-1 -> k. No action is inferred after the last
observation. Unobserved successes require an explicit censored-data recipe.
Collector pauses do not
split episodes; strict tick continuity is an explicit legacy option.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .timeline import span_is_continuous, validate_timeline_mode


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@dataclass(frozen=True)
class TransitionConfig:
    chunk_size: int = 8
    stride: int = 2
    gamma_tick: float = 0.997
    # "mixed" learns from all executed behavior, including human recovery.
    # "censor" stops policy credit before entry into a human segment.
    intervention_mode: str = "mixed"
    temporal_mode: str = "tick"
    timeline_mode: str = "episode_frames"

    def __post_init__(self):
        validate_timeline_mode(self.timeline_mode)
        if self.chunk_size < 1 or self.stride < 1 or not 0 < self.gamma_tick <= 1:
            raise ValueError("Invalid chunk_size, stride or gamma_tick")
        if self.intervention_mode not in {"mixed", "censor"}:
            raise ValueError("intervention_mode must be mixed or censor")
        if self.temporal_mode not in {"tick", "macro", "skip_hold"}:
            raise ValueError("temporal_mode must be tick, macro or skip_hold")


def inspect_episode(root: Path, row: dict, manifest: dict, info: dict):
    ep = int(row["episode_index"])
    fields = dict(episode_index=ep, episode_chunk=ep // int(info.get("chunks_size", 1000)))
    path = root / info["data_path"].format(**fields)
    table = pd.read_parquet(path)
    length = len(table)
    if length != row["length"] or table.frame_index.tolist() != list(range(length)):
        raise ValueError(f"Invalid frame indices/length: {path}")
    if set(table.episode_index) != {ep}:
        raise ValueError(f"Invalid episode indices: {path}")
    if np.any(np.diff(table.monotonic_ns.to_numpy()) <= 0) or np.any(np.diff(table.tick_index.to_numpy()) <= 0):
        raise ValueError(f"Non-increasing control timestamps/ticks: {path}")
    identity = dict(episode_index=ep, logical_episode_id=manifest.get("logical_episode_id"),
                    num_frames=length, parquet_path=path.relative_to(root).as_posix(),
                    parquet_sha256=sha256(path))
    annotation_path = root / f"recording_context/frame_annotations/episode_{ep:06d}.json"
    first_success = None
    annotation = None
    if annotation_path.exists():
        annotation = json.loads(annotation_path.read_text())
        if annotation.get("format") != "so101_frame_annotations/v1" or annotation.get("source") != identity:
            raise ValueError(f"Stale annotation identity: {annotation_path}")
        positives = []
        for interval in annotation["intervals"]:
            a, b = interval["start_frame"], interval["end_frame"]
            if type(a) is not int or type(b) is not int or not 0 <= a <= b < length:
                raise ValueError(f"Invalid interval: {annotation_path}")
            if interval["label"] == "success":
                positives.append(a)
        first_success = min(positives, default=None)
    outcome = str(manifest.get("outcome", "")).lower()
    if first_success is not None and outcome == "failure":
        raise ValueError(f"Failure manifest conflicts with success annotation: {path}")
    if first_success is None and outcome != "failure":
        status = "pending_success_annotation" if outcome == "success" else "unknown_outcome"
    else:
        status = "ready"
    for name in ("action", "executed_action_arm", "observation.state"):
        values = np.stack(table[name]).astype(np.float32)
        if values.ndim != 2 or not np.isfinite(values).all():
            raise ValueError(f"Non-finite or invalid {name}: {path}")
    if not np.allclose(np.stack(table.action), np.stack(table.executed_action_arm), atol=1e-5):
        raise ValueError(f"action differs from actual executed targets: {path}")
    if not set(table.control_source).issubset({0, 1, 2}):
        raise ValueError(f"Unknown control source: {path}")
    report = dict(episode_uid=f"{root.name}/{manifest.get('logical_episode_id') or ep}",
                  root=str(root), episode_index=ep, frames=length, status=status,
                  outcome=outcome, outcome_class=manifest.get("outcome_class"),
                  first_success=first_success,
                  success_source=None if first_success is None else int(table.control_source.iloc[first_success]),
                  sources={str(int(k)): int(v) for k, v in table.control_source.value_counts().items()},
                  tick_gaps=int((np.diff(table.tick_index.to_numpy()) != 1).sum()),
                  dropped_frames=int(table.dropped_tick.sum()),
                  annotation_sha256=sha256(annotation_path) if annotation else None,
                  success_reviewed=bool(annotation and "success" in annotation.get("reviewed_labels", [])),
                  source=identity)
    return table, report


def episode_transitions(table: pd.DataFrame, report: dict, cfg: TransitionConfig):
    """Tick-space chunks, including real SYSTEM_HOLD targets and elapsed ticks.

    Terminal suffixes repeat the final executed target only for tensor shape;
    action_valid records the actual prefix. Nonterminal short tails are dropped.
    Neither valid length nor future intervention labels enter the policy state.
    """
    if report["status"] != "ready":
        return []
    if cfg.temporal_mode == "skip_hold":
        return skip_hold_transitions(table, report, cfg)
    if cfg.temporal_mode == "macro":
        return macro_transitions(table, report, cfg)
    success = report["first_success"]
    last = len(table) - 1 if success is None else success
    actions = np.stack(table.executed_action_arm).astype(np.float32)
    states = np.stack(table["observation.state"]).astype(np.float32)
    source = table.control_source.to_numpy()
    rows = []
    for t in range(0, last, cfg.stride):
        end = min(t + cfg.chunk_size, last)
        boundary = None
        if cfg.intervention_mode == "censor" and source[t] != 1:
            entries = np.flatnonzero(source[t + 1:end + 1] == 1)
            if len(entries):
                end = t + 1 + int(entries[0])
                boundary = "intervention_censored"
        k = end - t
        if not span_is_continuous(table, t, end, cfg.timeline_mode):
            continue
        is_success = success is not None and end == success
        is_failure = success is None and end == last
        terminal = is_success or is_failure
        if terminal:
            boundary = None  # The terminal observation precedes any new controller's command.
        if k < cfg.chunk_size and not terminal and boundary is None:
            continue
        executed = actions[t:end]
        padded = np.concatenate([executed, np.repeat(executed[-1:], cfg.chunk_size - k, axis=0)])
        reward = cfg.gamma_tick ** (k - 1) if is_success else 0.0
        terminated = terminal
        truncated = boundary is not None
        rows.append(dict(
            episode_uid=report["episode_uid"], root=report["root"],
            episode_index=report["episode_index"], start_frame=t, next_frame=end,
            observation_key=f"{report['episode_uid']}:{t}",
            next_observation_key=f"{report['episode_uid']}:{end}",
            state=states[t], next_state=states[end], action=padded,
            action_valid=np.arange(cfg.chunk_size) < k,
            control_source=np.pad(source[t:end], (0, cfg.chunk_size-k), constant_values=255),
            reward=reward, discount=0.0 if terminal or truncated else cfg.gamma_tick ** k,
            duration_ticks=k, duration_seconds=(int(table.monotonic_ns.iloc[end]) - int(table.monotonic_ns.iloc[t])) / 1e9,
            terminated=terminated, truncated=truncated,
            end_reason=boundary or ("success" if is_success else "failure_end" if is_failure else "continuing"),
            assisted_episode=report.get("outcome_class") == "SUCCESS_WITH_INTERVENTION",
        ))
    return rows


def effective_success_frame(table, success):
    """Move success whose incoming command is HOLD to that hold's first image."""
    if success is None:
        return None
    source = table.control_source.to_numpy()
    # First HOLD image already belongs to the preceding active command.
    if success > 0 and source[success - 1] == 2:
        first = success - 1
        while first > 0 and source[first - 1] == 2:
            first -= 1
        return first
    return success


def skip_hold_transitions(table, report, cfg):
    """Full natural active-action windows, single-event terminal credit.

    Retain POLICY and HUMAN; omit only SYSTEM_HOLD commands. The endpoint
    is the *raw* observation immediately after the C-th command, including
    a first-HOLD image. Success changes return/continuation, never the
    natural command tail or actor labels. No fabricated terminal padding.
    Discount is measured in retained commands (not elapsed wall time).
    """
    success = effective_success_frame(table, report["first_success"])
    source = table.control_source.to_numpy()
    active = np.flatnonzero(source[:-1] != 2)  # each command needs a following image
    actions = np.stack(table.executed_action_arm).astype(np.float32)
    states = np.stack(table["observation.state"]).astype(np.float32)
    ticks = table.tick_index.to_numpy()
    out = []
    for pos in range(0, len(active), cfg.stride):
        t = int(active[pos])
        if success is not None and t >= success:
            continue
        indices = active[pos:pos + cfg.chunk_size]
        if len(indices) != cfg.chunk_size:
            continue  # no unknown/padded actor labels, even at a terminal
        end = int(indices[-1]) + 1
        if not span_is_continuous(table, t, end, cfg.timeline_mode):
            continue
        if cfg.intervention_mode == "censor" and source[t] != 1 and (source[indices] == 1).any():
            continue
        hit = success is not None and t < success <= end
        # effective success is an observation, so action[success-1] earns +1.
        reward_index = int(np.searchsorted(indices, success) - 1) if hit else None
        if hit and int(indices[reward_index]) != success - 1:
            continue  # no incoming active action observed (e.g. recording begins in HOLD)
        reaches_recording_end = pos + cfg.chunk_size == len(active)
        censored = bool(report.get("censored_recording", False))
        failure_end = success is None and reaches_recording_end and not censored
        terminal = hit or failure_end
        out.append(dict(episode_uid=report["episode_uid"], root=report["root"],
            episode_index=report["episode_index"], start_frame=t, next_frame=end,
            observation_key=f"{report['episode_uid']}:{t}", next_observation_key=f"{report['episode_uid']}:{end}",
            action_frames=indices.tolist(), annotated_success_frame=report["first_success"],
            effective_success_frame=success, reward_action_index=reward_index,
            state=states[t], next_state=states[end], action=actions[indices],
            action_valid=np.ones(cfg.chunk_size, dtype=bool), control_source=source[indices],
            reward=cfg.gamma_tick ** reward_index if hit else 0.0,
            discount=0.0 if terminal else cfg.gamma_tick ** cfg.chunk_size,
            duration_ticks=int(ticks[end]-ticks[t]),
            duration_seconds=(int(table.monotonic_ns.iloc[end])-int(table.monotonic_ns.iloc[t]))/1e9,
            terminated=terminal, truncated=bool(censored and reaches_recording_end),
            end_reason="success" if hit else "failure_end" if failure_end else
                "censored_recording_end" if censored and reaches_recording_end else "continuing",
            assisted_episode=report.get("outcome_class") == "SUCCESS_WITH_INTERVENTION"))
    return out


def macro_transitions(table: pd.DataFrame, report: dict, cfg: TransitionConfig):
    """Actual command blocks followed by logged holds as environment latency.

    POLICY blocks must begin at slot 0 and share a logged chunk ID. HUMAN
    blocks use at most C contiguous commands. Short nonterminal blocks are
    excluded rather than inventing their unexecuted tail. Holds must repeat
    the last target exactly; unmodelled control changes are rejected.
    """
    from dataclasses import replace
    success = report["first_success"]
    last = len(table)-1 if success is None else success
    actions = np.stack(table.executed_action_arm).astype(np.float32)
    source = table.control_source.to_numpy()
    output = []
    t = 0
    while t < last:
        if source[t] == 2:
            t += 1
            continue
        active_end = t+1
        while active_end < last and active_end-t < cfg.chunk_size:
            if source[active_end] != source[t]:
                break
            if source[t] == 0 and table.policy_chunk_id.iloc[active_end] != table.policy_chunk_id.iloc[t]:
                break
            active_end += 1
        end = active_end
        while end < last and source[end] == 2:
            end += 1
        c = active_end-t
        valid = span_is_continuous(table, t, end, cfg.timeline_mode)
        if source[t] == 0:
            valid &= (int(table.action_index_in_chunk.iloc[t]) == 0 and
                      table.action_index_in_chunk.iloc[t:active_end].tolist() == list(range(c)))
        if end > active_end:
            valid &= bool(np.allclose(actions[active_end:end], actions[active_end-1], atol=1e-5))
        if valid and (c == cfg.chunk_size or end == last):
            # Reuse checked reward/terminal construction for the elapsed span,
            # then replace its action with only the commanded prefix.
            sub = table.iloc[t:end+1].copy().reset_index(drop=True)
            sub_report = dict(report, first_success=(end-t) if success == end else None)
            # Avoid falsely terminating a continuing macro at this local end.
            record = episode_transitions(sub, sub_report,
                replace(cfg, temporal_mode="tick", chunk_size=end-t, stride=end-t,
                        intervention_mode="mixed"))[0]
            terminal = end == last
            censored = not terminal and cfg.intervention_mode == "censor" and source[t] != 1 and source[end] == 1
            record.update(start_frame=t, next_frame=end,
                observation_key=f"{report['episode_uid']}:{t}",
                next_observation_key=f"{report['episode_uid']}:{end}",
                action=np.concatenate([actions[t:active_end], np.repeat(actions[active_end-1:active_end], cfg.chunk_size-c, axis=0)]),
                action_valid=np.arange(cfg.chunk_size)<c,
                control_source=np.pad(source[t:active_end], (0,cfg.chunk_size-c), constant_values=255),
                terminated=terminal, truncated=censored,
                reward=cfg.gamma_tick**(end-t-1) if success == end else 0.0,
                discount=0.0 if terminal or censored else cfg.gamma_tick**(end-t),
                end_reason="intervention_censored" if censored else "success" if success == end else "failure_end" if terminal else "continuing")
            output.append(record)
        t = end
    return output


def build_dataset(roots: list[Path], cfg: TransitionConfig, recipe: dict | None = None):
    records, episodes = [], []
    seen = set()
    if recipe is not None:
        if recipe.get("format") != "ahawam_episode_recipe/v1" or cfg.temporal_mode != "skip_hold" or cfg.timeline_mode != "episode_frames":
            raise ValueError("Episode recipe requires skip_hold and continuous episode_frames")
        rules = recipe["episodes"]
    for root in roots:
        root = root.resolve()
        info = json.loads((root / "meta/info.json").read_text())
        if info["codebase_version"] != "v2.1":
            raise ValueError("Only LeRobot v2.1 supported")
        manifests = {int(x["data_episode_index"]): x for x in jsonl(root / "recording_context/episodes_manifest.jsonl")}
        for row in jsonl(root / "meta/episodes.jsonl"):
            table, report = inspect_episode(root, row, manifests.get(int(row["episode_index"]), {}), info)
            if report["episode_uid"] in seen:
                raise ValueError("Duplicate episode_uid across roots")
            seen.add(report["episode_uid"])
            if recipe is not None:
                rule = rules[report["episode_uid"]]
                if rule["source"] != report["source"]:
                    raise ValueError("Episode recipe source identity changed")
                retained = rule["retained_frames"]
                if type(retained) is not int or not 2 <= retained <= len(table):
                    raise ValueError("Invalid retained frame limit")
                if rule["split"] not in {"train", "val"}:
                    raise ValueError("Invalid explicit split")
                censored = rule["reward_mode"] == "observed_zero_nonterminal"
                if rule["reward_mode"] not in {"annotated_or_failure", "observed_zero_nonterminal"}:
                    raise ValueError("Invalid reward mode")
                if censored and not (report["outcome"] == "success" and report["first_success"] is None):
                    raise ValueError("Only unobserved successes may use the censored-success rule")
                if retained < len(table) and not censored:
                    raise ValueError("Prefix cuts are restricted to explicitly censored recordings")
                report.update(original_status=report["status"], retained_frames=retained,
                    split=rule["split"], censored_recording=censored, treatment=rule["treatment"])
                if censored:
                    report["status"] = "ready"
                table = table.iloc[:retained].copy()
            if cfg.temporal_mode == "skip_hold":
                report["effective_success_frame"] = effective_success_frame(table, report["first_success"])
            episodes.append(report)
            episode_rows = episode_transitions(table, report, cfg)
            report["transitions"] = len(episode_rows)
            report["success_windows"] = sum(r["end_reason"] == "success" for r in episode_rows)
            records.extend(episode_rows)
    if recipe is not None and seen != set(rules):
        raise ValueError("Recipe does not exactly match input episodes")
    numeric = ("state", "next_state", "action", "action_valid", "control_source", "reward", "discount",
               "duration_ticks", "duration_seconds", "terminated", "truncated")
    tensors = {k: torch.from_numpy(np.asarray([r[k] for r in records])) for k in numeric} if records else {}
    metadata = [{k: v for k, v in r.items() if k not in numeric} for r in records]
    result = dict(format="ahawam_offline_transitions/v1", config=asdict(cfg), tensors=tensors,
                records=metadata, episodes=episodes,
                conventions=dict(reward=("first marked success relocated to first HOLD image when incoming action is HOLD; incoming active command earns +1"
                                         if cfg.temporal_mode == "skip_hold" else "first marked success observation rewards its incoming transition"),
                                 action=("natural POLICY+HUMAN commands; HOLD omitted; full tails past success retained; discount per active command"
                                         if cfg.temporal_mode == "skip_hold" else "active absolute target blocks; trailing hold included in environment duration"
                                         if cfg.temporal_mode == "macro" else "actual absolute joint targets, hold included"),
                                 terminal="success or finite-horizon manifest failure end; censored success recordings bootstrap with no invented reward",
                                 timeline=("episode storage order; tick jumps and dropped_tick flags do not split history/transitions; pause time omitted"
                                           if cfg.timeline_mode == "episode_frames" else "legacy strict tick continuity"),
                                 annotation="positive marks accepted as first-success events by task contract"))
    if recipe is not None:
        result["episode_recipe"] = recipe
    return result
