#!/usr/bin/env python3
"""Export evaluation windows anchored at the START of every human correction.

The DAgger recordings interleave policy control with human takeovers. The
question this supports is narrow and specific: *at the exact moment the human
grabbed the arm, does the video model predict the corrected future, or does it
keep hallucinating the policy's failing continuation?*

So each window starts on the first HUMAN frame of a correction segment and runs
the usual 49 control frames forward. Ground truth over that window is what the
human actually did (plus whatever follows once they hand back).

Output is the same manifest layout `export_reference_samples.py` writes, so
`wan22_base/run_video_branch_preview.py` and `compare_runs.py` consume it
unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ahawam.runtime import build_datasets  # noqa: E402
from ahawam.utils.video_io import save_mp4  # noqa: E402
from scripts.so101_pipeline.hil_human_only_dataset import (  # noqa: E402
    _human_segments_from_columns,
    _to_numpy,
)
from scripts.so101_pipeline.baseline_video_experiments.common.sample_selection import (  # noqa: E402
    safe_name,
    save_manifest,
)
from scripts.so101_pipeline.baseline_video_experiments.export_reference_samples import (  # noqa: E402
    history_strip,
    tensor_video_to_frames,
)

CONTROL_COLUMNS = [
    "episode_index", "frame_index", "tick_index", "control_source",
    "intervention", "human_action_valid", "dropped_tick",
]


def scan_human_segments(dataset_dirs: list[str]) -> list[dict]:
    """Return one record per human correction segment, in dataset_dirs order."""
    records: list[dict] = []
    for session, raw_root in enumerate(dataset_dirs):
        root = Path(raw_root)
        for parquet_path in sorted((root / "data").glob("chunk-*/*.parquet")):
            names = list(CONTROL_COLUMNS)
            if "action_supervision_mask" in set(pq.read_schema(parquet_path).names):
                names.append("action_supervision_mask")
            table = pq.read_table(parquet_path, columns=names)
            columns = {name: _to_numpy(table[name]) for name in names}
            frame_index = columns["frame_index"].astype(int)
            episode_index = columns["episode_index"].astype(int)
            for row_start, row_end in _human_segments_from_columns(columns):
                records.append(
                    {
                        "session": session,
                        "session_dir": str(root),
                        "source_episode": int(episode_index[row_start]),
                        # Anchor on the recorded frame_index, not the row offset,
                        # so a dropped tick cannot silently shift the window.
                        "segment_start_frame": int(frame_index[row_start]),
                        "segment_frames": int(row_end - row_start),
                    }
                )
    return records


def build_episode_slot_lookup(dataset) -> dict[tuple[int, int], int]:
    """(session, episode_index) -> global episode slot in the concatenated dataset."""
    starts = dataset.lerobot_dataset.episode_data_index["from"].tolist()
    subsets = dataset.lerobot_dataset.multi_dataset._datasets
    lookup: dict[tuple[int, int], int] = {}
    for slot, start in enumerate(starts):
        raw, local = dataset._resolve_lerobot_dataset_for_global_index(int(start))
        session = next(i for i, d in enumerate(subsets) if d is raw)
        episode = int(raw.hf_dataset[int(local)]["episode_index"].item())
        lookup[(session, episode)] = slot
    return lookup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True,
                        help="Training run config.yaml supplying geometry + norm stats.")
    parser.add_argument("--dataset-dirs", nargs="+", required=True,
                        help="HIL raw recording session directories.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-segment-frames", type=int, default=16,
                        help="Skip corrections shorter than this; a 1-2 frame twitch "
                             "carries no visible correction to predict.")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = every segment")
    parser.add_argument(
        "--episode-allowlist-config", type=Path,
        help="An OVCR experiment config; keep only segments from episodes listed in its "
             "`data.train.correction_episode_selection`, i.e. the episodes that run "
             "actually trained on. Directories absent from that map contribute nothing.",
    )
    parser.add_argument(
        "--spread-samples", type=int, default=0,
        help="Keep this many windows, evenly spaced over the usable list ordered by "
             "session/episode/frame, so a subset still covers every session. 0 = all.",
    )
    parser.add_argument(
        "--allow-truncated", action="store_true",
        help="Keep windows that run past the end of their episode. LeRobot clamps "
             "out-of-range queries to the last frame, so ground truth simply freezes "
             "on the final frame instead of the window being dropped.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.data.train.skip_padding_as_possible = False
    cfg.data.train.dataset_dirs = list(args.dataset_dirs)
    cfg.data.train.val_set_proportion = 0.0
    print(f"[data] pretrained_norm_stats = {cfg.data.train.pretrained_norm_stats}", flush=True)
    dataset, _ = build_datasets(cfg.data, cfg.model)

    num_frames = int(dataset.num_frames)
    video_stride_ctrl = int(cfg.data.train.action_video_freq_ratio)
    horizon = int(num_frames - 1) * int(dataset.global_sample_stride)
    starts = dataset.lerobot_dataset.episode_data_index["from"].tolist()
    ends = dataset.lerobot_dataset.episode_data_index["to"].tolist()
    lookup = build_episode_slot_lookup(dataset)

    segments = scan_human_segments(list(args.dataset_dirs))
    print(f"[scan] {len(segments)} human segments found", flush=True)
    if args.episode_allowlist_config is not None:
        allow_cfg = OmegaConf.load(args.episode_allowlist_config)
        raw_allow = allow_cfg.data.train.get("correction_episode_selection")
        if raw_allow is None:
            raise KeyError(
                f"{args.episode_allowlist_config} has no data.train.correction_episode_selection"
            )
        allow = {
            str(Path(key)): {int(e) for e in episodes}
            for key, episodes in OmegaConf.to_container(raw_allow, resolve=True).items()
        }
        before = len(segments)
        segments = [
            r for r in segments
            if int(r["source_episode"]) in allow.get(str(Path(r["session_dir"])), set())
        ]
        print(f"[scan] episode allowlist: {before} -> {len(segments)} segments", flush=True)

    selections = []
    skipped_short = skipped_nofit = 0
    for record in segments:
        if record["segment_frames"] < args.min_segment_frames:
            skipped_short += 1
            continue
        slot = lookup.get((record["session"], record["source_episode"]))
        if slot is None:
            raise KeyError(f"No episode slot for {record}")
        episode_length = int(ends[slot]) - int(starts[slot])
        relative = int(record["segment_start_frame"])
        # The full 49-frame window normally has to stay inside the episode.
        if relative + num_frames > episode_length and not args.allow_truncated:
            skipped_nofit += 1
            continue
        selections.append(
            {
                **record,
                "sample_index": int(starts[slot]) + relative,
                "episode_slot": slot,
                "episode_length": episode_length,
                "relative": relative,
                "source_frame": relative,
                "task": None,
                "history_steps_back": relative // horizon,
                "history_frames": min(relative // horizon, int(dataset.num_history_frames)),
                # Video frames sit at control offsets 0,4,...,48; anything at or past
                # the episode end is the last real frame held.
                "gt_valid_video_frames": sum(
                    1 for o in range(0, num_frames, video_stride_ctrl)
                    if relative + o < episode_length
                ),
            }
        )
    print(f"[scan] 可用 {len(selections)}（跳过 {skipped_short} 段太短、"
          f"{skipped_nofit} 段窗口越界）", flush=True)
    if args.max_samples > 0:
        selections = selections[: args.max_samples]
    if 0 < args.spread_samples < len(selections):
        picks = np.linspace(0, len(selections) - 1, args.spread_samples).round().astype(int)
        selections = [selections[i] for i in sorted(set(picks.tolist()))]
        print(f"[scan] spread down to {len(selections)} windows", flush=True)
    if not selections:
        raise SystemExit("No usable correction window.")

    args.output.mkdir(parents=True, exist_ok=True)
    control_fps = 15.0
    video_stride = int(cfg.data.train.action_video_freq_ratio)
    video_fps = control_fps / video_stride

    manifest_samples = []
    for order, sel in enumerate(selections):
        sample = dataset[int(sel["sample_index"])]
        task = str(sample["prompt"])
        frames = tensor_video_to_frames(sample["video"])
        frozen = len(frames) - int(sel["gt_valid_video_frames"])
        tag = (f"{order:02d}_s{sel['session']}_ep{sel['source_episode']:03d}"
               f"_f{sel['relative']:04d}_seg{sel['segment_frames']:03d}"
               + (f"_trunc{frozen}" if frozen else ""))
        sample_dir = args.output / tag
        (sample_dir / "frames").mkdir(parents=True, exist_ok=True)

        for index, frame in enumerate(frames):
            frame.save(sample_dir / "frames" / f"gt_{index:03d}.png")
        frames[0].save(sample_dir / "first_frame.png")
        save_mp4(frames, str(sample_dir / "ground_truth.mp4"), fps=video_fps)
        history_strip(sample).save(sample_dir / "history_strip.png")

        meta = {
            "tag": tag,
            "prompt": task,
            "task": safe_name(task)[:48],
            "sample_index": int(sel["sample_index"]),
            "episode_slot": int(sel["episode_slot"]),
            "session": int(sel["session"]),
            "session_dir": sel["session_dir"],
            "source_episode": int(sel["source_episode"]),
            "source_frame": int(sel["source_frame"]),
            "episode_length": int(sel["episode_length"]),
            "episode_relative_frame": int(sel["relative"]),
            "segment_frames": int(sel["segment_frames"]),
            "segment_seconds": round(sel["segment_frames"] / control_fps, 2),
            "gt_valid_video_frames": int(sel["gt_valid_video_frames"]),
            "gt_frozen_video_frames": len(frames) - int(sel["gt_valid_video_frames"]),
            "history_frames": int(sel["history_frames"]),
            "history_steps_back": int(sel["history_steps_back"]),
            "history_rope_indices": torch.as_tensor(
                sample["video_history_frame_indices"]).reshape(-1).tolist(),
            "current_rope_index": int(torch.as_tensor(
                sample["video_current_frame_index"]).reshape(-1)[0].item()),
            "temporal_position_ids": torch.as_tensor(
                sample["video_temporal_position_ids"]).reshape(-1).tolist(),
            "num_video_frames": len(frames),
            "frame_height": frames[0].height,
            "frame_width": frames[0].width,
            "video_fps": video_fps,
            "control_fps": control_fps,
            "video_stride_control_frames": video_stride,
            "clip_seconds": (len(frames) - 1) / video_fps,
            "frame_times_seconds": [i / video_fps for i in range(len(frames))],
        }
        (sample_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        manifest_samples.append(meta)
        print(f"[export] {order + 1}/{len(selections)} {tag}", flush=True)

    manifest = {
        "config": str(args.config),
        "dataset_dirs": list(args.dataset_dirs),
        "selection": "hil_human_correction_segment_start",
        "min_segment_frames": args.min_segment_frames,
        "allow_truncated": bool(args.allow_truncated),
        "episode_allowlist_config": (
            str(args.episode_allowlist_config) if args.episode_allowlist_config else None
        ),
        "spread_samples": int(args.spread_samples),
        "num_samples": len(manifest_samples),
        "video_fps": video_fps,
        "control_fps": control_fps,
        "num_video_frames": manifest_samples[0]["num_video_frames"],
        "frame_height": manifest_samples[0]["frame_height"],
        "frame_width": manifest_samples[0]["frame_width"],
        "clip_seconds": manifest_samples[0]["clip_seconds"],
        "samples": manifest_samples,
    }
    save_manifest(args.output / "manifest.json", manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != "samples"}, indent=2,
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
