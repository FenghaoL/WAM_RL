#!/usr/bin/env python3
"""Generate many history-conditioned VideoDiT predictions against ground truth."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ahawam.runtime import build_datasets
from ahawam.utils.video_io import save_mp4
from scripts.so101_pipeline.archive.video_path_experiments.exact_video_path import (
    _batch_sample,
    _comparison_frames,
    _frame_metrics,
    _generate,
    _load_model,
    _make_robocoin_sample,
    _tensor_video_to_frames,
)


def _safe_name(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")


def _read_episode_rows(dataset_dir: Path) -> list[dict[str, Any]]:
    path = dataset_dir / "meta" / "episodes.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _select_so101_indices(
    dataset: Any,
    count: int,
    *,
    minimum_history_span: int = 0,
) -> list[dict[str, Any]]:
    starts = dataset.lerobot_dataset.episode_data_index["from"].tolist()
    ends = dataset.lerobot_dataset.episode_data_index["to"].tolist()
    raw = dataset.lerobot_dataset.multi_dataset._datasets[0]
    horizon = int(dataset.num_frames - 1) * int(dataset.global_sample_stride)
    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for episode_slot, (start, end) in enumerate(zip(starts, ends)):
        length = int(end) - int(start)
        latest = length - int(dataset.num_frames)
        if latest < max(horizon, minimum_history_span):
            continue
        relative = min(latest, int(dataset.num_history_frames) * horizon)
        relative = max(horizon, minimum_history_span, (relative // horizon) * horizon)
        sample_index = int(start) + relative
        raw_item = raw.hf_dataset[sample_index]
        task_index = int(raw_item["task_index"].item())
        task = str(raw.meta.tasks[task_index])
        candidates[task].append(
            {
                "sample_index": sample_index,
                "episode_slot": episode_slot,
                "source_episode": int(raw_item["episode_index"].item()),
                "source_frame": int(raw_item["frame_index"].item()),
                "task": task,
                "history_frames": dataset.get_video_history_valid_len_for_index(sample_index),
            }
        )

    for items in candidates.values():
        items.sort(key=lambda item: (-item["history_frames"], item["source_episode"]))

    selected: list[dict[str, Any]] = []
    tasks = sorted(candidates)
    round_index = 0
    while len(selected) < count:
        added = False
        for task in tasks:
            if round_index < len(candidates[task]) and len(selected) < count:
                selected.append(candidates[task][round_index])
                added = True
        if not added:
            break
        round_index += 1
    if len(selected) < count:
        raise ValueError(f"Only found {len(selected)} valid SO101 samples, requested {count}.")
    return selected


def _replace_with_online_history(
    dataset: Any,
    sample_index: int,
    episode_slot: int,
    sample: dict[str, Any],
    *,
    history_count: int,
    planner_control_stride: int,
    video_rope_frame_stride: int,
) -> dict[str, Any]:
    """Use the same observation cadence and RoPE positions as online action inference."""
    episode_start = int(
        dataset.lerobot_dataset.episode_data_index["from"][episode_slot].item()
    )
    available_history = (sample_index - episode_start) // planner_control_stride
    effective_history = min(history_count, available_history)
    if effective_history <= 0:
        raise ValueError(f"Sample {sample_index} has no online history observation.")
    history_sample_indices = [
        sample_index - planner_control_stride * offset
        for offset in range(effective_history, 0, -1)
    ]

    history_frames = [
        dataset[index]["video"][:, 0] for index in history_sample_indices
    ]
    current_frame_index = int(sample["video_current_frame_index"].item())
    rope_step = planner_control_stride // video_rope_frame_stride
    if rope_step * video_rope_frame_stride != planner_control_stride:
        raise ValueError(
            "action_horizon must be divisible by video_rope_frame_stride for online history."
        )
    history_rope_indices = [
        current_frame_index - rope_step * offset
        for offset in range(effective_history, 0, -1)
    ]
    if history_rope_indices[0] < 0:
        raise ValueError(
            f"Sample {sample_index} produces negative history RoPE positions: "
            f"{history_rope_indices}"
        )

    pad_count = history_count - effective_history
    padded_history_frames = [history_frames[0]] * pad_count + history_frames
    padded_rope_indices = [0] * pad_count + history_rope_indices

    sample = dict(sample)
    sample["video_history"] = torch.stack(padded_history_frames, dim=1)
    sample["video_history_valid_len"] = torch.tensor(
        effective_history, dtype=torch.long
    )
    sample["video_history_frame_indices"] = torch.tensor(
        padded_rope_indices, dtype=torch.long
    )
    return sample


def _select_robocoin_windows(dataset_dir: Path, count: int) -> list[dict[str, Any]]:
    rows = _read_episode_rows(dataset_dir)
    if count > len(rows):
        raise ValueError(f"Requested {count} RoboCOIN samples from only {len(rows)} episodes.")
    episode_positions = np.linspace(0, len(rows) - 1, num=count, dtype=np.int64)
    selected = []
    for output_index, row_index in enumerate(episode_positions.tolist()):
        row = rows[int(row_index)]
        length = int(row["length"])
        latest_start = length - 65
        if latest_start < 384:
            raise ValueError(
                f"Episode {row['episode_index']} has only {length} frames; full six-frame history "
                "and a 65-frame future require at least 449."
            )
        phase = (output_index % 4) / 3.0
        start_frame = int(round(384 + phase * (latest_start - 384)))
        selected.append(
            {
                "episode": int(row["episode_index"]),
                "start_frame": start_frame,
                "task": str(row["tasks"][0]),
                "episode_length": length,
                "history_frames": 6,
            }
        )
    return selected


def _history_strip(sample: dict[str, Any]) -> Image.Image:
    valid = int(sample["video_history_valid_len"].reshape(-1)[0].item())
    if valid <= 0:
        return Image.new("RGB", (320, 80), (0, 0, 0))
    frames = _tensor_video_to_frames(sample["video_history"])[-valid:]
    thumb_width = 160
    thumb_height = int(round(frames[0].height * thumb_width / frames[0].width))
    canvas = Image.new("RGB", (thumb_width * valid, thumb_height + 28), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    indices = sample["video_history_frame_indices"].reshape(-1).tolist()[-valid:]
    for index, (frame, frame_position) in enumerate(zip(frames, indices)):
        frame = frame.resize((thumb_width, thumb_height), Image.Resampling.BILINEAR)
        canvas.paste(frame, (index * thumb_width, 28))
        draw.text((index * thumb_width + 5, 7), f"history rope={frame_position}", fill="white")
    return canvas


def _contact_sheet(images: list[tuple[str, Image.Image]], output: Path) -> None:
    thumb_width = 480
    rendered: list[tuple[str, Image.Image]] = []
    for label, image in images:
        height = int(round(image.height * thumb_width / image.width))
        rendered.append((label, image.resize((thumb_width, height), Image.Resampling.BILINEAR)))
    row_height = max(image.height for _, image in rendered) + 30
    canvas = Image.new("RGB", (thumb_width, row_height * len(rendered)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for row, (label, image) in enumerate(rendered):
        y = row * row_height
        draw.text((6, y + 7), label, fill="white")
        canvas.paste(image, (0, y + 30))
    canvas.save(output)


def _save_summary(
    output: Path,
    mode: str,
    history_mode: str,
    rows: list[dict[str, Any]],
) -> None:
    summary = {
        "mode": mode,
        "history_mode": history_mode,
        "num_samples": len(rows),
        "mean_future_rgb_mse": round(float(np.mean([x["future_rgb_mse"] for x in rows])), 3),
        "mean_future_rgb_psnr": round(float(np.mean([x["future_rgb_psnr"] for x in rows])), 3),
        "mean_pred_temporal_abs_diff": round(
            float(np.mean([x["pred_temporal_abs_diff"] for x in rows])), 3
        ),
        "mean_target_temporal_abs_diff": round(
            float(np.mean([x["target_temporal_abs_diff"] for x in rows])), 3
        ),
        "mean_generation_seconds": round(float(np.mean([x["generation_seconds"] for x in rows])), 3),
        "max_peak_allocated_gib": round(float(max(x["peak_allocated_gib"] for x in rows)), 3),
        "samples": rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    fields = list(rows[0])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    if history_mode == "online":
        path_description = (
            "This prediction uses sequential online rolling history K/V and the shared "
            "online future-video denoising implementation."
        )
    else:
        path_description = (
            "This prediction uses the training-style one-shot causal history prefill and "
            "the shared online future-video denoising implementation."
        )
    lines = [
        f"# {mode.upper()} history-conditioned VideoDiT batch report",
        "",
        f"- Samples: {summary['num_samples']}",
        f"- Mean future PSNR: {summary['mean_future_rgb_psnr']} dB",
        f"- Mean prediction temporal difference: {summary['mean_pred_temporal_abs_diff']}",
        f"- Mean ground-truth temporal difference: {summary['mean_target_temporal_abs_diff']}",
        f"- Mean generation time: {summary['mean_generation_seconds']} s",
        f"- Max allocated GPU memory: {summary['max_peak_allocated_gib']} GiB",
        "",
        "Each sample directory contains `history_strip.png`, `ground_truth.mp4`, "
        "`prediction_with_history.mp4`, `comparison_gt_vs_prediction.mp4`, and `metrics.json`.",
        "",
        path_description,
        "The released configuration has `action_conditioned=false`, so no action tensor "
        "is consumed by the video branch.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "samples"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("so101", "robocoin"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--history-mode",
        choices=("training", "online"),
        default="training",
        help="Use one-shot training prefill or sequential online rolling prefill.",
    )
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--text-cache-dir", type=Path)
    args = parser.parse_args()
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    args.output.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.load(args.config)
    dataset = None
    if args.mode == "so101":
        cfg.data.train.skip_padding_as_possible = False
        dataset, _ = build_datasets(cfg.data, cfg.model)
        history_count = int(cfg.model.num_history_frames)
        planner_control_stride = int(cfg.model.action_horizon)
        minimum_history_span = (
            planner_control_stride if args.history_mode == "online" else 0
        )
        selections = _select_so101_indices(
            dataset,
            args.num_samples,
            minimum_history_span=minimum_history_span,
        )
    else:
        if args.dataset_dir is None or args.text_cache_dir is None:
            parser.error("robocoin mode requires --dataset-dir and --text-cache-dir")
        selections = _select_robocoin_windows(args.dataset_dir, args.num_samples)

    print(f"[selection] {json.dumps(selections, ensure_ascii=False)}", flush=True)
    print(f"[model] loading {args.checkpoint}", flush=True)
    model = _load_model(cfg, args.checkpoint, args.device)
    model.eval()
    if bool(model.video_expert.action_conditioned):
        raise RuntimeError("This diagnostic requires video_expert.action_conditioned=false.")
    print("[model] ready", flush=True)

    rows: list[dict[str, Any]] = []
    middle_images: list[tuple[str, Image.Image]] = []
    last_images: list[tuple[str, Image.Image]] = []
    for output_index, selection in enumerate(selections):
        if args.mode == "so101":
            assert dataset is not None
            sample_index = int(selection["sample_index"])
            raw_sample = dataset[sample_index]
            if args.history_mode == "online":
                raw_sample = _replace_with_online_history(
                    dataset,
                    sample_index,
                    int(selection["episode_slot"]),
                    raw_sample,
                    history_count=int(cfg.model.num_history_frames),
                    planner_control_stride=int(cfg.model.action_horizon),
                    video_rope_frame_stride=int(cfg.model.video_rope_frame_stride),
                )
            sample = _batch_sample(raw_sample)
            tag = f"{output_index:02d}_idx_{selection['sample_index']}_{_safe_name(selection['task'])[:36]}"
        else:
            assert args.dataset_dir is not None and args.text_cache_dir is not None
            sample = _make_robocoin_sample(
                args.dataset_dir,
                args.text_cache_dir,
                args.device,
                int(selection["episode"]),
                int(selection["start_frame"]),
                str(selection["task"]),
            )
            tag = f"{output_index:02d}_ep_{selection['episode']:06d}_frame_{selection['start_frame']}"

        sample_dir = args.output / tag
        sample_dir.mkdir(parents=True, exist_ok=True)
        ground_truth = _tensor_video_to_frames(sample["video"])
        print(f"[generate] {output_index + 1}/{len(selections)} {tag}", flush=True)
        prediction, generation_info = _generate(
            model,
            sample,
            steps=args.steps,
            seed=args.seed + output_index,
            use_history=True,
            history_mode=args.history_mode,
        )
        metrics = _frame_metrics(prediction, ground_truth)
        prediction_label = (
            "online video-only prediction"
            if args.history_mode == "online"
            else "training-history video prediction"
        )
        comparison = _comparison_frames(
            [("ground truth", ground_truth), (prediction_label, prediction)]
        )
        save_mp4(ground_truth, str(sample_dir / "ground_truth.mp4"), fps=args.fps)
        save_mp4(prediction, str(sample_dir / "prediction_with_history.mp4"), fps=args.fps)
        save_mp4(
            comparison,
            str(sample_dir / "comparison_gt_vs_prediction.mp4"),
            fps=args.fps,
        )
        _history_strip(sample).save(sample_dir / "history_strip.png")
        comparison[len(comparison) // 2].save(sample_dir / "comparison_middle.png")
        comparison[-1].save(sample_dir / "comparison_last.png")

        row = {
            "sample": tag,
            "prompt": sample["prompt"][0],
            "sample_index": selection.get("sample_index", ""),
            "episode": selection.get("source_episode", selection.get("episode", "")),
            "start_frame": selection.get("source_frame", selection.get("start_frame", "")),
            "history_rope_indices": sample["video_history_frame_indices"][0].tolist(),
            "current_rope_index": int(
                sample["video_current_frame_index"].reshape(-1)[0].item()
            ),
            **generation_info,
            **metrics,
        }
        (sample_dir / "metrics.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        rows.append(row)
        middle_images.append((tag, comparison[len(comparison) // 2]))
        last_images.append((tag, comparison[-1]))
        del prediction, comparison, ground_truth, sample
        gc.collect()
        torch.cuda.empty_cache()

    _contact_sheet(middle_images, args.output / "contact_sheet_middle.png")
    _contact_sheet(last_images, args.output / "contact_sheet_last.png")
    _save_summary(args.output, args.mode, args.history_mode, rows)


if __name__ == "__main__":
    main()
