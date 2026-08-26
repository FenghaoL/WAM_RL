#!/usr/bin/env python3
"""Stage 1: pick the shared SO101 evaluation windows and export them to disk.

Run this once in the `ahawam` conda environment. Everything downstream reads
the exported directory:

* `wan22_base/run_video_branch_preview.py` re-opens the dataset and uses the
  manifest so the AHA-WAM video branch scores exactly these windows;
* `minimax_h3/run_minimax_h3_preview.py` runs in its own conda environment and
  only ever touches the exported PNGs and `meta.json`, so it never has to
  import `ahawam`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ahawam.runtime import build_datasets  # noqa: E402
from ahawam.utils.video_io import save_mp4  # noqa: E402
from scripts.so101_pipeline.baseline_video_experiments.common.sample_selection import (  # noqa: E402
    sample_tag,
    save_manifest,
    select_so101_samples,
)


def tensor_video_to_frames(video: torch.Tensor) -> list[Image.Image]:
    """[3,T,H,W] or [1,3,T,H,W] in [-1,1] -> list of RGB frames."""
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4 or int(video.shape[0]) != 3:
        raise ValueError(f"Expected [3,T,H,W], got {tuple(video.shape)}")
    uint8 = ((video.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    return [
        Image.fromarray(uint8[:, index].permute(1, 2, 0).numpy())
        for index in range(int(uint8.shape[1]))
    ]


def history_strip(sample: dict, *, thumb_width: int = 160) -> Image.Image:
    valid = int(torch.as_tensor(sample["video_history_valid_len"]).reshape(-1)[0].item())
    if valid <= 0:
        return Image.new("RGB", (thumb_width, 80), (18, 18, 18))
    frames = tensor_video_to_frames(sample["video_history"])[-valid:]
    thumb_height = int(round(frames[0].height * thumb_width / frames[0].width))
    canvas = Image.new("RGB", (thumb_width * valid, thumb_height + 28), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    indices = torch.as_tensor(sample["video_history_frame_indices"]).reshape(-1).tolist()[-valid:]
    for index, (frame, rope) in enumerate(zip(frames, indices)):
        canvas.paste(
            frame.resize((thumb_width, thumb_height), Image.Resampling.BILINEAR),
            (index * thumb_width, 28),
        )
        draw.text((index * thumb_width + 5, 7), f"rope={rope}", fill="white")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="A training run's model/config.yaml; provides the dataset and model geometry.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        help="Override `data.train.dataset_dirs`, e.g. to score held-out validation "
             "recordings with the training run's own geometry and norm stats.",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        help="How densely to place windows inside an episode, in control frames. "
             "Default = one clip (48). Smaller = more windows per episode.",
    )
    parser.add_argument(
        "--per-episode-start",
        action="store_true",
        help="Reserve one window at frame 0 of every episode, so no episode's "
             "initial state goes unscored. Raises --num-samples if needed.",
    )
    parser.add_argument(
        "--per-episode-end",
        action="store_true",
        help="Also reserve the last window every episode admits, covering its "
             "final 3.2 s -- where the task actually completes.",
    )
    parser.add_argument(
        "--val-set-proportion",
        type=float,
        help="Override `data.train.val_set_proportion`. Pass 0 to use every episode, "
             "which is what you want when the directories already are the held-out set.",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    # Deterministic windows: never let the loader hop to a random index.
    cfg.data.train.skip_padding_as_possible = False
    if args.dataset_dirs:
        cfg.data.train.dataset_dirs = list(args.dataset_dirs)
        print(f"[data] dataset_dirs overridden -> {list(args.dataset_dirs)}", flush=True)
    if args.val_set_proportion is not None:
        cfg.data.train.val_set_proportion = float(args.val_set_proportion)
        print(
            f"[data] val_set_proportion overridden -> {args.val_set_proportion}", flush=True
        )
    # Normalisation stats must stay the training ones; recomputing them on a
    # held-out set would silently change the target the model is scored against.
    print(f"[data] pretrained_norm_stats = {cfg.data.train.pretrained_norm_stats}", flush=True)
    dataset, _ = build_datasets(cfg.data, cfg.model)

    selections = select_so101_samples(
        dataset,
        args.num_samples,
        per_episode_start=args.per_episode_start,
        per_episode_end=args.per_episode_end,
        window_stride=args.window_stride,
    )
    args.output.mkdir(parents=True, exist_ok=True)

    control_fps = 15.0
    video_stride = int(cfg.data.train.action_video_freq_ratio)
    video_fps = control_fps / video_stride

    manifest_samples = []
    for order, selection in enumerate(selections):
        tag = sample_tag(order, selection)
        sample_dir = args.output / tag
        (sample_dir / "frames").mkdir(parents=True, exist_ok=True)

        sample = dataset[int(selection["sample_index"])]
        frames = tensor_video_to_frames(sample["video"])
        for index, frame in enumerate(frames):
            frame.save(sample_dir / "frames" / f"gt_{index:03d}.png")
        frames[0].save(sample_dir / "first_frame.png")
        save_mp4(frames, str(sample_dir / "ground_truth.mp4"), fps=video_fps)
        history_strip(sample).save(sample_dir / "history_strip.png")

        meta = {
            "tag": tag,
            "prompt": sample["prompt"],
            "task": str(selection["task"]),
            "sample_index": int(selection["sample_index"]),
            "episode_slot": int(selection["episode_slot"]),
            "session": int(selection.get("session", 0)),
            "source_episode": int(selection["source_episode"]),
            "source_frame": int(selection["source_frame"]),
            "episode_length": int(selection["episode_length"]),
            "history_frames": int(selection["history_frames"]),
            "history_steps_back": int(selection["history_steps_back"]),
            "episode_relative_frame": int(selection.get("relative", 0)),
            "history_rope_indices": torch.as_tensor(
                sample["video_history_frame_indices"]
            ).reshape(-1).tolist(),
            "current_rope_index": int(
                torch.as_tensor(sample["video_current_frame_index"]).reshape(-1)[0].item()
            ),
            "temporal_position_ids": torch.as_tensor(
                sample["video_temporal_position_ids"]
            ).reshape(-1).tolist(),
            "num_video_frames": len(frames),
            "frame_height": frames[0].height,
            "frame_width": frames[0].width,
            "video_fps": video_fps,
            "control_fps": control_fps,
            "video_stride_control_frames": video_stride,
            "clip_seconds": (len(frames) - 1) / video_fps,
            "frame_times_seconds": [index / video_fps for index in range(len(frames))],
        }
        (sample_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest_samples.append(meta)
        print(f"[export] {order + 1}/{len(selections)} {tag}", flush=True)

    manifest = {
        "config": str(args.config),
        "dataset_dirs": list(cfg.data.train.dataset_dirs),
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
    print(json.dumps({k: v for k, v in manifest.items() if k != "samples"}, indent=2))


if __name__ == "__main__":
    main()
