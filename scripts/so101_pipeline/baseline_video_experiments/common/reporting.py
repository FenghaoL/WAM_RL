#!/usr/bin/env python3
"""Metrics, comparison images and run summaries shared by every baseline.

This module is deliberately free of `ahawam` and `torch` imports so the
MiniMax-H3 experiment can import it from its own conda environment.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw

# The SO101 model frame is a 384x320 composite: the fixed camera fills the top
# 256 rows, the wrist camera the bottom-left 128x160 block, and the bottom-right
# block is the black placeholder for RoboCOIN's missing second wrist camera.
SO101_REGIONS: dict[str, tuple[int, int, int, int]] = {
    "fixed": (0, 0, 320, 256),
    "wrist": (0, 256, 160, 384),
}


def save_mp4(frames: Sequence[Image.Image], path: Path | str, *, fps: float) -> None:
    """Write frames at their exact size (no macro-block resize), h264/yuv420p.

    yuv420p needs even dimensions, so odd sides are edge-padded by one pixel.
    """
    import imageio

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        str(path),
        fps=max(float(fps), 1e-3),
        codec="libx264",
        format="FFMPEG",
        pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for frame in frames:
            array = np.asarray(frame.convert("RGB"))
            pad_h = array.shape[0] % 2
            pad_w = array.shape[1] % 2
            if pad_h or pad_w:
                array = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
            writer.append_data(array)
    finally:
        writer.close()


def _stack(frames: Sequence[Image.Image]) -> np.ndarray:
    return np.stack([np.asarray(frame.convert("RGB"), dtype=np.float32) for frame in frames])


def _mse_psnr(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    mse = float(np.mean((pred - target) ** 2))
    psnr = float(10.0 * np.log10((255.0**2) / max(mse, 1e-9)))
    return mse, psnr


def frame_metrics(
    pred: Sequence[Image.Image],
    target: Sequence[Image.Image],
    *,
    regions: dict[str, tuple[int, int, int, int]] | None = None,
    skip_first: bool = True,
) -> dict[str, float]:
    """Compare two equal-length frame lists.

    `skip_first` drops frame 0 because it is the clean conditioning frame that
    every model is handed for free; including it inflates PSNR identically for
    all baselines and hides the part that is actually predicted.
    """
    if len(pred) != len(target):
        raise ValueError(f"Frame count mismatch: pred={len(pred)}, target={len(target)}")
    start = 1 if skip_first and len(pred) > 1 else 0
    pred_arr = _stack(pred[start:])
    target_arr = _stack(target[start:])
    if pred_arr.shape != target_arr.shape:
        raise ValueError(
            f"Frame shape mismatch: pred={pred_arr.shape}, target={target_arr.shape}"
        )

    mse, psnr = _mse_psnr(pred_arr, target_arr)
    metrics: dict[str, float] = {
        "future_rgb_mse": round(mse, 3),
        "future_rgb_psnr": round(psnr, 3),
        "pred_temporal_abs_diff": round(
            float(np.mean(np.abs(np.diff(pred_arr, axis=0)))) if len(pred_arr) > 1 else 0.0, 3
        ),
        "target_temporal_abs_diff": round(
            float(np.mean(np.abs(np.diff(target_arr, axis=0)))) if len(target_arr) > 1 else 0.0, 3
        ),
    }
    for name, (left, top, right, bottom) in (regions or {}).items():
        region_mse, region_psnr = _mse_psnr(
            pred_arr[:, top:bottom, left:right], target_arr[:, top:bottom, left:right]
        )
        metrics[f"{name}_rgb_mse"] = round(region_mse, 3)
        metrics[f"{name}_rgb_psnr"] = round(region_psnr, 3)
    return metrics


def comparison_frames(columns: Sequence[tuple[str, Sequence[Image.Image]]]) -> list[Image.Image]:
    """Lay out several frame lists side by side with a text header per column."""
    if not columns:
        raise ValueError("`columns` must not be empty")
    count = min(len(frames) for _, frames in columns)
    output: list[Image.Image] = []
    for index in range(count):
        width = sum(frames[index].width for _, frames in columns)
        height = max(frames[index].height for _, frames in columns) + 28
        canvas = Image.new("RGB", (width, height), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for label, frames in columns:
            canvas.paste(frames[index].convert("RGB"), (x, 28))
            draw.text((x + 6, 7), label, fill=(255, 255, 255))
            x += frames[index].width
        output.append(canvas)
    return output


def contact_sheet(images: Sequence[tuple[str, Image.Image]], output: Path) -> None:
    """Stack one labelled thumbnail per sample into a single overview image."""
    if not images:
        return
    thumb_width = 640
    rendered: list[tuple[str, Image.Image]] = []
    for label, image in images:
        height = max(1, int(round(image.height * thumb_width / image.width)))
        rendered.append((label, image.resize((thumb_width, height), Image.Resampling.BILINEAR)))
    row_height = max(image.height for _, image in rendered) + 30
    canvas = Image.new("RGB", (thumb_width, row_height * len(rendered)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for row, (label, image) in enumerate(rendered):
        y = row * row_height
        draw.text((6, y + 7), label, fill="white")
        canvas.paste(image, (0, y + 30))
    canvas.save(output)


def _mean(rows: Sequence[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if isinstance(row.get(key), (int, float))]
    if not values:
        return None
    return round(float(np.mean(values)), 3)


def save_summary(
    output: Path,
    rows: Sequence[dict[str, Any]],
    *,
    run_meta: dict[str, Any],
    mean_keys: Iterable[str] = (
        "future_rgb_mse",
        "future_rgb_psnr",
        "fixed_rgb_psnr",
        "wrist_rgb_psnr",
        "pred_temporal_abs_diff",
        "target_temporal_abs_diff",
        "generation_seconds",
    ),
) -> dict[str, Any]:
    """Write summary.json / summary.csv / README.md for one baseline run."""
    output.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("`rows` must not be empty")

    summary: dict[str, Any] = dict(run_meta)
    summary["num_samples"] = len(rows)
    for key in mean_keys:
        value = _mean(rows, key)
        if value is not None:
            summary[f"mean_{key}"] = value
    peaks = [row["peak_allocated_gib"] for row in rows if "peak_allocated_gib" in row]
    if peaks:
        summary["max_peak_allocated_gib"] = round(float(max(peaks)), 3)
    summary["samples"] = list(rows)

    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    headline = [f"# {run_meta.get('run_name', 'baseline')} video preview report", ""]
    for key, value in summary.items():
        if key in {"samples"}:
            continue
        headline.append(f"- {key}: {value}")
    headline += [
        "",
        "Every sample directory holds `ground_truth.mp4`, one `prediction_*.mp4` per "
        "condition, `comparison.mp4`, and `metrics.json`.",
        "",
        "`future_rgb_psnr` skips frame 0, which every model receives as a clean "
        "conditioning frame. `fixed_rgb_psnr` and `wrist_rgb_psnr` score the two "
        "camera regions of the SO101 composite separately.",
    ]
    (output / "README.md").write_text("\n".join(headline) + "\n", encoding="utf-8")

    printable = {key: value for key, value in summary.items() if key != "samples"}
    print(json.dumps(printable, indent=2, ensure_ascii=False), flush=True)
    return summary
