#!/usr/bin/env python3
"""Merge several baseline runs into one table and one side-by-side video set.

Every run directory produced in this folder writes a `summary.json` with the
same schema, so this script works for the AHA-WAM video-branch runs and the
MiniMax-H3 run alike. It only needs numpy/PIL/imageio, so it runs in either
conda environment.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import sys

import imageio.v3 as iio
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import reporting  # noqa: E402

HEADLINE_KEYS = [
    "mean_future_rgb_psnr",
    "mean_fixed_rgb_psnr",
    "mean_wrist_rgb_psnr",
    "mean_future_rgb_mse",
    "mean_pred_temporal_abs_diff",
    "mean_target_temporal_abs_diff",
    "mean_generation_seconds",
]


def read_frames(path: Path) -> list[Image.Image]:
    return [Image.fromarray(np.asarray(frame)) for frame in iio.imiter(path, plugin="pyav")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True, help="summary.json paths")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-video", action="store_true", help="Skip the cross-run videos.")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for path in args.runs:
        summary = json.loads(Path(path).read_text(encoding="utf-8"))
        summary["_dir"] = Path(path).parent
        runs.append(summary)
    runs.sort(key=lambda run: str(run.get("run_name", run["_dir"].name)))

    # --- headline table -----------------------------------------------------
    table_rows = []
    for run in runs:
        row = {"run": str(run.get("run_name", run["_dir"].name)),
               "weights": str(run.get("weights", run.get("model_id", "")))}
        for key in HEADLINE_KEYS:
            row[key] = run.get(key, "")
        row["num_samples"] = run.get("num_samples", "")
        table_rows.append(row)

    fields = list(table_rows[0])
    with (args.output / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table_rows)

    lines = ["# Baseline comparison", "", "| " + " | ".join(fields) + " |",
             "| " + " | ".join("---" for _ in fields) + " |"]
    for row in table_rows:
        lines.append("| " + " | ".join(str(row[key]) for key in fields) + " |")

    # --- per-sample PSNR table ---------------------------------------------
    sample_tags: list[str] = []
    for run in runs:
        for sample in run.get("samples", []):
            if sample["sample"] not in sample_tags:
                sample_tags.append(sample["sample"])
    per_sample = {
        str(run.get("run_name", run["_dir"].name)): {
            sample["sample"]: sample.get("future_rgb_psnr", "")
            for sample in run.get("samples", [])
        }
        for run in runs
    }
    run_names = list(per_sample)
    lines += ["", "## Per-sample future PSNR (dB)", "",
              "| sample | " + " | ".join(run_names) + " |",
              "| --- | " + " | ".join("---" for _ in run_names) + " |"]
    for tag in sample_tags:
        lines.append(
            "| " + tag + " | " + " | ".join(str(per_sample[name].get(tag, "")) for name in run_names) + " |"
        )
    (args.output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)

    if args.no_video:
        return

    # --- cross-run side-by-side videos --------------------------------------
    video_dir = args.output / "side_by_side"
    video_dir.mkdir(parents=True, exist_ok=True)
    sheet: list[tuple[str, Image.Image]] = []
    for tag in sample_tags:
        columns: list[tuple[str, list[Image.Image]]] = []
        fps = 3.75
        for run in runs:
            name = str(run.get("run_name", run["_dir"].name))
            fps = float(run.get("video_fps", fps))
            sample_dir = run["_dir"] / tag
            if not columns:
                gt = sample_dir / "ground_truth.mp4"
                if gt.is_file():
                    columns.append(("ground truth", read_frames(gt)))
            prediction = sample_dir / str(run.get("primary_prediction_file", "prediction.mp4"))
            if prediction.is_file():
                columns.append((name, read_frames(prediction)))
        if len(columns) < 2:
            continue
        merged = reporting.comparison_frames(columns)
        reporting.save_mp4(merged, video_dir / f"{tag}.mp4", fps=fps)
        sheet.append((tag, merged[-1]))
    reporting.contact_sheet(sheet, args.output / "contact_sheet_last.png")
    print(f"[compare] wrote {len(sheet)} side-by-side videos to {video_dir}", flush=True)


if __name__ == "__main__":
    main()
