#!/usr/bin/env python3
"""Run the AHA-WAM VideoDiT branch on the shared SO101 windows, any weights.

`--weights wan22-base` is the point of this script: it builds the very same
AHA-WAM graph the training run uses, but loads the video expert straight from
the `Wan-AI/Wan2.2-TI2V-5B` release instead of from an AHA-WAM checkpoint. That
is the "untrained" reference — the initialisation every AHA-WAM run starts from.

`--weights <path.pt>` loads an AHA-WAM checkpoint instead, so the pretrained and
finetuned checkpoints can be scored on exactly the same windows.

The denoising itself goes through `deploy/server/ahawam_policy.py`'s
`denoise_future_video_latents()`, the same function the online policy server
calls, via the existing helpers in
`scripts/so101_pipeline/archive/video_path_experiments/exact_video_path.py`.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ahawam.runtime import build_datasets  # noqa: E402
from ahawam.utils.video_io import save_mp4  # noqa: E402
from scripts.so101_pipeline.archive.video_path_experiments.batch.batch_exact_video_path import (  # noqa: E402
    _replace_with_online_history,
)
from scripts.so101_pipeline.archive.video_path_experiments.exact_video_path import (  # noqa: E402
    _batch_sample,
    _generate,
    _tensor_video_to_frames,
)
from scripts.so101_pipeline.baseline_video_experiments.common import reporting  # noqa: E402
from scripts.so101_pipeline.baseline_video_experiments.common.sample_selection import (  # noqa: E402
    load_manifest,
)

WAN22_BASE = "wan22-base"

MODE_LABELS = {
    "none": "no history KV",
    "training": "training-cadence history (RoPE stride 12)",
    "online": "deploy-cadence history (RoPE stride 10)",
    "wan_native": "Wan2.2-native (bidirectional mask, contiguous RoPE)",
}


def load_model(cfg: Any, weights: str, device: str) -> Any:
    """Instantiate AHA-WAM, then either keep the Wan2.2 init or load a checkpoint."""
    if weights == WAN22_BASE:
        # The SO101 run config sets `skip_dit_load_from_pretrain: true` because it
        # overrides every weight from `init_checkpoint`. Flip it back so the video
        # expert is populated from the Wan2.2-TI2V-5B release.
        cfg.model.skip_dit_load_from_pretrain = False
        cfg.model.action_dit_pretrained_path = None
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
        source = str(getattr(model, "model_paths", {}).get("video_dit", "unknown"))
        print(f"[model] video expert initialised from Wan2.2 release: {source}", flush=True)
    else:
        checkpoint = Path(weights)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
        model.load_checkpoint(str(checkpoint))
        print(f"[model] loaded checkpoint {checkpoint}", flush=True)
    model.eval()
    if bool(model.video_expert.action_conditioned):
        raise RuntimeError(
            "This preview only runs the video branch, which requires "
            "`video_dit_config.action_conditioned=false`."
        )
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True, help="export_reference_samples.py output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--weights",
        required=True,
        help=f"`{WAN22_BASE}` for the raw Wan2.2 initialisation, or a path to an AHA-WAM .pt",
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--history-modes",
        default="online,training,none",
        help="Comma-separated subset of none,training,online.",
    )
    parser.add_argument("--limit", type=int, default=0, help="0 = every sample in the manifest")
    parser.add_argument(
        "--dataset-dirs",
        nargs="+",
        help="Override `data.train.dataset_dirs`, e.g. to score held-out validation "
             "recordings with the training run's own geometry and norm stats.",
    )
    parser.add_argument(
        "--val-set-proportion",
        type=float,
        help="Override `data.train.val_set_proportion`. Pass 0 to use every episode, "
             "which is what you want when the directories already are the held-out set.",
    )
    args = parser.parse_args()

    modes = [mode.strip() for mode in args.history_modes.split(",") if mode.strip()]
    unknown = sorted(set(modes) - set(MODE_LABELS))
    if unknown:
        parser.error(f"Unknown history modes: {unknown}")

    manifest = load_manifest(args.samples_dir / "manifest.json")
    samples = manifest["samples"]
    if args.limit > 0:
        samples = samples[: args.limit]
    video_fps = float(manifest["video_fps"])

    cfg = OmegaConf.load(args.config)
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

    model = load_model(cfg, args.weights, args.device)
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    sheet_middle: list[tuple[str, Image.Image]] = []
    sheet_last: list[tuple[str, Image.Image]] = []
    started_run = time.perf_counter()

    for order, meta in enumerate(samples):
        tag = str(meta["tag"])
        sample_dir = args.output / tag
        sample_dir.mkdir(parents=True, exist_ok=True)

        raw_sample = dataset[int(meta["sample_index"])]
        ground_truth = _tensor_video_to_frames(raw_sample["video"])
        save_mp4(ground_truth, str(sample_dir / "ground_truth.mp4"), fps=video_fps)

        row: dict[str, Any] = {
            "sample": tag,
            "prompt": str(meta["prompt"]),
            "sample_index": int(meta["sample_index"]),
            "episode": int(meta["source_episode"]),
            "start_frame": int(meta["source_frame"]),
            "current_rope_index": int(meta["current_rope_index"]),
        }
        columns: list[tuple[str, list[Image.Image]]] = [("ground truth", ground_truth)]

        for mode in modes:
            sample = raw_sample
            restore_mask_mode: str | None = None
            if mode == "wan_native":
                # AHA-WAM runs the video expert with a per-frame-causal mask, a
                # history K/V prefix and sparse temporal RoPE positions. Wan2.2 was
                # never trained that way, so scoring its released weights in that
                # configuration measures the configuration, not the weights. This
                # mode puts the expert back into Wan2.2's own setting -- full
                # bidirectional attention over the clip and contiguous RoPE from 0 --
                # which is the fair "what can vanilla Wan2.2 do here" reference.
                restore_mask_mode = str(model.video_expert.video_attention_mask_mode)
                model.video_expert.video_attention_mask_mode = "bidirectional"
                latent_frames = (
                    int(raw_sample["video"].shape[1]) - 1
                ) // int(model.vae.temporal_downsample_factor) + 1
                sample = dict(raw_sample)
                sample["video_temporal_position_ids"] = torch.arange(
                    latent_frames, dtype=torch.long
                )
            elif mode == "online":
                sample = _replace_with_online_history(
                    dataset,
                    int(meta["sample_index"]),
                    int(meta["episode_slot"]),
                    raw_sample,
                    history_count=int(cfg.model.num_history_frames),
                    planner_control_stride=int(cfg.model.action_horizon),
                    video_rope_frame_stride=int(cfg.model.video_rope_frame_stride),
                )
            batched = _batch_sample(sample)
            print(f"[generate] {order + 1}/{len(samples)} {tag} mode={mode}", flush=True)
            try:
                prediction, info = _generate(
                    model,
                    batched,
                    steps=args.steps,
                    seed=args.seed + order,
                    use_history=mode in {"training", "online"},
                    history_mode="training" if mode in {"none", "wan_native"} else mode,
                )
            finally:
                if restore_mask_mode is not None:
                    model.video_expert.video_attention_mask_mode = restore_mask_mode
            metrics = reporting.frame_metrics(
                prediction, ground_truth, regions=reporting.SO101_REGIONS
            )
            save_mp4(prediction, str(sample_dir / f"prediction_{mode}.mp4"), fps=video_fps)
            columns.append((MODE_LABELS[mode], prediction))
            row[f"{mode}_history_frames"] = int(info["history_frames"])
            row[f"{mode}_generation_seconds"] = float(info["generation_seconds"])
            row[f"{mode}_peak_allocated_gib"] = float(info["peak_allocated_gib"])
            row[f"{mode}_history_rope_indices"] = (
                batched["video_history_frame_indices"][0].tolist()
            )
            for key, value in metrics.items():
                row[f"{mode}_{key}"] = value
            del prediction

        comparison = reporting.comparison_frames(columns)
        save_mp4(comparison, str(sample_dir / "comparison.mp4"), fps=video_fps)
        comparison[len(comparison) // 2].save(sample_dir / "comparison_middle.png")
        comparison[-1].save(sample_dir / "comparison_last.png")
        sheet_middle.append((tag, comparison[len(comparison) // 2]))
        sheet_last.append((tag, comparison[-1]))

        # Mirror the primary mode onto unprefixed keys so summaries stay readable.
        primary = modes[0]
        for key in (
            "future_rgb_mse",
            "future_rgb_psnr",
            "fixed_rgb_psnr",
            "wrist_rgb_psnr",
            "pred_temporal_abs_diff",
            "target_temporal_abs_diff",
            "generation_seconds",
            "peak_allocated_gib",
        ):
            source_key = f"{primary}_{key}"
            if source_key in row:
                row[key] = row[source_key]

        (sample_dir / "metrics.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        rows.append(row)
        del comparison, ground_truth
        gc.collect()
        torch.cuda.empty_cache()

    reporting.contact_sheet(sheet_middle, args.output / "contact_sheet_middle.png")
    reporting.contact_sheet(sheet_last, args.output / "contact_sheet_last.png")

    run_meta: dict[str, Any] = {
        "run_name": args.run_name,
        "model_family": "ahawam-video-branch",
        "weights": args.weights,
        "config": str(args.config),
        "samples_dir": str(args.samples_dir),
        "dataset_dirs": list(cfg.data.train.dataset_dirs),
        "history_modes": modes,
        "primary_history_mode": modes[0],
        "primary_prediction_file": f"prediction_{modes[0]}.mp4",
        "num_inference_steps": args.steps,
        "seed": args.seed,
        "video_fps": video_fps,
        "num_video_frames": int(manifest["num_video_frames"]),
        "frame_size": [int(manifest["frame_height"]), int(manifest["frame_width"])],
        "wall_seconds": round(time.perf_counter() - started_run, 1),
    }
    mean_keys = ["future_rgb_mse", "future_rgb_psnr", "fixed_rgb_psnr", "wrist_rgb_psnr",
                 "pred_temporal_abs_diff", "target_temporal_abs_diff", "generation_seconds"]
    mean_keys += [f"{mode}_{key}" for mode in modes for key in
                  ("future_rgb_psnr", "fixed_rgb_psnr", "wrist_rgb_psnr",
                   "pred_temporal_abs_diff", "generation_seconds")]
    reporting.save_summary(args.output, rows, run_meta=run_meta, mean_keys=mean_keys)


if __name__ == "__main__":
    main()
