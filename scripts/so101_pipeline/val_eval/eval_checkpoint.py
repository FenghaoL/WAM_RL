#!/usr/bin/env python3
"""Evaluate ONE saved checkpoint on the held-out validation set.

Standalone on purpose: no DeepSpeed, no optimizer, one GPU. Run several copies in
parallel (one checkpoint per GPU) via run_all.sh.

Every checkpoint is scored on the SAME fixed, evenly-strided subset of the val set,
so differences between checkpoints are the model changing and nothing else.

Reported per checkpoint:
  val_loss            teacher-forced loss (same objective as training)
  action_l1/action_l2 action error in DE-NORMALISED units, i.e. real robot units
                      -- the number that actually reflects policy quality
  history_len         mean number of history frames in the scored samples

AHAWAM.infer returns actions only (no video), so evaluate_validation returns before
the PSNR/SSIM block -- there are no video-quality metrics for this model.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from hydra.utils import instantiate

from ahawam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    build_datasets,
)
from ahawam.trainer import Wan22Trainer
from ahawam.utils import misc
from ahawam.utils.logging_config import setup_logging
from scripts.so101_pipeline.config_utils import load_training_config


def fixed_indices(n: int, count: int, seed: int = 0) -> list[int]:
    """A fixed random subset of [0, n). Same for every checkpoint.

    A seeded permutation rather than a stride: striding lands on the same phase of
    every episode (index 0 is always an episode start with history_len 0), which
    biases both the loss and the action error.
    """
    count = min(count, n)
    if count <= 0:
        return []
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randperm(n, generator=generator)[:count].tolist()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--settings", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--num-samples", type=int, default=24)
    ap.add_argument("--num-inference-steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    setup_logging(is_main_process=True)
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(out_path.parent))

    cfg = load_training_config(args.settings)
    if cfg.data.get("val") is None:
        raise ValueError("settings must define data.val for validation scoring")

    model_dtype = _mixed_precision_to_model_dtype(
        _normalize_mixed_precision(cfg.mixed_precision)
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    print(f"[eval] checkpoint = {args.checkpoint}", flush=True)
    model = instantiate(cfg.model, model_dtype=model_dtype, device=device)
    model.load_checkpoint(str(args.checkpoint), optimizer=None)
    model.eval()

    _, val_ds = build_datasets(cfg.data, cfg.model)
    indices = fixed_indices(len(val_ds), args.num_samples, seed=args.seed)
    print(f"[eval] val_size={len(val_ds)} scoring {len(indices)} samples", flush=True)

    per_sample: list[dict] = []
    t0 = time.time()
    for n, index in enumerate(indices):
        sample = Wan22Trainer._to_batched_eval_sample(val_ds[index])
        sample["_eval_dataset"] = val_ds
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            metrics = model.evaluate_validation(
                sample,
                eval_num_inference_steps=args.num_inference_steps,
                eval_dir=None,
                global_step=index,
                process_index=0,
            )
        row = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        row["index"] = index
        per_sample.append(row)
        print(f"[eval] {n+1}/{len(indices)} idx={index} "
              f"val_loss={row.get('val_loss', float('nan')):.4f} "
              f"action_l2={row.get('action_l2', float('nan')):.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)

    keys = sorted({k for row in per_sample for k in row if k != "index"})
    summary = {}
    for key in keys:
        vals = [row[key] for row in per_sample if key in row]
        if not vals:
            continue
        summary[key] = {
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
            "n": len(vals),
        }

    payload = {
        "checkpoint": str(args.checkpoint),
        "settings": str(args.settings),
        "val_dataset_dirs": [str(d) for d in cfg.data.val.dataset_dirs],
        "val_size": len(val_ds),
        "num_scored": len(indices),
        "indices": indices,
        "num_inference_steps": args.num_inference_steps,
        "seed": args.seed,
        "elapsed_s": round(time.time() - t0, 1),
        "summary": summary,
        "per_sample": per_sample,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[eval] wrote {out_path} in {payload['elapsed_s']}s", flush=True)


if __name__ == "__main__":
    main()
