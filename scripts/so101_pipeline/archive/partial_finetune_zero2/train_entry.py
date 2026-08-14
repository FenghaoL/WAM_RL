#!/usr/bin/env python3
"""Launch one run and derive exact epoch boundaries from dataset metadata."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from ahawam.runtime import run_training
from scripts.so101_pipeline.config_utils import load_training_config


HERE = Path(__file__).resolve().parent


def selected_frame_count(data_cfg, seed: int) -> tuple[int, int, int]:
    total_frames = 0
    selected_frames = 0
    selected_episodes = 0
    for dataset_dir in data_cfg.dataset_dirs:
        root = Path(str(dataset_dir))
        rows = [
            json.loads(line)
            for line in (root / "meta" / "episodes.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        total_frames += sum(int(row["length"]) for row in rows)
        indices = np.arange(len(rows))
        np.random.default_rng(seed).shuffle(indices)
        split_index = int(len(rows) * (1.0 - float(data_cfg.val_set_proportion)))
        if bool(data_cfg.is_training_set):
            selected = indices[:split_index]
        else:
            selected = indices[split_index:]
        selected_episodes += len(selected)
        selected_frames += sum(int(rows[int(index)]["length"]) for index in selected)
    return selected_frames, selected_episodes, total_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=str(HERE / "settings.yaml"))
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--smoke-steps", type=int, default=None)
    args = parser.parse_args()

    cfg = load_training_config(args.settings)
    cfg.output_dir = args.output_dir
    cfg.wandb.name = args.wandb_name
    frames, episodes, total_frames = selected_frame_count(cfg.data.train, int(cfg.seed))
    micro_steps = math.ceil(frames / (int(cfg.batch_size) * args.num_gpus))
    steps_per_epoch = math.ceil(micro_steps / int(cfg.gradient_accumulation_steps))

    if args.smoke_steps is None:
        cfg.max_steps = None
        cfg.save_every = steps_per_epoch
        cfg.eval_every = steps_per_epoch
    else:
        if args.smoke_steps <= 0:
            raise ValueError("--smoke-steps must be positive")
        cfg.max_steps = args.smoke_steps
        cfg.save_every = args.smoke_steps + 1
        cfg.eval_every = 0
        cfg.save_at_end = False
        cfg.wandb.enabled = False

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[schedule] "
            f"train_episodes={episodes} train_frames={frames}/{total_frames} "
            f"num_gpus={args.num_gpus} micro_batch={cfg.batch_size} "
            f"gradient_accumulation={cfg.gradient_accumulation_steps} "
            f"steps_per_epoch={steps_per_epoch} epochs={cfg.num_epochs} "
            f"total_steps={steps_per_epoch * int(cfg.num_epochs)}"
        )
    run_training(cfg)


if __name__ == "__main__":
    main()

