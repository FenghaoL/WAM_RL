#!/usr/bin/env python3
"""Four-GPU ZeRO-2 offload entry point with exact epoch checkpoints."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ahawam.runtime import (
    _mixed_precision_to_model_dtype,
    _normalize_mixed_precision,
    _resolve_train_device,
    build_datasets,
)
from ahawam.utils import misc
from ahawam.utils.logging_config import setup_logging
from scripts.so101_pipeline.config_utils import load_training_config
from scripts.so101_pipeline.full_finetune_zero2_offload.zero2_trainer import (
    Zero2WeightOnlyTrainer,
)


HERE = Path(__file__).resolve().parent


def selected_frame_count(data_cfg, seed: int) -> tuple[int, int, int]:
    total_frames = 0
    selected_frames = 0
    selected_episodes = 0
    for dataset_dir in data_cfg.dataset_dirs:
        root = Path(str(dataset_dir))
        rows = [
            json.loads(line)
            for line in (root / "meta" / "episodes.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        total_frames += sum(int(row["length"]) for row in rows)
        indices = np.arange(len(rows))
        np.random.default_rng(seed).shuffle(indices)
        split_index = int(len(rows) * (1.0 - float(data_cfg.val_set_proportion)))
        selected = indices[:split_index]
        selected_episodes += len(selected)
        selected_frames += sum(int(rows[int(index)]["length"]) for index in selected)
    return selected_frames, selected_episodes, total_frames


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=str(HERE / "settings.yaml"))
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()

    cfg = load_training_config(args.settings)
    cfg.output_dir = args.output_dir
    cfg.wandb.name = args.wandb_name
    if bool(cfg.freeze_video_dit):
        raise ValueError("Full fine-tuning requires `freeze_video_dit: false`.")

    frames, episodes, total_frames = selected_frame_count(
        cfg.data.train, int(cfg.seed)
    )
    steps_per_epoch = math.ceil(
        math.ceil(frames / (int(cfg.batch_size) * args.num_gpus))
        / int(cfg.gradient_accumulation_steps)
    )
    if args.smoke_steps is None:
        cfg.max_steps = steps_per_epoch * int(cfg.num_epochs)
        cfg.save_every = steps_per_epoch
        cfg.save_at_end = False
    else:
        if args.smoke_steps <= 0:
            raise ValueError("--smoke-steps must be positive")
        cfg.max_steps = args.smoke_steps
        cfg.save_every = args.smoke_steps + 1
        cfg.eval_every = 0
        cfg.save_at_end = False
        cfg.wandb.enabled = False
        cfg.zero2_save_training_state = False

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[schedule] "
            f"train_episodes={episodes} train_frames={frames}/{total_frames} "
            f"num_gpus={args.num_gpus} micro_batch={cfg.batch_size} "
            f"gradient_accumulation={cfg.gradient_accumulation_steps} "
            f"effective_batch={int(cfg.batch_size) * args.num_gpus * int(cfg.gradient_accumulation_steps)} "
            f"steps_per_epoch={steps_per_epoch} epochs={cfg.num_epochs} "
            f"total_steps={cfg.max_steps} init={cfg.init_checkpoint}"
        )

    setup_logging(
        is_main_process=torch.distributed.get_rank() == 0
        if torch.distributed.is_initialized()
        else True
    )
    misc.register_work_dir(cfg.output_dir)
    Path(cfg.output_dir, "config.yaml").write_text(
        OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
    )
    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    model_dtype = _mixed_precision_to_model_dtype(
        _normalize_mixed_precision(cfg.mixed_precision)
    )
    model = instantiate(
        cfg.model, model_dtype=model_dtype, device=_resolve_train_device()
    )
    train_ds, val_ds = build_datasets(cfg.data, cfg.model)
    trainer = Zero2WeightOnlyTrainer(
        cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds
    )
    trainer.train()


if __name__ == "__main__":
    main()

