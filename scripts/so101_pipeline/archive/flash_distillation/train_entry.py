#!/usr/bin/env python3
"""Two-GPU SO101 AHA-WAM ODE distillation entry point."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

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
from scripts.so101_pipeline.archive.flash_distillation.config_builder import (
    build_distillation_config,
)
from scripts.so101_pipeline.full_finetune_zero2_offload.train_entry import (
    selected_frame_count,
)
from scripts.so101_pipeline.full_finetune_zero2_offload.zero2_trainer import (
    Zero2WeightOnlyTrainer,
)


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=str(HERE / "settings.yaml"))
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    args = parser.parse_args()

    cfg, settings = build_distillation_config(
        args.settings,
        output_dir=args.output_dir,
        wandb_name=args.wandb_name,
    )
    counts = selected_frame_count(cfg.data.train, int(cfg.seed))
    train_samples = int(counts["natural_frames"])
    micro_steps_per_epoch = math.ceil(
        train_samples / (int(cfg.batch_size) * int(args.num_gpus))
    )
    steps_per_epoch = math.ceil(
        micro_steps_per_epoch / int(cfg.gradient_accumulation_steps)
    )
    cfg.max_steps = steps_per_epoch * int(cfg.num_epochs)
    save_every_epochs = int(settings.training.save_every_epochs)
    cfg.save_every = steps_per_epoch * save_every_epochs
    eval_times = int(settings.training.eval_times_per_epoch)
    cfg.eval_every = 0 if eval_times <= 0 else max(1, steps_per_epoch // eval_times)

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(
            "[flash-schedule] "
            f"samples={train_samples} num_gpus={args.num_gpus} "
            f"micro_batch={cfg.batch_size} grad_accum={cfg.gradient_accumulation_steps} "
            f"effective_batch={int(cfg.batch_size) * args.num_gpus * int(cfg.gradient_accumulation_steps)} "
            f"steps_per_epoch={steps_per_epoch} epochs={cfg.num_epochs} "
            f"max_steps={cfg.max_steps} save_every={cfg.save_every} "
            f"teacher={cfg.model.teacher.checkpoint_path} "
            f"teacher_steps={cfg.model.teacher_num_inference_steps} "
            f"student_steps={cfg.model.student.default_action_inference_steps}"
        )

    setup_logging(
        is_main_process=torch.distributed.get_rank() == 0
        if torch.distributed.is_initialized()
        else rank == 0
    )
    misc.register_work_dir(cfg.output_dir)
    if rank == 0:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        Path(cfg.output_dir, "config.yaml").write_text(
            OmegaConf.to_yaml(cfg, resolve=True), encoding="utf-8"
        )

    if torch.cuda.is_available():
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    model_dtype = _mixed_precision_to_model_dtype(
        _normalize_mixed_precision(cfg.mixed_precision)
    )
    model = instantiate(
        cfg.model,
        model_dtype=model_dtype,
        device=_resolve_train_device(),
    )
    train_ds, val_ds = build_datasets(cfg.data, cfg.model)
    trainer = Zero2WeightOnlyTrainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()


if __name__ == "__main__":
    main()

