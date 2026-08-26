#!/usr/bin/env python3
"""Continue NW2 training from a weight checkpoint with Video DiT frozen."""

from __future__ import annotations

import argparse
import json
import math
import os
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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


class ActionOnlyWeightTrainer(Zero2WeightOnlyTrainer):
    """Train Action DiT and small conditioning modules, not the large KV editor."""

    @staticmethod
    def _apply_dit_only_train_mode(model, *, freeze_video_dit: bool = False):
        if not freeze_video_dit:
            raise ValueError("ActionOnlyWeightTrainer requires freeze_video_dit=true")
        model.eval()
        model.requires_grad_(False)
        dit = model.dit
        mixtures = getattr(dit, "mixtures", None)
        if mixtures is None or "action" not in mixtures or "video" not in mixtures:
            raise ValueError("Expected model.dit.mixtures with video/action experts")

        # Keep the MoT container in train mode for its checkpointed action path, but
        # only optimize the Action expert and the tiny compatibility embedding.
        dit.train()
        action_expert = mixtures["action"]
        video_expert = mixtures["video"]
        action_expert.train()
        action_expert.requires_grad_(True)
        video_expert.eval()
        video_expert.requires_grad_(False)
        action_branch_embedding = getattr(dit, "action_branch_embedding", None)
        if isinstance(action_branch_embedding, nn.Parameter):
            action_branch_embedding.requires_grad_(True)

        # This layerwise editor is roughly billion-scale. It remains active in the
        # forward path, initialized from the source checkpoint, but is not updated.
        chunk_editor = getattr(dit, "chunk_kv_cache_editor", None)
        if chunk_editor is not None:
            chunk_editor.eval()
            chunk_editor.requires_grad_(False)

        returned_modules = model.get_additional_trainable_modules()
        if isinstance(returned_modules, dict):
            extra_modules = list(returned_modules.values())
        elif isinstance(returned_modules, Iterable):
            extra_modules = list(returned_modules)
        elif returned_modules is None:
            extra_modules = []
        else:
            extra_modules = [returned_modules]
        for module in extra_modules:
            module.train()
            module.requires_grad_(True)


def parameter_breakdown(model) -> dict[str, int]:
    dit = model.dit
    extra_modules = model.get_additional_trainable_modules()
    if not isinstance(extra_modules, dict):
        raise TypeError("Expected named additional trainable modules")
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "video_expert_frozen": sum(
            parameter.numel() for parameter in dit.mixtures["video"].parameters()
        ),
        "action_expert": sum(
            parameter.numel() for parameter in dit.mixtures["action"].parameters()
        ),
        "chunk_kv_editor_frozen": sum(
            parameter.numel()
            for parameter in dit.chunk_kv_cache_editor.parameters()
        ),
        **{
            f"extra_{name}": sum(parameter.numel() for parameter in module.parameters())
            for name, module in extra_modules.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True)
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--init-checkpoint", required=True)
    parser.add_argument("--num-epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--smoke-steps", type=int)
    args = parser.parse_args()

    cfg = load_training_config(args.settings)
    cfg.output_dir = args.output_dir
    cfg.wandb.name = args.wandb_name
    cfg.freeze_video_dit = True
    cfg.init_checkpoint = args.init_checkpoint
    cfg.resume = None
    cfg.num_epochs = args.num_epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if cfg.get("base_init_checkpoint") is not None:
        raise ValueError("Continuation must use the selected NW2 checkpoint alone.")
    if not Path(str(cfg.init_checkpoint)).is_file():
        raise FileNotFoundError(cfg.init_checkpoint)

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
        eval_times = int(cfg.get("eval_times_per_epoch", 0) or 0)
        cfg.eval_every = max(1, steps_per_epoch // eval_times) if eval_times else 0
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
            f"total_steps={cfg.max_steps} init={cfg.init_checkpoint} "
            f"freeze_video_dit={cfg.freeze_video_dit} eval_every={cfg.eval_every}",
            flush=True,
        )

    setup_logging(
        is_main_process=torch.distributed.get_rank() == 0
        if torch.distributed.is_initialized()
        else True
    )
    misc.register_work_dir(cfg.output_dir)
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
        cfg.model, model_dtype=model_dtype, device=_resolve_train_device()
    )
    ActionOnlyWeightTrainer._apply_dit_only_train_mode(
        model, freeze_video_dit=True
    )
    breakdown = parameter_breakdown(model)
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[parameters] " + " ".join(
                f"{name}={value}" for name, value in breakdown.items()
            ) + f" trainable_pct={100.0 * breakdown['trainable'] / breakdown['total']:.3f}",
            flush=True,
        )
    train_ds, val_ds = build_datasets(cfg.data, cfg.model)
    trainer = ActionOnlyWeightTrainer(
        cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds
    )
    trainer.train()


if __name__ == "__main__":
    main()
