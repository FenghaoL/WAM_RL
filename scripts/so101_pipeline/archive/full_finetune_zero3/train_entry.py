#!/usr/bin/env python3
"""SO101 full WAM fine-tuning entry point using the isolated ZeRO-3 trainer."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import types
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
from scripts.so101_pipeline.full_finetune_zero3.aligned_sampler import (
    ZeRO3HistoryAlignedSampler,
)
from scripts.so101_pipeline.full_finetune_zero3.zero3_trainer import (
    Zero3Wan22Trainer,
)


HERE = Path(__file__).resolve().parent


def enable_zero3_mot_parameter_coordination(model) -> int:
    """Coordinate parameters read by custom MoT methods outside child forwards.

    AHA-WAM manually executes transformer block pieces to form mixed attention.
    Consequently, a few parameters are read without calling their owning child
    module. ZeRO-3 needs those parameters registered on the enclosing MoT call.
    """
    import deepspeed

    mot = model.mot
    original_forward = mot.forward
    original_prior_forward = mot.forward_prior_action_with_chunk_updated_kv
    original_video_prefill = mot.prefill_video_cache

    def coordinated_forward(
        self,
        *args,
        zero3_operation=None,
        zero3_operation_kwargs=None,
        **kwargs,
    ):
        if zero3_operation == "prior_action_with_chunk_updated_kv":
            if zero3_operation_kwargs is None:
                raise ValueError("Missing `zero3_operation_kwargs`.")
            return original_prior_forward(**zero3_operation_kwargs)
        if zero3_operation == "prefill_video_cache":
            if zero3_operation_kwargs is None:
                raise ValueError("Missing `zero3_operation_kwargs`.")
            return original_video_prefill(**zero3_operation_kwargs)
        return original_forward(*args, **kwargs)

    def routed_prior_forward(self, **kwargs):
        return self(
            zero3_operation="prior_action_with_chunk_updated_kv",
            zero3_operation_kwargs=kwargs,
        )

    def routed_video_prefill(self, **kwargs):
        return self(
            zero3_operation="prefill_video_cache",
            zero3_operation_kwargs=kwargs,
        )

    mot.forward = types.MethodType(coordinated_forward, mot)
    mot.forward_prior_action_with_chunk_updated_kv = types.MethodType(
        routed_prior_forward, mot
    )
    mot.prefill_video_cache = types.MethodType(routed_video_prefill, mot)

    parameters = [mot.action_branch_embedding]
    for expert in mot.mixtures.values():
        for block in expert.blocks:
            parameters.append(block.modulation)
    editor = mot.chunk_kv_cache_editor
    if editor is not None and editor.delta_gate is not None:
        parameters.append(editor.delta_gate)

    unique_parameters = {id(parameter): parameter for parameter in parameters}
    for parameter in unique_parameters.values():
        deepspeed.zero.register_external_parameter(mot, parameter)
    return len(unique_parameters)


def enable_zero3_history_branch_guard(model) -> None:
    """Fail fast if ranks would take different history-dependent branches."""
    original_training_loss = model.training_loss

    def guarded_training_loss(self, sample, *args, **kwargs):
        valid_len = sample.get("video_history_valid_len")
        effective_len = 0
        if valid_len is not None:
            effective_len = int(torch.as_tensor(valid_len).max().item())
        if torch.distributed.is_initialized():
            device = torch.device("cuda", torch.cuda.current_device())
            local = torch.tensor([effective_len], device=device, dtype=torch.int64)
            gathered = [torch.zeros_like(local) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(gathered, local)
            lengths = [int(value.item()) for value in gathered]
            if len(set(lengths)) != 1:
                raise RuntimeError(
                    "ZeRO-3 requires identical history branches across ranks; "
                    f"got effective history lengths {lengths}. Disable dataset "
                    "sample-index retries and use history-aware batching."
                )
        return original_training_loss(sample, *args, **kwargs)

    model.training_loss = types.MethodType(guarded_training_loss, model)


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
        selected = indices[:split_index]
        selected_episodes += len(selected)
        selected_frames += sum(int(rows[int(index)]["length"]) for index in selected)
    return selected_frames, selected_episodes, total_frames


def run_training(cfg) -> None:
    setup_logging(
        log_level=logging.INFO,
        is_main_process=torch.distributed.get_rank() == 0
        if torch.distributed.is_initialized()
        else True,
    )
    misc.register_work_dir(cfg.output_dir)
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    model_dtype = _mixed_precision_to_model_dtype(
        _normalize_mixed_precision(cfg.mixed_precision)
    )
    model = instantiate(cfg.model, model_dtype=model_dtype, device=_resolve_train_device())
    coordinated_count = enable_zero3_mot_parameter_coordination(model)
    enable_zero3_history_branch_guard(model)
    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[zero3-external-params] "
            f"registered={coordinated_count} owner=MoT "
            "kinds=action_branch_embedding,block_modulation,chunk_editor_delta_gate "
            "coordinated_methods=prior_action,video_history_prefill "
            "history_branch_guard=enabled"
        )
    train_ds, val_ds = build_datasets(cfg.data, cfg.model)

    # This replacement is process-local to the isolated ZeRO-3 entry point.
    # It leaves the repository's default trainer and ZeRO-2 path unchanged.
    import ahawam.trainer as trainer_module

    priority_history_len = cfg.get("zero3_sampler_first_history_len", None)
    ZeRO3HistoryAlignedSampler.priority_history_len = (
        int(priority_history_len) if priority_history_len is not None else None
    )
    trainer_module.HistoryAwareResumableEpochSampler = ZeRO3HistoryAlignedSampler
    if cfg.max_steps is None:
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        expected_world_size = int(cfg.get("zero3_expected_world_size", world_size))
        if world_size != expected_world_size:
            raise RuntimeError(
                "Accelerate WORLD_SIZE does not match --num-gpus: "
                f"{world_size} vs {expected_world_size}"
            )
        schedule_sampler = ZeRO3HistoryAlignedSampler(
            dataset=train_ds,
            seed=int(cfg.seed),
            batch_size=int(cfg.batch_size),
            num_processes=world_size,
        )
        aligned_samples = len(schedule_sampler)
        global_micro_batch = int(cfg.batch_size) * world_size
        micro_steps_per_epoch = math.ceil(aligned_samples / global_micro_batch)
        optimizer_steps_per_epoch = math.ceil(
            micro_steps_per_epoch / int(cfg.gradient_accumulation_steps)
        )
        cfg.max_steps = optimizer_steps_per_epoch * int(cfg.num_epochs)
        cfg.save_every = optimizer_steps_per_epoch
        if int(os.environ.get("RANK", "0")) == 0:
            print(
                "[zero3-aligned-schedule] "
                f"original_samples={len(train_ds)} aligned_samples={aligned_samples} "
                f"repeated_samples={aligned_samples - len(train_ds)} "
                f"optimizer_steps_per_epoch={optimizer_steps_per_epoch} "
                f"total_steps={cfg.max_steps}"
            )

    config_payload = OmegaConf.to_container(cfg, resolve=True)
    with (Path(cfg.output_dir) / "config.yaml").open("w", encoding="utf-8") as handle:
        OmegaConf.save(config_payload, handle)

    trainer = Zero3Wan22Trainer(
        cfg=cfg,
        model=model,
        train_dataset=train_ds,
        val_dataset=val_ds,
    )
    trainer.train()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=str(HERE / "settings.yaml"))
    parser.add_argument("--num-gpus", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--wandb-name", required=True)
    parser.add_argument("--smoke-steps", type=int)
    parser.add_argument("--smoke-save", action="store_true")
    parser.add_argument("--smoke-gradient-accumulation", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--smoke-first-history-len", type=int)
    args = parser.parse_args()

    cfg = load_training_config(args.settings)
    cfg.output_dir = args.output_dir
    cfg.wandb.name = args.wandb_name
    cfg.zero3_expected_world_size = args.num_gpus
    if args.micro_batch_size is not None:
        if args.micro_batch_size <= 0:
            raise ValueError("--micro-batch-size must be positive")
        cfg.batch_size = args.micro_batch_size
    if bool(cfg.freeze_video_dit):
        raise ValueError("Full fine-tuning requires `freeze_video_dit: false`.")

    frames, episodes, total_frames = selected_frame_count(cfg.data.train, int(cfg.seed))
    if args.smoke_steps is not None:
        if args.smoke_steps <= 0:
            raise ValueError("--smoke-steps must be positive")
        cfg.max_steps = args.smoke_steps
        if args.smoke_gradient_accumulation <= 0:
            raise ValueError("--smoke-gradient-accumulation must be positive")
        cfg.gradient_accumulation_steps = args.smoke_gradient_accumulation
        cfg.log_every = 1
        cfg.save_every = args.smoke_steps if args.smoke_save else args.smoke_steps + 1
        cfg.eval_every = 0
        cfg.save_at_end = False
        cfg.wandb.enabled = False
        cfg.zero3_save_portable_weights = bool(args.smoke_save)
        cfg.zero3_save_training_state = False
        if args.smoke_first_history_len is not None:
            if args.smoke_first_history_len < 0:
                raise ValueError("--smoke-first-history-len must be non-negative")
            cfg.zero3_sampler_first_history_len = args.smoke_first_history_len

    micro_steps = math.ceil(frames / (int(cfg.batch_size) * args.num_gpus))
    nominal_steps_per_epoch = math.ceil(
        micro_steps / int(cfg.gradient_accumulation_steps)
    )
    if args.smoke_steps is None:
        cfg.max_steps = None
        cfg.eval_every = 0
        cfg.save_at_end = False

    if int(os.environ.get("RANK", "0")) == 0:
        print(
            "[zero3-schedule] "
            f"train_episodes={episodes} train_frames={frames}/{total_frames} "
            f"num_gpus={args.num_gpus} micro_batch={cfg.batch_size} "
            f"gradient_accumulation={cfg.gradient_accumulation_steps} "
            f"nominal_steps_per_epoch={nominal_steps_per_epoch} epochs={cfg.num_epochs} "
            f"full_wam_finetune={not bool(cfg.freeze_video_dit)} "
            f"portable_weights={bool(cfg.zero3_save_portable_weights)} "
            f"optimizer_state={bool(cfg.zero3_save_training_state)}"
        )
    run_training(cfg)


if __name__ == "__main__":
    main()
