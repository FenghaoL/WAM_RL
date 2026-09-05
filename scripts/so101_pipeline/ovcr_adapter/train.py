"""Train native OVCR adapters using the existing boundary-safe HIL dataset."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from scripts.so101_pipeline.config_utils import load_training_config


def build_config(args):
    cfg = load_training_config(args.settings)
    cfg.model.ovcr_adapter = OmegaConf.load(args.variant)
    cfg.model.ovcr_anchor_weight = args.anchor_weight
    cfg.model.loss.lambda_video = 0.0
    cfg.freeze_video_dit = True
    cfg.model.checkpoint_shape_adapt = False
    cfg.init_checkpoint = args.init_checkpoint
    cfg.base_init_checkpoint = None
    cfg.resume = None
    cfg.output_dir = args.output_dir
    cfg.learning_rate = args.learning_rate
    cfg.max_steps = args.steps
    cfg.save_every = args.save_every
    cfg.eval_every = getattr(args, "eval_every", 250)
    cfg.seed = getattr(args, "seed", 42)
    cfg.save_at_end = True
    cfg.zero2_save_training_state = False
    cfg.wandb.enabled = False
    cfg.num_workers = args.num_workers
    if args.sampling:
        cfg.data.train.sampling_strategy = args.sampling
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if not str(cfg.data.train._target_).endswith("SO101DemoHumanCorrectionDataset"):
        raise ValueError("This entry point requires the audited demo + HUMAN dataset")
    if cfg.data.train.sampling_strategy == "balanced_2_2" and cfg.batch_size != 4:
        raise ValueError("balanced_2_2 requires per-device batch_size=4")
    if not Path(args.init_checkpoint).is_file():
        raise FileNotFoundError(args.init_checkpoint)
    if args.steps <= 0 or args.save_every <= 0 or cfg.eval_every < 0 or args.anchor_weight < 0 or args.learning_rate <= 0:
        raise ValueError("Invalid steps/save_every/anchor_weight/learning_rate")
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", default="scripts/so101_pipeline/hil_human_only_training/settings_mixed_from5988.yaml")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--init-checkpoint", required=True, help="A task-finetuned 6D SO101 P0 checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--eval-every", type=int, default=250, help="0 disables periodic held-out evaluation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--anchor-weight", type=float, default=1.0)
    parser.add_argument("--sampling", choices=("natural", "balanced_2_2"), default="balanced_2_2")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Resolve/write configuration without loading model or running training")
    args = parser.parse_args()
    cfg = build_config(args)
    output = Path(args.output_dir)
    if int(os.environ.get("RANK", "0")) == 0:
        output.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output / "config.yaml", resolve=True)
        print(json.dumps(dict(mode=cfg.model.ovcr_adapter.mode, init_checkpoint=cfg.init_checkpoint,
                              steps=cfg.max_steps, sampling=cfg.data.train.sampling_strategy,
                              anchor_weight=cfg.model.ovcr_anchor_weight, dry_run=args.dry_run)))
    if args.dry_run:
        return
    import torch
    from hydra.utils import instantiate
    from ahawam.runtime import build_datasets, _mixed_precision_to_model_dtype, _normalize_mixed_precision
    from ahawam.models.wan22.ovcr_adapter import freeze_for_adapter_training
    from ahawam.utils.logging_config import setup_logging
    from scripts.so101_pipeline.full_finetune_zero2_offload.zero2_trainer import Zero2WeightOnlyTrainer

    class AdapterTrainer(Zero2WeightOnlyTrainer):
        @staticmethod
        def _apply_dit_only_train_mode(model, *, freeze_video_dit=False):
            freeze_for_adapter_training(model)

    if not torch.cuda.is_available():
        raise RuntimeError("Full AHA training requires a working CUDA driver; use --dry-run for CPU preflight")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    # Seed before constructing zero-up/down-random adapters on every rank.
    torch.manual_seed(int(cfg.seed))
    setup_logging(is_main_process=int(os.environ.get("RANK", "0")) == 0)
    model = instantiate(cfg.model, device=f"cuda:{local_rank}",
                        model_dtype=_mixed_precision_to_model_dtype(_normalize_mixed_precision(cfg.mixed_precision)))
    freeze_for_adapter_training(model)
    print("adapter_trainable_parameters=", sum(p.numel() for p in model.parameters() if p.requires_grad))
    train_ds, val_ds = build_datasets(cfg.data, cfg.model)
    trainer = AdapterTrainer(cfg=cfg, model=model, train_dataset=train_ds, val_dataset=val_ds)
    trainer.train()


if __name__ == "__main__":
    main()
