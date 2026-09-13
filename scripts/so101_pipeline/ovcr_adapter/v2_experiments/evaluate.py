"""Evaluate a saved adapter on larger fixed demo/HUMAN validation subsets."""
import argparse
import json
import os
from pathlib import Path
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from ahawam.runtime import build_datasets, _configure_dataset_history, _mixed_precision_to_model_dtype
from scripts.so101_pipeline.ovcr_adapter.v2_experiments.run import AdapterTrainer, FullTrainer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--loss-samples-per-rank', type=int, default=128)
    p.add_argument('--action-samples-per-rank', type=int, default=16)
    args = p.parse_args()
    if int(os.environ.get('WORLD_SIZE','1')) != 3: raise ValueError('Requires three ranks')
    output = Path(args.output_dir)
    if (output/'validation.jsonl').exists(): raise FileExistsError(output)
    cfg = OmegaConf.load(args.config)
    cfg.init_checkpoint = str(Path(args.checkpoint).resolve())
    cfg.output_dir = str(output)
    cfg.eval_num_val_batches = args.loss_samples_per_rank
    cfg.eval_num_action_batches = args.action_samples_per_rank
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    torch.manual_seed(int(cfg.seed))
    train, demo = build_datasets(cfg.data, cfg.model)
    human = instantiate(cfg.correction_val)
    _configure_dataset_history(human, cfg.model)
    model = instantiate(cfg.model, device=f'cuda:{local}', model_dtype=_mixed_precision_to_model_dtype(cfg.mixed_precision))
    trainer_cls = FullTrainer if cfg.experiment.get('training_scope', 'adapter') == 'full' else AdapterTrainer
    trainer_cls._apply_dit_only_train_mode(model, freeze_video_dit=bool(cfg.freeze_video_dit))
    trainer = trainer_cls(cfg=cfg, model=model, train_dataset=train, val_dataset=demo)
    trainer.correction_val_dataset = human
    trainer.evaluate()
    if trainer.accelerator.is_main_process:
        (output/'evaluation_source.json').write_text(json.dumps(vars(args),indent=2)+'\n')
    trainer.accelerator.wait_for_everyone()
    if torch.distributed.is_initialized(): torch.distributed.destroy_process_group()


if __name__ == '__main__': main()
