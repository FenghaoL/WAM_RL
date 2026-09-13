"""Run a v2 adapter or full-SFT experiment with fixed, separate validation sets."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from ahawam.runtime import build_datasets, _mixed_precision_to_model_dtype, _normalize_mixed_precision
from ahawam.models.wan22.ovcr_adapter import freeze_for_adapter_training
from ahawam.utils.logging_config import setup_logging
from scripts.so101_pipeline.ovcr_adapter.v2_experiments.zero2_trainer import Zero2WeightOnlyTrainer
from scripts.so101_pipeline.ovcr_adapter.v2_experiments.prepare import RUN_ROOT
from ahawam.trainer import Wan22Trainer


class AdapterTrainer(Zero2WeightOnlyTrainer):
    @staticmethod
    def _apply_dit_only_train_mode(model, *, freeze_video_dit=False):
        freeze_for_adapter_training(model)

    def _fixed_val_indices(self):
        training_seed = self.seed
        try:
            self.seed = 42  # validation subset is independent of the training repeat
            return super()._fixed_val_indices()
        finally:
            self.seed = training_seed

    def evaluate(self):
        original = self.val_dataset
        np_state, py_state = np.random.get_state(), random.getstate()
        metrics = {}
        model = self.accelerator.unwrap_model(self.model)
        anchor_weight = model.ovcr_anchor_weight
        model.ovcr_anchor_weight = 0.0  # report supervised validation loss, independent of regularizer
        try:
            with torch.random.fork_rng(devices=[self.accelerator.device]):
                for name, dataset in (("demo", original), ("human", self.correction_val_dataset)):
                    self.val_dataset = dataset
                    seed = 73000 + self.accelerator.process_index
                    torch.manual_seed(seed)
                    np.random.seed(seed)
                    random.seed(seed)
                    metrics.update({f"{name}_{k}": v for k, v in self._averaged_val_metrics().items()})
        finally:
            self.val_dataset = original
            np.random.set_state(np_state)
            random.setstate(py_state)
            model.ovcr_anchor_weight = anchor_weight
            # The frozen backbone is intentionally in eval mode even during training.
            self._set_dit_only_train_mode()
        metrics["val_loss"] = metrics.get("demo_loss", 0.0)
        # The training loop logs its own eval payload; this covers the standalone
        # calls before the first update and after the last one, at the same keys.
        self._wandb_log({f"eval/{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
        if self.accelerator.is_main_process:
            with (Path(self.output_dir) / "validation.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=self.global_step, **metrics)) + "\n")
            print("VALIDATION " + json.dumps(dict(step=self.global_step, **metrics)), flush=True)
        return metrics


class FullTrainer(AdapterTrainer):
    """Reuse full-SFT parameter selection and the two fixed validation sets."""
    @staticmethod
    def _apply_dit_only_train_mode(model, *, freeze_video_dit=False):
        if freeze_video_dit or model.mot.chunk_kv_cache_editor.adapter is not None:
            raise ValueError("Full baseline requires unfrozen video DiT and no adapter")
        Wan22Trainer._apply_dit_only_train_mode(model, freeze_video_dit=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke-steps", type=int, default=0)
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    run_root = Path(cfg.experiment.get("run_root", RUN_ROOT))
    # Variants whose data view differs (e.g. different start filters) keep one manifest each.
    manifest_path = run_root / cfg.experiment.get("manifest_file", "data_manifest.json")
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest() != cfg.experiment.manifest_sha256:
        raise ValueError("Data manifest changed after configuration generation")
    manifest = json.loads(manifest_path.read_text())
    stats = manifest["norm_stats"]
    if hashlib.sha256(Path(stats["path"]).read_bytes()).hexdigest() != stats["sha256"]:
        raise ValueError("P0 normalization statistics changed")
    train, val = build_datasets(cfg.data, cfg.model)
    correction_val = instantiate(cfg.correction_val)
    correction_val.configure_video_history_memory(num_history_frames=int(cfg.model.num_history_frames))
    # demo_only: plain task SFT on the demo sessions; HUMAN data is validation only.
    demo_only = cfg.experiment.get("training_data", "demo_human") == "demo_only"
    human_train = 0 if demo_only else manifest["train"]["human_frames"]
    train_lengths = (len(train), 0) if demo_only else (train.demo_length, train.correction_length)
    expected = (manifest["demo_train_frames"], human_train, manifest["demo_val_frames"], manifest["val"]["human_frames"])
    actual = (*train_lengths, len(val), len(correction_val))
    if actual != expected:
        raise ValueError(f"Dataset size mismatch: {actual} != {expected}")
    print("DATA_LENGTHS " + json.dumps(actual), flush=True)
    if args.data_only:
        probes = [(train, 0)] + ([] if demo_only else [(train, train.demo_length)]) + [(val, 0), (correction_val, 0)]
        for ds, index in probes:
            sample = ds[index]
            print("SAMPLE", index, {k: list(v.shape) for k,v in sample.items() if isinstance(v, torch.Tensor)}, flush=True)
        return
    if int(os.environ.get("WORLD_SIZE", "1")) != cfg.experiment.expected_world_size:
        raise ValueError("This experiment requires exactly three distributed processes")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    torch.manual_seed(int(cfg.seed))
    setup_logging(is_main_process=int(os.environ.get("RANK", "0")) == 0)
    if args.smoke_steps:
        cfg.max_steps = args.smoke_steps
        cfg.output_dir = str(run_root / (cfg.experiment.name + "_smoke") / "model")
        cfg.eval_every = 0
        cfg.save_every = args.smoke_steps
        cfg.log_every = 1
        cfg.eval_num_val_batches = 1
        cfg.eval_num_action_batches = 0
        if cfg.wandb.enabled:  # keep engineering checks out of the formal run name
            cfg.wandb.name = f"{cfg.experiment.name}_smoke"
    if int(os.environ.get("RANK", "0")) == 0:
        Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, Path(cfg.output_dir) / "experiment_config.yaml", resolve=True)
    model = instantiate(cfg.model, device=f"cuda:{local_rank}", model_dtype=_mixed_precision_to_model_dtype(_normalize_mixed_precision(cfg.mixed_precision)))
    trainer_cls = FullTrainer if cfg.experiment.get("training_scope", "adapter") == "full" else AdapterTrainer
    trainer_cls._apply_dit_only_train_mode(model, freeze_video_dit=bool(cfg.freeze_video_dit))
    print("TRAINABLE_PARAMETERS", sum(p.numel() for p in model.parameters() if p.requires_grad), flush=True)
    trainer = trainer_cls(cfg=cfg, model=model, train_dataset=train, val_dataset=val)
    trainer.correction_val_dataset = correction_val
    trainer.evaluate()  # P0 baseline, before the first update
    trainer.train()
    trainer.evaluate()
    trainer.accelerator.wait_for_everyone()
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
