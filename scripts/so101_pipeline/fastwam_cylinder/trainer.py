"""Weights-only FastWAM trainer, bounded checkpoint retention and two val sets."""
from __future__ import annotations

import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from ahawam.trainer import Wan22Trainer
from scripts.so101_pipeline.full_finetune_zero2_offload.zero2_trainer import Zero2WeightOnlyTrainer


class FastWAMTrainer(Zero2WeightOnlyTrainer):
    def _init_wandb(self):
        super()._init_wandb()
        if self.wandb_run is not None:
            self.wandb_run.define_metric("global_step")
            self.wandb_run.define_metric("*",step_metric="global_step")

    def _wandb_log(self,payload):
        if self.wandb_run is not None:
            # Validation and train can arrive separately at the same optimizer
            # step. A custom x-axis prevents W&B dropping the second payload.
            self.wandb_run.log(dict(global_step=self.global_step,**payload))

    @staticmethod
    def _apply_dit_only_train_mode(model,*,freeze_video_dit=False):
        Wan22Trainer._apply_dit_only_train_mode(model,freeze_video_dit=freeze_video_dit)
        # Native FastWAM does not use AHA's compatibility embedding.
        if hasattr(model.mot,"action_branch_embedding"):
            model.mot.action_branch_embedding.requires_grad_(False)

    def evaluate(self):
        original=self.val_dataset;np_state=np.random.get_state();py_state=random.getstate()
        metrics={}
        try:
            with torch.random.fork_rng(devices=[self.accelerator.device]):
                for name,ds in (("demo",original),("human",getattr(self,"correction_val_dataset",None))):
                    if ds is None:continue
                    self.val_dataset=ds
                    seed=73000+self.accelerator.process_index
                    torch.manual_seed(seed);np.random.seed(seed);random.seed(seed)
                    values=self._averaged_val_metrics()
                    metrics.update({name+"_"+k:float(v) for k,v in values.items()})
        finally:
            self.val_dataset=original;np.random.set_state(np_state);random.setstate(py_state)
            self._set_dit_only_train_mode()
        metrics["val_loss"]=metrics.get("demo_loss",0.)
        if self.accelerator.is_main_process:
            with (Path(self.output_dir)/"validation.jsonl").open("a") as f:
                f.write(json.dumps(dict(step=self.global_step,**metrics))+"\n")
            print("VALIDATION "+json.dumps(dict(step=self.global_step,**metrics)),flush=True)
        return metrics

    def _save_weights_checkpoint(self,step_tag):
        model=self.accelerator.unwrap_model(self.model)
        path=Path(self.weights_dir)/f"{step_tag}.pt"
        temporary=path.with_suffix(".tmp.pt")
        model.save_checkpoint(temporary,optimizer=None,step=self.global_step)
        temporary.replace(path)
        for old in Path(self.weights_dir).glob("step_*.pt"):
            if old!=path:old.unlink()
        return str(path)

    def save_checkpoint(self):
        if self.zero2_save_training_state:raise ValueError("Optimizer-state saves disabled by experiment contract")
        # The base loop calls save twice when the final step is also periodic.
        if getattr(self,"_last_saved_step",None)==self.global_step:
            return dict(weights_path=str(Path(self.weights_dir)/f"step_{self.global_step:06d}.pt")
                if self.accelerator.is_main_process else None,state_path=None)
        result=super().save_checkpoint()
        self._last_saved_step=self.global_step
        if self.accelerator.is_main_process:
            for path in Path(self.checkpoint_root).glob("step_*_trainer_state.json"):
                if path.name!=f"step_{self.global_step:06d}_trainer_state.json":path.unlink()
        return result

    def _update_profile_window(self,**kwargs):
        super()._update_profile_window(**kwargs)
        # Separate cheap machine-readable progress, including the first update.
        if self.accelerator.is_main_process:
            from ahawam.offline_rl.cache import atomic_json,memory_stats
            atomic_json(Path(self.output_dir)/"progress.json",dict(step=self.global_step,max_steps=self.max_steps,
                time_unix=time.time(),memory=memory_stats(),timing=kwargs))
