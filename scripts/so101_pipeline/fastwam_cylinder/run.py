"""Train or audit one native FastWAM cylinder stage using resolved YAML."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ahawam.runtime import build_datasets
from ahawam.utils import misc
from ahawam.utils.logging_config import setup_logging
from scripts.so101_pipeline.fastwam_cylinder.trainer import FastWAMTrainer


def build_data(cfg):
    train,val=build_datasets(cfg.data,cfg.model)
    human_val=instantiate(cfg.correction_val) if cfg.get("correction_val") else None
    if human_val is not None:human_val.configure_video_history_memory(num_history_frames=0)
    return train,val,human_val


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--data-only",action="store_true")
    p.add_argument("--smoke-steps",type=int,default=0)
    args=p.parse_args();cfg=OmegaConf.load(args.config)
    if cfg.resume is not None or cfg.zero2_save_training_state:
        raise ValueError("Use only init_checkpoint and model-weight saves")
    if cfg.data.train.max_action_offset!=0 or cfg.data.train.num_history_frames!=0:
        raise ValueError("Native FastWAM uses current observation, no AHA offset/history")
    if args.smoke_steps:
        cfg.output_dir=str(Path(cfg.output_dir).parent/"smoke_model")
        cfg.max_steps=args.smoke_steps;cfg.save_every=0;cfg.save_at_end=False;cfg.eval_every=0
        cfg.wandb.enabled=False
    Path(cfg.output_dir).mkdir(parents=True,exist_ok=True)
    misc.register_work_dir(cfg.output_dir)
    train,val,human_val=build_data(cfg)
    lengths=dict(train=len(train),demo_val=len(val),human_val=len(human_val) if human_val else 0,
        demo_train=getattr(train,"demo_length",len(train)),human_train=getattr(train,"correction_length",0))
    if args.data_only:
        probes=[("train",train,0),("demo_val",val,0)]
        if hasattr(train,"demo_length"):probes.append(("human_train",train,train.demo_length))
        if human_val is not None:probes.append(("human_val",human_val,0))
        for name,ds,index in probes:
            sample=ds[index]
            if sample["video"].shape!=(3,13,384,320) or sample["action"].shape!=(48,6):
                raise ValueError(f"Unexpected native FastWAM sample shape: {name}")
            if any(k in sample for k in ("chunk_obs_images","history_video")):
                raise ValueError("AHA-only conditions leaked into native FastWAM dataset")
            lengths[name+"_sample"]={k:list(v.shape) for k,v in sample.items() if isinstance(v,torch.Tensor)}
        audit=Path(cfg.output_dir)/"data_audit.json"
        audit.write_text(json.dumps(lengths,indent=2)+"\n")
        print(json.dumps(lengths,indent=2));return
    if int(os.environ.get("WORLD_SIZE","1"))!=3:raise ValueError("Use three GPU processes")
    torch.set_num_threads(2);local=int(os.environ.get("LOCAL_RANK","0"));torch.cuda.set_device(local)
    torch.manual_seed(int(cfg.seed));np.random.seed(int(cfg.seed))
    setup_logging(is_main_process=local==0)
    if local==0:OmegaConf.save(cfg,Path(cfg.output_dir)/"config.yaml",resolve=True)
    started=time.perf_counter()
    model=instantiate(cfg.model,model_dtype=torch.bfloat16,device=f"cuda:{local}")
    trainer=FastWAMTrainer(model,train,val,cfg=cfg)
    trainer.correction_val_dataset=human_val
    if local==0:
        params={name:sum(p.numel() for p in module.parameters() if p.requires_grad)
            for name,module in [("video",model.video_expert),("action",model.action_expert),("proprio",model.proprio_encoder)]}
        (Path(cfg.output_dir)/"initialization.json").write_text(json.dumps(dict(
            **model.initialization_report,trainable_parameters=params,dataset=lengths,
            optimizer_state_saved=False,load_seconds=time.perf_counter()-started),indent=2)+"\n")
    trainer.train()
    if not args.smoke_steps:
        final_metrics=trainer.evaluate()
        trainer._wandb_log({f"eval/{key}":value for key,value in final_metrics.items()})
        trainer._finish_wandb()
    if trainer.accelerator.is_main_process:
        from ahawam.offline_rl.cache import atomic_json,memory_stats
        atomic_json(Path(cfg.output_dir)/"complete.json",dict(step=trainer.global_step,
            wall_seconds=time.perf_counter()-started,memory=memory_stats(),smoke=bool(args.smoke_steps),
            optimizer_saved=False))
    trainer.accelerator.wait_for_everyone()


if __name__=="__main__":main()
