"""Generate native FastWAM SO101 SFT + two corrections + RoboCOIN SFT control."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from omegaconf import OmegaConf

from scripts.so101_pipeline.config_utils import load_training_config


FRANKA="checkpoints/FastWAM-Franka-Joint/joint/fastwam/train90_joint_job4415/checkpoints/weights/step_058620.pt"
ROBOCOIN="checkpoints/FastWAM-RoboCOIN/Fast-WAM-pretrained.pt"


def native_dataset(ds):
    ds.max_action_offset=0;ds.action_chunk_size=8;ds.action_horizon=48
    ds.num_history_frames=0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",type=Path,default=Path("runs/fastwam_cylinder_20261105"))
    args=p.parse_args()
    if args.output.exists():raise ValueError("Use a new experiment directory")
    root=args.output;root.mkdir();(root/"configs").mkdir()
    demo=load_training_config("scripts/so101_pipeline/full_finetune_zero2_offload/settings_cylinder_v1.yaml")
    model=OmegaConf.load("configs/model/fastwam.yaml")
    model._target_="scripts.so101_pipeline.fastwam_cylinder.model.create_model"
    model.allow_interface_adaptation=True
    model.skip_dit_load_from_pretrain=True;model.action_dit_pretrained_path=None
    model.loss.lambda_video=1.0
    demo.model=model
    demo.resume=None;demo.base_init_checkpoint=None;demo.zero2_save_training_state=False
    demo.wandb.enabled=True
    demo.wandb.project="fastwam-so101-cylinder";demo.wandb.group="cylinder-20261105"
    demo.batch_size=4;demo.gradient_accumulation_steps=1;demo.num_workers=2
    demo.save_at_end=True;demo.eval_num_val_batches=4;demo.eval_num_action_batches=0
    demo.history_aware_batching=False
    for ds in [demo.data.train,demo.data.val]:native_dataset(ds)
    frames=sum(json.loads(s)["length"] for d in demo.data.train.dataset_dirs
        for s in (Path(d)/"meta/episodes.jsonl").read_text().splitlines() if s.strip())
    # Preserve the original three-GPU batch=4 budget for comparable data exposure.
    sft_steps=math.ceil(frames/12)*4
    p0=root/"franka_demo_sft/model/checkpoints/weights"/f"step_{sft_steps:06d}.pt"
    cfgs={}
    for name,weight in [("franka_demo_sft",FRANKA),("robocoin_demo_sft",ROBOCOIN)]:
        c=copy.deepcopy(demo);c.init_checkpoint=weight;c.max_steps=sft_steps
        c.save_every=math.ceil(frames/12);c.eval_every=c.save_every
        c.output_dir=str(root/name/"model");c.experiment=dict(name=name,expected_world_size=3,
            training_scope="full",initialization_only=True)
        c.wandb.name=name
        cfgs[name]=c
    source=OmegaConf.load("runs/ovcr_adapter_cylinder_hil_v2_full/configs/p1_full_balanced_video_hilv2_vidext.yaml")
    for name,action_only in [("franka_action_only",True),("franka_full_vidext",False)]:
        c=copy.deepcopy(source);c.model=copy.deepcopy(model)
        c.model.allow_interface_adaptation=False;c.model.loss.lambda_video=0. if action_only else 1.
        c.freeze_video_dit=action_only;c.init_checkpoint=str(p0);c.resume=None;c.base_init_checkpoint=None
        c.zero2_save_training_state=False;c.save_at_end=True;c.save_every=c.max_steps
        c.num_workers=2;c.eval_every=max(1,int(c.max_steps)//4)
        c.eval_num_val_batches=4;c.eval_num_action_batches=0
        c.wandb.enabled=True;c.wandb.project="fastwam-so101-cylinder"
        c.wandb.group="cylinder-20261105";c.wandb.name=name
        c.output_dir=str(root/name/"model");c.history_aware_batching=False
        for ds in [c.data.train,c.data.val,c.correction_val]:native_dataset(ds)
        c.data.train.correction_history_crosses_segment=False
        c.data.train.correction_history_skip_system_hold=False
        c.correction_val.history_crosses_segment=False;c.correction_val.history_skip_system_hold=False
        # Both branches use the same video-extended data view; only joint branch
        # uses its eligible video labels. HUMAN action boundaries remain strict.
        c.data.train.correction_future_extension="video"
        c.experiment=dict(name=name,expected_world_size=3,training_scope="action_only" if action_only else "full",
            source_ahawam_config="runs/ovcr_adapter_cylinder_hil_v2_full/configs/p1_full_balanced_video_hilv2_vidext.yaml")
        cfgs[name]=c
    for name,c in cfgs.items():OmegaConf.save(c,root/"configs"/f"{name}.yaml",resolve=True)
    manifest=dict(date="2026-11-05",stages=list(cfgs),sft_steps=sft_steps,
        correction_steps=int(source.max_steps),demo_frames=frames,
        demo_train_dirs=list(demo.data.train.dataset_dirs),demo_val_dirs=list(demo.data.val.dataset_dirs),
        correction_train_selection=OmegaConf.to_container(source.data.train.correction_episode_selection),
        correction_val_selection=OmegaConf.to_container(source.correction_val.episode_selection),
        differences=dict(history="Native FastWAM has no history input; disabled rather than silently ignored",
            action_horizon="native 48 actions / 13 video frames, no asynchronous offset; AHA references used 40 actions + offset 0..7",
            supervision="Both correction datasets extend video past HUMAN on successful rollouts only; action targets stay HUMAN",
            initialization="Franka joint epoch20 main line; RoboCOIN reproduction independent demo-SFT control",
            optimizer="Fresh AdamW/scheduler each stage, init_checkpoint only; save latest weights only"))
    (root/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=="__main__":main()
