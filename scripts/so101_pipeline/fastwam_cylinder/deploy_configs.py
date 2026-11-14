"""Publish SO101 configs for the four final native FastWAM weights."""
from pathlib import Path

from omegaconf import OmegaConf


def main():
    root=Path("runs/fastwam_cylinder_20261105")
    destination=Path("scripts/so101_pipeline/online_deploy/deployments")
    names=["franka_demo_sft","franka_action_only","franka_full_vidext","robocoin_demo_sft"]
    for i,name in enumerate(names):
        train=OmegaConf.load(root/"configs"/f"{name}.yaml")
        model=OmegaConf.to_container(train.model,resolve=True)
        model["allow_interface_adaptation"]=False;model["mot_checkpoint_mixed_attn"]=False
        model["video_dit_config"]["use_gradient_checkpointing"]=False
        model["action_dit_config"]["use_gradient_checkpointing"]=False
        data=train.data.train
        config=dict(checkpoint_path=str(Path(train.output_dir)/"checkpoints/weights"/f"step_{int(train.max_steps):06d}.pt"),
            dataset_stats_path=str(data.pretrained_norm_stats),text_embedding_cache_dir=str(data.text_embedding_cache_dir),
            load_text_encoder=False,project_root=".",hydra_config_name="train",task=None,
            hydra_overrides=["data=robotwin","model=fastwam"],
            hydra_patch=dict(model=model,data=dict(train=dict(shape_meta=OmegaConf.to_container(data.shape_meta),
                num_frames=49,action_video_freq_ratio=4,video_size=[384,320],
                processor=OmegaConf.to_container(data.processor,resolve=True)))),
            serving=dict(gpu_id=i,port=12140+i,host="127.0.0.1",action_dim=6,max_return_actions=8,
                policy_module="deploy.server.ahawam_policy",policy_class="FastWAMPolicy",
                instruction="Pick up the cylinder and insert it into the matching hole."),
            device="cuda",mixed_precision="bf16",action_horizon=48,num_inference_steps=10,sigma_shift=5.,
            seed=42,text_cfg_scale=1.,negative_prompt="",rand_device="cpu",tiled=False,warmup_rounds=0,
            video_height=384,video_width=320,input_control_hz=15.,action_video_freq_ratio=4)
        path=destination/f"cylinder_fastwam_{name}.yaml"
        OmegaConf.save(config,path)
        print(path)


if __name__=="__main__":main()
