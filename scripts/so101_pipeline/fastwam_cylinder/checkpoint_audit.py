"""CPU/meta architecture compatibility audit, without allocating a 6B model."""
from __future__ import annotations

import torch
from omegaconf import OmegaConf

from ahawam.models.wan22.action_dit import ActionDiT
from ahawam.models.wan22.mot import MoT
from ahawam.models.wan22.wan_video_dit import WanVideoDiT
from scripts.so101_pipeline.fastwam_cylinder.model import adapted_state


def audit_checkpoint(config):
    with torch.device("meta"):
        video=WanVideoDiT(**OmegaConf.to_container(config.model.video_dit_config))
        action=ActionDiT(**OmegaConf.to_container(config.model.action_dit_config))
        mot=MoT(dict(video=video,action=action),mot_checkpoint_mixed_attn=False)
    source=torch.load(config.init_checkpoint,map_location="cpu",weights_only=True,mmap=True)
    state,skipped=adapted_state(source["mot"],mot.state_dict(),bool(config.model.allow_interface_adaptation))
    return dict(path=str(config.init_checkpoint),source_step=source.get("step"),
        source_action_dim=int(source["mot"]["mixtures.action.action_encoder.weight"].shape[1]),
        source_proprio_dim=int(source["proprio_encoder"]["weight"].shape[1]),
        target_action_dim=6,target_proprio_dim=6,loaded_mot_tensors=len(state),skipped_action_interfaces=skipped,
        backbone_compatible=True,optimizer_in_source="optimizer" in source)
