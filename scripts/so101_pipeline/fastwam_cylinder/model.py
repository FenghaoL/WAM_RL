"""Native FastWAM with SO101 checkpoint adaptation and masked video labels.

No history/OVCR modules are added. Base FastWAM attention and inference remain
unchanged. All non-interface weight mismatches are fatal.
"""
from __future__ import annotations

from pathlib import Path

import torch

from ahawam.models.wan22.fastwam import FastWAM


ACTION_INTERFACE_KEYS={"mixtures.action.action_encoder.weight", "mixtures.action.head.weight", "mixtures.action.head.bias"}


def masked_video_losses(loss,mask):
    if mask is None:return loss
    mask=torch.as_tensor(mask,device=loss.device,dtype=loss.dtype).reshape(-1)
    if mask.shape!=loss.shape:raise ValueError("Video supervision mask batch mismatch")
    return loss*mask*(mask.numel()/mask.sum().clamp_min(1))


def adapted_state(source, target, allow_interface_adaptation):
    filtered={};skipped={}
    for key,value in source.items():
        if key not in target:
            raise ValueError(f"Unexpected FastWAM tensor: {key}")
        if value.shape!=target[key].shape:
            if not allow_interface_adaptation or key not in ACTION_INTERFACE_KEYS:
                raise ValueError(f"Incompatible FastWAM tensor {key}: {tuple(value.shape)} vs {tuple(target[key].shape)}")
            skipped[key]=dict(source=list(value.shape),target=list(target[key].shape))
        else:filtered[key]=value
    missing=set(target)-set(filtered)
    allowed=set(skipped)|{"action_branch_embedding"}
    if missing-allowed:raise ValueError(f"Missing backbone tensors: {sorted(missing-allowed)[:20]}")
    return filtered,skipped


class SO101FastWAM(FastWAM):
    def load_checkpoint(self,path,optimizer=None):
        if optimizer is not None:raise ValueError("This experiment loads weights only")
        # GPU load uses ordinary CPU storage: directly copying mmap-backed
        # network-filesystem pages to CUDA causes very slow concurrent page
        # faults. Shape-only audits keep mmap; this temporary RAM is released
        # before DeepSpeed constructs optimizer state.
        payload=torch.load(path,map_location="cpu",weights_only=True,mmap=False)
        allow=bool(getattr(self,"allow_interface_adaptation",False))
        state,skipped=adapted_state(payload["mot"],self.mot.state_dict(),allow)
        self.mot.load_state_dict(state,strict=False)
        proprio=payload.get("proprio_encoder")
        current=self.proprio_encoder.state_dict()
        # An 8D Franka proprio or 14D dual-arm vector does not share SO101 joint
        # semantics. Reinitialize the whole interface, even a same-shaped bias.
        proprio_adapt=proprio is None or any(k not in proprio or v.shape!=proprio[k].shape for k,v in current.items())
        if proprio_adapt:
            if not allow:raise ValueError("Proprio interface mismatch in a same-task checkpoint")
        else:self.proprio_encoder.load_state_dict(proprio,strict=True)
        self.initialization_report=dict(checkpoint=str(Path(path).resolve()),source_step=payload.get("step"),
            loaded_mot_tensors=len(state),reinitialized_action_tensors=skipped,
            proprio_reinitialized=proprio_adapt,optimizer_loaded=False)
        print("FASTWAM_INIT "+str(self.initialization_report),flush=True)
        return {"step":payload.get("step")}

    def training_loss(self,sample,tiled=False):
        self._supervision_mask=sample.get("video_supervision_mask")
        try:return super().training_loss(sample,tiled=tiled)
        finally:self._supervision_mask=None

    def _compute_video_loss_per_sample(self,**kwargs):
        loss=super()._compute_video_loss_per_sample(**kwargs)
        mask=getattr(self,"_supervision_mask",None)
        # Native FastWAM reduces mean afterwards; rescale so only eligible
        # samples determine the video-loss mean, matching the AHA recipe.
        return masked_video_losses(loss,mask)


def create_model(*,allow_interface_adaptation=False,**kwargs):
    from ahawam.runtime import create_fastwam
    model=create_fastwam(**kwargs)
    model.__class__=SO101FastWAM
    model.allow_interface_adaptation=bool(allow_interface_adaptation)
    return model
