"""External actor API; does not contact or command a robot."""
import torch

from .networks import ChunkActor, NetworkConfig


class FrozenKVActor:
    def __init__(self,checkpoint,device="cuda",base_checkpoint_sha256=None):
        # Serving needs only the actor. Do not construct critics/optimizers or
        # restore a training RNG state in a running policy service.
        saved=torch.load(checkpoint,map_location="cpu",weights_only=True)
        if saved["format"]!="ahawam_offline_learner/v1":
            raise ValueError("Unsupported offline actor checkpoint")
        self.provenance=saved["provenance"]
        self.steps=saved["steps"]
        self.algorithm=saved["config"]["algorithm"]
        self.actor=ChunkActor(NetworkConfig(**saved["network"]))
        self.actor.load_state_dict(saved["models"]["actor"])
        self.actor=self.actor.to(device).eval().requires_grad_(False)
        self.spec=self.provenance["feature_spec"]
        if base_checkpoint_sha256 and self.spec["base_checkpoint_sha256"]!=base_checkpoint_sha256:
            raise ValueError("Actor and base WAM checkpoint do not match")
        self.device=device

    @torch.inference_mode()
    def action(self,last_k,last_v,normalized_state,normalized_reference):
        """Batched input, matching cache conditioning and normalization protocol."""
        features=torch.cat([last_k,last_v],dim=-1).to(self.device)
        valid=torch.ones(features.shape[:2],device=self.device,dtype=torch.bool)
        normalized=self.actor(features,valid,normalized_state.to(self.device).float(),
                              normalized_reference.to(self.device).float())
        scale=torch.tensor(self.spec["action_scale"],device=self.device)
        offset=torch.tensor(self.spec["action_offset"],device=self.device)
        return dict(normalized=normalized,absolute_targets=(normalized-offset)/scale)
