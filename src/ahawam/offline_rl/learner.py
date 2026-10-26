"""IQL chunk adaptation and TD3+BC comparison on frozen features.

IQL extracts a fixed-variance Gaussian mean via advantage-weighted MSE.
It never evaluates a generated action in its value backup. TD3+BC is an
explicit separate comparison, including target smoothing and delayed updates.
"""
from __future__ import annotations

import copy
import math
import os
from pathlib import Path
from dataclasses import asdict, dataclass

import torch
from torch.nn import functional as F

from .networks import ChunkActor, NetworkConfig, StateValue, TwinChunkQ


@dataclass(frozen=True)
class LearnerConfig:
    algorithm: str = "iql"
    lr: float = 3e-4
    tau: float = .005
    expectile: float = .7
    inverse_temperature: float = 3.0
    max_weight: float = 20.0
    reference_penalty: float = 0.0
    reference_dropout: float = .1
    actor_delay: int = 2
    td3_bc_alpha: float = .1
    target_noise: float = .1
    noise_clip: float = .2
    max_grad_norm: float = 10.0

    def __post_init__(self):
        if self.algorithm not in {"iql", "td3_bc", "bc"}:
            raise ValueError("Unknown algorithm")
        if not 0 < self.expectile < 1 or self.actor_delay < 1 or not 0 < self.tau <= 1:
            raise ValueError("Invalid expectile, actor_delay or tau")
        if self.max_weight <= 0 or self.inverse_temperature < 0:
            raise ValueError("Invalid advantage weighting")


def masked_mse(pred, target, valid):
    mask = valid[..., None].to(pred.dtype)
    return ((pred-target).square() * mask).sum((1, 2)) / (valid.sum(1) * pred.shape[-1]).clamp_min(1)


def canonical_tail(action, valid):
    """Repeat last real target on an absorbing terminal prefix, never zeros."""
    last = valid.sum(1).long() - 1
    tail = action[torch.arange(len(action), device=action.device), last]
    return torch.where(valid[..., None], action, tail[:, None])


def expectile_loss(diff, expectile):
    return (torch.where(diff > 0, expectile, 1-expectile) * diff.square()).mean()


class OfflineLearner:
    def __init__(self, network: NetworkConfig, config: LearnerConfig, device="cpu"):
        self.network_config, self.config = network, config
        self.device, self.steps = torch.device(device), 0
        self.actor = ChunkActor(network).to(device)
        self.critic = TwinChunkQ(network).to(device)
        self.value = StateValue(network).to(device)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False).eval()
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False).eval()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=config.lr)

    def _step(self, loss, optimizer, module):
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(module.parameters(), self.config.max_grad_norm,
                                             error_if_nonfinite=True)
        optimizer.step()
        return float(norm)

    @torch.no_grad()
    def _soft_update(self, source, target):
        for p, tp in zip(source.parameters(), target.parameters()):
            tp.lerp_(p, self.config.tau)

    def update(self, batch, actor_only=False):
        cfg = self.config
        b = {k: v.to(self.device) for k, v in batch.items()}
        obs = (b["features"], b["feature_valid"], b["state"])
        next_obs = (b["next_features"], b["next_feature_valid"], b["next_state"])
        self.steps += 1
        metrics = {}
        if cfg.algorithm != "bc" and not actor_only:
            if cfg.algorithm == "iql":
                with torch.no_grad():
                    q = torch.stack(self.target_critic(*obs, b["action"])).amin(0)
                value_loss = expectile_loss(q - self.value(*obs), cfg.expectile)
                metrics["value_grad"] = self._step(value_loss, self.value_optimizer, self.value)
                metrics["value_loss"] = float(value_loss.detach())
                with torch.no_grad():
                    y = b["reward"] + b["discount"] * self.value(*next_obs)
            else:
                with torch.no_grad():
                    a_next = self.target_actor(*next_obs, b["next_reference"])
                    noise = (torch.randn_like(a_next)*cfg.target_noise).clamp(-cfg.noise_clip, cfg.noise_clip)
                    a_next = (a_next+noise).clamp(-self.network_config.action_limit, self.network_config.action_limit)
                    y = b["reward"] + b["discount"] * torch.stack(self.target_critic(*next_obs, a_next)).amin(0)
            q1, q2 = self.critic(*obs, b["action"])
            loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
            metrics["critic_grad"] = self._step(loss_q, self.critic_optimizer, self.critic)
            metrics.update(critic_loss=float(loss_q.detach()), q_data=float(torch.minimum(q1,q2).mean().detach()),
                           target_mean=float(y.mean()), q_disagreement=float((q1-q2).abs().mean().detach()),
                           q_outside_0_1=float(((q1<0)|(q1>1)|(q2<0)|(q2>1)).float().mean().detach()))
        if cfg.algorithm == "bc" or actor_only or self.steps % cfg.actor_delay == 0:
            pi = self.actor(*obs, b["reference"], reference_dropout=cfg.reference_dropout)
            bc = masked_mse(pi, b["action"], b["action_valid"])
            if cfg.algorithm == "iql" and not actor_only:
                with torch.no_grad():
                    advantage = torch.stack(self.target_critic(*obs, b["action"])).amin(0) - self.value(*obs)
                    weight = (cfg.inverse_temperature*advantage).clamp(max=math.log(cfg.max_weight)).exp()
                actor_loss = (weight*bc).mean()
                metrics.update(weight_mean=float(weight.mean()), weight_max=float(weight.max()),
                               weight_ess=float(weight.sum().square() / weight.square().sum().clamp_min(1e-8)))
            elif cfg.algorithm == "td3_bc" and not actor_only:
                self.critic.requires_grad_(False)
                try:
                    q_pi = self.critic.first(*obs, canonical_tail(pi, b["action_valid"]))
                    scale = cfg.td3_bc_alpha / q_pi.abs().mean().detach().clamp_min(1e-3)
                    actor_loss = bc.mean() - scale*q_pi.mean()
                    actor_loss += cfg.reference_penalty * masked_mse(pi, b["reference"], b["action_valid"]).mean()
                    metrics["actor_grad"] = self._step(actor_loss, self.actor_optimizer, self.actor)
                finally:
                    self.critic.requires_grad_(True)
                metrics["q_pi"] = float(q_pi.mean().detach())
            else:
                actor_loss = bc.mean()
            if cfg.algorithm != "td3_bc" or actor_only:
                actor_loss += cfg.reference_penalty * masked_mse(pi, b["reference"], b["action_valid"]).mean()
                metrics["actor_grad"] = self._step(actor_loss, self.actor_optimizer, self.actor)
            metrics.update(actor_loss=float(actor_loss.detach()), bc_mse=float(bc.mean().detach()))
        if not actor_only and (cfg.algorithm == "iql" or (cfg.algorithm == "td3_bc" and self.steps % cfg.actor_delay == 0)):
            self._soft_update(self.critic, self.target_critic)
            if cfg.algorithm == "td3_bc" and self.steps % cfg.actor_delay == 0:
                self._soft_update(self.actor, self.target_actor)
        return metrics

    def save(self, path, provenance):
        path = Path(path)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        torch.save(dict(format="ahawam_offline_learner/v1", network=asdict(self.network_config),
                        config=asdict(self.config), steps=self.steps, provenance=provenance,
                        models={k: getattr(self,k).state_dict() for k in
                                ("actor", "critic", "value", "target_actor", "target_critic")},
                        optimizers={k: getattr(self,k+"_optimizer").state_dict() for k in ("actor","critic","value")},
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), temporary)
        os.replace(temporary, path)

    @classmethod
    def load(cls, path, device="cpu"):
        data = torch.load(path, map_location=device, weights_only=True)
        learner = cls(NetworkConfig(**data["network"]), LearnerConfig(**data["config"]), device)
        for k, v in data["models"].items():
            getattr(learner,k).load_state_dict(v)
        for k, v in data["optimizers"].items():
            getattr(learner,k+"_optimizer").load_state_dict(v)
        learner.steps = data["steps"]
        torch.set_rng_state(data["rng"].cpu())
        if torch.cuda.is_available() and data.get("cuda_rng"):
            torch.cuda.set_rng_state_all([s.cpu() for s in data["cuda_rng"]])
        return learner, data["provenance"]
