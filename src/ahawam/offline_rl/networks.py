"""Ordered chunk actor and critics; independent trainable state encoders."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class NetworkConfig:
    feature_dim: int
    state_dim: int = 6
    action_dim: int = 6
    chunk_size: int = 8
    width: int = 256
    num_heads: int = 4
    max_tokens: int = 64
    architecture: str = "mlp"
    action_limit: float = 5.0  # Existing z-score normalizer clamps to [-5,5].

    def __post_init__(self):
        if min(self.feature_dim,self.state_dim,self.action_dim,self.chunk_size,self.width,self.num_heads,self.max_tokens)<=0:
            raise ValueError("Network dimensions must be positive")
        if self.width % self.num_heads or self.architecture not in {"mlp","attention"} or self.action_limit<=0:
            raise ValueError("Invalid attention width, architecture or action limit")


def mlp(input_dim: int, width: int, output_dim: int):
    return nn.Sequential(nn.Linear(input_dim, width), nn.LayerNorm(width), nn.SiLU(),
                         nn.Linear(width, width), nn.LayerNorm(width), nn.SiLU(),
                         nn.Linear(width, output_dim))


class StateEncoder(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.architecture == "mlp":
            self.output_dim = cfg.feature_dim + cfg.state_dim
            return
        self.output_dim = cfg.width
        self.projection = nn.Linear(cfg.feature_dim, cfg.width)
        self.position = nn.Parameter(torch.randn(1, cfg.max_tokens, cfg.width) * .02)
        self.query = nn.Parameter(torch.randn(1, 1, cfg.width) * .02)
        self.attention = nn.MultiheadAttention(cfg.width, cfg.num_heads, batch_first=True)
        self.output = mlp(cfg.width + cfg.state_dim, cfg.width, cfg.width)

    def forward(self, features, valid, state):
        if features.ndim != 3 or valid.shape != features.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("Expected features [B,M,D] and boolean valid [B,M]")
        if features.shape[1] > self.cfg.max_tokens or not valid.any(-1).all():
            raise ValueError("Invalid feature length or entirely padded sample")
        x = features.detach().masked_fill(~valid[..., None], 0)
        if self.cfg.architecture == "mlp":
            # M=1 is a frozen RLTokenReadout embedding. Mean pooling for M>1
            # is a deliberately weaker diagnostic, not the RLT bottleneck.
            z = (x * valid[..., None]).sum(1) / valid.sum(1, keepdim=True)
            return torch.cat([z,state],dim=-1)
        x = self.projection(x) + self.position[:, :x.shape[1]]
        z, _ = self.attention(self.query.expand(x.shape[0], -1, -1), x, x,
                              key_padding_mask=~valid, need_weights=False)
        z = z[:, 0]
        return self.output(torch.cat([z, state], dim=-1))


class ChunkActor(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = StateEncoder(cfg)
        self.head = mlp(self.encoder.output_dim + cfg.chunk_size * cfg.action_dim, cfg.width,
                        cfg.chunk_size * cfg.action_dim)

    def forward(self, features, valid, state, reference, reference_dropout=0.0):
        z = self.encoder(features, valid, state)
        ref = reference
        if self.training and reference_dropout > 0:
            keep = torch.rand(len(ref), 1, 1, device=ref.device) >= reference_dropout
            ref = ref * keep
        x = self.head(torch.cat([z, ref.flatten(1)], dim=-1))
        return self.cfg.action_limit * torch.tanh(x.view(-1, self.cfg.chunk_size, self.cfg.action_dim))


class ChunkQ(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.encoder = StateEncoder(cfg)
        # Flattening retains temporal order. Reference is actor-only, matching
        # the RLT interface; Q does not consume future labels or source masks.
        self.head = mlp(self.encoder.output_dim + cfg.chunk_size * cfg.action_dim, cfg.width, 1)

    def forward(self, features, valid, state, action):
        z = self.encoder(features, valid, state)
        return self.head(torch.cat([z, action.flatten(1)], dim=-1)).squeeze(-1)


class TwinChunkQ(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.q1, self.q2 = ChunkQ(cfg), ChunkQ(cfg)

    def forward(self, features, valid, state, action):
        return self.q1(features, valid, state, action), self.q2(features, valid, state, action)


class StateValue(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.encoder = StateEncoder(cfg)
        self.head = mlp(self.encoder.output_dim, cfg.width, 1)

    def forward(self, features, valid, state):
        return self.head(self.encoder(features, valid, state)).squeeze(-1)
