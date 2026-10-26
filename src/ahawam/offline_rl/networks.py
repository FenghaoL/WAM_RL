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
    depth: int = 2
    shared_critic: bool = False

    def __post_init__(self):
        if min(self.feature_dim,self.state_dim,self.action_dim,self.chunk_size,self.width,self.num_heads,self.max_tokens)<=0:
            raise ValueError("Network dimensions must be positive")
        if self.width % self.num_heads or self.architecture not in {"mlp","attention","raw_kv"} or self.action_limit<=0 or self.depth < 1:
            raise ValueError("Invalid attention width, architecture or action limit")
        if self.architecture == "raw_kv" and self.feature_dim % 2:
            raise ValueError("raw_kv features must concatenate equal-width K and V")


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
        if cfg.architecture == "raw_kv":
            self.raw_encoder = KVDecoder(cfg)
            self.raw_head = mlp(cfg.width, cfg.width, cfg.action_dim)
            nn.init.zeros_(self.raw_head[-1].weight)
            nn.init.zeros_(self.raw_head[-1].bias)
            return
        self.encoder = StateEncoder(cfg)
        self.head = mlp(self.encoder.output_dim + cfg.chunk_size * cfg.action_dim, cfg.width,
                        cfg.chunk_size * cfg.action_dim)

    def forward(self, features, valid, state, reference, reference_dropout=0.0):
        ref = reference
        if self.training and reference_dropout > 0:
            keep = torch.rand(len(ref), 1, 1, device=ref.device) >= reference_dropout
            ref = ref * keep
        if self.cfg.architecture == "raw_kv":
            x = self.raw_head(self.raw_encoder(features, valid, state, ref)[:, 1:])
            # Zero initialization starts at the frozen policy; direct deterministic
            # output with an unconstrained learned residual and hard action bounds.
            return (ref + x).clamp(-self.cfg.action_limit, self.cfg.action_limit)
        z = self.encoder(features, valid, state)
        x = self.head(torch.cat([z, ref.flatten(1)], dim=-1))
        return self.cfg.action_limit * torch.tanh(x.view(-1, self.cfg.chunk_size, self.cfg.action_dim))


class ChunkQ(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.architecture == "raw_kv":
            self.raw_encoder = KVDecoder(cfg)
            self.head = mlp((cfg.chunk_size+1)*cfg.width, cfg.width, 1)
            return
        self.encoder = StateEncoder(cfg)
        # Flattening retains temporal order. Reference is actor-only, matching
        # the RLT interface; Q does not consume future labels or source masks.
        self.head = mlp(self.encoder.output_dim + cfg.chunk_size * cfg.action_dim, cfg.width, 1)

    def forward(self, features, valid, state, action):
        if self.cfg.architecture == "raw_kv":
            return self.head(self.raw_encoder(features, valid, state, action).flatten(1)).squeeze(-1)
        z = self.encoder(features, valid, state)
        return self.head(torch.cat([z, action.flatten(1)], dim=-1)).squeeze(-1)


class TwinChunkQ(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.shared = cfg.shared_critic and cfg.architecture == "raw_kv"
        if self.shared:
            self.encoder = KVDecoder(cfg)
            self.head1 = mlp((cfg.chunk_size+1)*cfg.width, cfg.width, 1)
            self.head2 = mlp((cfg.chunk_size+1)*cfg.width, cfg.width, 1)
        else:
            self.q1, self.q2 = ChunkQ(cfg), ChunkQ(cfg)

    def first(self, features, valid, state, action):
        if self.shared:
            return self.head1(self.encoder(features,valid,state,action).flatten(1)).squeeze(-1)
        return self.q1(features,valid,state,action)

    def forward(self, features, valid, state, action):
        if self.shared:
            z = self.encoder(features,valid,state,action).flatten(1)
            return self.head1(z).squeeze(-1), self.head2(z).squeeze(-1)
        return self.q1(features, valid, state, action), self.q2(features, valid, state, action)


class StateValue(nn.Module):
    def __init__(self, cfg: NetworkConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.architecture == "raw_kv":
            self.raw_encoder = KVDecoder(cfg, state_only=True)
            self.head = mlp(cfg.width, cfg.width, 1)
            return
        self.encoder = StateEncoder(cfg)
        self.head = mlp(self.encoder.output_dim, cfg.width, 1)

    def forward(self, features, valid, state):
        if self.cfg.architecture == "raw_kv":
            return self.head(self.raw_encoder(features,valid,state)[:,0]).squeeze(-1)
        return self.head(self.encoder(features, valid, state)).squeeze(-1)


class KVBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.self_norm = nn.LayerNorm(cfg.width)
        self.cross_norm = nn.LayerNorm(cfg.width)
        self.self_attn = nn.MultiheadAttention(cfg.width,cfg.num_heads,batch_first=True)
        self.cross_attn = nn.MultiheadAttention(cfg.width,cfg.num_heads,batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(cfg.width),nn.Linear(cfg.width,4*cfg.width),
                                nn.GELU(),nn.Linear(4*cfg.width,cfg.width))

    def forward(self,x,k,v,valid):
        z = self.self_norm(x)
        x = x + self.self_attn(z,z,z,need_weights=False)[0]
        x = x + self.cross_attn(self.cross_norm(x),k,v,key_padding_mask=~valid,need_weights=False)[0]
        return x + self.ff(x)


class KVDecoder(nn.Module):
    """Small learned decoder over frozen last-layer K/V, without denoising.

    The 3072-channel planner does not force a 3072-channel actor: K and V
    have separate learned input projections shared by the decoder blocks.
    Ordered action queries and spatial positions remain distinguishable.
    """
    def __init__(self,cfg,state_only=False):
        super().__init__()
        d = cfg.feature_dim//2
        self.cfg,self.state_only = cfg,state_only
        self.k_proj = nn.Sequential(nn.LayerNorm(d),nn.Linear(d,cfg.width))
        self.v_proj = nn.Sequential(nn.LayerNorm(d),nn.Linear(d,cfg.width))
        self.s_proj = nn.Linear(cfg.state_dim,cfg.width)
        self.a_proj = nn.Linear(cfg.action_dim,cfg.width)
        self.position = nn.Parameter(torch.randn(1,cfg.max_tokens,cfg.width)*.02)
        self.queries = nn.Parameter(torch.randn(1,1 if state_only else cfg.chunk_size+1,cfg.width)*.02)
        self.blocks = nn.ModuleList(KVBlock(cfg) for _ in range(cfg.depth))
        self.norm = nn.LayerNorm(cfg.width)

    def forward(self,features,valid,state,action=None):
        if features.ndim!=3 or features.shape[-1]!=self.cfg.feature_dim or features.shape[1]>self.cfg.max_tokens:
            raise ValueError("Invalid raw KV shape")
        if valid.dtype!=torch.bool or not valid.any(-1).all():
            raise ValueError("Invalid raw KV mask")
        k,v = features.detach().float().chunk(2,dim=-1)
        k,v = self.k_proj(k)+self.position[:,:k.shape[1]],self.v_proj(v)
        x = self.queries.expand(len(state),-1,-1) + self.s_proj(state)[:,None]
        if not self.state_only:
            x = torch.cat([x[:,:1],x[:,1:]+self.a_proj(action)],dim=1)
        for block in self.blocks:
            x = block(x,k,v,valid)
        return self.norm(x)
