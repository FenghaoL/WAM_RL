"""RLT-inspired readout for cached AHA observation/routing token sequences.

This reader does not write to OVCR, predict actions, or implement online RL.
Unlike paper RLT's full-width token, we optionally project to a smaller width.
"""
from __future__ import annotations

import torch
from torch import nn


class RLTokenReadout(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 2, max_tokens: int = 64):
        super().__init__()
        if min(input_dim, hidden_dim, num_heads, num_layers, max_tokens) <= 0 or hidden_dim % num_heads:
            raise ValueError("Dimensions must be positive and hidden_dim divisible by num_heads")
        self.config = dict(input_dim=input_dim, hidden_dim=hidden_dim, num_heads=num_heads,
                           num_layers=num_layers, max_tokens=max_tokens)
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.randn(1, max_tokens + 1, hidden_dim) * .02)
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim) * .02)
        def transformer():
            block = nn.TransformerEncoderLayer(hidden_dim, num_heads, 4 * hidden_dim,
                                               dropout=0.0, batch_first=True, norm_first=True)
            return nn.TransformerEncoder(block, num_layers, enable_nested_tensor=False)
        self.encoder = transformer()
        self.decoder = transformer()
        self.output = nn.Linear(hidden_dim, input_dim)

    def _validate(self, tokens, valid):
        if tokens.ndim != 3 or tokens.shape[-1] != self.config["input_dim"]:
            raise ValueError("tokens must be [B,M,input_dim]")
        if not 0 < tokens.shape[1] <= self.config["max_tokens"]:
            raise ValueError("Invalid token sequence length")
        if valid.shape != tokens.shape[:2] or valid.dtype != torch.bool:
            raise ValueError("valid must be boolean [B,M], True means valid")
        if not valid.any(dim=1).all():
            raise ValueError("Each sample must have at least one valid token")

    def encode(self, tokens, valid):
        self._validate(tokens, valid)
        # NaN padding must not contaminate attention or the reconstruction loss.
        target = tokens.detach().masked_fill(~valid[..., None], 0)
        embedded = self.projection(target)
        seq = torch.cat((embedded, self.query.expand(tokens.shape[0], -1, -1)), dim=1)
        padding = torch.cat((~valid, torch.zeros_like(valid[:, :1])), dim=1)
        return self.encoder(seq + self.position[:, :seq.shape[1]], src_key_padding_mask=padding)[:, -1]

    def forward(self, tokens, valid):
        z = self.encode(tokens, valid)
        target = tokens.detach().masked_fill(~valid[..., None], 0)
        seq = torch.cat((z[:, None], self.projection(target[:, :-1])), dim=1)
        # Decoder input position i contains target i-1, so padding must shift too.
        padding = torch.cat((torch.zeros_like(valid[:, :1]), ~valid[:, :-1]), dim=1)
        length = seq.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=seq.device).triu(1)
        predicted = self.output(self.decoder(seq + self.position[:, :length], mask=causal,
                                             src_key_padding_mask=padding))
        mse = (predicted.float() - target.float()).square().mean(-1)
        loss = (mse * valid).sum() / valid.sum()
        return loss, z, predicted
