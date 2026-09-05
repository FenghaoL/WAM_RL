"""Small, identity-initialized corrections at the native OVCR interface.

These are supervised routing adapters, not the actor-critic from the RLT paper.
No module changes the planner cache in place or adds attention sequence positions.
"""
from __future__ import annotations

import math
from contextlib import contextmanager

import torch
from torch import nn
from torch.nn import functional as F


class ZeroResidual(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, rank: int):
        super().__init__()
        self.down = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, rank), nn.SiLU())
        self.up = nn.Linear(rank, output_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        return self.up(self.down(x))


class TokenWriter(nn.Module):
    """Compress obs/routed queries, then decode spatially to existing KV slots."""
    def __init__(self, query_dim: int, hidden_dim: int, rank: int, num_tokens: int):
        super().__init__()
        self.obs = nn.Sequential(nn.LayerNorm(query_dim), nn.Linear(query_dim, rank))
        self.route = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, rank))
        self.spatial = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, rank))
        self.read_queries = nn.Parameter(torch.randn(num_tokens, rank) / math.sqrt(rank))
        self.up = nn.Linear(rank, 2 * hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, queries, routed, decoded):
        source = torch.cat((self.obs(queries), self.route(routed)), dim=-2)
        weights = torch.softmax(self.read_queries @ source.transpose(-1, -2) / math.sqrt(source.shape[-1]), dim=-1)
        tokens = weights @ source
        spatial = self.spatial(decoded)
        write_weights = torch.softmax(spatial @ tokens.transpose(-1, -2) / math.sqrt(tokens.shape[-1]), dim=-1)
        return self.up(write_weights @ tokens).chunk(2, dim=-1)


class OVCRAdapter(nn.Module):
    MODES = ("query", "value", "token_kv")

    def __init__(self, *, query_dim: int, hidden_dim: int, num_layers: int,
                 mode: str = "value", rank: int = 32, num_tokens: int = 4,
                 layers=None, scale: float = 1.0):
        super().__init__()
        if mode not in self.MODES or rank <= 0 or num_tokens <= 0:
            raise ValueError("Expected mode=query/value/token_kv and positive rank/num_tokens")
        selected = list(range(num_layers)) if layers is None else list(layers)
        if not selected or len(set(selected)) != len(selected) or any(type(i) is not int or i < 0 or i >= num_layers for i in selected):
            raise ValueError("layers must contain distinct valid zero-based layer indices")
        if not math.isfinite(scale) or scale < 0:
            raise ValueError("scale must be finite and nonnegative")
        self.mode, self.scale, self.enabled = mode, float(scale), True
        self.config = dict(mode=mode, rank=rank, num_tokens=num_tokens, layers=selected, scale=float(scale))
        self.blocks = nn.ModuleDict()
        for i in selected:
            if mode == "query":
                block = ZeroResidual(query_dim, query_dim, rank)
            elif mode == "value":
                block = ZeroResidual(hidden_dim, hidden_dim, rank)
            else:
                block = TokenWriter(query_dim, hidden_dim, rank, num_tokens)
            self.blocks[str(i)] = block

    def active(self, layer_idx):
        return self.enabled and self.scale != 0 and str(layer_idx) in self.blocks

    def edit_queries(self, layer_idx, queries):
        if self.mode == "query" and self.active(layer_idx):
            return queries + self.scale * self.blocks[str(layer_idx)](queries)
        return queries

    def edit_cache(self, layer_idx, queries, routed, decoded, keys, values):
        if not self.active(layer_idx) or self.mode == "query":
            return keys, values
        if self.mode == "value":
            return keys, values + self.scale * self.blocks[str(layer_idx)](decoded)
        dk, dv = self.blocks[str(layer_idx)](queries, routed, decoded)
        return keys + self.scale * dk, values + self.scale * dv

    @contextmanager
    def disabled(self):
        previous = self.enabled
        self.enabled = False
        try:
            yield
        finally:
            self.enabled = previous


def configure_adapter(editor, config):
    if editor is None:
        raise ValueError("OVCR editor must be configured before its adapter")
    if getattr(editor, "adapter", None) is not None:
        raise ValueError("An OVCR adapter is already installed")
    parameter = next(editor.parameters())
    editor.adapter = OVCRAdapter(query_dim=editor.query_dim, hidden_dim=editor.attn_hidden_dim,
                                 num_layers=editor.num_layers, **dict(config)).to(parameter)
    return editor.adapter


def freeze_for_adapter_training(model):
    model.eval()
    model.requires_grad_(False)
    adapter = model.mot.chunk_kv_cache_editor.adapter
    if adapter is None:
        raise ValueError("Adapter-only trainer requires model.ovcr_adapter")
    # MoT owns non-reentrant checkpointing. Frozen experts stay in eval mode,
    # but autograd through their operations is still required.
    model.mot.training = True
    adapter.train()
    adapter.requires_grad_(True)
    model.loss_lambda_video = 0.0


def training_loss_with_anchor(model, sample, base_loss, tiled=False):
    """Same-noise P0 velocity distillation on demo samples, plus native HIL BC.

    fork_rng restores sampling state so teacher/student see identical noise,
    offsets and timesteps. Teacher is the same frozen model, adapter bypassed.
    Temporary context is only used by the training thread, never deployment.
    """
    weight = float(getattr(model, "ovcr_anchor_weight", 0.0))
    editor = model.mot.chunk_kv_cache_editor
    adapter = getattr(editor, "adapter", None)
    if weight <= 0 or adapter is None or not adapter.enabled or not torch.is_grad_enabled():
        return base_loss(sample, tiled=tiled)
    if any(p.requires_grad for name, p in model.named_parameters() if ".adapter." not in name):
        raise ValueError("Same-model P0 anchor requires all base parameters frozen")
    source = sample.get("data_source")
    if source is None:
        raise ValueError("Anchor requires data_source: 0=demo, 1=HUMAN")
    devices = sorted({p.device.index for p in model.parameters() if p.is_cuda})
    model._ovcr_anchor_context = {"capture": True}
    try:
        with torch.random.fork_rng(devices=devices), torch.no_grad(), adapter.disabled():
            base_loss(sample, tiled=tiled)
        if "teacher" not in model._ovcr_anchor_context:
            raise RuntimeError("Action flow was not captured for the anchor")
        model._ovcr_anchor_context.update(capture=False, demo_mask=(source == 0), weight=weight)
        loss, metrics = base_loss(sample, tiled=tiled)
        anchor = model._ovcr_anchor_context["anchor"]
        metrics["loss_anchor"] = float(anchor.detach())
        return loss + weight * anchor, metrics
    finally:
        del model._ovcr_anchor_context


def capture_anchor_flow(model, pred, action_is_pad):
    context = getattr(model, "_ovcr_anchor_context", None)
    if context is None:
        return
    if context["capture"]:
        context["teacher"] = pred.detach()
        return
    valid = torch.ones_like(pred[..., 0], dtype=torch.bool) if action_is_pad is None else ~action_is_pad
    valid = valid & context["demo_mask"].to(device=pred.device, dtype=torch.bool).reshape(-1, 1)
    error = F.mse_loss(pred.float(), context["teacher"].float(), reduction="none").mean(-1)
    context["anchor"] = (error * valid).sum() / valid.sum().clamp_min(1)
