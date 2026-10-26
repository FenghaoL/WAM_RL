"""Advantage-weighted FM and frozen, causal conditions for the native Action DiT."""
from __future__ import annotations

import math
from pathlib import Path
import json

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .cache import conditioning_frames
from .data import sha256


class LoRALinear(nn.Module):
    """Frozen BF16 projection plus FP32 low-rank parameters/optimizer state."""
    def __init__(self, base: nn.Linear, rank=16, alpha=16.):
        super().__init__()
        if rank < 1 or alpha <= 0:
            raise ValueError("Invalid LoRA rank/alpha")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        result = self.base(x)
        # Keep the adapter arithmetic FP32 even under a caller's autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            delta = F.linear(F.linear(x.float(), self.lora_a), self.lora_b) * (self.alpha / self.rank)
        return result + delta.to(result.dtype)


def install_action_lora(model, rank=16, alpha=16.):
    """Only Action DiT self-attention q/k/v/o; planner/OVCR/encoders stay frozen."""
    model.requires_grad_(False)
    names = []
    for index, block in enumerate(model.action_expert.blocks):
        for key in ("q", "k", "v", "o"):
            base = getattr(block.self_attn, key)
            if not isinstance(base, nn.Linear):
                raise ValueError(f"Already adapted or unsupported Action DiT projection: {index}/{key}")
            setattr(block.self_attn, key, LoRALinear(base, rank, alpha))
            names.append(f"blocks.{index}.self_attn.{key}")
    return names


def adapter_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def restore_adapter(model, state):
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if set(parameters) != set(state):
        raise ValueError("Adapter parameter names differ")
    if any(p.shape != state[name].shape or not torch.isfinite(state[name]).all() for name, p in parameters.items()):
        raise ValueError("Invalid adapter tensor shape or values")
    with torch.no_grad():
        for name, p in parameters.items():
            p.copy_(state[name])


def load_action_lora(model, checkpoint_path, base_checkpoint_sha256):
    """Attach a saved adapter to an already loaded, matching collector SFT model.

    The caller must supply the SHA256 of the actual base weights, not a filename.
    This does not connect to a robot or alter the planner's parameters.
    """
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if saved["format"] != "ahawam_iql_flow_lora/v1":
        raise ValueError("Unsupported Action DiT adapter checkpoint")
    provenance = saved["provenance"]
    if provenance["base_checkpoint_sha256"] != base_checkpoint_sha256:
        raise ValueError("Adapter base checkpoint mismatch")
    targets = install_action_lora(model, provenance["rank"], provenance["alpha"])
    if targets != provenance["targets"]:
        raise ValueError("Adapter target architecture mismatch")
    restore_adapter(model, saved["adapter"])
    model.eval()
    return dict(step=saved["step"], provenance=provenance)


def weighted_flow_loss(prediction, target, valid, timestep_weight, advantage_weight):
    """One detached weight per clean 8-action data chunk; no Q action gradient."""
    if prediction.shape != target.shape or valid.shape != prediction.shape[:2]:
        raise ValueError("Flow target/mask shape mismatch")
    batch = len(prediction)
    if advantage_weight.shape != (batch,):
        raise ValueError("Expected exactly one advantage weight per chunk")
    error = (prediction.float() - target.float()).square().mean(-1)
    mask = valid.to(error.dtype)
    if (mask.sum(-1) == 0).any():
        raise ValueError("Empty action chunk")
    if timestep_weight.ndim == 0:
        timestep_weight = timestep_weight.expand(batch)
    if timestep_weight.shape == (batch,):
        error = error * timestep_weight[:, None]
    elif timestep_weight.shape == valid.shape:
        error = error * timestep_weight
    else:
        raise ValueError("Timestep weight shape mismatch")
    per_chunk = (error * mask).sum(-1) / mask.sum(-1)
    if not torch.isfinite(per_chunk).all() or not torch.isfinite(advantage_weight).all() or (advantage_weight < 0).any():
        raise FloatingPointError("Invalid weighted flow loss")
    return (per_chunk * advantage_weight.detach()).mean(), per_chunk.detach()


def action_editor_cache(kv, dtype):
    """Serving casts edited KV to action-Q dtype before attention; match it."""
    return [{key: value.detach().to(dtype=dtype).unsqueeze(1) for key, value in layer.items()} for layer in kv]


def predict_native_flow(model, noisy, timestep, state):
    """Gradient-enabled counterpart of serving's single-chunk flow predictor.

    Reuse serving's attention mask/backend as well as its layers, rather than
    introducing differences between a mask-free distillation kernel and serving.
    """
    pre = model.action_expert.pre_dit(action_tokens=noisy, timestep=timestep,
        context=state["context"], context_mask=state["context_mask"],
        obs_context=state["proprio_context"], obs_context_mask=state["proprio_mask"],
        clean_action_tokens=None, clean_timestep=None, chunk_size=model.action_chunk_size,
        noisy_position_offset=0, single_branch_chunk_causal=True, obs_chunk_offset=0,
        obs_context_causal=model.obs_context_causal,
        obs_proprio_tokens_per_chunk=model._require_obs_proprio_tokens_per_chunk())
    kv = [{key: value[:, 0] for key, value in layer.items()} for layer in state["editor_cache"]]
    video_length = int(kv[0]["k"].shape[1])
    attention = model._build_mot_attention_mask(video_seq_len=video_length,
        action_seq_len=int(pre["tokens"].shape[1]), video_tokens_per_frame=video_length,
        device=noisy.device, action_self_attn_mask=model._build_prefilled_action_attention_mask(
            current_action_seq_len=int(pre["tokens"].shape[1]), action_history_seq_len=0,
            chunk_start=0, device=noisy.device))
    tokens = pre["tokens"] + model.mot.action_branch_embedding[1].to(pre["tokens"]).view(1, 1, -1)
    tokens = model.mot.forward_action_with_video_cache(action_tokens=tokens, action_freqs=pre["freqs"],
        action_t_mod=pre["t_mod"], action_context_payload=dict(context=pre["context"], mask=pre["context_mask"]),
        video_kv_cache=kv, attention_mask=attention, video_seq_len=video_length,
        action_history_kv_cache=None, action_history_seq_len=0, cross_attn_kv_cache=None)
    return model.action_expert.post_dit(tokens, pre)


class FrozenConditionProvider:
    """Recompute all layer KV using exactly the last-layer export's history/offset.

    Keep only one episode's video readers and one planner anchor on GPU. Uses
    no_grad rather than inference_mode so frozen tensors can be saved by backward.
    """
    def __init__(self, policy, data, cache):
        self.policy, self.data, self.cache = policy, data, cache
        self.episodes = {e["episode_uid"]: e for e in data["episodes"]}
        self.uid = self.anchor = None
        self.videos = {}
        self.checked_videos = set()
        # Exporter records video digests separately from the shared spec.
        self.video_hashes = {}
        for path in cache.root.glob("done_rank*.json"):
            self.video_hashes.update(json.loads(path.read_text())["video_sha256"])

    def close(self):
        for video in self.videos.values():
            video.close()
        self.videos = {}

    def _open(self, uid):
        import pandas as pd
        from scripts.frame_annotation.annotate import Video
        self.close()
        episode = self.episodes[uid]
        root, ep = Path(episode["root"]), episode["episode_index"]
        info = json.loads((root / "meta/info.json").read_text())
        fields = dict(episode_index=ep, episode_chunk=ep // int(info.get("chunks_size", 1000)))
        path = root / info["data_path"].format(**fields)
        if sha256(path) != self.cache.spec["source_hashes"][uid]:
            raise ValueError("Raw source changed since export")
        self.table = pd.read_parquet(path)
        paths = {cam: root / info["video_path"].format(**fields, video_key=f"observation.images.{cam}") for cam in ("fixed", "wrist")}
        if uid not in self.checked_videos:
            if uid not in self.video_hashes or any(sha256(path) != self.video_hashes[uid][cam] for cam, path in paths.items()):
                raise ValueError("Missing or changed source video provenance")
            self.checked_videos.add(uid)
        self.videos = {cam: Video(path, len(self.table)) for cam, path in paths.items()}
        tasks = {r["task_index"]: r["task"] for r in map(json.loads, (root / "meta/tasks.jsonl").read_text().splitlines())}
        self.policy.set_language_instruction(tasks[int(self.table.task_index.iloc[0])])
        self.prompt = self.policy._DEFAULT_PROMPT.format(task=self.policy.current_instruction)
        self.context, self.context_mask = self.policy._resolve_text_condition(self.prompt)
        self.uid, self.anchor = uid, None

    def _observe(self, frame):
        images = {cam: v.frame(frame) for cam, v in self.videos.items()}
        state = np.array(self.table["observation.state"].iloc[frame], copy=True)
        self.policy.update_observation_window([self.policy.prepare_camera_images(images)], state)
        return self.policy._observation["image_tensor"], self.policy._normalize_state(state)

    @torch.no_grad()
    def prepare(self, record, verify=False):
        if record["episode_uid"] != self.uid:
            self._open(record["episode_uid"])
        spec = self.cache.spec
        frame = record["start_frame"]
        anchor, history = conditioning_frames(self.table, frame, spec["offset_period"], spec["history_count"], spec["history_stride"])
        policy, model = self.policy, self.policy.model
        if anchor != self.anchor:
            policy.reset_obsrvationwindows()
            for prior in [*history, anchor]:
                image, _ = self._observe(prior)
                model._observed_frame_index = int(prior // model.video_rope_frame_stride)
                policy._prefill_episode(prompt=self.prompt, image_tensor=image, context=self.context, context_mask=self.context_mask)
            self.anchor = anchor
        image, proprio = self._observe(frame)
        model._inference_state["next_chunk_index"] = 0
        obs, mask, _ = model._prepare_inference_chunk_conditioning(
            chunk_obs_image=image, chunk_proprio=proprio, chunk_index=0,
            inference_state=model._inference_state, tiled=policy.tiled)
        kv = model._inference_state["_chunk_video_kv_cache"]
        result = dict(editor_cache=action_editor_cache(kv, model.torch_dtype),
            context=model._inference_state["context"].detach(), context_mask=model._inference_state["context_mask"],
            proprio_context=obs.detach(), proprio_mask=mask, batch_size=1)
        if verify:
            saved = self.cache.sample(record["observation_key"])
            errors = {}
            for key in ("k", "v"):
                # Export explicitly quantizes post-OVCR outputs to BF16; some
                # editor arithmetic returns FP32 even with a BF16 backbone.
                actual = kv[-1][key][0].cpu().to(saved[key].dtype).float()
                expected = saved[key].float()
                errors[key] = float((actual - expected).abs().max())
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(proprio[0].cpu().float(), saved["state"], rtol=0, atol=0)
            result["cache_verification"] = errors
        return result
