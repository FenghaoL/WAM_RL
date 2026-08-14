#!/usr/bin/env python3
"""Numerically check that the preview really is the training/deploy video path.

Three claims are tested on one real SO101 window with real weights:

1. `video_expert.pre_dit` ignores the action tensor, because the released
   configuration sets `action_conditioned=false`. Training passes `action=...`
   there, the preview passes `action=None`; if the claim holds, the two calls
   produce identical tokens, freqs, t_mod and context.

2. The joint MoT step used in training, `forward_prior_action_with_chunk_updated_kv`,
   returns the same video tokens as the video-only `forward_video_with_history`
   the preview and the deployed policy call. Two different random action tensors
   are pushed through the joint call to show the video half does not move.

3. The history K/V cache the training loop builds in one shot
   (`prefill_video_cache` over all history frames with a causal frame mask) is
   the same tensor the deployed policy accumulates one observation at a time
   (`prefill_video_cache_with_prefix`). If it is, the online preview differs
   from training only in *which* frames it feeds and at which RoPE positions,
   not in the arithmetic.

bf16 matmuls are not bit-reproducible across different reduction orders, so the
report prints max/mean absolute differences and compares them to the tensor
scale rather than asserting exact equality.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ahawam.runtime import build_datasets  # noqa: E402
from deploy.server.ahawam_policy import predict_future_video_flow  # noqa: E402
from scripts.so101_pipeline.archive.video_path_experiments.exact_video_path import (  # noqa: E402
    _batch_sample,
)
from scripts.so101_pipeline.baseline_video_experiments.wan22_base.run_video_branch_preview import (  # noqa: E402
    load_model,
)


def _diff(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a32 = a.detach().float()
    b32 = b.detach().float()
    delta = (a32 - b32).abs()
    scale = a32.abs().mean().item()
    return {
        "max_abs_diff": float(delta.max().item()),
        "mean_abs_diff": float(delta.mean().item()),
        "reference_mean_abs": round(scale, 6),
        "relative_max": float(delta.max().item() / max(scale, 1e-9)),
        "exactly_equal": bool(torch.equal(a.detach(), b.detach())),
    }


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--sample-index", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.data.train.skip_padding_as_possible = False
    dataset, _ = build_datasets(cfg.data, cfg.model)
    sample = _batch_sample(dataset[args.sample_index])
    model = load_model(cfg, args.weights, args.device)

    # `MoT.configure_chunk_kv_cache_editor` picks its dtype from `next(mot.parameters())`,
    # which is the float32 `action_branch_embedding`, so a plain `instantiate()` leaves the
    # editor in float32 while every other expert is bf16. Training never notices because
    # accelerate casts the whole module. Claim 2 below drives the action branch, so cast
    # here the way the trainer would. See PATH_AUDIT_ZH.md section 8.
    model.mot.chunk_kv_cache_editor.to(dtype=model.torch_dtype)
    model.mot.action_branch_embedding.data = model.mot.action_branch_embedding.data.to(
        dtype=model.torch_dtype
    )

    inputs = model.build_inputs(sample)
    input_latents = inputs["input_latents"]
    batch_size = int(input_latents.shape[0])
    context = inputs["context"]
    context_mask = inputs["context_mask"]
    action = inputs["action"]

    generator = torch.Generator(device="cpu").manual_seed(1234)
    noise = torch.randn(
        input_latents.shape, generator=generator, dtype=torch.float32
    ).to(device=model.device, dtype=input_latents.dtype)
    timestep_video = torch.full(
        (batch_size,), 500.0, device=model.device, dtype=input_latents.dtype
    )
    latents = model.train_video_scheduler.add_noise(input_latents, noise, timestep_video)
    latents[:, :, 0:1] = inputs["first_frame_latents"]

    temporal_position_ids = model._get_main_video_temporal_position_ids(
        sample, batch_size=batch_size, current_clip_latent_frames=int(input_latents.shape[2])
    )

    report: dict[str, Any] = {
        "config": str(args.config),
        "weights": args.weights,
        "sample_index": args.sample_index,
        "video_action_conditioned": bool(model.video_expert.action_conditioned),
        "video_attention_mask_mode": str(model.video_expert.video_attention_mask_mode),
        "num_history_frames": int(model._configured_num_history_frames()),
        "latent_shape": list(input_latents.shape),
        "context_shape": list(context.shape),
        "temporal_position_ids": temporal_position_ids.reshape(-1).tolist(),
    }

    # --- claim 1: pre_dit ignores the action tensor -------------------------
    pre_with_action = model.video_expert.pre_dit(
        x=latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=action,
        fuse_vae_embedding_in_latents=True,
        temporal_position_ids=temporal_position_ids,
    )
    pre_without_action = model.video_expert.pre_dit(
        x=latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        action=None,
        fuse_vae_embedding_in_latents=True,
        temporal_position_ids=temporal_position_ids,
        clean_prefix_frames=1,
    )
    report["claim_1_pre_dit_ignores_action"] = {
        key: _diff(pre_with_action[key], pre_without_action[key])
        for key in ("tokens", "freqs", "t_mod", "context")
    }
    report["claim_1_context_len_equal"] = bool(
        pre_with_action["context"].shape == pre_without_action["context"].shape
    )

    tokens_per_frame = int(pre_with_action["meta"]["tokens_per_frame"])

    # --- history caches: training one-shot vs deploy incremental ------------
    valid_len = model._get_video_history_valid_len(
        sample, batch_size=batch_size, num_history_frames=model._configured_num_history_frames()
    )
    effective = int(valid_len.max().item())
    history_kv_training = None
    if effective > 0:
        frame_indices = model._get_video_history_frame_indices(
            sample,
            batch_size=batch_size,
            num_history_frames=model._configured_num_history_frames(),
        )[:, -effective:]
        history_latents = model._encode_training_history_latents(
            sample, batch_size=batch_size, num_history_frames=effective, tiled=False
        )
        zero_t = torch.zeros_like(timestep_video)
        history_pre = model.video_expert.pre_dit(
            x=history_latents,
            timestep=zero_t,
            context=context,
            context_mask=context_mask,
            action=None,
            temporal_position_ids=frame_indices,
            fuse_vae_embedding_in_latents=True,
            clean_prefix_frames=effective,
        )
        history_kv_training = model.mot.prefill_video_cache(
            video_tokens=history_pre["tokens"],
            video_freqs=history_pre["freqs"],
            video_t_mod=history_pre["t_mod"],
            video_context_payload={
                "context": history_pre["context"],
                "mask": history_pre["context_mask"],
            },
            video_attention_mask=model._expand_frame_mask_to_token_mask(
                frame_mask=model._build_history_video_frame_mask(
                    num_history_frames=effective,
                    valid_history_len=valid_len,
                    device=model.device,
                ),
                tokens_per_frame=tokens_per_frame,
            ),
        )

        # deploy-style: one observation at a time, each attending to all prior ones
        entries: list[list[dict[str, torch.Tensor]]] = []
        for offset in range(effective):
            frame_pre = model.video_expert.pre_dit(
                x=history_latents[:, :, offset : offset + 1],
                timestep=zero_t,
                context=context,
                context_mask=context_mask,
                action=None,
                temporal_position_ids=frame_indices[:, offset : offset + 1],
                fuse_vae_embedding_in_latents=True,
                clean_prefix_frames=1,
            )
            seq_len = int(frame_pre["tokens"].shape[1])
            frame_mask = model.video_expert.build_video_to_video_mask(
                video_seq_len=seq_len,
                video_tokens_per_frame=int(frame_pre["meta"]["tokens_per_frame"]),
                device=model.device,
            )
            payload = {"context": frame_pre["context"], "mask": frame_pre["context_mask"]}
            if entries:
                prefix = model._concat_history_kv_entries(entries)
                prefix_len = int(prefix[0]["k"].shape[1])
                frame_mask = torch.cat(
                    [
                        torch.ones((seq_len, prefix_len), dtype=torch.bool, device=model.device),
                        frame_mask,
                    ],
                    dim=1,
                )
                current = model.mot.prefill_video_cache_with_prefix(
                    video_tokens=frame_pre["tokens"],
                    video_freqs=frame_pre["freqs"],
                    video_t_mod=frame_pre["t_mod"],
                    video_context_payload=payload,
                    prefix_video_kv_cache=prefix,
                    prefix_video_seq_len=prefix_len,
                    video_attention_mask=frame_mask,
                )
            else:
                current = model.mot.prefill_video_cache(
                    video_tokens=frame_pre["tokens"],
                    video_freqs=frame_pre["freqs"],
                    video_t_mod=frame_pre["t_mod"],
                    video_context_payload=payload,
                    video_attention_mask=frame_mask,
                )
            entries.append([{"k": l["k"].detach(), "v": l["v"].detach()} for l in current])
        history_kv_online = model._concat_history_kv_entries(entries)

        report["claim_3_history_prefill_one_shot_vs_incremental"] = {
            "history_frames": effective,
            "history_rope_indices": frame_indices.reshape(-1).tolist(),
            "layer_0_k": _diff(history_kv_training[0]["k"], history_kv_online[0]["k"]),
            "layer_0_v": _diff(history_kv_training[0]["v"], history_kv_online[0]["v"]),
            "layer_last_k": _diff(history_kv_training[-1]["k"], history_kv_online[-1]["k"]),
            "layer_last_v": _diff(history_kv_training[-1]["v"], history_kv_online[-1]["v"]),
        }
    else:
        report["claim_3_history_prefill_one_shot_vs_incremental"] = "skipped: no history frames"

    # --- claim 2: the joint MoT step leaves the video half alone ------------
    (
        visual_obs_context,
        visual_obs_mask,
        proprio_context,
        proprio_mask,
    ) = model._split_visual_obs_and_proprio_context(
        obs_context=inputs["obs_context"], obs_context_mask=inputs["obs_context_mask"]
    )
    chunk_queries = model._build_chunk_kv_queries(
        obs_context=visual_obs_context, obs_context_mask=visual_obs_mask
    )

    video_attention_mask = model.video_expert.build_video_to_video_mask(
        video_seq_len=int(pre_with_action["tokens"].shape[1]),
        video_tokens_per_frame=tokens_per_frame,
        device=model.device,
    )
    if history_kv_training is not None:
        prefix_len = int(history_kv_training[0]["k"].shape[1])
        joint_mask = torch.cat(
            [
                torch.ones(
                    (int(pre_with_action["tokens"].shape[1]), prefix_len),
                    dtype=torch.bool,
                    device=model.device,
                ),
                video_attention_mask,
            ],
            dim=1,
        )
    else:
        joint_mask = video_attention_mask

    def joint_video_tokens(action_tokens: torch.Tensor) -> torch.Tensor:
        action_timestep = torch.full(
            action_tokens.shape[:2], 300.0, device=model.device, dtype=action_tokens.dtype
        )
        action_pre = model.action_expert.pre_dit(
            action_tokens=action_tokens,
            timestep=action_timestep,
            context=context,
            context_mask=context_mask,
            obs_context=proprio_context,
            obs_context_mask=proprio_mask,
            clean_action_tokens=None,
            clean_timestep=None,
            chunk_size=model.action_chunk_size,
            noisy_position_offset=0,
            single_branch_chunk_causal=True,
            obs_chunk_offset=0,
            obs_context_causal=True,
            obs_proprio_tokens_per_chunk=model._require_obs_proprio_tokens_per_chunk(),
        )
        out = model.mot.forward_prior_action_with_chunk_updated_kv(
            video_tokens=pre_with_action["tokens"],
            video_freqs=pre_with_action["freqs"],
            video_t_mod=pre_with_action["t_mod"],
            video_context_payload={
                "context": pre_with_action["context"],
                "mask": pre_with_action["context_mask"],
            },
            video_attention_mask=joint_mask,
            action_tokens=action_pre["tokens"],
            action_freqs=action_pre["freqs"],
            action_t_mod=action_pre["t_mod"],
            action_context_payload={
                "context": action_pre["context"],
                "mask": action_pre["context_mask"],
            },
            chunk_queries=chunk_queries,
            video_tokens_per_frame=tokens_per_frame,
            action_chunk_size=model.action_chunk_size,
            history_video_kv_cache=history_kv_training,
        )
        return model.video_expert.post_dit(out["video"], pre_with_action)

    joint_a = joint_video_tokens(torch.randn_like(action))
    joint_b = joint_video_tokens(torch.randn_like(action) * 3.0)
    preview = predict_future_video_flow(
        model=model,
        latents=latents,
        timestep=timestep_video,
        context=context,
        context_mask=context_mask,
        temporal_position_ids=temporal_position_ids,
        history_video_kv_cache=history_kv_training,
    )

    report["claim_2_joint_vs_video_only"] = {
        "joint_with_two_different_action_tensors": _diff(joint_a, joint_b),
        "joint_vs_preview_predict_future_video_flow": _diff(joint_a, preview),
        "flow_prediction_shape": list(preview.shape),
    }

    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
