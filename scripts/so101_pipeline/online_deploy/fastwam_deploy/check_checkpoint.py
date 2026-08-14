#!/usr/bin/env python3
"""Inspect a WAM checkpoint without allocating its tensors in RAM or VRAM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--expected-action-dim", type=int)
    parser.add_argument("--dataset-stats", type=Path)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    payload = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if "mot" not in payload:
        raise ValueError("Checkpoint has no 'mot' state dict.")

    mot = payload["mot"]
    encoder_key = "mixtures.action.action_encoder.weight"
    head_key = "mixtures.action.head.weight"
    if encoder_key not in mot or head_key not in mot:
        raise ValueError("Checkpoint is missing the FastWAM ActionDiT input/output layers.")

    encoder_shape = tuple(mot[encoder_key].shape)
    head_shape = tuple(mot[head_key].shape)
    action_dim = int(encoder_shape[1])
    proprio_weight = payload.get("proprio_encoder", {}).get("weight")
    proprio_dim = None if proprio_weight is None else int(proprio_weight.shape[1])
    aha_only = [
        key
        for key in ("action_obs_visual_proj", "chunk_obs_query_encoder")
        if key in payload
    ]

    print(f"checkpoint: {args.checkpoint}")
    print(f"size_bytes: {args.checkpoint.stat().st_size}")
    print(f"step: {payload.get('step')}")
    print(f"torch_dtype: {payload.get('torch_dtype')}")
    print(f"mot_tensor_count: {len(mot)}")
    print(f"action_encoder_shape: {encoder_shape}")
    print(f"action_head_shape: {head_shape}")
    print(f"action_dim: {action_dim}")
    print(f"proprio_dim: {proprio_dim}")
    print(f"has_action_branch_embedding: {'action_branch_embedding' in mot}")
    print(f"aha_only_payload_keys: {aha_only}")

    if head_shape[0] != action_dim:
        raise ValueError(
            f"Action input/output dimensions disagree: encoder={action_dim}, head={head_shape[0]}"
        )
    if args.expected_action_dim is not None and action_dim != args.expected_action_dim:
        raise SystemExit(
            f"Refusing to continue: checkpoint is {action_dim}-D, but this service "
            f"expects {args.expected_action_dim}-D actions."
        )

    if args.dataset_stats is not None:
        stats = json.loads(args.dataset_stats.read_text(encoding="utf-8"))
        stats_dims = {}
        for group in ("state", "action"):
            values = stats[group]["default"]["global_mean"]
            stats_dims[group] = len(values)
        print(f"dataset_stats: {args.dataset_stats}")
        print(f"dataset_stats_dims: {stats_dims}")
        if stats_dims["action"] != action_dim or stats_dims["state"] != proprio_dim:
            raise SystemExit(
                "Refusing to continue: checkpoint and dataset_stats dimensions disagree."
            )


if __name__ == "__main__":
    main()
