#!/usr/bin/env python3
"""Validate deployment paths and 6-D normalization metadata without loading a GPU model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def validate_offline_rl(cfg, repo_root):
    """Sidecar checkpoints are small; the full collector stays checkpoint_path."""
    if "offline_rl" not in cfg:
        return
    import torch
    options = cfg["offline_rl"]
    mode = options["mode"]
    if mode not in {"iql", "td3_bc", "iql_action_dit"}:
        raise ValueError("Invalid offline RL mode")
    path = Path(options["checkpoint_path"])
    if not path.is_absolute():
        path = repo_root / path
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            digest.update(block)
    if digest.hexdigest() != options["checkpoint_sha256"]:
        raise ValueError("Offline RL checkpoint SHA256 mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if mode == "iql_action_dit":
        if saved["format"] != "ahawam_iql_flow_lora/v1" or saved["provenance"]["uniform_weights"]:
            raise ValueError("Expected IQL weighted-FM LoRA")
        spec = saved["provenance"]["feature_spec"]
        steps = saved["step"]
    else:
        if saved["format"] != "ahawam_offline_learner/v1" or saved["config"]["algorithm"] != mode:
            raise ValueError("Offline actor algorithm mismatch")
        if saved["network"]["architecture"] != "raw_kv" or saved["network"]["chunk_size"] != 8:
            raise ValueError("Expected raw-KV 8-action actor")
        spec = saved["provenance"]["feature_spec"]
        steps = saved["steps"]
    if steps != int(options["expected_steps"]) or spec["base_checkpoint_sha256"] != options["base_checkpoint_sha256"]:
        raise ValueError("Wrong RL training step or collector base")
    serving = cfg["serving"]
    if (serving.get("policy_module") != "deploy.server.offline_rl_policy" or serving.get("policy_class") != "OfflineRLPolicy"
        or serving["max_return_actions"] != 8 or options["history_count"] != spec["history_count"]
        or options["history_stride_commands"] != spec["history_stride"]):
        raise ValueError("Offline RL serving interface/history mismatch")
    stats_path = Path(cfg["dataset_stats_path"])
    if not stats_path.is_absolute():
        stats_path = repo_root / stats_path
    if hashlib.sha256(stats_path.read_bytes()).hexdigest() != spec["stats_sha256"]:
        raise ValueError("Offline RL normalization stats mismatch")
    print(f"offline_rl={mode} steps={steps} sidecar={path} gpu={serving['gpu_id']} port={serving['port']}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--allow-placeholder-checkpoint", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    repo_root = Path(cfg.get("project_root", ".")).expanduser().resolve()

    checkpoint = Path(cfg["checkpoint_path"]).expanduser()
    stats_path = Path(cfg["dataset_stats_path"]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = repo_root / checkpoint
    if not stats_path.is_absolute():
        stats_path = repo_root / stats_path

    if not checkpoint.exists():
        if not args.allow_placeholder_checkpoint:
            raise FileNotFoundError(f"Missing fine-tuned checkpoint: {checkpoint}")
        print(f"WARNING: checkpoint placeholder is not ready: {checkpoint}")
    else:
        size_gib = checkpoint.stat().st_size / 2**30
        if size_gib < 1.0:
            raise ValueError(
                f"Checkpoint is unexpectedly small ({size_gib:.2f} GiB): {checkpoint}"
            )
        print(f"checkpoint={checkpoint} size_gib={size_gib:.2f}")

    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for group in ("state", "action"):
        if group not in stats or "default" not in stats[group]:
            raise KeyError(f"dataset_stats is missing {group}.default")
        values = stats[group]["default"].get("global_mean")
        if values is None or len(values) != 6:
            raise ValueError(f"{group}.default.global_mean must have 6 values")
    print(f"dataset_stats={stats_path} action_dim=6 state_dim=6")
    validate_offline_rl(cfg, repo_root)
    print("RESULT: deployment metadata is internally consistent")


if __name__ == "__main__":
    main()
