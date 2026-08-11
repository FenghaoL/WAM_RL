#!/usr/bin/env python3
"""Validate deployment paths and 6-D normalization metadata without loading a GPU model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


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
    print("RESULT: deployment metadata is internally consistent")


if __name__ == "__main__":
    main()
