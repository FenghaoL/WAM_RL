#!/usr/bin/env python3
"""Load the native FastWAM service wrapper and run one synthetic inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from deploy.server.ahawam_policy import FastWAMPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="scripts/so101_pipeline/fastwam_deploy/settings_native_robotwin.yaml",
    )
    args = parser.parse_args()

    config = Path(args.config)
    policy = FastWAMPolicy(str(config))
    policy.set_language_instruction("Pick up the white block.")

    stats_path = Path(
        "checkpoints/FastWAM-SharePoint/robotwin_uncond_3cam_384_dataset_stats.json"
    )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    state = np.asarray(stats["state"]["default"]["global_mean"], dtype=np.float32)
    image = np.zeros((384, 320, 3), dtype=np.uint8)

    policy.update_observation_window([image], state)
    policy.reset_cuda_peak_memory()
    started = time.perf_counter()
    action = policy.get_action()
    elapsed = time.perf_counter() - started

    if action.shape != (32, 14):
        raise ValueError(f"Expected action shape (32, 14), got {action.shape}")
    if not np.all(np.isfinite(action)):
        raise ValueError("Action contains NaN or Inf")

    print(f"action_shape: {action.shape}")
    print(f"action_finite: {bool(np.all(np.isfinite(action)))}")
    print(f"inference_seconds: {elapsed:.3f}")
    print(f"cuda_memory: {policy.get_cuda_memory_metrics()}")


if __name__ == "__main__":
    main()
