#!/usr/bin/env python3
"""Generate AHA-WAM action/state normalization statistics in one process."""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra.utils import instantiate

from ahawam.utils import misc
from scripts.so101_pipeline.config_utils import DEFAULT_SETTINGS, load_training_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default=str(DEFAULT_SETTINGS))
    parser.add_argument("--output-dir", default="runs/so101/stats")
    args = parser.parse_args()

    cfg = load_training_config(args.settings)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    misc.register_work_dir(str(output_dir))
    dataset = instantiate(cfg.data.train, pretrained_norm_stats=None)
    stats_path = output_dir / "dataset_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"Statistics were not written to {stats_path}")
    print(f"dataset_samples={len(dataset)}")
    print(f"dataset_stats={stats_path}")


if __name__ == "__main__":
    main()
