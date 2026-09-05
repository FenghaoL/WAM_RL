#!/usr/bin/env python3
"""Cheap but strict checks before loading two copies of the 7B model."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from omegaconf import OmegaConf

from ahawam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from scripts.so101_pipeline.flash_distillation.config_builder import (
    REPO_ROOT,
    build_distillation_config,
)


def repo_path(value) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", required=True)
    args = parser.parse_args()

    settings = OmegaConf.load(Path(args.settings).expanduser().resolve())
    state_path = repo_path(settings.source_trainer_state)
    source_config_path = repo_path(settings.source_model_config)
    teacher_path = repo_path(settings.teacher_checkpoint)
    for path in (state_path, source_config_path, teacher_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    state = json.loads(state_path.read_text(encoding="utf-8"))
    step = int(state["global_step"])
    name_match = re.search(r"step_(\d+)", teacher_path.name)
    if name_match is None or int(name_match.group(1)) != step:
        raise ValueError(
            f"trainer state step={step} does not match teacher {teacher_path.name}"
        )
    if teacher_path.stat().st_size < 1_000_000_000:
        raise ValueError(
            f"Teacher checkpoint is unexpectedly small: {teacher_path.stat().st_size} bytes"
        )
    with teacher_path.open("rb") as handle:
        if handle.read(4) != b"PK\x03\x04":
            raise ValueError(f"Teacher checkpoint is not a PyTorch zip archive: {teacher_path}")

    source = OmegaConf.load(source_config_path)
    expected = {
        "proprio_dim": 6,
        "action_horizon": 40,
        "action_chunk_size": 8,
        "max_action_offset": 7,
        "num_history_frames": 6,
        "video_rope_frame_stride": 4,
    }
    for key, wanted in expected.items():
        actual = int(source.model[key])
        if actual != wanted:
            raise ValueError(f"source.model.{key}={actual}, expected {wanted}")
    if int(source.data.train.processor.action_output_dim) != 6:
        raise ValueError("Training action_output_dim must be 6")
    if int(source.data.train.num_frames) != 49:
        raise ValueError("Training num_frames must be 49")
    if list(source.data.train.video_size) != [384, 320]:
        raise ValueError("Training video_size must be [384, 320]")

    prompts: set[str] = set()
    episodes = 0
    frames = 0
    for split in ("train", "val"):
        data_cfg = source.data.get(split)
        if data_cfg is None:
            continue
        for raw_root in data_cfg.dataset_dirs:
            root = repo_path(raw_root)
            info_path = root / "meta" / "info.json"
            episode_path = root / "meta" / "episodes.jsonl"
            task_path = root / "meta" / "tasks.jsonl"
            for path in (info_path, episode_path, task_path):
                if not path.is_file():
                    raise FileNotFoundError(path)
            info = json.loads(info_path.read_text(encoding="utf-8"))
            if int(info["fps"]) != 15:
                raise ValueError(f"{root}: expected 15 FPS")
            rows = [
                json.loads(line)
                for line in episode_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            episodes += len(rows)
            frames += sum(int(row["length"]) for row in rows)
            for line in task_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    prompts.add(DEFAULT_PROMPT.format(task=json.loads(line)["task"]))

    stats_path = repo_path(source.data.train.pretrained_norm_stats)
    cache_dir = repo_path(source.data.train.text_embedding_cache_dir)
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    for group in ("action", "state"):
        if len(stats[group]["default"]["global_mean"]) != 6:
            raise ValueError(f"{stats_path}: {group} stats are not 6-D")
    for prompt in prompts:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache_path = cache_dir / f"{digest}.t5_len128.wan22ti2v5b.pt"
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if "context" not in payload or "mask" not in payload:
            raise ValueError(f"Malformed text cache: {cache_path}")

    cfg, _ = build_distillation_config(
        args.settings,
        output_dir="runs/preflight-only",
        wandb_name="preflight-only",
    )
    if cfg.model._target_ != "ahawam.runtime.create_ahawam_ode":
        raise ValueError("Distillation config did not select create_ahawam_ode")
    if not bool(cfg.freeze_video_dit):
        raise ValueError("Flash distillation must freeze the student Video DiT")
    print(
        "preflight=ready "
        f"teacher_step={step} teacher_gib={teacher_path.stat().st_size / 2**30:.2f} "
        f"datasets_episodes={episodes} datasets_frames={frames} prompts={len(prompts)} "
        "shape=6D horizon=40 chunk=8 offset=7 video=384x320@15Hz "
        f"teacher_steps={cfg.model.teacher_num_inference_steps} "
        f"student_steps={cfg.model.student.default_action_inference_steps}"
    )


if __name__ == "__main__":
    main()

