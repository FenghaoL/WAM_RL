#!/usr/bin/env python3
"""Resolve the newest saved Flash student and materialize a deploy config."""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def step_number(path: Path) -> int:
    match = re.fullmatch(r"step_(\d+)\.pt", path.name)
    return -1 if match is None else int(match.group(1))


def resolve_checkpoint(explicit: str | None) -> Path:
    if explicit:
        path = Path(explicit).expanduser()
        path = path if path.is_absolute() else REPO_ROOT / path
    else:
        pointer = REPO_ROOT / "runs/so101_ahawam_padapter_v2/latest_flash_run.txt"
        if not pointer.is_file():
            raise FileNotFoundError(
                f"No Flash run pointer at {pointer}. Start distillation or set FLASH_CHECKPOINT."
            )
        run_dir = Path(pointer.read_text(encoding="utf-8").strip())
        run_dir = run_dir if run_dir.is_absolute() else REPO_ROOT / run_dir
        candidates = list((run_dir / "model/checkpoints/weights").glob("step_*.pt"))
        candidates = [path for path in candidates if step_number(path) >= 0]
        if not candidates:
            raise FileNotFoundError(
                f"The latest Flash run has not saved a student yet: {run_dir}. "
                "Wait for its first epoch checkpoint or set FLASH_CHECKPOINT explicitly."
            )
        path = max(candidates, key=step_number)
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size < 1_000_000_000:
        raise ValueError(f"Checkpoint is unexpectedly small: {path}")
    return path.resolve()


def validate_flash_checkpoint(path: Path) -> dict:
    """Reject a normal SFT checkpoint accidentally used with the 2-step sampler.

    ``mmap=True`` leaves the large tensor storages file-backed; only the small
    checkpoint dictionary and ODE metadata are inspected here.
    """
    try:
        payload = torch.load(
            path,
            map_location="cpu",
            mmap=True,
            weights_only=False,
        )
    except TypeError:  # Compatibility with older torch builds without mmap.
        payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a checkpoint dictionary, got {type(payload).__name__}: {path}")

    metadata = payload.get("ode_metadata")
    if not isinstance(metadata, dict):
        raise ValueError(
            f"Checkpoint has no ode_metadata and is not a verified Flash student: {path}"
        )
    teacher_steps = int(metadata.get("teacher_num_inference_steps", -1))
    capture_steps = tuple(int(value) for value in metadata.get("teacher_capture_step_indices", ()))
    if teacher_steps != 16 or not {0, 8, 16}.issubset(capture_steps):
        raise ValueError(
            "The 2-step deployment expects a 16-step teacher and ODE anchors "
            f"0/8/16, got teacher_steps={teacher_steps}, capture_steps={capture_steps}: {path}"
        )
    student_task = str(metadata.get("student_task_name", ""))
    if "flash" not in student_task.lower():
        raise ValueError(
            f"ode_metadata.student_task_name does not identify a Flash student: {student_task!r}"
        )
    return {
        "checkpoint_step": payload.get("step"),
        "teacher_steps": teacher_steps,
        "capture_steps": capture_steps,
        "student_task": student_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=os.environ.get("FLASH_CHECKPOINT"))
    parser.add_argument("--template", default=str(HERE / "settings.template.yaml"))
    parser.add_argument(
        "--output",
        default=str(
            REPO_ROOT
            / "runs/so101_ahawam_padapter_v2/flash_async_deploy/settings.resolved.yaml"
        ),
    )
    args = parser.parse_args()

    checkpoint = resolve_checkpoint(args.checkpoint)
    flash_metadata = validate_flash_checkpoint(checkpoint)
    template = yaml.safe_load(Path(args.template).read_text(encoding="utf-8"))
    if int(template.get("num_inference_steps", -1)) != 2:
        raise ValueError("Flash deploy template must set num_inference_steps: 2")
    template["checkpoint_path"] = str(checkpoint)
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(template, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        "Verified Flash checkpoint: "
        f"step={flash_metadata['checkpoint_step']} "
        f"student={flash_metadata['student_task']} "
        f"teacher_steps={flash_metadata['teacher_steps']} "
        f"anchors={flash_metadata['capture_steps']}",
        file=sys.stderr,
    )
    print(output)


if __name__ == "__main__":
    main()
