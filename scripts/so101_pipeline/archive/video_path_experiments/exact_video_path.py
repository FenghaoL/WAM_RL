#!/usr/bin/env python3
"""Diagnose AHA-WAM free video generation on real dataset observations.

This deliberately runs only the VideoDiT side of the joint MoT. The released
configuration has action_conditioned=false, so the video result is identical
to the video result inside the joint training forward for the same inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Optional

import imageio.v3 as iio
import numpy as np
import torch
import torchvision.transforms.functional as tvf
from hydra.utils import instantiate
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ahawam.runtime import build_datasets
from ahawam.utils.video_io import save_mp4


DEFAULT_PROMPT = (
    "A video recorded from a robot's point of view executing the following "
    "instruction: {task}"
)


def _batch_sample(sample: dict[str, Any]) -> dict[str, Any]:
    batched: dict[str, Any] = {}
    scalar_tensor_keys = {
        "action_offset",
        "video_history_valid_len",
        "video_current_frame_index",
    }
    for key, value in sample.items():
        if isinstance(value, torch.Tensor):
            batched[key] = value.unsqueeze(0)
        elif key == "prompt":
            batched[key] = [value]
        else:
            batched[key] = value
    for key in scalar_tensor_keys:
        if key in batched and batched[key].ndim != 1:
            raise ValueError(f"Unexpected batched scalar shape for {key}: {batched[key].shape}")
    return batched


def _tensor_video_to_frames(video: torch.Tensor) -> list[Image.Image]:
    if video.ndim == 5:
        video = video[0]
    if video.ndim != 4 or int(video.shape[0]) != 3:
        raise ValueError(f"Expected [3,T,H,W], got {tuple(video.shape)}")
    uint8 = ((video.detach().float().cpu().clamp(-1, 1) + 1.0) * 127.5).to(torch.uint8)
    return [
        Image.fromarray(uint8[:, index].permute(1, 2, 0).numpy())
        for index in range(int(uint8.shape[1]))
    ]


def _load_text_cache(cache_dir: Path, prompt: str, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    matches = sorted(cache_dir.glob(f"{digest}.t5_len128.*.pt"))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one text cache for prompt in {cache_dir}, found {len(matches)}. "
            "Run prepare_robocoin_text.sh first."
        )
    payload = torch.load(matches[0], map_location="cpu", weights_only=True)
    context = payload["context"].to(device=device, dtype=torch.bfloat16).unsqueeze(0)
    context_mask = payload["mask"].to(device=device, dtype=torch.bool).unsqueeze(0)
    context[:, ~context_mask[0]] = 0
    context_mask = torch.ones_like(context_mask)
    return context, context_mask


def _read_selected_frames(path: Path, indices: list[int]) -> list[torch.Tensor]:
    requested = set(int(index) for index in indices)
    selected: dict[int, torch.Tensor] = {}
    final_index = max(requested)
    for frame_index, rgb in enumerate(iio.imiter(path, plugin="pyav")):
        if frame_index > final_index:
            break
        if frame_index in requested:
            selected[frame_index] = (
                torch.from_numpy(np.asarray(rgb).copy()).permute(2, 0, 1).float() / 255.0
            )
    missing = sorted(requested - set(selected))
    if missing:
        raise RuntimeError(f"Missing frame indices {missing} in {path}")
    return [selected[int(index)] for index in indices]


def _compose_robotwin_frames(
    dataset_dir: Path, episode: int, indices: list[int]
) -> torch.Tensor:
    root = dataset_dir / "videos" / "chunk-000"
    names = (
        "observation.images.cam_high_rgb",
        "observation.images.cam_left_wrist_rgb",
        "observation.images.cam_right_wrist_rgb",
    )
    streams = [
        _read_selected_frames(root / name / f"episode_{episode:06d}.mp4", indices)
        for name in names
    ]
    composed = []
    for head, left, right in zip(*streams):
        head = tvf.resize(head, [256, 320], antialias=True)
        left = tvf.resize(left, [128, 160], antialias=True)
        right = tvf.resize(right, [128, 160], antialias=True)
        composed.append(torch.cat([head, torch.cat([left, right], dim=-1)], dim=-2))
    video = torch.stack(composed, dim=1)
    return video * 2.0 - 1.0


def _make_robocoin_sample(
    dataset_dir: Path,
    text_cache_dir: Path,
    device: str,
    episode: int,
    start_frame: int,
    task: str,
) -> dict[str, Any]:
    sample_offsets = list(range(0, 65, 8))
    current_indices = [start_frame + offset for offset in sample_offsets]
    history_indices = [start_frame - 64 * offset for offset in range(6, 0, -1)]
    if history_indices[0] < 0:
        raise ValueError("RoboCOIN start_frame must be at least 384 for six history frames.")
    video = _compose_robotwin_frames(dataset_dir, episode, current_indices)
    history = _compose_robotwin_frames(dataset_dir, episode, history_indices)
    prompt = DEFAULT_PROMPT.format(task=task)
    context, context_mask = _load_text_cache(text_cache_dir, prompt, device="cpu")
    current_rope_index = start_frame // 8
    return {
        "video": video.unsqueeze(0),
        "prompt": [prompt],
        "context": context,
        "context_mask": context_mask,
        "video_history": history.unsqueeze(0),
        "video_history_valid_len": torch.tensor([6], dtype=torch.long),
        "video_history_frame_indices": torch.tensor(
            [[index // 8 for index in history_indices]], dtype=torch.long
        ),
        "video_current_frame_index": torch.tensor([current_rope_index], dtype=torch.long),
        "video_temporal_position_ids": torch.tensor(
            [[current_rope_index + offset for offset in (0, 4, 8)]], dtype=torch.long
        ),
    }


def _prepare_history_cache(
    model: Any,
    sample: dict[str, Any],
    context: torch.Tensor,
    context_mask: torch.Tensor,
    tokens_per_frame: int,
) -> tuple[Optional[list[dict[str, torch.Tensor]]], int]:
    valid_len = model._get_video_history_valid_len(
        sample,
        batch_size=1,
        num_history_frames=int(model.num_history_frames),
    )
    effective = int(valid_len.max().item())
    if effective == 0:
        return None, 0
    frame_indices = model._get_video_history_frame_indices(
        sample,
        batch_size=1,
        num_history_frames=int(model.num_history_frames),
    )[:, -effective:]
    history_latents = model._encode_training_history_latents(
        sample,
        batch_size=1,
        num_history_frames=effective,
        tiled=False,
    )
    timestep = torch.zeros((1,), device=model.device, dtype=model.torch_dtype)
    history_pre = model.video_expert.pre_dit(
        x=history_latents,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        action=None,
        temporal_position_ids=frame_indices,
        fuse_vae_embedding_in_latents=True,
        clean_prefix_frames=effective,
    )
    frame_mask = model._build_history_video_frame_mask(
        num_history_frames=effective,
        valid_history_len=valid_len,
        device=model.device,
    )
    token_mask = model._expand_frame_mask_to_token_mask(
        frame_mask=frame_mask,
        tokens_per_frame=tokens_per_frame,
    )
    cache = model.mot.prefill_video_cache(
        video_tokens=history_pre["tokens"],
        video_freqs=history_pre["freqs"],
        video_t_mod=history_pre["t_mod"],
        video_context_payload={
            "context": history_pre["context"],
            "mask": history_pre["context_mask"],
        },
        video_attention_mask=token_mask,
    )
    return [{"k": item["k"].detach(), "v": item["v"].detach()} for item in cache], effective


@torch.no_grad()
def _generate(
    model: Any,
    sample: dict[str, Any],
    *,
    steps: int,
    seed: int,
    use_history: bool,
) -> tuple[list[Image.Image], dict[str, float | int]]:
    video = sample["video"].to(device=model.device, dtype=model.torch_dtype)
    context = sample["context"].to(device=model.device, dtype=model.torch_dtype)
    context_mask = sample["context_mask"].to(device=model.device, dtype=torch.bool)
    first_frame = model._encode_input_image_latents_tensor(video[:, :, 0], tiled=False)
    latent_frames = (int(video.shape[2]) - 1) // int(model.vae.temporal_downsample_factor) + 1
    spatial_factor = int(model.vae.upsampling_factor)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents = torch.randn(
        (
            1,
            int(model.vae.model.z_dim),
            latent_frames,
            int(video.shape[3]) // spatial_factor,
            int(video.shape[4]) // spatial_factor,
        ),
        generator=generator,
        dtype=torch.float32,
    ).to(device=model.device, dtype=model.torch_dtype)
    latents[:, :, :1] = first_frame
    temporal_ids = sample["video_temporal_position_ids"].to(device=model.device, dtype=torch.long)

    probe_t = torch.zeros((1,), device=model.device, dtype=model.torch_dtype)
    probe_pre = model.video_expert.pre_dit(
        x=latents,
        timestep=probe_t,
        context=context,
        context_mask=context_mask,
        action=None,
        temporal_position_ids=temporal_ids,
        fuse_vae_embedding_in_latents=True,
        clean_prefix_frames=1,
    )
    tokens_per_frame = int(probe_pre["meta"]["tokens_per_frame"])
    history_cache = None
    history_frames = 0
    if use_history:
        history_cache, history_frames = _prepare_history_cache(
            model, sample, context, context_mask, tokens_per_frame
        )

    timesteps, deltas = model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=steps,
        device=model.device,
        dtype=latents.dtype,
        shift_override=None,
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(model.device)
    for step_t, step_delta in zip(timesteps, deltas):
        timestep = step_t.unsqueeze(0).to(device=model.device, dtype=latents.dtype)
        pre = model.video_expert.pre_dit(
            x=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            temporal_position_ids=temporal_ids,
            fuse_vae_embedding_in_latents=True,
            clean_prefix_frames=1,
        )
        attention_mask = model.video_expert.build_video_to_video_mask(
            video_seq_len=int(pre["tokens"].shape[1]),
            video_tokens_per_frame=int(pre["meta"]["tokens_per_frame"]),
            device=model.device,
        )
        if history_cache is not None:
            prefix_len = int(history_cache[0]["k"].shape[1])
            attention_mask = torch.cat(
                [
                    torch.ones(
                        (int(pre["tokens"].shape[1]), prefix_len),
                        dtype=torch.bool,
                        device=model.device,
                    ),
                    attention_mask,
                ],
                dim=1,
            )
        tokens = model.mot.forward_video_with_history(
            video_tokens=pre["tokens"],
            video_freqs=pre["freqs"],
            video_t_mod=pre["t_mod"],
            video_context_payload={"context": pre["context"], "mask": pre["context_mask"]},
            video_attention_mask=attention_mask,
            history_video_kv_cache=history_cache,
        )
        pred_flow = model.video_expert.post_dit(tokens, pre)
        latents = model.infer_video_scheduler.step(pred_flow, step_delta, latents)
        latents[:, :, :1] = first_frame
    frames = model._decode_latents(latents, tiled=False)
    elapsed = time.perf_counter() - started
    peak_gib = torch.cuda.max_memory_allocated(model.device) / (1024**3)
    return frames, {
        "history_frames": history_frames,
        "steps": steps,
        "seed": seed,
        "generation_seconds": round(elapsed, 3),
        "peak_allocated_gib": round(peak_gib, 3),
    }


def _frame_metrics(pred: list[Image.Image], target: list[Image.Image]) -> dict[str, float]:
    pred_arr = np.stack([np.asarray(frame, dtype=np.float32) for frame in pred[1:]])
    target_arr = np.stack([np.asarray(frame, dtype=np.float32) for frame in target[1:]])
    mse = float(np.mean((pred_arr - target_arr) ** 2))
    motion = float(np.mean(np.abs(np.diff(pred_arr, axis=0)))) if len(pred_arr) > 1 else 0.0
    target_motion = float(np.mean(np.abs(np.diff(target_arr, axis=0)))) if len(target_arr) > 1 else 0.0
    return {
        "future_rgb_mse": round(mse, 3),
        "future_rgb_psnr": round(float(10 * np.log10((255.0**2) / max(mse, 1e-9))), 3),
        "pred_temporal_abs_diff": round(motion, 3),
        "target_temporal_abs_diff": round(target_motion, 3),
    }


def _comparison_frames(columns: list[tuple[str, list[Image.Image]]]) -> list[Image.Image]:
    count = min(len(frames) for _, frames in columns)
    output = []
    for index in range(count):
        width = sum(frames[index].width for _, frames in columns)
        height = max(frames[index].height for _, frames in columns) + 28
        canvas = Image.new("RGB", (width, height), (18, 18, 18))
        draw = ImageDraw.Draw(canvas)
        x = 0
        for label, frames in columns:
            canvas.paste(frames[index], (x, 28))
            draw.text((x + 6, 7), label, fill=(255, 255, 255))
            x += frames[index].width
        output.append(canvas)
    return output


def _load_so101_sample(config_path: Path, sample_index: int) -> tuple[Any, dict[str, Any]]:
    cfg = OmegaConf.load(config_path)
    cfg.data.train.skip_padding_as_possible = False
    train_dataset, _ = build_datasets(cfg.data, cfg.model)
    return cfg, _batch_sample(train_dataset[sample_index])


def _load_model(cfg: Any, checkpoint: Path, device: str) -> Any:
    model = instantiate(cfg.model, model_dtype=torch.bfloat16, device=device)
    model.load_checkpoint(str(checkpoint))
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("so101", "robocoin"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=float, default=8.0)
    parser.add_argument("--sample-index", type=int, default=8696)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--text-cache-dir", type=Path)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=384)
    parser.add_argument(
        "--task",
        default="grasp the yellow bread and place it into the yellow plate, then retrieve it.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    if args.mode == "so101":
        cfg, sample = _load_so101_sample(args.config, args.sample_index)
    else:
        if args.dataset_dir is None or args.text_cache_dir is None:
            parser.error("robocoin mode requires --dataset-dir and --text-cache-dir")
        cfg = OmegaConf.load(args.config)
        sample = _make_robocoin_sample(
            args.dataset_dir,
            args.text_cache_dir,
            args.device,
            args.episode,
            args.start_frame,
            args.task,
        )

    print(f"[model] loading {args.checkpoint}", flush=True)
    model = _load_model(cfg, args.checkpoint, args.device)
    print("[model] ready", flush=True)
    if bool(model.video_expert.action_conditioned):
        raise RuntimeError("This diagnostic assumes video_expert.action_conditioned=false.")
    ground_truth = _tensor_video_to_frames(sample["video"])
    print("[generate] no history KV", flush=True)
    no_history, no_history_info = _generate(
        model, sample, steps=args.steps, seed=args.seed, use_history=False
    )
    print("[generate] training-style history KV", flush=True)
    with_history, with_history_info = _generate(
        model, sample, steps=args.steps, seed=args.seed, use_history=True
    )
    print("[save] videos and metrics", flush=True)

    save_mp4(ground_truth, str(args.output / "ground_truth.mp4"), fps=args.fps)
    save_mp4(no_history, str(args.output / "prediction_no_history.mp4"), fps=args.fps)
    save_mp4(with_history, str(args.output / "prediction_with_history.mp4"), fps=args.fps)
    comparison = _comparison_frames(
        [("ground truth", ground_truth), ("no history KV", no_history), ("training-style history KV", with_history)]
    )
    save_mp4(comparison, str(args.output / "comparison.mp4"), fps=args.fps)
    comparison[len(comparison) // 2].save(args.output / "comparison_middle.png")

    report = {
        "mode": args.mode,
        "checkpoint": str(args.checkpoint),
        "prompt": sample["prompt"][0],
        "sample_index": args.sample_index if args.mode == "so101" else None,
        "episode": args.episode if args.mode == "robocoin" else None,
        "start_frame": args.start_frame if args.mode == "robocoin" else None,
        "video_shape": list(sample["video"].shape),
        "temporal_position_ids": sample["video_temporal_position_ids"].tolist(),
        "history_frame_indices": sample["video_history_frame_indices"].tolist(),
        "no_history": {**no_history_info, **_frame_metrics(no_history, ground_truth)},
        "with_history": {**with_history_info, **_frame_metrics(with_history, ground_truth)},
        "video_action_conditioned": bool(model.video_expert.action_conditioned),
        "path_equivalence": (
            "Same VideoDiT pre_dit, per-frame-causal self-attention, text cross-attention, "
            "post_dit, scheduler and VAE decode as the joint model. The action branch is omitted "
            "because action_conditioned=false and video blocks execute before action blocks."
        ),
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
