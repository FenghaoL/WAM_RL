#!/usr/bin/env python3
"""MiniMax-H3 image-to-video baseline on the exported SO101 windows.

Runs in the `minimax_h3` conda environment and never imports `ahawam`: it only
reads what `export_reference_samples.py` wrote (first frame, ground-truth PNGs,
prompt, timing metadata) and writes a run directory with the same layout as the
AHA-WAM video-branch runs, so `compare_runs.py` can merge them.

What is and is not comparable
-----------------------------
MiniMax-H3 is a general video+audio generator. It has no robot history K/V, no
per-frame-causal video mask, no action branch and no notion of the 3.75 Hz
control clock. The only conditioning it shares with AHA-WAM is *the same first
frame and the same instruction text*, so that is exactly what it is given.

MiniMax-H3 generates at a fixed 24 fps and refuses anything shorter than five
seconds, while one SO101 window is 3.2 s. So the request is rounded up to the
shortest legal length and the generated clip is resampled back onto the 13
ground-truth timestamps before scoring; the untruncated clip is kept next to it.
"""

from __future__ import annotations

import argparse
import gc
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import reporting  # noqa: E402

MIN_SECONDS = 5.0
MAX_SECONDS = 15.0
NATIVE_FPS = 24.0


def snap_num_frames(seconds: float) -> int:
    """Round a duration up to the next `17*n + 5` frame count MiniMax-H3 decodes."""
    target = max(seconds, MIN_SECONDS) * NATIVE_FPS
    n = 0
    while 17 * n + 5 < target:
        n += 1
    frames = 17 * n + 5
    duration = frames / NATIVE_FPS
    if duration > MAX_SECONDS:
        raise ValueError(
            f"{seconds:.2f}s rounds to {frames} frames ({duration:.2f}s), "
            f"beyond MiniMax-H3's {MAX_SECONDS}s ceiling."
        )
    return frames


def resample_to_timestamps(
    frames: list[Image.Image], times: list[float], *, fps: float
) -> list[Image.Image]:
    """Pick the generated frame nearest each ground-truth timestamp."""
    picked = []
    for t in times:
        index = int(round(t * fps))
        picked.append(frames[min(index, len(frames) - 1)])
    return picked


# Layers torchao must leave in bfloat16. These are the entry/exit projections and
# norms the MiniMax-H3 card calls out, and they also happen to be the only two
# transformer weights whose input dim is not a multiple of the int4 group size
# (`proj_in` 96, `audio_proj_in` 32), so one list serves both int8 and int4.
TRANSFORMER_SKIP = [
    "proj_in", "audio_proj_in", "context_embedder", "time_embedder",
    "time_proj", "token_refiner", "norm_out", "proj_out", "audio_proj_out",
]
# Same story on the conditioner: every int4-incompatible weight (27 of them,
# in_features 4304) lives under `model.visual`, which is skipped anyway.
TEXT_ENCODER_SKIP = [
    "model.visual",
    "model.language_model.embed_tokens",
    "model.language_model.norm",
    "lm_head",
]


def torchao_weight_config(quantization: str):
    """The weight-only torchao config for A100 (sm_80).

    A100 has no FP8 and no NVFP4 tensor cores, so the community `-FP8` / `-nvfp4`
    releases cannot run here. int8 and int4 both quantize on the fly from the
    bfloat16 checkpoint already on disk -- no extra download.

    int4 has exactly one packing format that works on sm_80: `tile_packed_to_4d`,
    the tinygemm `_weight_int4pack_mm` path. `plain` and `preshuffled` need mslk,
    and `plain_int32` has no CUDA implementation.
    """
    from torchao.quantization import Int4WeightOnlyConfig, Int8WeightOnlyConfig
    from torchao.quantization.quantize_.workflows.int4.int4_packing_format import (
        Int4PackingFormat,
    )

    if quantization == "int8":
        return Int8WeightOnlyConfig(version=2)
    if quantization == "int4":
        return Int4WeightOnlyConfig(
            group_size=128,
            int4_packing_format=Int4PackingFormat.TILE_PACKED_TO_4D,
        )
    raise ValueError(f"Unsupported quantization: {quantization}")


def _real_bytes(tensor: Any) -> int:
    """Storage a tensor actually occupies, seeing through torchao subclasses.

    A quantized torchao tensor still reports the *logical* shape and the bf16
    dtype, so `numel * element_size` would report the unquantized size. The inner
    packed tensors are reachable through `__tensor_flatten__`.
    """
    flatten = getattr(tensor, "__tensor_flatten__", None)
    if flatten is None or type(tensor) in (torch.Tensor, torch.nn.Parameter):
        return tensor.numel() * tensor.element_size()
    try:
        names, _ = flatten()
    except Exception:
        return tensor.numel() * tensor.element_size()
    return sum(_real_bytes(getattr(tensor, name)) for name in names)


def component_footprint(pipe: Any) -> dict[str, float]:
    """Bytes each big component actually occupies once loaded, in GiB.

    This is the number that decides whether a component fits on the card -- unlike
    a parameter count, which says nothing once the weights are quantized.
    """
    sizes: dict[str, float] = {}
    for name in ("transformer", "text_encoder", "vae", "audio_vae"):
        module = getattr(pipe, name, None)
        if module is None or not hasattr(module, "parameters"):
            continue
        seen: set[int] = set()
        total = 0
        for tensor in list(module.parameters()) + list(module.buffers()):
            inner = getattr(tensor, "_data", tensor)
            key = id(inner)
            if key in seen:
                continue
            seen.add(key)
            total += _real_bytes(tensor)
        sizes[name] = round(total / (1024**3), 2)
    sizes["total"] = round(sum(sizes.values()), 2)
    return sizes


def _park_on_cpu(module: Any) -> Any:
    """Move a just-quantized component back to host RAM.

    Components loaded with `device_map=` come back carrying accelerate hooks, and
    `.to()` refuses to move a hooked module. The hooks only exist to place weights
    during loading, which is already done, so dropping them is safe here.
    """
    try:
        return module.to("cpu")
    except (RuntimeError, ValueError, NotImplementedError):
        from accelerate.hooks import remove_hook_from_module

        remove_hook_from_module(module, recurse=True)
        for attr in ("hf_device_map", "_hf_hook"):
            if hasattr(module, attr):
                try:
                    delattr(module, attr)
                except AttributeError:
                    pass
        return module.to("cpu")


def build_pipeline(args: argparse.Namespace) -> Any:
    from diffusers import ComponentsManager, MiniMaxH3Transformer3DModel, ModularPipeline
    from diffusers import TorchAoConfig
    from transformers import Qwen3VLForConditionalGeneration
    from transformers import TorchAoConfig as TransformersTorchAoConfig

    model_path = str(args.model_path)
    manager = ComponentsManager()
    pipe = ModularPipeline.from_pretrained(model_path, components_manager=manager)

    if args.quantization != "none":
        weight_config = torchao_weight_config(args.quantization)
        # int4 packing goes through `aten::_convert_weight_to_int4pack`, which only
        # has a CUDA kernel, so those weights have to be quantized on the card.
        # int8 packs fine on CPU, and loading it there keeps the card empty.
        quantize_device = args.device if args.quantization == "int4" else None
        load_kwargs = {} if quantize_device is None else {"device_map": quantize_device}
        print(
            f"[load] quantizing transformer and conditioner to {args.quantization}"
            f" on {quantize_device or 'cpu'}",
            flush=True,
        )
        # Built one at a time and parked in host RAM: at int4 the two of them are
        # ~33 GiB together, which would not leave room for the VAE on a 40 GB card.
        transformer = MiniMaxH3Transformer3DModel.from_pretrained(
            model_path,
            subfolder="transformer",
            dtype=torch.bfloat16,
            quantization_config=TorchAoConfig(
                weight_config, modules_to_not_convert=TRANSFORMER_SKIP
            ),
            **load_kwargs,
        )
        if quantize_device is not None:
            transformer = _park_on_cpu(transformer)
            torch.cuda.empty_cache()
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            subfolder="text_encoder",
            dtype=torch.bfloat16,
            quantization_config=TransformersTorchAoConfig(
                weight_config, modules_to_not_convert=TEXT_ENCODER_SKIP
            ),
            **load_kwargs,
        )
        if quantize_device is not None:
            text_encoder = _park_on_cpu(text_encoder)
            torch.cuda.empty_cache()
        pipe.update_components(transformer=transformer, text_encoder=text_encoder)

    pipe.load_components(workflow="fl2va", dtype=torch.bfloat16)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    manager.enable_auto_cpu_offload(
        device=args.device, memory_reserve_margin=args.memory_reserve_margin
    )
    return pipe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--run-name", default="minimax_h3")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quantization", choices=("int4", "int8", "none"), default="int8")
    # The int8 transformer is ~31 GB. On a 40 GB A100 the reserve margin has to
    # leave more than that, or the ComponentsManager evicts the transformer on
    # every denoising step ("no combination of models to offload to cpu is
    # found") and a step costs 90 s instead of a few.
    parser.add_argument("--memory-reserve-margin", default="3GB")
    parser.add_argument("--height", type=int, default=576, help="multiple of 32")
    parser.add_argument("--width", type=int, default=480, help="multiple of 32")
    parser.add_argument("--num-inference-steps", type=int, default=0, help="0 = block default")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=6)
    parser.add_argument("--prompt-suffix", default="", help="Appended to the dataset prompt.")
    parser.add_argument("--dry-run", action="store_true", help="Load the pipeline and print pipe.doc.")
    args = parser.parse_args()

    if args.height % 32 or args.width % 32:
        parser.error("--height and --width must be multiples of 32")

    manifest = json.loads((args.samples_dir / "manifest.json").read_text(encoding="utf-8"))
    samples = manifest["samples"]
    if args.limit > 0:
        samples = samples[: args.limit]
    video_fps = float(manifest["video_fps"])
    gt_height = int(manifest["frame_height"])
    gt_width = int(manifest["frame_width"])

    print(f"[load] MiniMax-H3 from {args.model_path}", flush=True)
    load_started = time.perf_counter()
    pipe = build_pipeline(args)
    load_seconds = round(time.perf_counter() - load_started, 1)
    footprint = component_footprint(pipe)
    print(f"[load] ready in {load_seconds}s", flush=True)
    print(f"[load] resident weight footprint (GiB): {footprint}", flush=True)
    if args.dry_run:
        print(pipe.doc)
        return

    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    sheet: list[tuple[str, Image.Image]] = []
    run_started = time.perf_counter()

    for order, meta in enumerate(samples):
        tag = str(meta["tag"])
        source_dir = args.samples_dir / tag
        sample_dir = args.output / tag
        sample_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_dir / "ground_truth.mp4", sample_dir / "ground_truth.mp4")

        ground_truth = [
            Image.open(source_dir / "frames" / f"gt_{index:03d}.png").convert("RGB")
            for index in range(int(meta["num_video_frames"]))
        ]
        first_frame = Image.open(source_dir / "first_frame.png").convert("RGB")
        prompt = str(meta["prompt"]) + args.prompt_suffix
        num_frames = snap_num_frames(float(meta["clip_seconds"]))

        call_kwargs: dict[str, Any] = {
            "prompt": prompt,
            "image": first_frame,
            "num_frames": num_frames,
            "height": args.height,
            "width": args.width,
            "generator": torch.Generator().manual_seed(args.seed + order),
            "output": ["videos"],
        }
        if args.num_inference_steps > 0:
            call_kwargs["num_inference_steps"] = args.num_inference_steps

        print(
            f"[generate] {order + 1}/{len(samples)} {tag} "
            f"{args.height}x{args.width} {num_frames}f ({num_frames / NATIVE_FPS:.2f}s)",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        results = pipe(**call_kwargs)
        elapsed = time.perf_counter() - started
        peak_gib = torch.cuda.max_memory_allocated(args.device) / (1024**3)

        generated = [frame.convert("RGB") for frame in results["videos"][0]]
        reporting.save_mp4(generated, sample_dir / "prediction_full_24fps.mp4", fps=NATIVE_FPS)

        matched = resample_to_timestamps(
            generated, list(meta["frame_times_seconds"]), fps=NATIVE_FPS
        )
        matched = [
            frame.resize((gt_width, gt_height), Image.Resampling.BILINEAR) for frame in matched
        ]
        reporting.save_mp4(matched, sample_dir / "prediction_minimax_h3.mp4", fps=video_fps)

        metrics = reporting.frame_metrics(
            matched, ground_truth, regions=reporting.SO101_REGIONS
        )
        comparison = reporting.comparison_frames(
            [("ground truth", ground_truth), ("MiniMax-H3 i2v", matched)]
        )
        reporting.save_mp4(comparison, sample_dir / "comparison.mp4", fps=video_fps)
        comparison[len(comparison) // 2].save(sample_dir / "comparison_middle.png")
        comparison[-1].save(sample_dir / "comparison_last.png")
        sheet.append((tag, comparison[-1]))

        row: dict[str, Any] = {
            "sample": tag,
            "prompt": prompt,
            "sample_index": int(meta["sample_index"]),
            "episode": int(meta["source_episode"]),
            "start_frame": int(meta["source_frame"]),
            "generated_frames": len(generated),
            "generated_seconds": round(len(generated) / NATIVE_FPS, 3),
            "canvas": f"{args.height}x{args.width}",
            "generation_seconds": round(elapsed, 3),
            "peak_allocated_gib": round(peak_gib, 3),
            **metrics,
        }
        (sample_dir / "metrics.json").write_text(
            json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        rows.append(row)
        del results, generated, matched, comparison
        gc.collect()
        torch.cuda.empty_cache()

    reporting.contact_sheet(sheet, args.output / "contact_sheet_last.png")
    run_meta = {
        "run_name": args.run_name,
        "model_family": "minimax-h3-fl2va",
        "weights": f"MiniMaxAI/MiniMax-H3 ({args.quantization})",
        "model_path": str(args.model_path),
        "samples_dir": str(args.samples_dir),
        "primary_prediction_file": "prediction_minimax_h3.mp4",
        "canvas": f"{args.height}x{args.width}",
        "native_fps": NATIVE_FPS,
        "video_fps": video_fps,
        "num_video_frames": int(manifest["num_video_frames"]),
        "frame_size": [gt_height, gt_width],
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps or "block default",
        "quantization": args.quantization,
        "weight_footprint_gib": footprint,
        "memory_reserve_margin": args.memory_reserve_margin,
        "load_seconds": load_seconds,
        "wall_seconds": round(time.perf_counter() - run_started, 1),
    }
    reporting.save_summary(args.output, rows, run_meta=run_meta)


if __name__ == "__main__":
    main()
