#!/usr/bin/env python3
"""AHAWAM policy wrapper for real-robot deployment."""

from __future__ import annotations

import copy
import hashlib
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"", "none", "null"}
    return False


def _parse_optional_int(value: Any) -> Optional[int]:
    if _is_none_like(value):
        return None
    return int(value)


def _parse_optional_float(value: Any) -> Optional[float]:
    if _is_none_like(value):
        return None
    return float(value)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def predict_future_video_flow(
    *,
    model: Any,
    latents: torch.Tensor,
    timestep: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    temporal_position_ids: torch.Tensor,
    history_video_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
) -> torch.Tensor:
    """Run the video-only MoT branch used by online future-video inference."""
    expert = model.video_expert
    pre = expert.pre_dit(
        x=latents,
        timestep=timestep,
        context=context,
        context_mask=context_mask,
        action=None,
        temporal_position_ids=temporal_position_ids,
        fuse_vae_embedding_in_latents=True,
        clean_prefix_frames=1,
    )
    tokens = pre["tokens"]
    attention_mask = expert.build_video_to_video_mask(
        video_seq_len=int(tokens.shape[1]),
        video_tokens_per_frame=int(pre["meta"]["tokens_per_frame"]),
        device=tokens.device,
    )
    if history_video_kv_cache is not None:
        prefix_len = int(history_video_kv_cache[0]["k"].shape[1])
        attention_mask = torch.cat(
            [
                torch.ones(
                    (int(tokens.shape[1]), prefix_len),
                    dtype=torch.bool,
                    device=tokens.device,
                ),
                attention_mask,
            ],
            dim=1,
        )
    tokens = model.mot.forward_video_with_history(
        video_tokens=tokens,
        video_freqs=pre["freqs"],
        video_t_mod=pre["t_mod"],
        video_context_payload={
            "context": pre["context"],
            "mask": pre["context_mask"],
        },
        video_attention_mask=attention_mask,
        history_video_kv_cache=history_video_kv_cache,
    )
    return expert.post_dit(tokens, pre)


@torch.no_grad()
def denoise_future_video_latents(
    *,
    model: Any,
    latents: torch.Tensor,
    first_frame_latents: torch.Tensor,
    context: torch.Tensor,
    context_mask: torch.Tensor,
    temporal_position_ids: torch.Tensor,
    history_video_kv_cache: Optional[list[dict[str, torch.Tensor]]],
    num_inference_steps: int,
    sigma_shift: Optional[float],
) -> torch.Tensor:
    """Denoise future latents through the exact shared online/offline path."""
    timesteps, deltas = model.infer_video_scheduler.build_inference_schedule(
        num_inference_steps=int(num_inference_steps),
        device=model.device,
        dtype=latents.dtype,
        shift_override=sigma_shift,
    )
    for step_t, step_delta in zip(timesteps, deltas):
        timestep = step_t.unsqueeze(0).to(
            device=model.device, dtype=latents.dtype
        )
        pred_flow = predict_future_video_flow(
            model=model,
            latents=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            temporal_position_ids=temporal_position_ids,
            history_video_kv_cache=history_video_kv_cache,
        )
        latents = model.infer_video_scheduler.step(
            pred_flow, step_delta, latents
        )
        latents[:, :, 0:1] = first_frame_latents
    return latents


class AHAWAMPolicy:
    """AHAWAM policy wrapper used by ``deploy/server/wam_policy_server.py``.

    ``policy_path`` points to a deployment YAML file. The public keys are:
    ``checkpoint_path``, ``dataset_stats_path``, ``project_root``,
    ``hydra_config_name``, ``task``, device/precision fields, and inference
    sampling fields. The wrapper consumes one already-composed RGB image plus
    a robot state vector and returns a denormalized action chunk.
    """

    def __init__(self, policy_path: str, task_name: Optional[str] = None):
        init_t0 = time.perf_counter()
        self.task_name = task_name
        self.current_instruction = ""
        self._observation: Optional[dict[str, Any]] = None
        self._current_observation_hash: Optional[bytes] = None
        self._last_prefill_observation_hash: Optional[bytes] = None
        self._last_prefill_frame_index = 0
        self._preview_history_before_last_prefill: list[dict[str, torch.Tensor]] = []
        self._standalone_preview_history_entries: list[
            list[dict[str, torch.Tensor]]
        ] = []
        self._standalone_preview_frame_index = 0

        cfg = self._load_deploy_config(policy_path)
        project_root = cfg.get("project_root", Path(__file__).resolve().parents[2])
        self.project_root = Path(project_root).expanduser().resolve()
        src_root = self.project_root / "src"
        for path in (self.project_root, src_root):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from ahawam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
        from ahawam.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json

        self._DEFAULT_PROMPT = DEFAULT_PROMPT

        configs_root = (self.project_root / "configs").resolve()
        hydra_config_name = str(cfg.get("hydra_config_name", "deploy"))
        task = cfg.get("task")
        overrides = []
        if not _is_none_like(task):
            overrides.append(f"task={str(task)}")

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()
        hydra_overrides = cfg.get("hydra_overrides", [])
        if hydra_overrides is None:
            hydra_overrides = []
        if not isinstance(hydra_overrides, list):
            raise TypeError("`hydra_overrides` must be a YAML list of Hydra override strings.")
        overrides.extend(str(value) for value in hydra_overrides)

        with initialize_config_dir(version_base="1.3", config_dir=str(configs_root)):
            hydra_cfg = compose(config_name=hydra_config_name, overrides=overrides)

        hydra_patch = cfg.get("hydra_patch")
        if hydra_patch is not None:
            if not isinstance(hydra_patch, dict):
                raise TypeError("`hydra_patch` must be a YAML mapping.")
            OmegaConf.set_struct(hydra_cfg, False)
            hydra_cfg = OmegaConf.merge(hydra_cfg, OmegaConf.create(hydra_patch))

        logger.info(
            "Hydra config composed: config=%s task=%s extra_overrides=%d patched=%s",
            hydra_config_name,
            task,
            len(hydra_overrides),
            hydra_patch is not None,
        )

        device = str(os.environ.get("AHAWAM_DEPLOY_DEVICE_OVERRIDE") or cfg.get("device", "cuda"))
        if device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA unavailable; falling back to cpu.")
            device = "cpu"
        self.device = device

        mixed_precision = str(cfg.get("mixed_precision", "bf16"))
        model_dtype = _mixed_precision_to_dtype(mixed_precision)

        checkpoint_path = cfg.get("checkpoint_path")
        if _is_none_like(checkpoint_path):
            raise ValueError("`checkpoint_path` is required in the deploy config YAML.")
        checkpoint_path = Path(str(checkpoint_path)).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.project_root / checkpoint_path
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        dataset_stats_path = cfg.get("dataset_stats_path")
        if _is_none_like(dataset_stats_path):
            raise ValueError("`dataset_stats_path` is required in the deploy config YAML.")
        dataset_stats_path = Path(str(dataset_stats_path)).expanduser()
        if not dataset_stats_path.is_absolute():
            dataset_stats_path = self.project_root / dataset_stats_path
        if not dataset_stats_path.exists():
            raise FileNotFoundError(f"Dataset stats not found: {dataset_stats_path}")

        text_cache_dir = cfg.get("text_embedding_cache_dir")
        self.text_embedding_cache_dir: Optional[Path] = None
        if not _is_none_like(text_cache_dir):
            resolved_cache_dir = Path(str(text_cache_dir)).expanduser()
            if not resolved_cache_dir.is_absolute():
                resolved_cache_dir = self.project_root / resolved_cache_dir
            if not resolved_cache_dir.is_dir():
                raise FileNotFoundError(
                    f"Text embedding cache directory not found: {resolved_cache_dir}"
                )
            self.text_embedding_cache_dir = resolved_cache_dir

        model_cfg = OmegaConf.create(OmegaConf.to_container(hydra_cfg.model, resolve=True))

        # text_embedding_mode:
        #   cache  - disk cache only; an unknown instruction is an error (default,
        #            matches the historical behaviour and needs no text encoder)
        #   online - always encode with the T5 encoder; any instruction works
        #   hybrid - use the disk cache when it has the instruction, otherwise
        #            encode online; new embeddings are memoised in RAM and, when
        #            text_embedding_cache_write is true, written back to the cache dir
        self.text_embedding_mode = str(cfg.get("text_embedding_mode", "cache")).lower()
        if self.text_embedding_mode not in {"cache", "online", "hybrid"}:
            raise ValueError(
                "`text_embedding_mode` must be one of cache/online/hybrid, "
                f"got {self.text_embedding_mode!r}."
            )
        self.text_embedding_cache_write = bool(cfg.get("text_embedding_cache_write", False))
        needs_encoder = self.text_embedding_mode in {"online", "hybrid"}

        if self.text_embedding_mode == "cache" and self.text_embedding_cache_dir is None:
            raise ValueError(
                "`text_embedding_mode: cache` requires `text_embedding_cache_dir`."
            )
        if self.text_embedding_mode == "online":
            # Ignore any configured cache dir so every instruction takes the same path.
            self.text_embedding_cache_dir = None

        load_text_encoder = bool(cfg.get("load_text_encoder", needs_encoder))
        if needs_encoder and not load_text_encoder:
            raise ValueError(
                f"`text_embedding_mode: {self.text_embedding_mode}` requires "
                "`load_text_encoder: true` so unknown instructions can be encoded."
            )
        if not load_text_encoder and self.text_embedding_cache_dir is None:
            raise ValueError(
                "`load_text_encoder=false` requires `text_embedding_cache_dir`."
            )
        model_cfg.load_text_encoder = load_text_encoder
        self.context_len = int(model_cfg.get("tokenizer_max_len", 128))
        self._text_context_cache: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

        logger.info("Instantiating WAM model...")
        model_t0 = time.perf_counter()
        self.model = instantiate(model_cfg, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(str(checkpoint_path))
        self.model = self.model.to(device).eval()
        infer_action_params = inspect.signature(self.model.infer_action).parameters
        self.model_variant = (
            "ahawam" if "phase" in infer_action_params else "fastwam"
        )
        logger.info("Model loaded in %.1f ms", (time.perf_counter() - model_t0) * 1000.0)

        self.processor = instantiate(hydra_cfg.data.train.processor).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)
        self.proprio_dim = int(hydra_cfg.data.train.processor.proprio_output_dim)
        self.action_dim = int(hydra_cfg.data.train.processor.action_output_dim)
        logger.info("Processor and normalizer initialized from %s", dataset_stats_path)

        eval_cfg = hydra_cfg.get("EVALUATION", {})
        action_horizon = _parse_optional_int(cfg.get("action_horizon"))
        if action_horizon is None:
            action_horizon = _parse_optional_int(eval_cfg.get("action_horizon"))
        if action_horizon is None:
            action_horizon = int(hydra_cfg.data.train.num_frames) - 1
        if action_horizon <= 0:
            raise ValueError(f"`action_horizon` must be positive, got {action_horizon}")
        self.action_horizon = action_horizon

        self.num_inference_steps = _parse_optional_int(
            cfg.get("num_inference_steps", eval_cfg.get("num_inference_steps"))
        )
        if self.num_inference_steps is None:
            self.num_inference_steps = 10
        self.sigma_shift = _parse_optional_float(cfg.get("sigma_shift", eval_cfg.get("sigma_shift")))
        self.seed = _parse_optional_int(cfg.get("seed"))
        self.text_cfg_scale = float(cfg.get("text_cfg_scale", eval_cfg.get("text_cfg_scale", 1.0)))
        self.negative_prompt = str(cfg.get("negative_prompt", eval_cfg.get("negative_prompt", "")))
        self.rand_device = str(cfg.get("rand_device", eval_cfg.get("rand_device", "cpu")))
        self.tiled = bool(cfg.get("tiled", eval_cfg.get("tiled", False)))
        self.warmup_rounds = max(0, int(cfg.get("warmup_rounds", 1)))

        if self.model_variant == "ahawam":
            self.action_chunk_size = int(self.model.action_chunk_size)
            self.num_chunks = int(self.action_horizon // self.action_chunk_size)
        else:
            # FastWAM predicts one complete action horizon from each observation.
            self.action_chunk_size = self.action_horizon
            self.num_chunks = 1
        if self.num_chunks <= 0:
            raise ValueError("action_horizon must be at least one action chunk.")

        self._episode_prefilled = False
        self.video_height = int(cfg.get("video_height", 384))
        self.video_width = int(cfg.get("video_width", 320))
        self.input_control_hz = float(cfg.get("input_control_hz", 15.0))
        self.action_video_freq_ratio = int(cfg.get("action_video_freq_ratio", 4))
        self.preview_num_frames = int(cfg.get("preview_num_frames", 13))
        self.preview_num_inference_steps = int(
            cfg.get("preview_num_inference_steps", self.num_inference_steps)
        )
        self.preview_fps = float(
            cfg.get(
                "preview_fps",
                self.input_control_hz / self.action_video_freq_ratio,
            )
        )
        preview_output_dir = Path(
            str(cfg.get("preview_output_dir", "runs/so101_g05_square_v1/online_previews"))
        ).expanduser()
        if not preview_output_dir.is_absolute():
            preview_output_dir = self.project_root / preview_output_dir
        self.preview_output_dir = preview_output_dir
        self._preview_count = 0

        if self.preview_num_frames % 4 != 1:
            raise ValueError(
                f"`preview_num_frames` must satisfy T % 4 == 1, got {self.preview_num_frames}"
            )

        logger.info(
            "%s policy ready | device=%s dtype=%s | horizon=%d chunk_size=%d num_chunks=%d | image=%dx%d",
            self.model_variant,
            device,
            model_dtype,
            self.action_horizon,
            self.action_chunk_size,
            self.num_chunks,
            self.video_width,
            self.video_height,
        )
        logger.info(
            "Text conditioning: %s",
            (
                f"cached embeddings from {self.text_embedding_cache_dir}"
                if self.text_embedding_cache_dir is not None
                else "online UMT5 encoder"
            ),
        )
        if self.warmup_rounds > 0 and self.text_embedding_cache_dir is not None:
            logger.warning(
                "Skipping constructor warmup because cached text mode needs the "
                "instruction, which is set by the server after policy construction."
            )
        elif self.warmup_rounds > 0:
            self._warmup()
        logger.info("AHAWAM initialization complete in %.1f ms", (time.perf_counter() - init_t0) * 1000.0)

    def set_language_instruction(self, instruction: str):
        self.current_instruction = instruction
        logger.info("Instruction set: %r", instruction)

    def get_metadata(self) -> dict[str, Any]:
        return {
            "policy": type(self).__name__,
            "model_variant": self.model_variant,
            "action_dim": self.action_dim,
            "proprio_dim": self.proprio_dim,
            "action_chunk_size": self.action_chunk_size,
            "action_horizon": self.action_horizon,
            "num_action_chunks_per_planner_refresh": self.num_chunks,
            "model_image_shape_hwc": [self.video_height, self.video_width, 3],
            "accepted_image_keys": [
                ["front"],
                ["fixed", "wrist"],
                ["cam_high", "cam_left_wrist", "cam_right_wrist"],
            ],
            "text_mode": (
                "cached" if self.text_embedding_cache_dir is not None else "online_encoder"
            ),
            "preview_num_frames": self.preview_num_frames,
            "preview_fps": self.preview_fps,
        }

    def reset_cuda_peak_memory(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

    def get_cuda_memory_metrics(self) -> Optional[dict[str, float]]:
        if not self.device.startswith("cuda"):
            return None
        mib = 1024.0 * 1024.0
        return {
            "allocated_mib": torch.cuda.memory_allocated(self.device) / mib,
            "reserved_mib": torch.cuda.memory_reserved(self.device) / mib,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(self.device) / mib,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(self.device) / mib,
        }

    @staticmethod
    def _as_hwc_rgb(image: Any, name: str) -> np.ndarray:
        array = np.asarray(image)
        if array.ndim != 3:
            raise ValueError(f"images['{name}'] must be 3D, got {array.shape}")
        if array.shape[-1] == 3:
            pass
        elif array.shape[0] == 3:
            array = np.moveaxis(array, 0, -1)
        else:
            raise ValueError(
                f"images['{name}'] must be HWC or CHW RGB with 3 channels, got {array.shape}"
            )
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.size:
                max_value = float(np.nanmax(array))
                if max_value <= 1.0:
                    array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    def prepare_camera_images(self, images: dict[str, Any]) -> np.ndarray:
        """Return a 384x320 SO101 or native RoboTwin camera composite."""
        if images.get("front") is not None:
            return self._as_hwc_rgb(images["front"], "front")

        robotwin_keys = ("cam_high", "cam_left_wrist", "cam_right_wrist")
        if all(images.get(key) is not None for key in robotwin_keys):
            high = Image.fromarray(
                self._as_hwc_rgb(images["cam_high"], "cam_high"), mode="RGB"
            ).resize((320, 240), resample=Image.Resampling.BILINEAR)
            left = Image.fromarray(
                self._as_hwc_rgb(images["cam_left_wrist"], "cam_left_wrist"),
                mode="RGB",
            ).resize((320, 240), resample=Image.Resampling.BILINEAR)
            right = Image.fromarray(
                self._as_hwc_rgb(images["cam_right_wrist"], "cam_right_wrist"),
                mode="RGB",
            ).resize((320, 240), resample=Image.Resampling.BILINEAR)
            high = high.resize((320, 256), resample=Image.Resampling.BILINEAR)
            left = left.resize((160, 128), resample=Image.Resampling.BILINEAR)
            right = right.resize((160, 128), resample=Image.Resampling.BILINEAR)
            wrist_row = np.concatenate(
                [np.asarray(left, dtype=np.uint8), np.asarray(right, dtype=np.uint8)],
                axis=1,
            )
            return np.concatenate(
                [np.asarray(high, dtype=np.uint8), wrist_row], axis=0
            )

        if images.get("fixed") is None or images.get("wrist") is None:
            raise ValueError(
                "request images must contain ['front'], ['fixed', 'wrist'], or "
                "['cam_high', 'cam_left_wrist', 'cam_right_wrist']"
            )
        fixed = Image.fromarray(self._as_hwc_rgb(images["fixed"], "fixed"), mode="RGB")
        wrist = Image.fromarray(self._as_hwc_rgb(images["wrist"], "wrist"), mode="RGB")

        # Match training: fixed on top, SO101 wrist in RoboCOIN's bottom-left
        # slot, and black padding for the unavailable bottom-right wrist view.
        fixed = fixed.resize((320, 240), resample=Image.Resampling.BILINEAR)
        wrist = wrist.resize((320, 240), resample=Image.Resampling.BILINEAR)
        fixed = fixed.resize((320, 256), resample=Image.Resampling.BILINEAR)
        wrist = wrist.resize((160, 128), resample=Image.Resampling.BILINEAR)
        wrist_array = np.asarray(wrist, dtype=np.uint8)
        wrist_row = np.concatenate([wrist_array, np.zeros_like(wrist_array)], axis=1)
        return np.concatenate([np.asarray(fixed, dtype=np.uint8), wrist_row], axis=0)

    def update_observation_window(self, img_arr, state):
        """Buffer one composed camera observation and one state vector."""
        front_img = np.asarray(img_arr[0])
        if front_img.ndim != 3 or front_img.shape[2] != 3:
            raise ValueError(
                f"Expected an HWC RGB image with 3 channels, got {front_img.shape}"
            )
        if front_img.shape[:2] != (self.video_height, self.video_width):
            pil_img = Image.fromarray(front_img.astype(np.uint8), mode="RGB")
            pil_img = pil_img.resize((self.video_width, self.video_height), resample=Image.BILINEAR)
            front_img = np.asarray(pil_img, dtype=np.uint8)

        image_tensor = (
            torch.from_numpy(front_img.copy())
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=self.model.torch_dtype)
        )
        image_tensor = image_tensor * (2.0 / 255.0) - 1.0
        state_vector = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_vector.size != self.proprio_dim:
            raise ValueError(
                f"Expected {self.proprio_dim}-D state, got {state_vector.size} values"
            )
        if not np.all(np.isfinite(state_vector)):
            raise ValueError("State contains NaN or Inf")
        self._observation = {"image_tensor": image_tensor, "state_vector": state_vector}
        self._current_observation_hash = hashlib.sha256(front_img.tobytes()).digest()

    def get_action(self) -> np.ndarray:
        """Run one action chunk inference."""
        if self._observation is None:
            raise RuntimeError("Call update_observation_window before get_action.")

        if self.model_variant == "fastwam":
            return self._get_fastwam_action()

        image_tensor = self._observation["image_tensor"]
        state_vector = self._observation["state_vector"]

        if not hasattr(self.model, "_inference_state") or self.model._inference_state is None:
            next_chunk_index = 0
        else:
            next_chunk_index = int(self.model._inference_state.get("next_chunk_index", 0))
        if next_chunk_index >= self.num_chunks:
            self._soft_reset_for_new_observation()

        proprio = self._normalize_state(state_vector)
        prompt = self._DEFAULT_PROMPT.format(task=self.current_instruction)
        context, context_mask = self._resolve_text_condition(prompt)

        with torch.no_grad():
            if not self._episode_prefilled:
                self._prefill_episode(
                    prompt=prompt,
                    image_tensor=image_tensor,
                    context=context,
                    context_mask=context_mask,
                )
            pred = self._infer_action_chunk(image_tensor=image_tensor, proprio=proprio)

        action_tensor = pred["action_chunk"].unsqueeze(0)
        action = self._denormalize_action(action_tensor)[0]
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Model returned {action.shape[-1]} action dimensions; expected {self.action_dim}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Model action contains NaN or Inf")
        return action

    def _get_fastwam_action(self) -> np.ndarray:
        """Run FastWAM's native current-frame-to-action inference path."""
        image_tensor = self._observation["image_tensor"]
        state_vector = self._observation["state_vector"]
        proprio = self._normalize_state(state_vector)
        prompt = self._DEFAULT_PROMPT.format(task=self.current_instruction)
        context, context_mask = self._resolve_text_condition(prompt)

        with torch.no_grad():
            pred = self.model.infer_action(
                prompt=None if context is not None else prompt,
                input_image=image_tensor,
                action_horizon=self.action_horizon,
                proprio=proprio,
                context=context,
                context_mask=context_mask,
                negative_prompt=self.negative_prompt,
                text_cfg_scale=self.text_cfg_scale,
                num_inference_steps=self.num_inference_steps,
                sigma_shift=self.sigma_shift,
                seed=self.seed,
                rand_device=self.rand_device,
                tiled=self.tiled,
            )

        action = self._denormalize_action(pred["action"].unsqueeze(0))[0]
        if action.shape[-1] != self.action_dim:
            raise ValueError(
                f"Model returned {action.shape[-1]} action dimensions; expected {self.action_dim}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("Model action contains NaN or Inf")
        return action

    def reset_obsrvationwindows(self):
        self._observation = None
        self._hard_reset()
        logger.info("AHAWAM episode reset.")

    def eval(self):
        self.model.eval()

    def _prefill_episode(
        self,
        *,
        prompt: str,
        image_tensor: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        context_mask: Optional[torch.Tensor] = None,
    ) -> None:
        prior_entries = list(getattr(self.model, "_history_kv_entries", None) or [])
        self._preview_history_before_last_prefill = prior_entries
        self._last_prefill_observation_hash = self._current_observation_hash
        self._last_prefill_frame_index = int(
            getattr(self.model, "_observed_frame_index", 0)
        )
        video_kwargs = {
            "prompt": None if context is not None else prompt,
            "input_image": image_tensor,
            "action_horizon": self.action_horizon,
            "context": context,
            "context_mask": context_mask,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
            "phase": "video",
        }
        self.model.infer_action(**video_kwargs)
        self._episode_prefilled = True

    def _infer_action_chunk(self, *, image_tensor: torch.Tensor, proprio: torch.Tensor) -> dict[str, Any]:
        if not self._episode_prefilled:
            raise RuntimeError("Episode video state not initialized before action inference.")
        action_kwargs = {
            "chunk_obs_image": image_tensor,
            "chunk_proprio": proprio,
            "sigma_shift": self.sigma_shift,
            "tiled": self.tiled,
            "phase": "action",
        }
        if self.num_inference_steps is not None:
            action_kwargs["num_inference_steps"] = self.num_inference_steps
        return self.model.infer_action(**action_kwargs)

    def prefill_video_only(self, image_tensor: torch.Tensor, prompt: Optional[str] = None, deep_copy: bool = True) -> dict:
        """Run only the video phase and return a state snapshot for async serving."""
        if prompt is None:
            prompt = self._DEFAULT_PROMPT.format(task=self.current_instruction)
        context, context_mask = self._resolve_text_condition(prompt)
        with torch.no_grad():
            self._prefill_episode(
                prompt=prompt,
                image_tensor=image_tensor,
                context=context,
                context_mask=context_mask,
            )
        state = self.model._inference_state
        if state is None:
            raise RuntimeError("Video prefill did not produce inference state.")
        if deep_copy:
            return copy.deepcopy(state)
        return state

    def infer_action_chunk_only(self, inference_state: dict, image_tensor: torch.Tensor, proprio: torch.Tensor) -> np.ndarray:
        """Run only the action phase from a provided async state snapshot."""
        prev_state = self.model._inference_state
        self.model._inference_state = inference_state
        self._episode_prefilled = True
        try:
            with torch.no_grad():
                pred = self._infer_action_chunk(image_tensor=image_tensor, proprio=proprio)
        finally:
            self.model._inference_state = prev_state
        action_tensor = pred["action_chunk"].unsqueeze(0)
        return self._denormalize_action(action_tensor)[0]

    def _hard_reset(self) -> None:
        if hasattr(self.model, "reset_history"):
            self.model.reset_history()
        if hasattr(self.model, "_inference_state"):
            self.model._inference_state = None
        self._episode_prefilled = False
        self._last_prefill_observation_hash = None
        self._last_prefill_frame_index = 0
        self._preview_history_before_last_prefill = []
        self._standalone_preview_history_entries = []
        self._standalone_preview_frame_index = 0

    def _soft_reset_for_new_observation(self) -> None:
        self._episode_prefilled = False

    def _encode_text_condition_online(
        self, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode one instruction with the model's own T5 encoder.

        Returns unbatched tensors shaped like the on-disk cache payload so both
        paths can share the same memo dict and the same downstream code.
        """
        encode_t0 = time.perf_counter()
        with torch.no_grad():
            context, context_mask = self.model.encode_prompt(prompt)
        context = context[0].detach().to(dtype=self.model.torch_dtype).cpu()
        context_mask = context_mask[0].detach().bool().cpu()
        logger.info(
            "Encoded instruction online in %.3fs: %r",
            time.perf_counter() - encode_t0,
            prompt,
        )
        return context.contiguous(), context_mask.contiguous()

    def _load_text_condition_from_disk(
        self, digest: str
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if self.text_embedding_cache_dir is None:
            return None
        cache_path = self.text_embedding_cache_dir / (
            f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if not cache_path.is_file():
            return None
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        context = payload["context"].detach().to(dtype=self.model.torch_dtype)
        context_mask = payload["mask"].detach().bool()
        if context.ndim != 2 or context_mask.ndim != 1:
            raise ValueError(
                f"Invalid text cache shapes in {cache_path}: "
                f"context={tuple(context.shape)} mask={tuple(context_mask.shape)}"
            )
        # Match precompute semantics: zero the padded positions, then treat the
        # whole sequence as valid.
        context[~context_mask] = 0.0
        context_mask = torch.ones_like(context_mask)
        return context.contiguous(), context_mask.contiguous()

    def _persist_text_condition(
        self, digest: str, context: torch.Tensor, context_mask: torch.Tensor
    ) -> None:
        if not self.text_embedding_cache_write or self.text_embedding_cache_dir is None:
            return
        cache_path = self.text_embedding_cache_dir / (
            f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
        )
        if cache_path.exists():
            return
        try:
            torch.save({"context": context, "mask": context_mask}, cache_path)
            logger.info("Wrote text embedding cache entry: %s", cache_path)
        except OSError as err:  # a read-only cache dir must not kill the server
            logger.warning("Could not write %s: %s", cache_path, err)

    def _resolve_text_condition(
        self, prompt: str
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cached = self._text_context_cache.get(digest)
        if cached is None:
            cached = self._load_text_condition_from_disk(digest)
            if cached is None:
                if self.text_embedding_mode == "cache":
                    cache_path = self.text_embedding_cache_dir / (
                        f"{digest}.t5_len{self.context_len}.wan22ti2v5b.pt"
                    )
                    raise FileNotFoundError(
                        "No cached text embedding for this exact instruction. "
                        f"Expected: {cache_path}. Run scripts/precompute_text_embeds.py, "
                        "or set text_embedding_mode: hybrid to encode it online."
                    )
                cached = self._encode_text_condition_online(prompt)
                self._persist_text_condition(digest, *cached)
            self._text_context_cache[digest] = cached
        context, context_mask = cached
        return (
            context.unsqueeze(0).to(device=self.device, non_blocking=True),
            context_mask.unsqueeze(0).to(device=self.device, non_blocking=True),
        )

    @torch.no_grad()
    def generate_future_video(
        self, *, advance_preview_history: bool = False
    ) -> dict[str, Any]:
        """Explicitly denoise and VAE-decode a diagnostic future RGB clip."""
        if self._observation is None:
            raise RuntimeError("Call update_observation_window before generate_future_video.")

        from ahawam.utils.video_io import save_mp4

        started = time.perf_counter()
        image_tensor = self._observation["image_tensor"]
        prompt = self._DEFAULT_PROMPT.format(task=self.current_instruction)
        context, context_mask = self._resolve_text_condition(prompt)
        if context is None or context_mask is None:
            context, context_mask = self.model.encode_prompt(prompt)

        num_frames = self.preview_num_frames
        _, _, height, width = image_tensor.shape
        latent_t = (
            (num_frames - 1) // int(self.model.vae.temporal_downsample_factor) + 1
        )
        latent_h = height // int(self.model.vae.upsampling_factor)
        latent_w = width // int(self.model.vae.upsampling_factor)
        generator = torch.Generator(device=self.rand_device).manual_seed(
            0 if self.seed is None else self.seed
        )
        latents = torch.randn(
            (1, self.model.vae.model.z_dim, latent_t, latent_h, latent_w),
            generator=generator,
            device=self.rand_device,
            dtype=torch.float32,
        ).to(device=self.device, dtype=self.model.torch_dtype)
        first_frame_latents = self.model._encode_input_image_latents_tensor(
            input_image=image_tensor,
            tiled=self.tiled,
        )
        latents[:, :, 0:1] = first_frame_latents

        temporal_ids = (
            torch.arange(latent_t, device=self.model.device, dtype=torch.long)
            * int(self.model.vae.temporal_downsample_factor)
        )
        if advance_preview_history:
            history_entries = self._standalone_preview_history_entries
            current_frame_index = self._standalone_preview_frame_index
        elif self._current_observation_hash == self._last_prefill_observation_hash:
            history_entries = self._preview_history_before_last_prefill
            current_frame_index = self._last_prefill_frame_index
        else:
            history_entries = list(
                getattr(self.model, "_history_kv_entries", None) or []
            )
            current_frame_index = int(
                getattr(self.model, "_observed_frame_index", 0)
            )
        history_video_kv_cache = (
            self.model._concat_history_kv_entries(history_entries)
            if history_entries
            else None
        )
        temporal_ids = temporal_ids + current_frame_index
        latents = denoise_future_video_latents(
            model=self.model,
            latents=latents,
            first_frame_latents=first_frame_latents,
            context=context,
            context_mask=context_mask,
            temporal_position_ids=temporal_ids,
            history_video_kv_cache=history_video_kv_cache,
            num_inference_steps=self.preview_num_inference_steps,
            sigma_shift=self.sigma_shift,
        )

        frames = self.model._decode_latents(latents, tiled=self.tiled)
        if advance_preview_history:
            self._append_standalone_preview_history(
                first_frame_latents=first_frame_latents,
                context=context,
                context_mask=context_mask,
                current_frame_index=current_frame_index,
                history_video_kv_cache=history_video_kv_cache,
            )
        preview_frames = self._build_camera_preview_frames(frames)
        self.preview_output_dir.mkdir(parents=True, exist_ok=True)
        self._preview_count += 1
        stem = f"preview_{int(time.time())}_{self._preview_count:04d}"
        raw_path = self.preview_output_dir / f"{stem}_raw_composite.mp4"
        view_path = self.preview_output_dir / f"{stem}_fixed_wrist.mp4"
        save_mp4(frames, str(raw_path), fps=self.preview_fps)
        save_mp4(preview_frames, str(view_path), fps=self.preview_fps)
        latency_ms = (time.perf_counter() - started) * 1000.0
        horizon_seconds = (num_frames - 1) / self.preview_fps
        return {
            "encoding": "mp4",
            "mime_type": "video/mp4",
            "data": view_path.read_bytes(),
            "frame_count": len(frames),
            "fps": self.preview_fps,
            "horizon_seconds": horizon_seconds,
            "includes_current_frame": True,
            "layout": "display_fixed_left_wrist_right",
            "model_input_layout": "fixed_top_wrist_bottom_left_black_bottom_right",
            "server_path": str(view_path),
            "raw_composite_server_path": str(raw_path),
            "generation_latency_ms": latency_ms,
            "history_frames_used": len(history_entries),
            "video_current_frame_index": current_frame_index,
        }

    def _predict_future_video_flow(
        self,
        *,
        latents: torch.Tensor,
        timestep: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        temporal_position_ids: torch.Tensor,
        history_video_kv_cache: Optional[list[dict[str, torch.Tensor]]] = None,
    ) -> torch.Tensor:
        return predict_future_video_flow(
            model=self.model,
            latents=latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            temporal_position_ids=temporal_position_ids,
            history_video_kv_cache=history_video_kv_cache,
        )

    def _append_standalone_preview_history(
        self,
        *,
        first_frame_latents: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor,
        current_frame_index: int,
        history_video_kv_cache: Optional[list[dict[str, torch.Tensor]]],
    ) -> None:
        expert = self.model.video_expert
        timestep = torch.zeros(
            (1,), device=self.model.device, dtype=self.model.torch_dtype
        )
        pre = expert.pre_dit(
            x=first_frame_latents,
            timestep=timestep,
            context=context,
            context_mask=context_mask,
            action=None,
            temporal_position_ids=torch.tensor(
                [current_frame_index], device=self.model.device, dtype=torch.long
            ),
            fuse_vae_embedding_in_latents=True,
            clean_prefix_frames=1,
        )
        seq_len = int(pre["tokens"].shape[1])
        tokens_per_frame = int(pre["meta"]["tokens_per_frame"])
        attention_mask = expert.build_video_to_video_mask(
            video_seq_len=seq_len,
            video_tokens_per_frame=tokens_per_frame,
            device=self.model.device,
        )
        payload = {"context": pre["context"], "mask": pre["context_mask"]}
        if history_video_kv_cache is None:
            current_cache = self.model.mot.prefill_video_cache(
                video_tokens=pre["tokens"],
                video_freqs=pre["freqs"],
                video_t_mod=pre["t_mod"],
                video_context_payload=payload,
                video_attention_mask=attention_mask,
            )
        else:
            prefix_len = int(history_video_kv_cache[0]["k"].shape[1])
            attention_mask = torch.cat(
                [
                    torch.ones(
                        (seq_len, prefix_len),
                        dtype=torch.bool,
                        device=self.model.device,
                    ),
                    attention_mask,
                ],
                dim=1,
            )
            current_cache = self.model.mot.prefill_video_cache_with_prefix(
                video_tokens=pre["tokens"],
                video_freqs=pre["freqs"],
                video_t_mod=pre["t_mod"],
                video_context_payload=payload,
                prefix_video_kv_cache=history_video_kv_cache,
                prefix_video_seq_len=prefix_len,
                video_attention_mask=attention_mask,
            )
        current_cache = [
            {"k": layer["k"].detach(), "v": layer["v"].detach()}
            for layer in current_cache
        ]
        entries = [*self._standalone_preview_history_entries, current_cache]
        max_history = int(self.model._configured_num_history_frames())
        self._standalone_preview_history_entries = (
            entries[-max_history:] if max_history > 0 else []
        )
        self._standalone_preview_frame_index = (
            current_frame_index + self.preview_num_frames - 1
        )

    def _build_camera_preview_frames(self, frames: list[Image.Image]) -> list[Image.Image]:
        rendered: list[Image.Image] = []
        for frame_index, frame in enumerate(frames):
            composite = frame.convert("RGB")
            fixed = composite.crop((0, 0, self.video_width, 256))
            wrist = composite.crop((0, 256, self.video_width // 2, 384)).resize(
                (self.video_width, 256), resample=Image.Resampling.BILINEAR
            )
            canvas = Image.new("RGB", (self.video_width * 2, 288), color=(18, 18, 18))
            canvas.paste(fixed, (0, 32))
            canvas.paste(wrist, (self.video_width, 32))
            draw = ImageDraw.Draw(canvas)
            draw.text((8, 8), "predicted fixed", fill=(255, 255, 255))
            draw.text((self.video_width + 8, 8), "predicted wrist", fill=(255, 255, 255))
            draw.text(
                (self.video_width - 46, 8),
                f"t={frame_index / self.preview_fps:.2f}s",
                fill=(255, 220, 90),
            )
            rendered.append(canvas)
        return rendered

    def _normalize_state(self, state: np.ndarray) -> torch.Tensor:
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one merged state key in shape_meta['state'].")
        state_key = state_meta[0]["key"]
        state_batch = {"state": {state_key: torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)}}
        state_batch = self.processor.action_state_transform(state_batch)
        state_batch = self.processor.normalizer.forward(state_batch)
        return state_batch["state"][state_key]

    def _denormalize_action(self, action: torch.Tensor) -> np.ndarray:
        if action.ndim == 2:
            action = action.unsqueeze(0)
        if action.ndim != 3:
            raise ValueError(f"Expected action tensor [B,T,D], got {tuple(action.shape)}")
        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one merged action key in shape_meta['action'].")
        action_key = action_meta[0]["key"]
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action.to(dtype=torch.float32, device="cpu"))
        return denorm.numpy()

    def _warmup(self) -> None:
        """Run optional warmup inference passes during initialization."""
        logger.info("Running warmup inference (%d rounds)...", self.warmup_rounds)
        dummy_image = torch.zeros(
            (1, 3, self.video_height, self.video_width),
            device=self.device,
            dtype=self.model.torch_dtype,
        )
        dummy_proprio = None
        if self.model.proprio_dim is not None:
            dummy_proprio = torch.zeros((1, self.model.proprio_dim), device=self.device, dtype=torch.float32)

        warmup_prompt = self._DEFAULT_PROMPT.format(task="warmup")
        infer_params = inspect.signature(self.model.infer_action).parameters
        optional_kwargs = {}
        if self.num_inference_steps is not None:
            optional_kwargs["num_inference_steps"] = self.num_inference_steps
        if self.sigma_shift is not None:
            optional_kwargs["sigma_shift"] = self.sigma_shift

        with torch.no_grad():
            for round_idx in range(self.warmup_rounds):
                if "phase" in infer_params:
                    self.model.infer_action(
                        prompt=warmup_prompt,
                        input_image=dummy_image,
                        action_horizon=self.action_horizon,
                        negative_prompt=self.negative_prompt,
                        text_cfg_scale=self.text_cfg_scale,
                        seed=0,
                        rand_device=self.rand_device,
                        tiled=self.tiled,
                        phase="video",
                        **optional_kwargs,
                    )
                    self.model.infer_action(
                        prompt=None,
                        input_image=dummy_image,
                        action_horizon=self.action_horizon,
                        chunk_obs_image=dummy_image,
                        chunk_proprio=dummy_proprio,
                        negative_prompt=self.negative_prompt,
                        text_cfg_scale=self.text_cfg_scale,
                        seed=0,
                        rand_device=self.rand_device,
                        tiled=self.tiled,
                        phase="action",
                        **optional_kwargs,
                    )
                    if hasattr(self.model, "_inference_state"):
                        self.model._inference_state = None
                else:
                    self.model.infer_action(
                        prompt=warmup_prompt,
                        input_image=dummy_image,
                        action_horizon=self.action_horizon,
                        proprio=dummy_proprio,
                        negative_prompt=self.negative_prompt,
                        text_cfg_scale=self.text_cfg_scale,
                        seed=0,
                        rand_device=self.rand_device,
                        tiled=self.tiled,
                        **optional_kwargs,
                    )
                logger.info("Warmup round %d/%d complete.", round_idx + 1, self.warmup_rounds)
        self._hard_reset()

    def _load_deploy_config(self, policy_path: str) -> dict[str, Any]:
        path = Path(policy_path)
        if path.suffix in (".yml", ".yaml") and path.is_file():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        if path.is_dir():
            for name in ("deploy.yml", "deploy.yaml", "config.yaml"):
                cfg_file = path / name
                if cfg_file.exists():
                    with open(cfg_file, encoding="utf-8") as f:
                        return yaml.safe_load(f) or {}
        raise FileNotFoundError(
            f"No AHAWAM deploy config found at '{policy_path}'. "
            "Pass a YAML config file or place 'deploy.yml' in the directory."
        )


class FastWAMPolicy(AHAWAMPolicy):
    """FastWAM-specific name for the shared real-robot policy wrapper."""

    def __init__(self, policy_path: str, task_name: Optional[str] = None):
        super().__init__(policy_path=policy_path, task_name=task_name)
        if self.model_variant != "fastwam":
            raise ValueError(
                "FastWAMPolicy requires a deploy config that resolves to model=fastwam."
            )

    def generate_future_video(self, *args, **kwargs):
        del args, kwargs
        raise NotImplementedError(
            "FastWAM action inference intentionally skips future-video generation."
        )
