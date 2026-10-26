"""Synchronous TCP adapter for the three cylinder offline-RL policies.

Each request is one complete 8-command chunk. Live request observations cannot
reconstruct missing raw control ticks: history uses a declared command clock.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from deploy.server.ahawam_policy import AHAWAMPolicy
from ahawam.offline_rl.cache import asset_fingerprint
from ahawam.offline_rl.data import sha256
from ahawam.offline_rl.flow_actor import load_action_lora
from ahawam.offline_rl.inference import FrozenKVActor


def history_steps(current, history_count=6, stride=48):
    return [current-j*stride for j in range(history_count, 0, -1) if current-j*stride >= 0]


class OfflineRLPolicy(AHAWAMPolicy):
    """Reuse camera preparation, text, normalization and the existing TCP ABI."""
    def __init__(self, policy_path, task_name=None):
        self._rl_observations = {}
        self._rl_requests = 0
        self.last_conditioning = None
        super().__init__(policy_path, task_name)
        cfg = self._load_deploy_config(policy_path)
        options = cfg["offline_rl"]
        self.rl_mode = str(options["mode"])
        if self.rl_mode not in {"iql", "td3_bc", "iql_action_dit"}:
            raise ValueError("Unsupported offline RL deployment mode")
        def resolve(value):
            path = Path(value)
            return path if path.is_absolute() else self.project_root / path
        checkpoint = resolve(options["checkpoint_path"])
        if sha256(checkpoint) != options["checkpoint_sha256"]:
            raise ValueError("Offline RL checkpoint changed; update and review deployment YAML")
        base = resolve(cfg["checkpoint_path"])
        self.base_sha256 = asset_fingerprint(base, resolve(options["base_fingerprint_path"]))
        if self.base_sha256 != options["base_checkpoint_sha256"]:
            raise ValueError("Wrong collector SFT checkpoint")
        self.rl_checkpoint = str(checkpoint)
        self.external_actor = None
        if self.rl_mode == "iql_action_dit":
            metadata = load_action_lora(self.model, checkpoint, self.base_sha256)
            self.rl_steps = metadata["step"]
            self.feature_spec = metadata["provenance"]["feature_spec"]
            if metadata["provenance"]["uniform_weights"]:
                raise ValueError("Expected advantage-weighted FM, got plain BC adapter")
        else:
            self.external_actor = FrozenKVActor(checkpoint, self.device, self.base_sha256)
            if self.external_actor.algorithm != self.rl_mode:
                raise ValueError("Actor algorithm does not match deployment YAML")
            self.rl_steps = self.external_actor.steps
            self.feature_spec = self.external_actor.spec
        if self.rl_steps != int(options["expected_steps"]):
            raise ValueError("Unexpected RL training step")
        self.history_count = int(options["history_count"])
        self.history_stride_commands = int(options["history_stride_commands"])
        if (self.model_variant != "ahawam" or self.action_chunk_size != 8 or self.action_dim != 6
            or self.action_horizon != 40 or self.video_height != 384 or self.video_width != 320
            or int(cfg["serving"]["max_return_actions"]) != 8 or self.processor.action_state_transforms):
            raise ValueError("Offline RL expects the collector's absolute 8x6 cylinder model interface")
        spec = self.feature_spec
        if (spec["cache_protocol"] != "causal_independent_anchor_v2" or spec["dtype"] != "bfloat16"
            or spec["history_count"] != self.history_count or spec["history_stride"] != self.history_stride_commands
            or spec["base_checkpoint_sha256"] != self.base_sha256
            or sha256(resolve(cfg["dataset_stats_path"])) != spec["stats_sha256"]):
            raise ValueError("Conditioning/normalization provenance mismatch")
        if self.history_count != 6 or self.history_stride_commands != 48:
            raise ValueError("Use the trained history count/stride")
        for path, digest in spec["code_sha256"].items():
            # cache.py is an exporter, not part of the live model calculation.
            if Path(path).name != "cache.py" and sha256(resolve(path)) != digest:
                raise ValueError(f"Frozen model implementation changed since cache export: {path}")
        normalizer = self.processor.normalizer.normalizers["action"][self.processor.shape_meta["action"][0]["key"]]
        torch.testing.assert_close(normalizer.scale.cpu(), torch.tensor(spec["action_scale"]), rtol=0, atol=0)
        torch.testing.assert_close(normalizer.offset.cpu(), torch.tensor(spec["action_offset"]), rtol=0, atol=0)
        self.model.eval().requires_grad_(False)

    def reset_obsrvationwindows(self):
        super().reset_obsrvationwindows()
        self._rl_observations.clear()
        self._rl_requests = 0
        self.last_conditioning = None

    def set_language_instruction(self, instruction):
        if getattr(self, "current_instruction", instruction) != instruction and self._rl_observations:
            self.reset_obsrvationwindows()
        super().set_language_instruction(instruction)

    @torch.inference_mode()
    def get_action(self):
        if self._observation is None:
            raise RuntimeError("Call update_observation_window before get_action")
        # One observation per chunk is the legacy TCP client's input. It does
        # not carry HOLD duration, actual executed command count, or raw ticks.
        current_step = self._rl_requests * self.action_chunk_size
        current = dict(image=self._observation["image_tensor"].detach().cpu().clone(),
            state=self._observation["state_vector"].copy(), image_hash=self._current_observation_hash)
        prior_steps = history_steps(current_step, self.history_count, self.history_stride_commands)
        prompt = self._DEFAULT_PROMPT.format(task=self.current_instruction)
        context, context_mask = self._resolve_text_condition(prompt)
        # Independent replay reproduces the cache's causal history mechanism;
        # only the live history clock differs (commands vs raw recorded ticks).
        super()._hard_reset()
        for step in [*prior_steps, current_step]:
            observation = current if step == current_step else self._rl_observations[step]
            image = observation["image"].to(device=self.device, dtype=self.model.torch_dtype)
            self._current_observation_hash = observation["image_hash"]
            self.model._observed_frame_index = step // self.model.video_rope_frame_stride
            self._prefill_episode(prompt=prompt, image_tensor=image, context=context, context_mask=context_mask)
        self.model._inference_state["next_chunk_index"] = 0
        proprio = self._normalize_state(current["state"])
        prediction = self._infer_action_chunk(image_tensor=self._observation["image_tensor"], proprio=proprio)
        if int(prediction["chunk_index"]) != 0:
            raise RuntimeError("Offline RL only trained the independent chunk-index-zero route")
        if self.external_actor is None:
            normalized = prediction["action_chunk"].unsqueeze(0)
        else:
            kv = self.model._inference_state["_chunk_video_kv_cache"][-1]
            result = self.external_actor.action(kv["k"].to(torch.bfloat16), kv["v"].to(torch.bfloat16),
                proprio, prediction["action_chunk"].clamp(-5, 5).unsqueeze(0))
            normalized = result["normalized"].cpu()
        action = self._denormalize_action(normalized)[0]
        if action.shape != (8, 6) or not np.isfinite(action).all():
            raise ValueError("Offline policy must produce finite [8,6] absolute model-coordinate targets")
        # Commit the request clock only after a complete successful inference.
        self._rl_observations[current_step] = current
        keep_from = current_step-self.history_count*self.history_stride_commands
        self._rl_observations = {s: obs for s,obs in self._rl_observations.items() if s >= keep_from}
        self._rl_requests += 1
        self.last_conditioning = dict(request_index=self._rl_requests-1, command_step=current_step,
            history_steps=prior_steps, anchor_step=current_step, offset=0, chunk_index=0)
        return action

    def get_metadata(self):
        result = super().get_metadata()
        result.update(policy=type(self).__name__, offline_rl_mode=self.rl_mode,
            offline_rl_checkpoint=self.rl_checkpoint, offline_rl_steps=self.rl_steps,
            base_checkpoint_sha256=self.base_sha256, num_action_chunks_per_planner_refresh=1,
            conditioning_protocol="independent_request_anchor_offset0",
            history_clock="requested_full_8_command_chunks; not raw control ticks",
            history_count=self.history_count, history_stride_commands=self.history_stride_commands,
            last_conditioning=self.last_conditioning)
        return result

    def prefill_video_only(self, *args, **kwargs):
        raise NotImplementedError("Use the synchronous action server for this offline RL policy")

    def infer_action_chunk_only(self, *args, **kwargs):
        raise NotImplementedError("Async chunk state does not implement offline RL request history")

    def generate_future_video(self, *args, **kwargs):
        raise NotImplementedError("These comparison configs support action-only inference")
