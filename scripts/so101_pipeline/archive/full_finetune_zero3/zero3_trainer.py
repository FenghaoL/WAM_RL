"""ZeRO-3-specific checkpoint handling without changing the repository trainer."""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from ahawam.trainer import Wan22Trainer


class Zero3Wan22Trainer(Wan22Trainer):
    """Save ZeRO-3 weights collectively, then emit the normal AHA-WAM format."""

    def __init__(self, *args: Any, **kwargs: Any):
        cfg = kwargs["cfg"]
        self.zero3_save_portable_weights = bool(
            cfg.get("zero3_save_portable_weights", True)
        )
        self.zero3_save_training_state = bool(
            cfg.get("zero3_save_training_state", False)
        )
        super().__init__(*args, **kwargs)
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        stage = None
        if plugin is not None:
            stage = plugin.deepspeed_config.get("zero_optimization", {}).get("stage")
        if int(stage or -1) != 3:
            raise RuntimeError(
                "Zero3Wan22Trainer requires DeepSpeed ZeRO stage 3; "
                f"the active stage is {stage!r}."
            )

    @staticmethod
    def _strip_wrapper_prefix(key: str) -> str:
        prefixes = ("module.", "_forward_module.")
        prefix_removed = True
        while prefix_removed:
            prefix_removed = False
            for prefix in prefixes:
                if key.startswith(prefix):
                    key = key[len(prefix) :]
                    prefix_removed = True
                    break
        return key

    @classmethod
    def build_portable_payload(
        cls,
        state_dict: dict[str, torch.Tensor],
        *,
        step: int,
        torch_dtype: torch.dtype,
    ) -> dict[str, Any]:
        normalized = OrderedDict(
            (cls._strip_wrapper_prefix(key), value)
            for key, value in state_dict.items()
        )

        mot_prefix = None
        for candidate in ("mot.", "dit."):
            if any(key.startswith(candidate) for key in normalized):
                mot_prefix = candidate
                break
        if mot_prefix is None:
            preview = list(normalized)[:20]
            raise KeyError(
                "The consolidated ZeRO-3 state has neither `mot.` nor `dit.` keys. "
                f"First keys: {preview}"
            )

        def select(prefix: str) -> OrderedDict[str, torch.Tensor]:
            return OrderedDict(
                (key[len(prefix) :], value)
                for key, value in normalized.items()
                if key.startswith(prefix)
            )

        payload: dict[str, Any] = {
            "mot": select(mot_prefix),
            "step": int(step),
            "torch_dtype": str(torch_dtype),
        }
        for module_name in (
            "proprio_encoder",
            "action_obs_visual_proj",
            "chunk_obs_query_encoder",
        ):
            module_state = select(module_name + ".")
            if module_state:
                payload[module_name] = module_state
        if not payload["mot"]:
            raise RuntimeError("Portable checkpoint would contain an empty `mot` state.")
        return payload

    def _save_consolidated_weights(self, step_tag: str) -> str | None:
        # Every rank must enter get_state_dict: ZeRO-3 gathers one partitioned
        # layer at a time and returns the complete CPU dict on rank 0.
        state_dict = self.accelerator.get_state_dict(self.model)
        output_path = os.path.join(self.weights_dir, f"{step_tag}.pt")
        if self.accelerator.is_main_process:
            if state_dict is None:
                raise RuntimeError("Rank 0 did not receive consolidated ZeRO-3 weights.")
            model = self.accelerator.unwrap_model(self.model)
            payload = self.build_portable_payload(
                state_dict,
                step=self.global_step,
                torch_dtype=model.torch_dtype,
            )
            temporary_path = output_path + ".tmp"
            torch.save(payload, temporary_path)
            os.replace(temporary_path, output_path)
        del state_dict
        self.accelerator.wait_for_everyone()
        return output_path if self.accelerator.is_main_process else None

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"
        self.accelerator.wait_for_everyone()

        weights_path = None
        if self.zero3_save_portable_weights:
            weights_path = self._save_consolidated_weights(step_tag)

        state_path = None
        if self.zero3_save_training_state:
            state_path = os.path.join(self.state_dir, step_tag)
            Path(state_path).mkdir(parents=True, exist_ok=True)
            self.accelerator.save_state(output_dir=state_path)
            if self.accelerator.is_main_process:
                self._save_trainer_state(state_path)
            self.accelerator.wait_for_everyone()
        elif self.accelerator.is_main_process:
            metadata_path = Path(self.checkpoint_root) / f"{step_tag}_trainer_state.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "global_step": int(self.global_step),
                        "epoch": int(self.epoch),
                        "batch_in_epoch": int(self.batch_in_epoch),
                        "optimizer_state_saved": False,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
        self.accelerator.wait_for_everyone()
        return {"weights_path": weights_path, "state_path": state_path}
