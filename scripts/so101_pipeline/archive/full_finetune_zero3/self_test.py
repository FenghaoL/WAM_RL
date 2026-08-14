#!/usr/bin/env python3
"""Fast CPU checks for the isolated SO101 ZeRO-3 training configuration."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
import yaml

from scripts.so101_pipeline.full_finetune_zero3.aligned_sampler import (
    ZeRO3HistoryAlignedSampler,
)
from scripts.so101_pipeline.full_finetune_zero3.zero3_trainer import (
    Zero3Wan22Trainer,
)


HERE = Path(__file__).resolve().parent


def test_settings() -> None:
    settings = yaml.safe_load((HERE / "settings.yaml").read_text(encoding="utf-8"))
    assert int(settings["num_epochs"]) == 4
    assert int(settings["batch_size"]) == 1
    assert int(settings["gradient_accumulation_steps"]) == 4
    assert settings["mixed_precision"] == "bf16"
    assert settings["freeze_video_dit"] is False
    assert settings["zero3_save_portable_weights"] is True
    assert settings["zero3_save_training_state"] is False
    assert settings["wandb"]["enabled"] is True
    assert settings["wandb"]["mode"] == "online"
    assert settings["data"]["train"]["concat_multi_camera"] == "so101"
    assert settings["data"]["train"]["skip_padding_as_possible"] is False


def test_deepspeed_config() -> None:
    config = json.loads((HERE / "ds_zero3_hybrid.json").read_text(encoding="utf-8"))
    zero = config["zero_optimization"]
    assert int(zero["stage"]) == 3
    assert zero["stage3_gather_16bit_weights_on_model_save"] is True
    assert zero["allgather_sequential"] is True
    assert zero["offload_optimizer"]["device"] == "cpu"
    assert zero["offload_param"]["device"] == "none"
    assert float(zero["offload_optimizer"]["ratio"]) == 0.35
    assert int(zero["sub_group_size"]) == 100_000_000


def test_portable_payload() -> None:
    state = {
        "_forward_module.module.mot.layer.weight": torch.ones(2, 3),
        "module.proprio_encoder.weight": torch.ones(4, 6),
        "action_obs_visual_proj.0.bias": torch.ones(4),
        "chunk_obs_query_encoder.query": torch.ones(2, 4),
        "vae.frozen": torch.ones(1),
    }
    payload = Zero3Wan22Trainer.build_portable_payload(
        state, step=7, torch_dtype=torch.bfloat16
    )
    assert list(payload["mot"]) == ["layer.weight"]
    assert tuple(payload["proprio_encoder"]["weight"].shape) == (4, 6)
    assert "vae" not in payload
    assert payload["step"] == 7
    assert payload["torch_dtype"] == "torch.bfloat16"


def test_history_aligned_sampler() -> None:
    class Dataset:
        valid_lens = torch.tensor(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2], dtype=torch.long
        )

        def __len__(self) -> int:
            return int(self.valid_lens.numel())

        def get_video_history_valid_len_for_all_indices(self) -> torch.Tensor:
            return self.valid_lens

    dataset = Dataset()
    sampler = ZeRO3HistoryAlignedSampler(
        dataset=dataset,
        seed=42,
        batch_size=2,
        num_processes=4,
    )
    indices = list(sampler)
    chunk_size = 8
    assert len(indices) == len(sampler) == 24
    assert set(indices) == set(range(len(dataset)))
    assert all(count >= 1 for count in Counter(indices).values())
    for start in range(0, len(indices), chunk_size):
        chunk = indices[start : start + chunk_size]
        history_lens = {int(dataset.valid_lens[index]) for index in chunk}
        assert len(history_lens) == 1

    ZeRO3HistoryAlignedSampler.priority_history_len = 2
    priority_indices = list(sampler)
    assert {
        int(dataset.valid_lens[index]) for index in priority_indices[:chunk_size]
    } == {2}
    ZeRO3HistoryAlignedSampler.priority_history_len = None


def main() -> None:
    test_settings()
    test_deepspeed_config()
    test_portable_payload()
    test_history_aligned_sampler()
    print("SO101 ZeRO-3 settings and portable checkpoint tests passed.")


if __name__ == "__main__":
    main()
