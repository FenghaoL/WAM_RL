"""ZeRO-2 checkpoint policy for deployable weights without huge Adam states."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from ahawam.trainer import Wan22Trainer
from ahawam.utils.logging_config import get_logger


logger = get_logger(__name__)


class Zero2WeightOnlyTrainer(Wan22Trainer):
    """Use the repository trainer while making optimizer-state saves optional."""

    def __init__(self, *args, **kwargs):
        self.base_init_checkpoint = kwargs["cfg"].get(
            "base_init_checkpoint", None
        )
        self.zero2_save_training_state = bool(
            kwargs["cfg"].get("zero2_save_training_state", False)
        )
        self.eval_num_val_batches = int(
            kwargs["cfg"].get("eval_num_val_batches", 0) or 0
        )
        self.eval_num_action_batches = int(
            kwargs["cfg"].get("eval_num_action_batches", 0) or 0
        )
        super().__init__(*args, **kwargs)
        plugin = getattr(self.accelerator.state, "deepspeed_plugin", None)
        stage = None
        if plugin is not None:
            stage = plugin.deepspeed_config.get("zero_optimization", {}).get("stage")
        if int(stage or -1) != 2:
            raise RuntimeError(
                "Zero2WeightOnlyTrainer requires DeepSpeed ZeRO stage 2; "
                f"the active stage is {stage!r}."
            )

    def _load_weights_before_prepare(self):
        if self.base_init_checkpoint:
            base_path = Path(str(self.base_init_checkpoint))
            if not self.init_checkpoint:
                raise ValueError(
                    "`base_init_checkpoint` requires `init_checkpoint` for the overlay."
                )
            if not base_path.is_file():
                raise FileNotFoundError(
                    f"Base initialization checkpoint not found: {base_path}"
                )
            logger.info(
                "Loading base AHA initialization BEFORE FastWAM overlay: %s",
                base_path,
            )
            self.model.load_checkpoint(str(base_path), optimizer=None)
        super()._load_weights_before_prepare()

    def _fixed_val_indices(self):
        """A fixed, rank-disjoint slice of the val set, identical at every eval.

        Two properties matter here. (1) Step-invariance: the stock trainer draws a
        fresh random index per rank per eval, so the curve moves because the sample
        moved, not because the model did. (2) Unbiasedness: a plain stride would
        land on the same phase of every episode (index 0 is always an episode start,
        history_len 0), so we use a seeded permutation instead.
        """
        n = len(self.val_dataset)
        if n == 0 or self.eval_num_val_batches <= 0:
            return []
        world = max(1, self.accelerator.num_processes)
        rank = self.accelerator.process_index
        wanted = min(self.eval_num_val_batches * world, n)
        generator = torch.Generator(device="cpu").manual_seed(int(self.seed))
        order = torch.randperm(n, generator=generator).tolist()
        return order[:wanted][rank::world]

    @torch.no_grad()
    def _averaged_val_metrics(self):
        """Mean metrics over the fixed subset.

        Teacher-forced loss for every sample (one forward each, cheap). Action error
        additionally for the first `eval_num_action_batches` of them, because that
        needs a full `infer()` -- and for AHAWAM `infer` returns actions only, no
        video, so there are no PSNR/SSIM numbers to collect here.
        """
        indices = self._fixed_val_indices()
        if not indices:
            return {}

        model = self.accelerator.unwrap_model(self.model)
        was_training = model.dit.training
        model.eval()

        totals: dict[str, float] = {}
        counts: dict[str, int] = {}

        def accumulate(key, value):
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    return
                value = float(value.float().item())
            if isinstance(value, (int, float)):
                totals[key] = totals.get(key, 0.0) + float(value)
                counts[key] = counts.get(key, 0) + 1

        for position, index in enumerate(indices):
            sample = self._to_batched_eval_sample(self.val_dataset[index])
            sample["_eval_dataset"] = self.val_dataset
            with self.accelerator.autocast():
                loss, loss_dict = model.training_loss(sample)
            accumulate("loss", loss)
            for key, value in (loss_dict or {}).items():
                accumulate(str(key), value)

            if position < self.eval_num_action_batches:
                with self.accelerator.autocast():
                    pred = model.infer(
                        **model._build_eval_infer_kwargs(
                            sample=sample,
                            eval_num_inference_steps=self.eval_num_inference_steps,
                        )
                    )
                action_metrics = model._evaluate_action_metrics(
                    sample=sample, pred_action=pred.get("action")
                )
                for key, value in action_metrics.items():
                    accumulate(str(key), value)

        if was_training:
            self._set_dit_only_train_mode()
        if not totals:
            return {}

        out: dict[str, float] = {}
        for key, total in totals.items():
            local = torch.tensor(
                total / max(1, counts[key]),
                device=self.accelerator.device,
                dtype=torch.float32,
            ).reshape(1)
            out[key] = float(self.accelerator.gather_for_metrics(local).mean().item())
        world = max(1, self.accelerator.num_processes)
        out["num_loss_samples"] = float(len(indices) * world)
        out["num_action_samples"] = float(
            min(len(indices), self.eval_num_action_batches) * world
        )
        return out

    def evaluate(self):
        """Base-trainer single-sample metrics, plus much less noisy averaged ones."""
        metrics = super().evaluate()
        if metrics is None:
            return None
        averaged = self._averaged_val_metrics()
        if not averaged:
            return metrics

        # Keep the single-sample values for continuity, but let the headline
        # `val_loss` / `action_*` be the averaged ones -- those are what the
        # trainer prints and ships to wandb.
        for key in ("val_loss", "action_l1", "action_l2"):
            if key in metrics:
                metrics[f"{key}_single_sample"] = metrics[key]
        metrics["val_loss"] = averaged.pop("loss")
        metrics["val_num_loss_samples"] = averaged.pop("num_loss_samples", 0.0)
        metrics["val_num_action_samples"] = averaged.pop("num_action_samples", 0.0)
        for key in ("action_l1", "action_l2"):
            if key in averaged:
                metrics[key] = averaged.pop(key)
        for key, value in averaged.items():
            metrics[f"avg_{key}"] = value
        return metrics

    def save_checkpoint(self):
        step_tag = f"step_{self.global_step:06d}"
        self.accelerator.wait_for_everyone()

        weights_path = None
        if self.accelerator.is_main_process:
            weights_path = self._save_weights_checkpoint(step_tag=step_tag)
        self.accelerator.wait_for_everyone()

        state_path = None
        if self.zero2_save_training_state:
            state_path = str(Path(self.state_dir) / step_tag)
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
