"""ZeRO-2 checkpoint policy for deployable weights without huge Adam states."""

from __future__ import annotations

import json
from pathlib import Path

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
