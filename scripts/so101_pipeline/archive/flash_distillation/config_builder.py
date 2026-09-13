"""Build an ODE-distillation config from an exact completed SFT config."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[3]


def _repo_path(value: str | Path) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def build_distillation_config(
    settings_path: str | Path,
    *,
    output_dir: str | Path,
    wandb_name: str,
) -> tuple[DictConfig, DictConfig]:
    """Return ``(training_cfg, launcher_settings)``.

    Starting from the saved resolved SFT config avoids silently falling back to
    RoboTwin's 14-D model, horizon 80, or a different camera/data timebase.
    """

    settings = OmegaConf.load(Path(settings_path).expanduser().resolve())
    source_config_path = _repo_path(settings.source_model_config)
    source = OmegaConf.load(source_config_path)
    source = OmegaConf.create(OmegaConf.to_container(source, resolve=True))

    cfg = deepcopy(source)
    train = settings.training
    cfg.output_dir = str(Path(output_dir))
    cfg.batch_size = int(train.batch_size)
    cfg.num_workers = int(train.num_workers)
    cfg.gradient_accumulation_steps = int(train.gradient_accumulation_steps)
    cfg.num_epochs = int(train.num_epochs)
    cfg.learning_rate = float(train.learning_rate)
    cfg.lr_scheduler_type = str(train.lr_scheduler_type)
    cfg.warmup_ratio = float(train.warmup_ratio)
    cfg.weight_decay = float(train.weight_decay)
    cfg.max_grad_norm = float(train.max_grad_norm)
    cfg.mixed_precision = str(train.mixed_precision)
    cfg.log_every = int(train.log_every)
    cfg.max_steps = None
    cfg.eval_every = 0
    cfg.save_every = 0
    cfg.save_at_end = False
    cfg.freeze_video_dit = True
    cfg.resume = None
    cfg.init_checkpoint = None
    cfg.base_init_checkpoint = None
    cfg.resume_global_batch_size = None
    cfg.zero2_save_training_state = False
    cfg.eval_num_val_batches = 0
    cfg.eval_num_action_batches = 0
    cfg.seed = int(train.seed)

    cfg.wandb = OmegaConf.create(
        {
            "enabled": bool(settings.wandb.enabled),
            "workspace": None,
            "project": str(settings.wandb.project),
            "name": str(wandb_name),
            "group": str(settings.wandb.group),
            "mode": str(settings.wandb.mode),
        }
    )

    # Preserve the exact task-SFT AHA-WAM shape for both roles.  The teacher
    # loads step_005328; the student starts as an exact teacher copy inside
    # AHAWAMODE before optimization begins.
    teacher_model = deepcopy(source.model)
    student_model = deepcopy(source.model)
    teacher_model.mot_checkpoint_mixed_attn = True
    student_model.mot_checkpoint_mixed_attn = True
    distill = settings.distillation
    cfg.model = OmegaConf.create(
        {
            "_target_": "ahawam.runtime.create_ahawam_ode",
            "teacher_task_name": "so101_power_adapter_v2",
            "student_task_name": "so101_power_adapter_v2_flash",
            "mot_checkpoint_mixed_attn": True,
            "action_horizon": int(source.model.action_horizon),
            "action_chunk_size": int(source.model.action_chunk_size),
            "action_train_timestep_mode": str(
                source.model.action_train_timestep_mode
            ),
            "student_capture_timestep_mode": str(
                distill.student_capture_timestep_mode
            ),
            "teacher_num_inference_steps": int(
                distill.teacher_num_inference_steps
            ),
            "teacher_capture_step_indices": list(
                distill.teacher_capture_step_indices
            ),
            "teacher_sigma_shift": distill.teacher_sigma_shift,
            "num_history_frames": int(source.model.num_history_frames),
            "teacher": {
                "_target_": "ahawam.runtime.create_ahawam_teacher",
                "checkpoint_path": str(_repo_path(settings.teacher_checkpoint)),
                "model": OmegaConf.to_container(teacher_model, resolve=True),
            },
            "student": {
                "_target_": "ahawam.runtime.create_ahawam_student",
                "checkpoint_path": None,
                "prediction_type": "flow",
                "ode_denoise_step_indices": list(
                    distill.teacher_capture_step_indices
                ),
                "default_action_inference_steps": int(
                    distill.student_default_action_inference_steps
                ),
                "model": OmegaConf.to_container(student_model, resolve=True),
            },
        }
    )
    OmegaConf.resolve(cfg)
    return cfg, settings

