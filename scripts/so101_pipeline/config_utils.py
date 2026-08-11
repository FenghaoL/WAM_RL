"""Compose the repository Hydra config with an editable SO101 YAML file."""

from __future__ import annotations

from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from ahawam.utils.config_resolvers import register_default_resolvers


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SETTINGS = (
    Path(__file__).resolve().parent / "partial_finetune_zero2" / "settings.yaml"
)


def load_training_config(settings_path: str | Path = DEFAULT_SETTINGS) -> DictConfig:
    register_default_resolvers()
    settings_path = Path(settings_path).expanduser().resolve()
    settings = OmegaConf.load(settings_path)
    base_task = str(settings.pop("base_task", "robotwin_ahawam_offset"))

    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base="1.3", config_dir=str((REPO_ROOT / "configs").resolve())
    ):
        base_cfg = compose(config_name="train", overrides=[f"task={base_task}"])

    OmegaConf.set_struct(base_cfg, False)
    cfg = OmegaConf.merge(base_cfg, settings)
    OmegaConf.resolve(cfg)
    return cfg
