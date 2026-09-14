"""Prepare isolated, auditable paper controls; never launches training.

Run from the repository root with PYTHONPATH=src:. in the ahawam environment.
All optimization and data loading reuse ovcr_adapter/v2_experiments/run.py.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
CYLINDER_BASE = "runs/ovcr_adapter_cylinder_hil_v2/configs/p1_full_balanced_video_hilv2.yaml"
POWER_BASE = "runs/ovcr_adapter_padapter_v2/configs/p1_full_balanced_video.yaml"
CONTROLS = {
    "context_truncated": {"data.train.correction_history_crosses_segment": False},
    "video_off": {"model.loss.lambda_video": 0.0},
    "history_raw48": {"data.train.correction_history_skip_system_hold": False},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten(value, prefix=""):
    if not isinstance(value, dict):
        return {prefix: value}
    result = {}
    for key, item in value.items():
        result.update(flatten(item, f"{prefix}.{key}" if prefix else key))
    return result


def apply_control(base, name):
    """Validation context remains fixed; only training's requested factor changes."""
    cfg = copy.deepcopy(base)
    for key, value in CONTROLS[name].items():
        OmegaConf.update(cfg, key, value, force_add=True)
    before = flatten(OmegaConf.to_container(base, resolve=True))
    after = flatten(OmegaConf.to_container(cfg, resolve=True))
    changed = {k for k in before.keys() | after.keys() if before.get(k) != after.get(k)}
    if changed != set(CONTROLS[name]):
        raise ValueError(f"Control is a no-op or changed unexpected keys: {name}: {changed}")
    return cfg


def write_suite(run_root, manifest, configs, parent_config):
    out = (ROOT / run_root).resolve()
    out.relative_to(ROOT / "runs")
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}")
    # Validate before creating any output. Full trainer freezes VAE/text, trains DiTs/OVCR.
    for cfg in configs.values():
        if cfg.experiment.training_scope != "full" or cfg.model.get("ovcr_adapter") is not None:
            raise ValueError("This entry prepares full fine-tuning controls only")
        if not (ROOT / cfg.init_checkpoint).is_file():
            raise FileNotFoundError(cfg.init_checkpoint)
        if cfg.batch_size * cfg.gradient_accumulation_steps * 3 != 12:
            raise ValueError("Expected global batch 12 on three GPUs")
    out.mkdir(parents=True)
    (out / "configs").mkdir()
    mp = out / "data_manifest.json"
    mp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    deltas = {}
    for name, cfg in configs.items():
        original = OmegaConf.to_container(cfg, resolve=True)
        cfg.output_dir = str(out.relative_to(ROOT) / name / "model")
        cfg.experiment.name = name
        cfg.experiment.run_root = str(out.relative_to(ROOT))
        cfg.experiment.manifest_file = "data_manifest.json"
        cfg.experiment.manifest_sha256 = sha(mp)
        cfg.experiment.initialization = cfg.init_checkpoint
        cfg.experiment.reference_run = str(parent_config)
        cfg.wandb.name = name
        cfg.wandb.group = "icra2027-context-controls"
        OmegaConf.save(cfg, out / "configs" / f"{name}.yaml", resolve=True)
        deltas[name] = dict(planned=True, trained=False,
                            scientific_change=original["experiment"].get("paper_change", {}),
                            checkpoint=cfg.init_checkpoint, steps=cfg.max_steps,
                            expected_final_checkpoint=f"{cfg.output_dir}/checkpoints/weights/step_{cfg.max_steps:06d}.pt")
    (out / "plan.json").write_text(json.dumps(deltas, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"run_root": str(out.relative_to(ROOT)), "names": list(configs),
                      "status": "prepared_only", "launch_entry": "scripts/so101_pipeline/ovcr_adapter/v2_experiments/launch.sh"}, indent=2))


def controls(args):
    path = ROOT / args.base_config
    base = OmegaConf.load(path)
    mp = ROOT / base.experiment.run_root / base.experiment.get("manifest_file", "data_manifest.json")
    if sha(mp) != base.experiment.manifest_sha256:
        raise ValueError("Parent manifest changed")
    manifest = json.loads(mp.read_text())
    manifest["paper_parent"] = {"config": str(path.relative_to(ROOT)), "sha256": sha(path),
                                "manifest": str(mp.relative_to(ROOT)), "sha256_manifest": sha(mp)}
    configs = {}
    for variant in args.variants:
        cfg = apply_control(base, variant)
        cfg.experiment.paper_change = CONTROLS[variant]
        # Don't inherit a false claim that this control still crosses segments.
        cfg.experiment.history_crosses_segment = cfg.data.train.correction_history_crosses_segment
        cfg.experiment.history_skip_system_hold = cfg.data.train.correction_history_skip_system_hold
        configs[f"full_{variant}"] = cfg
    write_suite(args.run_root, manifest, configs, path.relative_to(ROOT))


def power(args):
    # Explicit new data only: no fallback to the camera-invalid legacy collections.
    hil = (ROOT / args.hil_root).resolve()
    hil.relative_to(ROOT / "data")
    for name in ("so101_ahawam_power_adapter_hil_v1", "so101_ahawam_power_adapter_hil_v2"):
        old = (ROOT / "data/ahawam_hil_raw" / name).resolve()
        if hil == old or old in hil.parents:
            raise ValueError("Legacy power-adapter HIL was invalidated by the user; provide a new calibrated collection")
    if not hil.is_dir():
        raise FileNotFoundError(f"New collection not available: {hil}")
    if not args.camera_note.strip():
        raise ValueError("Record the camera setup/calibration identity")
    from scripts.so101_pipeline.ovcr_adapter.v2_experiments.prepare import audit
    from scripts.so101_pipeline.ovcr_adapter.cylinder_experiments.prepare_hil_v2 import split_by_val_sessions
    report = split_by_val_sessions(audit(str(hil.relative_to(ROOT)), "20260913"))
    if not report["train"]["human_frames"] or not report["val"]["human_frames"]:
        raise ValueError("Need HUMAN training data and a separate session ending in _val")
    base_path = ROOT / POWER_BASE
    cfg = OmegaConf.load(base_path)
    # The old full config supplies ONLY the original task checkpoint/demo/normalization recipe.
    parent = json.loads((ROOT / cfg.experiment.run_root / "data_manifest.json").read_text())
    for key in ("checkpoint", "norm_stats", "demo_train_dirs", "demo_val_dirs", "demo_train_frames", "demo_val_frames"):
        report[key] = copy.deepcopy(parent[key])
    if sha(ROOT / report["norm_stats"]["path"]) != report["norm_stats"]["sha256"]:
        raise ValueError("Task normalization changed")
    report["camera_note"] = args.camera_note
    report["task"] = "power_adapter_new_camera_hil"
    report["excluded_legacy_hil"] = True
    selections = report["correction_episode_selection"]
    cfg.data.train.correction_dataset_dirs = list(selections["train"])
    cfg.data.train.correction_episode_selection = selections["train"]
    cfg.data.train.correction_clip_dataset_dirs = []
    cfg.data.train.correction_history_crosses_segment = True
    cfg.data.train.correction_history_frame_stride = 48
    cfg.data.train.correction_history_skip_system_hold = True
    cfg.data.train.correction_future_extension = "none"
    cfg.data.train.correction_exclude_static_starts = False
    cfg.correction_val.dataset_dirs = list(selections["val"])
    cfg.correction_val.episode_selection = selections["val"]
    cfg.correction_val.history_crosses_segment = True
    cfg.correction_val.history_frame_stride = 48
    cfg.correction_val.history_skip_system_hold = True
    cfg.correction_val.future_extension = "none"
    cfg.correction_val.video_supervision_require_full_window = True
    cfg.model.num_history_frames = 6
    cfg.experiment.task = report["task"]
    cfg.experiment.correction_sources = [str(hil.relative_to(ROOT))]
    cfg.experiment.history_crosses_segment = True
    cfg.experiment.history_frame_stride = 48
    cfg.experiment.history_skip_system_hold = True
    cfg.experiment.paper_change = {"new_camera_hil": str(hil.relative_to(ROOT)), "history": "cross-boundary executing48"}
    # Keep native task budget: power full uses 2664 steps, not cylinder's 3285.
    write_suite(args.run_root, report, {"full_new_camera_context": cfg}, POWER_BASE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("controls")
    p.add_argument("--base-config", default=CYLINDER_BASE)
    p.add_argument("--run-root", default="runs/icra2027_cylinder_controls")
    p.add_argument("--variants", nargs="+", choices=CONTROLS, default=["context_truncated", "video_off", "history_raw48"])
    p.set_defaults(func=controls)
    p = sub.add_parser("power")
    p.add_argument("--hil-root", required=True)
    p.add_argument("--camera-note", required=True)
    p.add_argument("--run-root", default="runs/icra2027_power_new_camera")
    p.set_defaults(func=power)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
