"""Freeze paper-relevant configs, checkpoint existence and offline metrics, CPU only.

No success rate is inferred from losses or from intervention-assisted outcomes.
Weight files are stat'ed, not loaded or hashed (each is about 15 GB).
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[3]
RUNS = ["runs/ovcr_adapter_cylinder_hil_v2", "runs/ovcr_adapter_cylinder_hil_v2_full"]
SFT = {
    "cylinder_sft": ("runs/so101_ahawam_cylinder_v1/full_zero2_offload_ahawam_base/formal-cylinderv1-sft-ahawambase-bs4-acc1-4ep-3gpu-20260905-022135/model", 6572),
    "power_sft": ("runs/so101_ahawam_padapter_v2/full_zero2_offload_ahawam_base/formal-padapterv2-sft-ahawambase-bs4-acc1-4ep-3gpu-20260901-001743/model", 5328),
}


def file_info(path, digest=False):
    p = ROOT / path
    result = {"path": str(p.relative_to(ROOT)), "exists": p.is_file()}
    if p.is_file():
        stat = p.stat()
        result.update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        if digest:
            result["sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    return result


def config_info(path):
    c = OmegaConf.load(ROOT / path)
    keys = ["init_checkpoint", "batch_size", "gradient_accumulation_steps", "learning_rate", "max_steps",
            "num_epochs", "seed", "save_every", "eval_every", "history_aware_batching", "model.num_history_frames",
            "model.loss", "model.ovcr_adapter", "model.ovcr_anchor_weight", "data.train.sampling_strategy",
            "data.train.pretrained_norm_stats", "data.train.dataset_dirs", "data.val.dataset_dirs",
            "experiment.training_scope", "data.train.correction_future_extension",
            "data.train.correction_extension_success_only", "data.train.correction_history_crosses_segment",
            "data.train.correction_history_frame_stride", "data.train.correction_history_skip_system_hold"]
    values = {k: OmegaConf.select(c, k) for k in keys}
    values = OmegaConf.to_container(OmegaConf.create(values), resolve=True)
    return dict(file=file_info(path, True), values=values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    out = ROOT / args.output
    if out.exists():
        raise FileExistsError(out)
    result = {"created_utc": datetime.now(timezone.utc).isoformat(), "real_robot_success_rates": None,
              "weight_hash_policy": "existence/size/mtime only; no tensor inspection", "sft": {}, "runs": {}}
    result["git_head"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["source_files"] = [file_info(path, True) for path in (
        "scripts/so101_pipeline/hil_human_only_dataset.py", "src/ahawam/models/wan22/ahawam.py",
        "src/ahawam/models/wan22/mot.py", "src/ahawam/models/wan22/ovcr_adapter.py",
        "src/ahawam/utils/samplers.py", "scripts/so101_pipeline/ovcr_adapter/v2_experiments/run.py",
        "scripts/so101_pipeline/ovcr_adapter/cylinder_experiments/prepare_hil_v2_full.py",
        "scripts/so101_pipeline/online_deploy/run_action_server.sh",
        "scripts/so101_pipeline/online_deploy/deployments/cylinder_v2_p1_full_balanced_video.yaml")]
    for name, (model, step) in SFT.items():
        result["sft"][name] = {"config": config_info(f"{model}/config.yaml"),
                               "checkpoint": file_info(f"{model}/checkpoints/weights/step_{step:06d}.pt")}
    for run in RUNS:
        root = ROOT / run
        status = json.loads((root / "suite_status.json").read_text())
        states = {r["name"]: r for r in status.get("experiments", [])}
        manifest = json.loads((root / "data_manifest.json").read_text())
        data = {k: v for k, v in manifest.items() if k not in ("episodes", "correction_episode_selection")}
        entries = {}
        for path in sorted((root / "configs").glob("*.yaml")):
            c = OmegaConf.load(path)
            runtime = ROOT / c.output_dir / "experiment_config.yaml"
            log = root / "logs" / f"{path.stem}.log"
            text = log.read_text(errors="replace") if log.exists() else ""
            vals = [json.loads(line[11:]) for line in text.splitlines() if line.startswith("VALIDATION ")]
            fractions = [float(v) for v in re.findall(r"(?<!_)video_supervised_fraction=([0-9.]+)", text)]
            entries[path.stem] = {
                "suite_record": states.get(path.stem, {"state": "not_started_in_snapshot"}),
                "generated_config": config_info(path.relative_to(ROOT)),
                "runtime_config": config_info(runtime.relative_to(ROOT)) if runtime.exists() else None,
                "last_logged_step": re.findall(r"step=(\d+)/\d+", text)[-1:],
                "first_validation": vals[0] if vals else None, "last_validation": vals[-1] if vals else None,
                "training_video_fraction_mean": sum(fractions)/len(fractions) if fractions else None,
                "training_video_fraction_records": len(fractions),
                "checkpoints": [file_info(w.relative_to(ROOT)) for w in sorted((ROOT / c.output_dir / "checkpoints/weights").glob("*.pt"))],
                "expected_final_checkpoint": file_info(Path(c.output_dir) / "checkpoints/weights" / f"step_{c.max_steps:06d}.pt"),
            }
        result["runs"][run] = dict(manifest=file_info(root.relative_to(ROOT)/"data_manifest.json", True), data=data, experiments=entries)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(out)


if __name__ == "__main__":
    main()
