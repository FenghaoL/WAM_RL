"""Power-adapter HIL v3 full fine-tunes: segment-as-demo and demo:DAgger 2:1.

(The cylinder segment-as-demo run is the already prepared, never trained
``runs/icra2027_cylinder_controls/configs/full_context_truncated.yaml``, which
differs from ``p1_full_balanced_video_hilv2`` only in not crossing the takeover
start; it is queued as is instead of being duplicated here.)

``*_segdemo`` treats every HUMAN takeover span like an ordinary demo episode:
history frames come only from inside the span (fewer when the span is too young),
the future action/video window stops at the span end, and video loss is used only
when the 13 sampled future images all lie inside the span. No history from the
policy-driven part before the takeover, no video extension after it.

``padapter_..._demo2to1`` is the power-adapter counterpart of the cylinder
``_demo2to1`` run: the plain hil_v3 configuration (history may cross the takeover,
48 executing frames per step) with 8 demo + 4 correction samples per 12-sample
global batch.

Each run is copied from the corresponding plain configuration and differs from it
only as described. Validation (demo val + HUMAN val, including its history
setting) is left identical to that plain run, so curves compare directly. Both
tasks share one run root and one queue; each run names its own manifest copy.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
RUN_ROOT_RELATIVE = "runs/ovcr_adapter_padapter_hil_v3_segdemo_2to1"
RUN_ROOT = ROOT / RUN_ROOT_RELATIVE
SOURCES = {
    "padapter_hil_v3": dict(root="runs/ovcr_adapter_padapter_hil_v3", config="p1_full_balanced_video_hilv3",
                            project="ahawam-so101-padapter-hil-v3", group="padapter-hil-v3-full"),
}
SEGMENT_AS_DEMO = dict(correction_history_crosses_segment=False, correction_history_skip_system_hold=False,
                       correction_history_frame_stride=None, correction_future_extension="none")
# queue order
EXPERIMENTS = [
    ("p1_full_balanced_video_hilv3_segdemo", "padapter_hil_v3", SEGMENT_AS_DEMO),
    ("p1_full_balanced_video_hilv3_demo2to1", "padapter_hil_v3", dict(sampling_strategy="balanced_2_1")),
]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    (RUN_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    manifests = {}
    for tag, source in SOURCES.items():
        target = RUN_ROOT / f"data_manifest_{tag}.json"
        if target.exists():
            raise FileExistsError(f"Refusing to replace {target}")
        shutil.copyfile(ROOT / source["root"] / "data_manifest.json", target)
        manifests[tag] = target
    summary = {}
    for name, tag, change in EXPERIMENTS:
        source = SOURCES[tag]
        cfg = OmegaConf.load(ROOT / source["root"] / "configs" / f"{source['config']}.yaml")
        if cfg.experiment.get("training_scope") != "full":
            raise ValueError(f"{source['config']} is not a full fine-tune configuration")
        for key, value in change.items():
            cfg.data.train[key] = value
        cfg.output_dir = f"{RUN_ROOT_RELATIVE}/{name}/model"
        cfg.wandb.enabled = True
        cfg.wandb.project = source["project"]
        cfg.wandb.group = source["group"]
        cfg.wandb.name = name
        cfg.wandb.mode = "online"
        cfg.experiment.name = name
        cfg.experiment.run_root = RUN_ROOT_RELATIVE
        cfg.experiment.manifest_file = manifests[tag].name
        cfg.experiment.manifest_sha256 = sha256(manifests[tag])
        cfg.experiment.reference_run = f"{source['root']}/{source['config']}"
        cfg.experiment.variant = {k: v for k, v in change.items()}
        OmegaConf.save(cfg, RUN_ROOT / "configs" / f"{name}.yaml", resolve=True)
        t = cfg.data.train
        summary[name] = dict(init=cfg.init_checkpoint.rsplit("/", 1)[-1], max_steps=cfg.max_steps,
                             sampling=t.sampling_strategy, crosses=t.get("correction_history_crosses_segment"),
                             skip_hold=t.get("correction_history_skip_system_hold"),
                             extension=t.get("correction_future_extension", "none"),
                             val_crosses=cfg.correction_val.get("history_crosses_segment"),
                             full_window_rule=t.get("video_supervision_require_full_window"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
