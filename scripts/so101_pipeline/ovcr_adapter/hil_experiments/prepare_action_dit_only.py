"""Action-DiT-only correction training for both tasks.

Same recipe as each task's ``p1_full_balanced_video`` run -- start from the task SFT,
1:1 demo/correction batches, history crossing the takeover, same step budget, lr and
schedule -- with one change: the video DiT is frozen and the video loss is switched
off, so only the action side is updated.

Trainable after ``freeze_video_dit`` (trainer._apply_dit_only_train_mode):
``dit.mixtures['action']`` (action expert), ``dit.chunk_kv_cache_editor`` (the native
OVCR routing), and the extra modules ``action_obs_visual_proj`` /
``chunk_obs_query_encoder``.  Frozen: ``dit.mixtures['video']`` (video DiT), VAE and
text encoder.  No adapter is installed; this is a full fine-tune restricted to the
action pathway, which is why the learning rate stays at the full-run 1e-5.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[4]
RUN_ROOT_RELATIVE = "runs/ovcr_adapter_action_dit_only"
RUN_ROOT = ROOT / RUN_ROOT_RELATIVE
WANDB_GROUP = "action-dit-only"
# (new experiment, source run root, source experiment, manifest copy name), in queue order
EXPERIMENTS = [
    ("p1_action_dit_only_hilv2", "runs/ovcr_adapter_cylinder_hil_v2",
     "p1_full_balanced_video_hilv2", "data_manifest_cylinder_hil_v2.json"),
    ("p1_action_dit_only_hilv3", "runs/ovcr_adapter_padapter_hil_v3",
     "p1_full_balanced_video_hilv3", "data_manifest_padapter_hil_v3.json"),
]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    (RUN_ROOT / "configs").mkdir(parents=True, exist_ok=True)
    plan = []
    for new_name, source_root, source_name, manifest_name in EXPERIMENTS:
        cfg = OmegaConf.load(ROOT / source_root / "configs" / f"{source_name}.yaml")
        # the recipe this variant is derived from must be the plain 1:1 full run
        if (cfg.data.train.sampling_strategy != "balanced_2_2"
                or bool(cfg.freeze_video_dit)
                or float(cfg.model.loss.lambda_video) != 1.0
                or cfg.experiment.get("training_scope") != "full"
                or cfg.model.ovcr_adapter is not None):
            raise ValueError(f"{source_name}: not the plain full 1:1 recipe")

        source_manifest = (ROOT / cfg.experiment.run_root /
                           cfg.experiment.get("manifest_file", "data_manifest.json"))
        if sha256(source_manifest) != cfg.experiment.manifest_sha256:
            raise ValueError(f"{source_name}: data manifest changed since it was trained")
        manifest_copy = RUN_ROOT / manifest_name
        if manifest_copy.exists() and sha256(manifest_copy) != sha256(source_manifest):
            raise ValueError(f"{manifest_copy} differs from {source_manifest}")
        shutil.copyfile(source_manifest, manifest_copy)

        cfg.freeze_video_dit = True
        cfg.model.loss.lambda_video = 0.0
        # routes run.py to ActionOnlyTrainer (the full baseline refuses a frozen video DiT)
        cfg.experiment.training_scope = "action_only"
        cfg.output_dir = f"{RUN_ROOT_RELATIVE}/{new_name}/model"
        cfg.wandb.name = new_name
        cfg.wandb.group = WANDB_GROUP
        cfg.experiment.name = new_name
        cfg.experiment.run_root = RUN_ROOT_RELATIVE
        cfg.experiment.manifest_file = manifest_name
        cfg.experiment.reference_run = f"{source_root}/{source_name}"
        cfg.experiment.training_pathway = dict(
            trainable="action expert + chunk_kv_cache_editor (native OVCR routing) + "
                      "action_obs_visual_proj + chunk_obs_query_encoder",
            frozen="video DiT expert, VAE, text encoder",
            video_loss="disabled (lambda_video=0); the video branch still runs in eval "
                       "mode to supply visual context",
        )
        OmegaConf.save(cfg, RUN_ROOT / "configs" / f"{new_name}.yaml")
        plan.append(dict(name=new_name, reference=f"{source_root}/{source_name}",
                         init_checkpoint=str(cfg.init_checkpoint), max_steps=int(cfg.max_steps),
                         learning_rate=float(cfg.learning_rate),
                         wandb_project=str(cfg.wandb.project)))
        print(json.dumps(plan[-1]))
    (RUN_ROOT / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")


if __name__ == "__main__":
    main()
