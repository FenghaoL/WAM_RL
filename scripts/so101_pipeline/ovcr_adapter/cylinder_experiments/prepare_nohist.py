"""Generate the two no-history (``num_history_frames=0``) cylinder-v1 experiments.

1. ``sft_nohist_pretrained_4ep``: the recipe of the cylinder task SFT
   ``formal-cylinderv1-sft-ahawambase-bs4-acc1-4ep-3gpu-20260905-022135``
   (AHA pretrained init with 14->6 shape adaptation, the same four demo sessions,
   4 epochs, global batch 12, lr 1e-5, video + action loss), with the video
   history memory switched off.  Its resolved config is copied and only
   ``model.num_history_frames`` plus bookkeeping keys are changed.
2. ``p1_full_balanced_video_nohist``: ``p1_full_balanced_video`` unchanged except
   that it starts from (1)'s final weights and also runs without history.

Both reuse the frozen cylinder-v1 manifest, so demo validation is the held-out
``20260904_024009_val`` session and HUMAN validation is the same 13 rollouts,
evaluated on the same fixed subsets as every history-6 run in this directory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math

from omegaconf import OmegaConf

from scripts.so101_pipeline.ovcr_adapter.cylinder_experiments.prepare import (
    BASE, PRETRAINED, ROOT, RUN_ROOT, RUN_ROOT_RELATIVE, WANDB_PROJECT)

SFT = "sft_nohist_pretrained_4ep"
P1 = "p1_full_balanced_video_nohist"
REFERENCE_P1 = "p1_full_balanced_video"
WANDB_GROUP = "cylinder-v1-nohist"


def main():
    configurations = RUN_ROOT / "configs"
    for name in (SFT, P1):
        if (configurations / f"{name}.yaml").exists():
            raise FileExistsError(f"Refusing to replace {configurations / f'{name}.yaml'}")
    manifest_path = RUN_ROOT / "data_manifest.json"
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    manifest = json.loads(manifest_path.read_text())
    reference = OmegaConf.load(configurations / f"{REFERENCE_P1}.yaml")
    if reference.experiment.manifest_sha256 != manifest_hash:
        raise ValueError("Frozen manifest no longer matches the history-6 configurations")

    # 1. Task SFT without history: the original resolved SFT config, one model change.
    original = OmegaConf.load(ROOT / BASE / "config.yaml")
    sft = copy.deepcopy(original)
    sft.model.num_history_frames = 0
    sft.model.ovcr_adapter = None
    sft.model.ovcr_anchor_weight = 0.0
    sft.model.loss.lambda_video = 1.0  # the original relied on the same default
    sft.init_checkpoint = PRETRAINED
    sft.resume = None
    sft.base_init_checkpoint = None
    sft.correction_val = copy.deepcopy(reference.correction_val)
    steps_per_epoch = math.ceil(manifest["demo_train_frames"] / 12)
    sft.num_epochs = 4
    sft.max_steps = 4 * steps_per_epoch
    sft.save_every = steps_per_epoch
    sft.eval_every = steps_per_epoch // 4  # train_entry.py: steps_per_epoch // eval_times_per_epoch
    sft.pop("eval_times_per_epoch", None)
    sft.save_at_end = False  # save_every already lands on the last step
    # Same fixed subsets as the history-6 correction runs, so numbers line up.
    sft.eval_num_val_batches = int(reference.eval_num_val_batches)
    sft.eval_num_action_batches = int(reference.eval_num_action_batches)
    sft.zero2_save_training_state = False
    sft.wandb.enabled = True
    sft.wandb.project = WANDB_PROJECT
    sft.wandb.group = WANDB_GROUP
    sft.wandb.name = SFT
    sft.wandb.mode = "online"
    sft.output_dir = f"{RUN_ROOT_RELATIVE}/{SFT}/model"
    sft.experiment = dict(
        name=SFT, task="cylinder_v1", run_root=RUN_ROOT_RELATIVE, training_scope="full",
        training_data="demo_only", initialization=PRETRAINED, num_history_frames=0,
        reference_run=BASE, steps_per_epoch=steps_per_epoch, planned_epochs=4,
        manifest_sha256=manifest_hash, expected_world_size=3, global_batch=12)
    # Only history, plus the explicit lambda_video=1 default, may differ in the model.
    changed = {k for k in original.model if original.model[k] != sft.model.get(k)}
    loss_changed = {k for k in sft.model.loss if original.model.loss.get(k, 1.0) != sft.model.loss[k]}
    if changed - {"num_history_frames", "loss"} or loss_changed:
        raise ValueError(f"Unexpected model changes vs the original SFT: {sorted(changed | loss_changed)}")
    final_sft = f"{RUN_ROOT_RELATIVE}/{SFT}/model/checkpoints/weights/step_{sft.max_steps:06d}.pt"

    # 2. The 1:1 full correction run on top of the no-history SFT.
    p1 = copy.deepcopy(reference)
    p1.init_checkpoint = final_sft
    p1.model.num_history_frames = 0
    p1.output_dir = f"{RUN_ROOT_RELATIVE}/{P1}/model"
    p1.wandb.name = P1
    p1.wandb.group = WANDB_GROUP
    p1.experiment.name = P1
    p1.experiment.initialization = final_sft
    p1.experiment.num_history_frames = 0
    p1.experiment.reference_run = REFERENCE_P1

    OmegaConf.save(sft, configurations / f"{SFT}.yaml", resolve=True)
    OmegaConf.save(p1, configurations / f"{P1}.yaml", resolve=True)
    for cfg in (sft, p1):
        print(json.dumps(dict(
            name=cfg.experiment.name, init=cfg.init_checkpoint, history=cfg.model.num_history_frames,
            shape_adapt=cfg.model.checkpoint_shape_adapt, train=cfg.data.train._target_.rsplit(".", 1)[-1],
            sampling=cfg.data.train.get("sampling_strategy", "natural"), lr=cfg.learning_rate,
            max_steps=cfg.max_steps, save_every=cfg.save_every, eval_every=cfg.eval_every,
            eval_batches=[cfg.eval_num_val_batches, cfg.eval_num_action_batches],
            history_aware_batching=cfg.history_aware_batching, num_workers=cfg.num_workers)))


if __name__ == "__main__":
    main()
