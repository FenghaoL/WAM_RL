"""Freeze v2 data splits and generate compute-matched three-GPU experiments."""
from __future__ import annotations
import copy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from scripts.so101_pipeline.hil_human_only_dataset import _human_segments_from_columns, _to_numpy

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RUN_ROOT = ROOT / "runs/ovcr_adapter_padapter_v2"
BASE = "runs/so101_ahawam_padapter_v2/full_zero2_offload_ahawam_base/formal-padapterv2-sft-ahawambase-bs4-acc1-4ep-3gpu-20260901-001743/model"
HIL = "data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v2"

# Fixed step count: one complete demo pass in the 1:1 condition (15984 / 6).
EXPERIMENTS = [
    ("p1_v32_balanced", "value", "balanced_2_2", 1.0, 1),
    ("p1_q32_balanced", "query", "balanced_2_2", 1.0, 1),
    ("p1_t4_balanced", "token_kv", "balanced_2_2", 1.0, 1),
    ("p2_v32_natural", "value", "natural", 1.0, 2),
    ("p2_v32_3to1", "value", "balanced_3_1", 1.0, 2),
    ("p2_v32_noanchor", "value", "balanced_2_2", 0.0, 2),
    ("p3_v32_human_only", "value", "human_only", 0.0, 3),
]


def audit(hil_root: str = HIL, split_tag: str = "20260905"):
    """Freeze the HUMAN split of one HIL collection; defaults reproduce the v2 manifest."""
    records, selections = [], {"train": {}, "val": {}}
    for root in sorted((ROOT / hil_root).iterdir()):
        relative = str(root.relative_to(ROOT))
        rows = []
        for path in sorted((root / "data").glob("chunk-*/*.parquet")):
            table = pq.read_table(path)
            names = ["episode_index", "frame_index", "tick_index", "control_source", "intervention",
                     "human_action_valid", "dropped_tick", "action_supervision_mask"]
            cols = {n: _to_numpy(table[n]) for n in names}
            segments = _human_segments_from_columns(cols)
            lengths = [b-a for a,b in segments]
            episode = int(cols["episode_index"][0])
            mask = np.zeros(len(table), dtype=bool)
            for a,b in segments: mask[a:b] = True
            action = np.array(table["action"].to_pylist(), dtype=np.float64)
            if not np.isfinite(action[mask]).all(): raise ValueError(f"Nonfinite HUMAN actions in {path}")
            rows.append(dict(root=relative, episode=episode, raw_frames=len(table),
                             human_frames=sum(lengths), segment_lengths=lengths,
                             clamped_human=int((_to_numpy(table["safety_clamped"]).astype(bool)&mask).sum()),
                             parquet_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        eligible = [r["episode"] for r in rows if r["human_frames"] > 0]
        # Per-session raw-episode split, independent of optimizer/model seed.
        ordered = sorted(eligible, key=lambda ep: hashlib.sha256(f"{split_tag}:{relative}:{ep}".encode()).hexdigest())
        nval = min(len(ordered)-1, max(1, round(.2 * len(ordered)))) if len(ordered) > 1 else 0
        val = set(ordered[:nval])
        selections["val"][relative] = sorted(val)
        selections["train"][relative] = sorted(set(eligible)-val)
        for row in rows:
            row["split"] = "val" if row["episode"] in val else "train"
            records.append(row)
    lengths = [n for row in records for n in row["segment_lengths"]]
    result = dict(schema_version=1, split_seed=int(split_tag), hil_root=hil_root, correction_episode_selection=selections,
                  episodes=records, raw_episodes=len(records), raw_frames=sum(r["raw_frames"] for r in records),
                  human_frames=sum(lengths), human_segments=len(lengths),
                  segment_quantiles=np.quantile(lengths,[0,.25,.5,.75,1]).tolist(),
                  segments_shorter_than_8=sum(n<8 for n in lengths),
                  segments_shorter_than_49=sum(n<49 for n in lengths))
    for split in ("train","val"):
        chosen = [r for r in records if r["split"] == split]
        result[split] = dict(episodes=sum(r["human_frames"]>0 for r in chosen),
                             human_frames=sum(r["human_frames"] for r in chosen),
                             human_segments=sum(len(r["segment_lengths"]) for r in chosen))
    return result


def main():
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    if (RUN_ROOT / "suite.pid").exists():
        raise RuntimeError("A suite PID file exists: do not regenerate configurations of an active suite")
    report = audit()
    cfg0 = OmegaConf.load(ROOT / BASE / "config.yaml")
    p0 = ROOT / BASE / "checkpoints/weights/step_005328.pt"
    report["checkpoint"] = dict(path=str(p0.relative_to(ROOT)), size=p0.stat().st_size, mtime_ns=p0.stat().st_mtime_ns)
    report["norm_stats"] = dict(path=cfg0.data.train.pretrained_norm_stats,
                                sha256=hashlib.sha256((ROOT/cfg0.data.train.pretrained_norm_stats).read_bytes()).hexdigest())
    report["demo_train_frames"] = sum(json.loads(line)["length"] for root in cfg0.data.train.dataset_dirs
                                     for line in (ROOT/root/"meta/episodes.jsonl").read_text().splitlines() if line.strip())
    report["demo_val_frames"] = sum(json.loads(line)["length"] for root in cfg0.data.val.dataset_dirs
                                   for line in (ROOT/root/"meta/episodes.jsonl").read_text().splitlines() if line.strip())
    report["natural_human_fraction_train"] = report["train"]["human_frames"]/(report["train"]["human_frames"]+report["demo_train_frames"])
    (RUN_ROOT / "data_manifest.json").write_text(json.dumps(report, indent=2)+"\n")
    manifest_hash = hashlib.sha256((RUN_ROOT/"data_manifest.json").read_bytes()).hexdigest()
    configurations = RUN_ROOT / "configs"
    configurations.mkdir(exist_ok=True)
    steps = math.ceil(report["demo_train_frames"]/6)
    for name, mode, sampling, anchor, priority in EXPERIMENTS:
        cfg = copy.deepcopy(cfg0)
        cfg.data.train._target_ = "scripts.so101_pipeline.hil_human_only_dataset.SO101DemoHumanCorrectionDataset"
        cfg.data.train.demo_dataset_dirs = list(cfg0.data.train.dataset_dirs)
        del cfg.data.train.dataset_dirs
        cfg.data.train.correction_dataset_dirs = list(report["correction_episode_selection"]["train"])
        cfg.data.train.correction_episode_selection = report["correction_episode_selection"]["train"]
        cfg.data.train.sampling_strategy = sampling
        cfg.data.train.video_supervision_min_segment_frames = 49
        correction_val = copy.deepcopy(cfg0.data.val)
        correction_val._target_ = "scripts.so101_pipeline.hil_human_only_dataset.SO101HumanCorrectionRobotVideoDataset"
        correction_val.dataset_dirs = list(report["correction_episode_selection"]["val"])
        correction_val.episode_selection = report["correction_episode_selection"]["val"]
        cfg.correction_val = correction_val
        cfg.init_checkpoint = str(p0.relative_to(ROOT))
        cfg.resume = None
        cfg.base_init_checkpoint = None
        cfg.model.ovcr_adapter = OmegaConf.load(HERE.parent / f"{mode}.yaml")
        cfg.model.ovcr_anchor_weight = anchor
        cfg.model.loss.lambda_video = 0.0
        cfg.model.checkpoint_shape_adapt = False
        cfg.freeze_video_dit = True
        cfg.history_aware_batching = False
        cfg.batch_size = 4
        cfg.gradient_accumulation_steps = 1
        cfg.learning_rate = 1e-4
        cfg.num_workers = 2
        cfg.max_steps = steps
        cfg.num_epochs = 20  # max_steps takes precedence in the stock trainer
        cfg.eval_every = 444
        cfg.save_every = 888
        cfg.eval_num_val_batches = 4
        cfg.eval_num_action_batches = 1
        cfg.save_at_end = True
        cfg.wandb.enabled = False
        cfg.zero2_save_training_state = False
        cfg.output_dir = str((RUN_ROOT/name/"model").relative_to(ROOT))
        cfg.experiment = dict(name=name, priority=priority, manifest_sha256=manifest_hash,
                              expected_world_size=3, global_batch=12)
        OmegaConf.save(cfg,configurations/f"{name}.yaml",resolve=True)
    print(json.dumps({k:v for k,v in report.items() if k not in {"episodes","correction_episode_selection"}},indent=2))


if __name__ == "__main__": main()
