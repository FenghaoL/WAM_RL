"""Correction runs whose HUMAN source adds the 2026-09-11 correction clips.

Presets, each with its own run root, frozen manifest and configurations:

* ``nohist``: ``p1_full`` / V32 / Q32 / T4 on the no-history cylinder SFT
  ``sft_nohist_pretrained_4ep`` with ``num_history_frames=0``
  (``runs/ovcr_adapter_cylinder_v1_nohist_clips``).
* ``hist6``: V32 / Q32 / T4 on the original cylinder task SFT
  ``formal-cylinderv1-sft-...-20260905-022135/step_006572.pt`` with its trained
  history of 6 frames (``runs/ovcr_adapter_cylinder_v1_clips``).

Each configuration is copied from its history-6 cylinder configuration.  Only
the initial checkpoint, the history length, the correction source, outputs and
wandb names change.  The HUMAN correction source is the frozen HIL HUMAN train
split **plus** every episode of the ``20260911_*`` sessions of the cylinder demo
directory: teleoperated recoveries from perturbed insertion poses, so each
episode is one correction segment (``data_source=1``), never a demo.  Same four
demo sessions, 1:1 demo/correction batches, 3285 steps, learning rates, anchor,
video rule (only corrections with a fully valid 49-frame window get video loss
in the full run; adapters have no video loss) and the same demo/HUMAN
validation subsets.  The HIL session ``20260910_152524_val`` is not used: the
frozen HIL split is copied verbatim and its parquet hashes are re-checked.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path

import pyarrow.parquet as pq
from omegaconf import OmegaConf

from scripts.so101_pipeline.ovcr_adapter.cylinder_experiments.prepare import (
    P0, ROOT, RUN_ROOT as PARENT_ROOT, RUN_ROOT_RELATIVE as PARENT_RELATIVE)

DEMO_ROOT = "data/so101_ahawam_cylinder_insertion_v1/01_pick_up_and_insert_cylinder"
CLIP_GLOB = "20260911_*"
TASK = "Pick up the cylinder and insert it into the matching hole."
SFT_NOHIST = f"{PARENT_RELATIVE}/sft_nohist_pretrained_4ep/model/checkpoints/weights/step_006572.pt"
PRESETS = {
    "recover": dict(run_root="runs/ovcr_adapter_cylinder_v1_recover", init=P0, history=6,
                    suffix="_recover", group="cylinder-v1-recover", task="cylinder_v1_recover",
                    experiments=["p1_v32_balanced", "p1_q32_balanced", "p1_t4_balanced"],
                    clip_glob="20260911_185932_recover", use_hil_human=False, trim_leading_seconds=0.5,
                    save_every=547),  # ~900 correction frames are replayed ~21x here: keep early checkpoints
    "nohist": dict(run_root="runs/ovcr_adapter_cylinder_v1_nohist_clips", init=SFT_NOHIST, history=0,
                   suffix="_nohist_clips", group="cylinder-v1-nohist-clips", task="cylinder_v1_nohist_clips",
                   experiments=["p1_full_balanced_video", "p1_v32_balanced", "p1_q32_balanced", "p1_t4_balanced"]),
    "hist6": dict(run_root="runs/ovcr_adapter_cylinder_v1_clips", init=P0, history=6,
                  suffix="_clips", group="cylinder-v1-clips", task="cylinder_v1_clips",
                  experiments=["p1_v32_balanced", "p1_q32_balanced", "p1_t4_balanced"]),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def clip_report(demo_dirs: list[str], glob_pattern: str = CLIP_GLOB,
                trim_leading_seconds: float = 0.0) -> dict:
    sessions = sorted((ROOT / DEMO_ROOT).glob(glob_pattern))
    if not sessions:
        raise FileNotFoundError(f"No {glob_pattern} sessions under {DEMO_ROOT}")
    rates = {json.loads((session / "meta/info.json").read_text())["fps"] for session in sessions}
    if len(rates) != 1:
        raise ValueError(f"Sessions disagree on fps: {rates}")
    fps = int(rates.pop())
    # Round up so at least the requested idle head is removed.
    trim = math.ceil(float(trim_leading_seconds) * fps)
    rows = []
    for session in sessions:
        relative = str(session.relative_to(ROOT))
        if relative in demo_dirs:
            raise ValueError(f"{relative} is already a demo training directory")
        tasks = [json.loads(line)["task"] for line in (session / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
        if tasks != [TASK]:
            raise ValueError(f"{relative}: unexpected task text {tasks}")
        lengths = [json.loads(line)["length"] for line in (session / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]
        parquets = sorted((session / "data").glob("chunk-*/*.parquet"))
        if sum(pq.read_metadata(p).num_rows for p in parquets) != sum(lengths) or len(parquets) != len(lengths):
            raise ValueError(f"{relative}: parquet rows disagree with meta/episodes.jsonl")
        if "control_source" in set(pq.read_schema(parquets[0]).names):
            raise ValueError(f"{relative} is a HIL recording; it belongs in the HIL loader")
        videos = sorted(session.glob("videos/chunk-*/*/*.mp4"))
        if len(videos) != 2 * len(lengths):
            raise ValueError(f"{relative}: expected two encoded videos per episode, found {len(videos)}")
        kept = [n - trim for n in lengths if n > trim]
        rows.append(dict(root=relative, episodes=len(lengths), frames=sum(lengths), episode_lengths=lengths,
                         episodes_after_trim=len(kept), frames_after_trim=sum(kept),
                         episode_lengths_after_trim=kept,
                         episodes_at_least_49_after_trim=sum(n >= 49 for n in kept),
                         full_video_window_starts_after_trim=sum(max(0, n - 48) for n in kept),
                         parquet_sha256=[sha256(p) for p in parquets]))
    lengths = [n for row in rows for n in row["episode_lengths"]]
    kept = [n for row in rows for n in row["episode_lengths_after_trim"]]
    return dict(glob=f"{DEMO_ROOT}/{glob_pattern}", sessions=rows, fps=fps,
                trim_leading_seconds=float(trim_leading_seconds), trim_leading_frames=trim,
                trim_mechanism=("load-time logical view: SO101CorrectionClipDataset drops the first "
                                "trim_leading_frames of every episode; stored frames are never modified"),
                episodes=len(lengths), frames=sum(lengths),
                episodes_after_trim=len(kept), frames_after_trim=sum(kept),
                dropped_episodes=len(lengths) - len(kept),
                episodes_at_least_49_after_trim=sum(n >= 49 for n in kept),
                full_video_window_starts_after_trim=sum(max(0, n - 48) for n in kept))


def verify_hil_unchanged(parent: dict) -> None:
    expected: dict[str, list[str]] = {}
    for row in parent["episodes"]:
        expected.setdefault(row["root"], []).append(row["parquet_sha256"])
    for root, hashes in expected.items():
        actual = [sha256(p) for p in sorted((ROOT / root / "data").glob("chunk-*/*.parquet"))]
        if sorted(actual) != sorted(hashes):
            raise ValueError(f"HIL recordings changed since the frozen split: {root}")


def generate(preset: str) -> None:
    spec = PRESETS[preset]
    run_root_relative = spec["run_root"]
    run_root = ROOT / run_root_relative
    if (run_root / "data_manifest.json").exists():
        raise FileExistsError(f"Refusing to replace {run_root / 'data_manifest.json'}")
    parent_path = PARENT_ROOT / "data_manifest.json"
    parent_hash = sha256(parent_path)
    parent = json.loads(parent_path.read_text())
    verify_hil_unchanged(parent)
    use_hil = bool(spec.get("use_hil_human", True))
    clips = clip_report(parent["demo_train_dirs"], spec.get("clip_glob", CLIP_GLOB),
                        spec.get("trim_leading_seconds", 0.0))
    manifest = copy.deepcopy(parent)
    manifest.update(
        schema_version=2, task=spec["task"], parent_manifest=str(parent_path.relative_to(ROOT)),
        parent_manifest_sha256=parent_hash, correction_clips=clips, init_checkpoint=spec["init"],
        num_history_frames=spec["history"],
        excluded_hil_sessions={"data/ahawam_hil_raw/so101_ahawam_cylinder_hil_v1/20260910_152524_val":
                               "added after the frozen split; named _val, never used for training"})
    hil_train = parent["train"]["human_frames"] if use_hil else 0
    manifest["train"] = dict(parent["train"], hil_human_frames=hil_train, hil_human_used=use_hil,
                             clip_frames=clips["frames_after_trim"], clip_episodes=clips["episodes_after_trim"],
                             human_frames=hil_train + clips["frames_after_trim"])
    manifest["natural_human_fraction_train"] = manifest["train"]["human_frames"] / (
        manifest["train"]["human_frames"] + manifest["demo_train_frames"])
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "data_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_hash = sha256(run_root / "data_manifest.json")

    configurations = run_root / "configs"
    configurations.mkdir(exist_ok=True)
    clip_dirs = [row["root"] for row in clips["sessions"]]
    summary = []
    for base in spec["experiments"]:
        name = base + spec["suffix"]
        cfg = OmegaConf.load(PARENT_ROOT / "configs" / f"{base}.yaml")
        if cfg.experiment.manifest_sha256 != parent_hash:
            raise ValueError(f"{base} was not generated from the frozen cylinder manifest")
        cfg.init_checkpoint = spec["init"]
        cfg.model.num_history_frames = spec["history"]
        cfg.data.train.correction_clip_dataset_dirs = clip_dirs
        cfg.data.train.correction_clip_trim_leading_frames = clips["trim_leading_frames"]
        if not use_hil:  # HIL HUMAN stays in validation only
            cfg.data.train.correction_dataset_dirs = []
            cfg.data.train.correction_episode_selection = {}
        cfg.output_dir = f"{run_root_relative}/{name}/model"
        cfg.wandb.name = name
        cfg.wandb.group = spec["group"]
        cfg.experiment.name = name
        cfg.experiment.run_root = run_root_relative
        cfg.experiment.manifest_sha256 = manifest_hash
        cfg.experiment.initialization = spec["init"]
        cfg.experiment.num_history_frames = spec["history"]
        cfg.experiment.correction_sources = (["hil_human_train_split"] if use_hil else []) + [
            f"{DEMO_ROOT}/{spec.get('clip_glob', CLIP_GLOB)}"]
        cfg.experiment.correction_clip_trim_leading_frames = clips["trim_leading_frames"]
        if spec.get("save_every"):
            cfg.save_every = int(spec["save_every"])
        cfg.experiment.reference_run = f"{PARENT_RELATIVE}/{base}"
        OmegaConf.save(cfg, configurations / f"{name}.yaml", resolve=True)
        summary.append(dict(name=name, scope=cfg.experiment.training_scope, init=cfg.init_checkpoint,
                            history=cfg.model.num_history_frames, lr=cfg.learning_rate,
                            lambda_video=cfg.model.loss.lambda_video, anchor=cfg.model.ovcr_anchor_weight,
                            max_steps=cfg.max_steps))
    repeat = cfg.max_steps * 6 / manifest["train"]["human_frames"]
    print(json.dumps(dict(preset=preset, clip_sessions=[r["root"].rsplit("/", 1)[-1] for r in clips["sessions"]],
                          trim_leading_frames=clips["trim_leading_frames"], fps=clips["fps"],
                          clip_frames_before_trim=clips["frames"], dropped_episodes=clips["dropped_episodes"],
                          hil_human_used=use_hil,
                          hil_human_train=hil_train, clip_frames=clips["frames_after_trim"],
                          clip_episodes=clips["episodes_after_trim"],
                          correction_train=manifest["train"]["human_frames"], demo_train=manifest["demo_train_frames"],
                          natural_correction_fraction=round(manifest["natural_human_fraction_train"], 4),
                          correction_passes_at_1to1=round(repeat, 2)), indent=2))
    print(json.dumps(summary, indent=2))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preset", choices=sorted(PRESETS), required=True)
    generate(parser.parse_args(argv).preset)


if __name__ == "__main__":
    main()
