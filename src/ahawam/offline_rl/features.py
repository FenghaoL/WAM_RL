"""Causal offline feature extraction using the existing deployment policy.

This defines a NEW reproducible conditioning protocol. It does not reconstruct
the original collector's asynchronous hidden state from RGB or request hashes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .data import sha256
from .timeline import history_start


def pool_routed_values(cache, layer_indices=(-1,), num_tokens=64):
    """Read post-OVCR V at chosen layers; retain a fixed ordered token grid.

    1D adaptive spatial pooling is an explicit baseline. It can blur camera
    boundaries and should be compared with view-aware 2D pooling later.
    """
    values = []
    for layer in layer_indices:
        v = cache[layer]["v"].detach().float()
        if v.ndim != 3 or v.shape[1] < num_tokens:
            raise ValueError("Expected routed V [B,N,D] with N >= num_tokens")
        values.append(F.adaptive_avg_pool1d(v.transpose(1,2), num_tokens).transpose(1,2))
    return torch.cat(values, dim=-1).cpu()


def causal_history_frames(table, frame, count, stride, timeline_mode="episode_frames"):
    """Use earlier stored episode observations, without resetting at a pause."""
    lower = history_start(table, frame, timeline_mode)
    return sorted([frame] + [frame-j*stride for j in range(1,count+1) if frame-j*stride >= lower])


def export_features(data, deploy_config: Path, output: Path, history_count=0,
                    history_stride=8, num_tokens=64, layers=(-1,), limit=None):
    if not torch.cuda.is_available():
        raise RuntimeError("Full WAM feature export requires CUDA; CPU learner tests are separate")
    import json
    import pandas as pd
    from deploy.server.ahawam_policy import AHAWAMPolicy
    from scripts.frame_annotation.annotate import Video

    if history_count < 0 or history_stride < 1:
        raise ValueError("Invalid history count/stride")
    policy = AHAWAMPolicy(str(deploy_config))
    policy.eval()
    policy.seed = 42 if policy.seed is None else policy.seed
    if policy.processor.action_state_transforms:
        raise ValueError("Exporter requires absolute targets without extra action/state transforms")
    if data["config"]["chunk_size"] > policy.model.action_chunk_size:
        raise ValueError("Requested chunk exceeds the deployment reference horizon")
    if policy.model_variant != "ahawam":
        raise ValueError("This feature source requires AHA-WAM's routed KV cache")
    requests = {}
    for r in data["records"]:
        for key, frame in ((r["observation_key"],r["start_frame"]), (r["next_observation_key"],r["next_frame"])):
            requests[key] = (r["root"],r["episode_index"],frame)
    needed = sorted(requests.items(),key=lambda x:x[1])
    if limit is not None:
        needed = needed[:limit]
    samples, opened, video_hashes = {}, {}, {}
    source_hashes = {e["episode_uid"]:e["source"]["parquet_sha256"] for e in data["episodes"]}
    action_key = policy.processor.shape_meta["action"][0]["key"]
    normalizer = policy.processor.normalizer.normalizers["action"][action_key]
    if normalizer.scale.ndim != 1:
        raise ValueError("Feature exporter requires global, non-stepwise action normalization")
    try:
        for index, (key,(root_name,ep,frame)) in enumerate(needed):
            root = Path(root_name)
            if (root_name,ep) not in opened:
                info = json.loads((root/"meta/info.json").read_text())
                fields = dict(episode_index=ep,episode_chunk=ep//int(info.get("chunks_size",1000)))
                table_path = root/info["data_path"].format(**fields)
                uid = key.rsplit(":",1)[0]
                if sha256(table_path) != source_hashes[uid]:
                    raise ValueError("Source data changed after transition preparation")
                table = pd.read_parquet(table_path)
                videos = {cam:Video(root/info["video_path"].format(**fields,video_key=f"observation.images.{cam}"),len(table))
                          for cam in ("fixed","wrist")}
                video_hashes[uid] = {cam:sha256(root/info["video_path"].format(**fields,video_key=f"observation.images.{cam}")) for cam in videos}
                tasks = {x["task_index"]:x["task"] for x in map(json.loads,(root/"meta/tasks.jsonl").read_text().splitlines())}
                task = tasks[int(table.task_index.iloc[0])]
                opened[root_name,ep] = (table,videos,task)
            table,videos,task = opened[root_name,ep]
            policy.reset_obsrvationwindows()
            torch.manual_seed(policy.seed)
            policy.set_language_instruction(task)
            with torch.inference_mode():
                for history_frame in causal_history_frames(table,frame,history_count,history_stride,
                        data["config"].get("timeline_mode", "strict_ticks")):
                    images = {cam:video.frame(history_frame) for cam,video in videos.items()}
                    composite = policy.prepare_camera_images(images)
                    policy.update_observation_window([composite],table["observation.state"].iloc[history_frame])
                    ref = policy.get_action()
                kv = policy.model._inference_state["_chunk_video_kv_cache"]
                tokens = pool_routed_values(kv,layers,num_tokens)[0]
                ref_norm = normalizer.forward(torch.as_tensor(ref,dtype=torch.float32).cpu())
                samples[key] = dict(tokens=tokens,valid=torch.ones(num_tokens,dtype=torch.bool),
                                    reference=ref_norm[:data["config"]["chunk_size"]],
                                    state=policy._normalize_state(np.asarray(table["observation.state"].iloc[frame])).cpu()[0])
            print(f"feature {index+1}/{len(needed)} {key}",flush=True)
    finally:
        for _,videos,_ in opened.values():
            for video in videos.values():
                video.close()
    # Exact scale/offset are obtained from the actual processor, not guessed.
    deployment = policy._load_deploy_config(str(deploy_config))
    def asset_path(name):
        path = Path(deployment[name])
        return path if path.is_absolute() else policy.project_root/path
    spec = dict(base_checkpoint=str(asset_path("checkpoint_path")),
                base_checkpoint_sha256=sha256(asset_path("checkpoint_path")),
                stats_sha256=sha256(asset_path("dataset_stats_path")), seed=policy.seed,
                deploy_config_sha256=sha256(deploy_config),
                normalization="deployment processor affine scale/offset, clamped [-5,5]",
                token_source=f"post_OVCR_V/layers={list(layers)}/adaptive_1d_pool={num_tokens}",
                cache_protocol=f"canonical_causal_window_v1/history={history_count}/stride={history_stride}",
                timeline_mode=data["config"].get("timeline_mode", "strict_ticks"),
                source_hashes=source_hashes, video_hashes=video_hashes, synthetic=False)
    code_paths = [Path(__file__),Path("deploy/server/ahawam_policy.py"),
                  Path("src/ahawam/offline_rl/timeline.py"),
                  Path("src/ahawam/models/wan22/ahawam.py"),Path("src/ahawam/models/wan22/ahawam_chunk_base.py"),
                  Path("src/ahawam/models/wan22/mot.py")]
    spec["code_sha256"] = {str(path):sha256(path) for path in code_paths}
    output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(format="ahawam_feature_cache/v1",samples=samples,feature_spec=spec,
                    action_scale=normalizer.scale.cpu(),action_offset=normalizer.offset.cpu()),output)
