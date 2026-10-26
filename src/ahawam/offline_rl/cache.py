"""Sharded BF16 last-layer KV export; bounded memory and restartable shards."""
from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .data import sha256


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")
    os.replace(temporary,path)


def conditioning_frames(table,frame,offset_period=8,history_count=6,history_stride=48):
    """Deterministic uniform offset coverage, reset at every missing tick.

    Planner sees anchor t-offset; local OVCR/action sees t. No future images.
    Each observation has one variant, rather than an eightfold cache expansion.
    """
    gaps = np.flatnonzero(np.diff(table.tick_index.to_numpy()[:frame+1]) != 1)
    lower = int(gaps[-1])+1 if len(gaps) else 0
    anchor = lower + ((frame-lower)//offset_period)*offset_period
    history = [anchor-j*history_stride for j in range(history_count,0,-1)
               if anchor-j*history_stride >= lower]
    return anchor,history


def memory_stats():
    if not torch.cuda.is_available():
        return {}
    return {name:func()/2**20 for name,func in (
        ("allocated_mib",torch.cuda.memory_allocated),("reserved_mib",torch.cuda.memory_reserved),
        ("peak_allocated_mib",torch.cuda.max_memory_allocated),("peak_reserved_mib",torch.cuda.max_memory_reserved))}


def asset_fingerprint(path,record_path):
    """Reuse a local digest only while absolute path, size and mtime match."""
    path=Path(path).resolve();record_path=Path(record_path)
    stat=path.stat();identity=dict(path=str(path),size=stat.st_size,mtime_ns=stat.st_mtime_ns)
    if record_path.exists():
        previous=json.loads(record_path.read_text())
        if previous.get("identity")==identity:return previous["sha256"]
    digest=sha256(path)
    atomic_json(record_path,dict(identity=identity,sha256=digest))
    return digest


def export_shards(data,deploy_config,output,rank=0,world_size=1,shard_size=128,limit=None,
                  history_count=6,history_stride=48,offset_period=8):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for real WAM export")
    if not 0<=rank<world_size or min(shard_size,offset_period,history_stride)<1 or history_count<0:
        raise ValueError("Invalid exporter settings")
    import pandas as pd
    from deploy.server.ahawam_policy import AHAWAMPolicy
    from scripts.frame_annotation.annotate import Video

    output = Path(output); output.mkdir(parents=True,exist_ok=True)
    started = time.perf_counter()
    policy = AHAWAMPolicy(str(deploy_config)); policy.eval()
    policy.model.requires_grad_(False)
    print(json.dumps(dict(rank=rank,stage="model_loaded",seconds=time.perf_counter()-started,memory=memory_stats())),flush=True)
    if policy.model_variant!="ahawam" or policy.processor.action_state_transforms:
        raise ValueError("Expected absolute-action AHA-WAM")
    if offset_period>policy.model.action_chunk_size:
        raise ValueError("Offset must remain within one action chunk")
    if data["config"]["chunk_size"] != policy.model.action_chunk_size:
        raise ValueError("Reference and dataset chunk lengths differ")
    editor = policy.model.mot.chunk_kv_cache_editor
    editor.offline_capture_layer = editor.num_layers-1
    key = policy.processor.shape_meta["action"][0]["key"]
    normalizer = policy.processor.normalizer.normalizers["action"][key]
    if normalizer.scale.ndim!=1:
        raise ValueError("Expected global affine normalization")
    deployment = policy._load_deploy_config(str(deploy_config))
    def asset(name):
        p=Path(deployment[name]);return p if p.is_absolute() else policy.project_root/p
    spec = dict(format="ahawam_raw_kv/v2",base_checkpoint=str(asset("checkpoint_path")),
        base_checkpoint_sha256=asset_fingerprint(asset("checkpoint_path"),output.parent/"checkpoint_fingerprint.json"),
        deploy_config_sha256=sha256(Path(deploy_config)),stats_sha256=sha256(asset("dataset_stats_path")),
        normalization="deployment affine z-score, clamp [-5,5]",token_source="last-layer post-OVCR K+V; routed weighted V and projected query",
        cache_protocol="causal_independent_anchor_v2",offset_period=offset_period,
        offset_schedule="(raw_frame - contiguous_segment_start) modulo offset_period",history_count=history_count,
        history_stride=history_stride,dtype="bfloat16",seed=policy.seed,synthetic=False,
        action_scale=normalizer.scale.cpu().tolist(),action_offset=normalizer.offset.cpu().tolist(),
        code_sha256={p:sha256(Path(p)) for p in (__file__,"src/ahawam/models/wan22/mot.py",
            "src/ahawam/models/wan22/ahawam.py","deploy/server/ahawam_policy.py")},
        source_hashes={e["episode_uid"]:e["source"]["parquet_sha256"] for e in data["episodes"]})
    spec_path=output/f"spec_rank{rank}.json"
    if spec_path.exists() and json.loads(spec_path.read_text())!=spec:
        raise ValueError("Cache provenance changed; choose a new output directory")
    atomic_json(spec_path,spec)
    print(json.dumps(dict(rank=rank,stage="provenance_ready",seconds=time.perf_counter()-started)),flush=True)
    # Features depend on observations, not reward annotations: export all 103
    # episodes, including temporarily unlabelled successes, for later reuse.
    episodes=sorted(data["episodes"],key=lambda e:e["episode_uid"])
    episodes=episodes[rank::world_size]
    count=0; measurements=[];shards=[];video_hashes={}
    model_load_seconds=time.perf_counter()-started
    for episode in episodes:
        root=Path(episode["root"]);ep=episode["episode_index"];uid=episode["episode_uid"]
        info=json.loads((root/"meta/info.json").read_text());fields=dict(episode_index=ep,episode_chunk=ep//int(info.get("chunks_size",1000)))
        path=root/info["data_path"].format(**fields)
        if sha256(path)!=episode["source"]["parquet_sha256"]:
            raise ValueError("Source changed after prepare")
        table=pd.read_parquet(path)
        tasks={r["task_index"]:r["task"] for r in map(json.loads,(root/"meta/tasks.jsonl").read_text().splitlines())}
        policy.set_language_instruction(tasks[int(table.task_index.iloc[0])])
        videos={cam:Video(root/info["video_path"].format(**fields,video_key=f"observation.images.{cam}"),len(table)) for cam in ("fixed","wrist")}
        video_hashes[uid]={cam:sha256(v.path) if hasattr(v,"path") else sha256(root/info["video_path"].format(**fields,video_key=f"observation.images.{cam}")) for cam,v in videos.items()}
        def observe(frame):
            images={cam:v.frame(frame) for cam,v in videos.items()}
            composite=policy.prepare_camera_images(images)
            policy.update_observation_window([composite],np.array(table["observation.state"].iloc[frame],copy=True))
            return policy._observation["image_tensor"]
        prompt=policy._DEFAULT_PROMPT.format(task=policy.current_instruction)
        context,context_mask=policy._resolve_text_condition(prompt)
        anchor_cached=None
        try:
            for first in range(0,len(table),shard_size):
                if limit is not None and count>=limit:
                    break
                stop=min(first+shard_size,len(table))
                if limit is not None:
                    stop=min(stop,first+limit-count)
                stem=f"{root.name}_ep{ep:06d}_{first:06d}_{stop:06d}"
                metadata_path=output/f"{stem}.json";tensor_path=output/f"{stem}.safetensors"
                if metadata_path.exists() and tensor_path.exists():
                    saved=json.loads(metadata_path.read_text())
                    if saved["spec_sha256"]!=sha256(spec_path) or saved["tensor_sha256"]!=sha256(tensor_path):
                        raise ValueError("Completed shard integrity mismatch")
                    shards.append(metadata_path.name);count+=len(saved["rows"]);continue
                rows=[];buffers={name:[] for name in ("k","v","routed","projected_query","state","reference")}
                for frame in range(first,stop):
                    tick=time.perf_counter();anchor,history=conditioning_frames(table,frame,offset_period,history_count,history_stride)
                    with torch.inference_mode():
                        if anchor!=anchor_cached:
                            policy.reset_obsrvationwindows()
                            for prior in [*history,anchor]:
                                image=observe(prior)
                                # Explicit original-frame RoPE positions, not one increment per export request.
                                policy.model._observed_frame_index=int(prior//policy.model.video_rope_frame_stride)
                                policy._prefill_episode(prompt=prompt,image_tensor=image,context=context,context_mask=context_mask)
                            anchor_cached=anchor
                        image=observe(frame)
                        policy.model._inference_state["next_chunk_index"]=0
                        current_state=np.array(table["observation.state"].iloc[frame],copy=True)
                        pred=policy._infer_action_chunk(image_tensor=image,proprio=policy._normalize_state(current_state))
                        kv=policy.model._inference_state["_chunk_video_kv_cache"][-1]
                        capture=editor.offline_capture
                        payload=dict(k=kv["k"][0],v=kv["v"][0],routed=capture["routed"][0,0],
                            projected_query=capture["projected_query"][0,0],
                            state=policy._normalize_state(current_state)[0],
                            reference=pred["action_chunk"][:data["config"]["chunk_size"]].clamp(-5,5))
                        for name,value in payload.items():
                            if not torch.isfinite(value).all():raise FloatingPointError(f"Nonfinite {name}")
                            buffers[name].append(value.detach().to("cpu",dtype=torch.bfloat16 if name in {"k","v","routed","projected_query"} else torch.float32).contiguous())
                    seconds=time.perf_counter()-tick;measurements.append(seconds);count+=1
                    rows.append(dict(key=f"{uid}:{frame}",episode_uid=uid,frame=frame,planner_frame=anchor,
                        offset=frame-anchor,history_frames=history,control_source=int(table.control_source.iloc[frame])))
                    if count==1 or count%16==0:
                        print(json.dumps(dict(rank=rank,observations=count,last_seconds=seconds,mean_seconds=sum(measurements)/len(measurements),memory=memory_stats())),flush=True)
                tensors={name:torch.stack(values) for name,values in buffers.items()}
                temporary=tensor_path.with_suffix(".tmp")
                save_file(tensors,str(temporary));os.replace(temporary,tensor_path)
                atomic_json(metadata_path,dict(spec_sha256=sha256(spec_path),tensor_file=tensor_path.name,
                    tensor_sha256=sha256(tensor_path),rows=rows,shapes={k:list(v.shape) for k,v in tensors.items()}))
                shards.append(metadata_path.name)
                atomic_json(output/f"progress_rank{rank}.json",dict(rank=rank,observations=count,shards=shards,
                    model_load_seconds=model_load_seconds,mean_seconds=sum(measurements)/len(measurements),
                    memory=memory_stats(),wall_seconds=time.perf_counter()-started,video_sha256=video_hashes))
        finally:
            for video in videos.values():video.close()
        if limit is not None and count>=limit:break
    atomic_json(output/f"done_rank{rank}.json",dict(rank=rank,world_size=world_size,observations=count,
        shards=shards,model_load_seconds=model_load_seconds,wall_seconds=time.perf_counter()-started,
        memory=memory_stats(),mean_seconds=sum(measurements)/max(1,len(measurements)),video_sha256=video_hashes))


class ShardedCache:
    """Memory-map only requested shard rows; bounded LRU of open mappings."""
    def __init__(self,root,max_open=16):
        self.root=Path(root);self.index={};self.opened=OrderedDict();self.max_open=max_open
        specs=[json.loads(p.read_text()) for p in sorted(self.root.glob("spec_rank*.json"))]
        if not specs or any(s!=specs[0] for s in specs):raise ValueError("Missing/inconsistent cache specs")
        self.spec=specs[0]
        spec_hashes={sha256(p) for p in self.root.glob("spec_rank*.json")}
        for path in sorted(self.root.glob("*_ep*.json")):
            meta=json.loads(path.read_text())
            if meta["spec_sha256"] not in spec_hashes:
                raise ValueError("Shard provenance does not match cache spec")
            for i,row in enumerate(meta["rows"]):
                if row["key"] in self.index:raise ValueError("Duplicate cached observation")
                self.index[row["key"]]=(meta["tensor_file"],i)

    def sample(self,key):
        path,index=self.index[key]
        if path not in self.opened:
            self.opened[path]=safe_open(str(self.root/path),framework="pt",device="cpu")
            if len(self.opened)>self.max_open:self.opened.popitem(last=False)
        self.opened.move_to_end(path);handle=self.opened[path]
        return {name:handle.get_slice(name)[index:index+1][0] for name in ("k","v","state","reference")}

    def batch(self,data,indices):
        tensors={k:v[indices] for k,v in data["tensors"].items()}
        records=[data["records"][int(i)] for i in indices]
        for prefix,key in (("","observation_key"),("next_","next_observation_key")):
            samples=[self.sample(r[key]) for r in records]
            tensors[prefix+"features"]=torch.stack([torch.cat([s["k"],s["v"]],dim=-1) for s in samples])
            tensors[prefix+"feature_valid"]=torch.ones(tensors[prefix+"features"].shape[:2],dtype=torch.bool)
            for name in ("state","reference"):
                tensors[prefix+name]=torch.stack([s[name] for s in samples]).float()
        tensors["action"]=(tensors["action"].float()*torch.tensor(self.spec["action_scale"])+torch.tensor(self.spec["action_offset"])).clamp(-5,5)
        for name in ("reward","discount"):tensors[name]=tensors[name].float()
        return tensors
