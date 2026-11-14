"""Persistent four-GPU cache rebuild followed by independent IQL/TD3+BC runs.

No previous cache or model weights are consumed. A failed child stops the next
stage. Status, exact commands, resource reports and logs remain on disk.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch

from ahawam.offline_rl.cache import ShardedCache, atomic_json, conditioning_from_spec
from ahawam.offline_rl.data import sha256
from scripts.so101_pipeline.offline_rl.system import require_complete, split


def verify_cache(data, directory):
    import pandas as pd
    cache=ShardedCache(directory)
    require_complete(data,cache)
    expected=sum(e["retained_frames"] for e in data["episodes"])
    if len(cache.index)!=expected:raise ValueError("Cache observation count differs from recipe")
    for source,digest in cache.spec["code_sha256"].items():
        if sha256(Path(source))!=digest:raise ValueError(f"Exporter/model source changed during export: {source}")
    episodes={e["episode_uid"]:e for e in data["episodes"]};tables={};shards=0
    for metadata in sorted(directory.glob("*_ep*.json")):
        item=json.loads(metadata.read_text())
        if sha256(directory/item["tensor_file"])!=item["tensor_sha256"]:
            raise ValueError("Cache tensor hash mismatch")
        for row in item["rows"]:
            uid=row["episode_uid"];e=episodes[uid];frame=row["frame"]
            if not 0<=frame<e["retained_frames"]:raise ValueError("Cache leaked a discarded tail frame")
            if uid not in tables:
                root=Path(e["root"]);info=json.loads((root/"meta/info.json").read_text());ep=e["episode_index"]
                tables[uid]=pd.read_parquet(root/info["data_path"].format(episode_index=ep,
                    episode_chunk=ep//int(info.get("chunks_size",1000))))
            anchor,history=conditioning_from_spec(tables[uid],frame,cache.spec)
            if anchor!=row["planner_frame"] or history!=row["history_frames"] or frame-anchor!=row["offset"]:
                raise ValueError("Exported history/offset differs from continuous episode recipe")
        shards+=1
    result=dict(observations=expected,shards=shards,tensor_hashes_verified=True,history_rows_verified=True,
        excluded_tails_absent=True,stored_bytes=sum(p.stat().st_size for p in directory.glob("*.safetensors")))
    atomic_json(directory/"complete.json",result)
    return result


def run(args):
    root=args.run_dir.resolve();data_path=root/"data/transitions.pt";cache=root/"cache";out=root/"experiments"
    if (root/"pipeline.json").exists():raise ValueError("Pipeline already registered; inspect existing status")
    data=torch.load(data_path,map_location="cpu",weights_only=True)
    if data["config"]["timeline_mode"]!="episode_frames" or "episode_recipe" not in data:
        raise ValueError("Use the prepared episode recipe")
    train,val,_=split(data)
    cache.mkdir();out.mkdir()
    state=dict(supervisor_pid=os.getpid(),status="starting",started_unix=time.time(),
        transitions_sha256=sha256(data_path),recipe_sha256=sha256(root/"data/episode_recipe.json"),
        train_transitions=len(train),validation_transitions=len(val),jobs={},
        training_scope="IQL and TD3+BC only; no Action DiT fine-tuning in this rebuild")
    for name,gpu in (("iql",args.gpus[0]),("td3_bc",args.gpus[1])):
        state["jobs"][name]=dict(status="waiting_for_verified_cache_and_smoke",gpu=gpu,steps=args.steps)
    active={}
    script=Path(__file__).resolve().parent
    def save():atomic_json(root/"pipeline.json",state)
    def start(name,gpu,command,log):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="2",PYTHONPATH="src:.")
        with log.open("a") as stream:
            child=subprocess.Popen(command,env=env,stdout=stream,stderr=subprocess.STDOUT)
        active[name]=child
        state["jobs"][name]=dict(status="running",pid=child.pid,gpu=gpu,command=command,
            log=str(log),started_unix=time.time());save()
    def stop(*_):raise InterruptedError("Supervisor interrupted")
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    def wait_children():
        while active:
            for name,child in list(active.items()):
                code=child.poll()
                if code is not None:
                    state["jobs"][name].update(status="complete" if code==0 else "failed",exit_code=code,ended_unix=time.time())
                    del active[name];save()
                    if code:raise RuntimeError(f"{name} failed, exit {code}; see its log")
            time.sleep(3)
    try:
        state["status"]="exporting_cache";save()
        for rank,gpu in enumerate(args.gpus):
            command=[sys.executable,str(script/"system.py"),"export-worker","--transitions",str(data_path),
                "--cache",str(cache),"--rank",str(rank),"--world-size","4","--shard-size","128"]
            start(f"cache_rank{rank}",gpu,command,cache/f"worker_rank{rank}.log")
        wait_children()
        state["status"]="verifying_cache";save()
        state["cache_verification"]=verify_cache(data,cache);save()
        state["status"]="smoke_training";save()
        for name,gpu in (("iql",args.gpus[0]),("td3_bc",args.gpus[1])):
            command=[sys.executable,str(script/"system.py"),"train","--transitions",str(data_path),
                "--cache",str(cache),"--output",str(out/f"smoke_{name}"),"--algorithm",name,
                "--steps","4","--batch-size","8","--bc-warmup","0","--log-every","1",
                "--save-every","4","--val-samples","128"]
            start(f"smoke_{name}",gpu,command,out/f"smoke_{name}.log")
        wait_children()
        for name in ("iql","td3_bc"):
            row=json.loads((out/f"smoke_{name}"/"metrics.jsonl").read_text().splitlines()[-1])
            if not {"critic_loss","actor_loss"} <= set(row["train"]) or \
               not all(math.isfinite(v) for v in row["train"].values()):
                raise ValueError("Real-cache smoke failed to exercise finite actor/critic updates")
        # All models initialize from scratch; the frozen WAM reference in the
        # freshly exported cache still comes from the collector's SFT model.
        state["status"]="training";save()
        for name,gpu in (("iql",args.gpus[0]),("td3_bc",args.gpus[1])):
            command=[sys.executable,str(script/"system.py"),"train","--transitions",str(data_path),
                "--cache",str(cache),"--output",str(out/name),"--algorithm",name,"--steps",str(args.steps),
                "--batch-size","64","--lr","0.0001","--bc-warmup","200","--log-every","100",
                "--save-every","1000","--val-samples",str(len(val))]
            start(name,gpu,command,out/f"{name}.log")
        wait_children()
        # A zero exit alone is insufficient; check final finite losses, step
        # count, exact data provenance and held-out episode identities.
        for name in ("iql","td3_bc"):
            checkpoint=torch.load(out/name/"last.pt",map_location="cpu",weights_only=True)
            if checkpoint["steps"]!=args.steps or checkpoint["provenance"]["transitions_sha256"]!=state["transitions_sha256"]:
                raise ValueError("Final checkpoint steps/provenance mismatch")
            if set(checkpoint["provenance"]["validation_episodes"])!=set(split(data)[2]):
                raise ValueError("Final checkpoint validation split mismatch")
            final=json.loads((out/name/"metrics.jsonl").read_text().splitlines()[-1])
            if final["step"]!=args.steps or not all(math.isfinite(v) for v in final["train"].values()):
                raise ValueError("Final training metrics are missing or nonfinite")
            if not all(math.isfinite(v) for v in final["validation"].values()):
                raise ValueError("Final validation metrics are nonfinite")
            state["jobs"][name]["checkpoint_sha256"]=sha256(out/name/"last.pt")
        state.update(status="complete",ended_unix=time.time());save()
    except BaseException as error:
        for name,child in active.items():
            if child.poll() is None:child.terminate()
        for name,child in active.items():
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            state["jobs"][name].update(status="stopped_after_pipeline_failure",exit_code=child.returncode)
        state.update(status="failed",error=repr(error),ended_unix=time.time());save()
        raise


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--gpus",type=int,nargs=4,default=[0,1,2,3])
    p.add_argument("--steps",type=int,default=10000)
    args=p.parse_args()
    if len(set(args.gpus))!=4 or min(args.gpus)<0 or args.steps<1:p.error("Use four distinct nonnegative GPU IDs and positive steps")
    run(args)
