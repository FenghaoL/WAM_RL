"""Real-data raw-KV pipeline. See docs/offline_rl/08_SYSTEM_IMPLEMENTATION_ZH.md."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

from ahawam.offline_rl.cache import ShardedCache,atomic_json,export_shards,memory_stats
from ahawam.offline_rl.data import sha256
from ahawam.offline_rl.learner import LearnerConfig,OfflineLearner,masked_mse
from ahawam.offline_rl.networks import NetworkConfig


def load_data(path):
    return torch.load(path,map_location="cpu",weights_only=True)


def split(data,seed=42):
    from scripts.so101_pipeline.offline_rl.run import episode_split
    records=data["records"]
    assigned={e["episode_uid"]:e.get("split") for e in data["episodes"]}
    if any(v is not None for v in assigned.values()):
        if any(v not in {"train","val"} for v in assigned.values()):
            raise ValueError("Incomplete explicit episode split")
        val=sorted(uid for uid,part in assigned.items() if part=="val")
        return (torch.tensor([i for i,r in enumerate(records) if assigned[r["episode_uid"]]=="train"],dtype=torch.long),
                torch.tensor([i for i,r in enumerate(records) if assigned[r["episode_uid"]]=="val"],dtype=torch.long),val)
    val={r["episode_uid"] for r in records if Path(r["root"]).name.endswith("_val")}
    if not val:return episode_split(records,seed)
    return (torch.tensor([i for i,r in enumerate(records) if r["episode_uid"] not in val]),
            torch.tensor([i for i,r in enumerate(records) if r["episode_uid"] in val]),sorted(val))


def require_complete(data,cache):
    if data["config"].get("timeline_mode", "strict_ticks") != cache.spec.get("timeline_mode", "strict_ticks"):
        raise ValueError("Transition/cache timeline mismatch; regenerate cache in a new directory")
    limits={e["episode_uid"]:e.get("retained_frames",e["frames"]) for e in data["episodes"]}
    if "episode_frame_limits" in cache.spec and cache.spec["episode_frame_limits"]!=limits:
        raise ValueError("Cache/transition frame-limit recipe mismatch")
    for e in data["episodes"]:
        if cache.spec["source_hashes"].get(e["episode_uid"])!=e["source"]["parquet_sha256"]:
            raise ValueError("Cache source mismatch")
    keys={r[k] for r in data["records"] for k in ("observation_key","next_observation_key")}
    missing=keys-cache.index.keys()
    if missing:raise ValueError(f"Cache incomplete: {len(missing)} observations missing")


@torch.no_grad()
def validate(learner,data,cache,indices,batch_size=32):
    learner.actor.eval();learner.critic.eval();learner.value.eval()
    metrics=Counter()
    for ids in indices.split(batch_size):
        b={k:v.to(learner.device) for k,v in cache.batch(data,ids).items()}
        obs=b["features"],b["feature_valid"],b["state"]
        nxt=b["next_features"],b["next_feature_valid"],b["next_state"]
        pi=learner.actor(*obs,b["reference"])
        q1,q2=learner.critic(*obs,b["action"])
        if learner.config.algorithm=="iql":
            next_v=learner.value(*nxt)
        else:
            next_a=learner.target_actor(*nxt,b["next_reference"])
            next_v=torch.stack(learner.target_critic(*nxt,next_a)).amin(0)
        target=b["reward"]+b["discount"]*next_v
        metrics["bc_mse"]+=float(masked_mse(pi,b["action"],b["action_valid"]).sum())
        metrics["reference_bc_mse"]+=float(masked_mse(b["reference"],b["action"],b["action_valid"]).sum())
        metrics["bellman_mse"]+=float(((q1-target).square()+(q2-target).square()).sum())
        metrics["q_disagreement"]+=float((q1-q2).abs().sum())
    learner.actor.train();learner.critic.train();learner.value.train()
    return {k:v/len(indices) for k,v in metrics.items()}


def train(args):
    torch.set_num_threads(args.cpu_threads);torch.manual_seed(args.seed)
    if args.device.startswith("cuda") and not torch.cuda.is_available():raise RuntimeError("CUDA unavailable")
    data=load_data(args.transitions);cache=ShardedCache(args.cache);require_complete(data,cache)
    train_ids,val_ids,val_episodes=split(data,args.seed)
    if min(len(train_ids),len(val_ids))==0:raise ValueError("Empty train/validation split")
    sample=cache.sample(data["records"][0]["observation_key"])
    net=NetworkConfig(feature_dim=sample["k"].shape[-1]*2,max_tokens=sample["k"].shape[0],
        width=args.width,depth=args.depth,num_heads=4,architecture="raw_kv",shared_critic=not args.independent_q,
        chunk_size=data["config"]["chunk_size"])
    cfg=LearnerConfig(algorithm=args.algorithm,lr=args.lr,actor_delay=1 if args.algorithm=="iql" else 2,
        reference_dropout=args.reference_dropout,td3_bc_alpha=args.td3_alpha)
    provenance=dict(transitions_sha256=sha256(args.transitions),feature_spec=cache.spec,
        validation_episodes=val_episodes,seed=args.seed,bc_warmup=args.bc_warmup,
        batch_size=args.batch_size,warning="Offline losses do not establish real robot success")
    learner=OfflineLearner(net,cfg,args.device)
    if args.resume:
        learner,previous=OfflineLearner.load(args.resume,args.device)
        if learner.config!=cfg or learner.network_config!=net or previous!=provenance:
            raise ValueError("Resume settings/provenance mismatch")
    args.output.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output/"provenance.json",provenance)
    params={name:sum(p.numel() for p in getattr(learner,name).parameters()) for name in ("actor","critic","value")}
    atomic_json(args.output/"config.json",dict(network=asdict(net),learner=asdict(cfg),parameters=params,
        train_transitions=len(train_ids),validation_transitions=len(val_ids),requested_steps=args.steps))
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    times=[];load_times=[];begin=time.perf_counter()
    with (args.output/"metrics.jsonl").open("a") as log:
        for i in range(args.steps):
            tick=time.perf_counter()
            ids=train_ids[torch.randint(len(train_ids),(args.batch_size,))]
            batch=cache.batch(data,ids);load_times.append(time.perf_counter()-tick)
            actor_only=learner.steps<args.bc_warmup
            metrics=learner.update(batch,actor_only=actor_only)
            if actor_only and learner.steps==args.bc_warmup:
                learner.target_actor.load_state_dict(learner.actor.state_dict())
            if torch.cuda.is_available():torch.cuda.synchronize()
            times.append(time.perf_counter()-tick)
            if i==0 or (i+1)%args.log_every==0 or i+1==args.steps:
                # Fixed subset for frequent diagnostics; full held-out evaluation on request.
                validation=validate(learner,data,cache,val_ids[:args.val_samples],args.validation_batch_size)
                row=dict(step=learner.steps,actor_only=actor_only,train=metrics,validation=validation,
                    seconds_per_update=sum(times[-100:])/len(times[-100:]),
                    seconds_per_batch_load=sum(load_times[-100:])/len(load_times[-100:]),memory=memory_stats())
                log.write(json.dumps(row)+"\n");log.flush();print(json.dumps(row),flush=True)
            if (i+1)%args.save_every==0:
                learner.save(args.output/"last.pt",provenance)
        learner.save(args.output/"last.pt",provenance)
    atomic_json(args.output/"profile.json",dict(steps=args.steps,total_steps=learner.steps,
        seconds_per_update=sum(times[1:] or times)/len(times[1:] or times),
        total_seconds=time.perf_counter()-begin,memory=memory_stats(),parameters=params,
        batch_size=args.batch_size,train_transitions=len(train_ids),validation_episodes=val_episodes))


def launch(args):
    args.cache.mkdir(parents=True,exist_ok=True)
    state=[]
    for rank,gpu in enumerate(args.gpus):
        command=[sys.executable,__file__,"export-worker","--transitions",str(args.transitions),
            "--cache",str(args.cache),"--deploy-config",str(args.deploy_config),"--rank",str(rank),
            "--world-size",str(len(args.gpus)),"--shard-size",str(args.shard_size)]
        if args.limit:command += ["--limit",str(args.limit)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="2",PYTHONPATH="src:.")
        with (args.cache/f"worker_rank{rank}.log").open("a") as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.append(dict(rank=rank,gpu=gpu,pid=child.pid,command=command))
    atomic_json(args.cache/"launch.json",dict(workers=state,transitions=str(args.transitions.resolve())))
    print(json.dumps(state),flush=True)


def sequence(args):
    """Own child processes through export, then run bounded real-data pilots."""
    args.cache.mkdir(parents=True,exist_ok=True)
    children=[]
    for rank,gpu in enumerate(args.gpus):
        command=[sys.executable,__file__,"export-worker","--transitions",str(args.transitions),
            "--cache",str(args.cache),"--deploy-config",str(args.deploy_config),"--rank",str(rank),
            "--world-size",str(len(args.gpus)),"--shard-size",str(args.shard_size)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS="2",PYTHONPATH="src:.")
        with (args.cache/f"worker_rank{rank}.log").open("a") as log:
            child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        children.append(child)
    atomic_json(args.cache/"launch.json",dict(supervisor_pid=os.getpid(),workers=[
        dict(rank=i,gpu=gpu,pid=child.pid) for i,(gpu,child) in enumerate(zip(args.gpus,children))]))
    while any(child.poll() is None for child in children):
        failed=[child for child in children if child.poll() not in (None,0)]
        if failed:
            for child in children:
                if child.poll() is None:child.terminate()
            raise RuntimeError("Cache worker failed; inspect worker logs before resuming")
        time.sleep(5)
    if any(child.returncode!=0 for child in children):raise RuntimeError("Cache export failed")
    cache=ShardedCache(args.cache);require_complete(load_data(args.transitions),cache)
    atomic_json(args.cache/"complete.json",dict(observations=len(cache.index),stored_bytes=sum(
        p.stat().st_size for p in args.cache.glob("*.safetensors"))))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpus[0]),OMP_NUM_THREADS="2",PYTHONPATH="src:.")
    if args.pilot_steps:
        for algorithm in ("iql","td3_bc"):
            out=args.cache.parent/f"pilot_{algorithm}"
            command=[sys.executable,__file__,"train","--transitions",str(args.transitions),"--cache",str(args.cache),
                "--output",str(out),"--algorithm",algorithm,"--steps",str(args.pilot_steps),"--batch-size","64"]
            with (args.cache.parent/f"pilot_{algorithm}.log").open("a") as log:
                subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    atomic_json(args.cache.parent/"sequence_done.json",dict(pilot_steps=args.pilot_steps,algorithms=["iql","td3_bc"]))


def status(args):
    cache=args.cache
    result={p.stem:json.loads(p.read_text()) for p in sorted(cache.glob("*rank*.json")) if p.name.startswith(("progress","done"))}
    result["stored_bytes"]=sum(p.stat().st_size for p in cache.glob("*.safetensors"))
    if (cache/"launch.json").exists():result["launch"]=json.loads((cache/"launch.json").read_text())
    print(json.dumps(result,indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest="command",required=True)
    for name in ("export-worker","launch-cache","sequence","status","train"):
        q=sub.add_parser(name);q.add_argument("--cache",type=Path,required=True)
        if name=="status":continue
        q.add_argument("--transitions",type=Path,required=True)
        if name in {"export-worker","launch-cache","sequence"}:
            q.add_argument("--deploy-config",type=Path,default=Path("scripts/so101_pipeline/offline_rl/configs/cylinder_sft_cache.yaml"))
            q.add_argument("--shard-size",type=int,default=128)
            q.add_argument("--limit",type=int)
            if name=="export-worker":
                q.add_argument("--rank",type=int,default=0);q.add_argument("--world-size",type=int,default=1)
                q.add_argument("--history-count",type=int,default=6);q.add_argument("--history-stride",type=int,default=48)
                q.add_argument("--offset-period",type=int,default=8)
            else:
                q.add_argument("--gpus",type=int,nargs="+",default=[0,1,2,3])
                if name=="sequence":q.add_argument("--pilot-steps",type=int,default=500)
        else:
            q.add_argument("--output",type=Path,required=True);q.add_argument("--algorithm",choices=["iql","td3_bc"],default="iql")
            q.add_argument("--batch-size",type=int,default=64);q.add_argument("--steps",type=int,default=5000)
            q.add_argument("--width",type=int,default=256);q.add_argument("--depth",type=int,default=2)
            q.add_argument("--lr",type=float,default=1e-4);q.add_argument("--bc-warmup",type=int,default=200)
            q.add_argument("--reference-dropout",type=float,default=.1);q.add_argument("--td3-alpha",type=float,default=.1)
            q.add_argument("--independent-q",action="store_true");q.add_argument("--device",default="cuda")
            q.add_argument("--seed",type=int,default=42);q.add_argument("--cpu-threads",type=int,default=2)
            q.add_argument("--log-every",type=int,default=100);q.add_argument("--save-every",type=int,default=500)
            q.add_argument("--val-samples",type=int,default=256);q.add_argument("--validation-batch-size",type=int,default=32)
            q.add_argument("--resume",type=Path)
    args=p.parse_args()
    for name in ("steps","batch_size","width","depth","cpu_threads","log_every","save_every",
                 "val_samples","validation_batch_size","world_size","shard_size","history_stride","offset_period"):
        if hasattr(args,name) and getattr(args,name)<=0:p.error(f"{name} must be positive")
    if hasattr(args,"limit") and args.limit is not None and args.limit<=0:p.error("limit must be positive")
    if hasattr(args,"gpus") and (len(set(args.gpus))!=len(args.gpus) or min(args.gpus)<0):
        p.error("Use distinct nonnegative GPU indices")
    if hasattr(args,"bc_warmup") and args.bc_warmup<0:p.error("bc_warmup must be nonnegative")
    if hasattr(args,"pilot_steps") and args.pilot_steps<0:p.error("pilot_steps must be nonnegative")
    if hasattr(args,"reference_dropout") and not 0<=args.reference_dropout<1:p.error("reference_dropout must be in [0,1)")
    if args.command=="export-worker":
        torch.set_num_threads(2)
        export_shards(load_data(args.transitions),args.deploy_config,args.cache,args.rank,args.world_size,
            args.shard_size,args.limit,args.history_count,args.history_stride,args.offset_period)
    elif args.command=="launch-cache":launch(args)
    elif args.command=="sequence":sequence(args)
    elif args.command=="train":train(args)
    else:status(args)


if __name__=="__main__":main()
