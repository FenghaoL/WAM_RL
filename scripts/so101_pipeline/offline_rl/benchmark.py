"""Measure real cached observation batches, without claiming policy quality."""
import argparse
import json
from pathlib import Path
import time

import torch

from ahawam.offline_rl.cache import ShardedCache,atomic_json,memory_stats
from ahawam.offline_rl.learner import LearnerConfig,OfflineLearner
from ahawam.offline_rl.networks import NetworkConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transitions",type=Path,required=True);p.add_argument("--cache",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--steps",type=int,default=20)
    p.add_argument("--batch-sizes",type=int,nargs="+",default=[32,64,128])
    args=p.parse_args();torch.set_num_threads(2);torch.manual_seed(42)
    if not torch.cuda.is_available():raise RuntimeError("GPU profiling requires CUDA")
    data=torch.load(args.transitions,weights_only=True);cache=ShardedCache(args.cache)
    covered=torch.tensor([i for i,r in enumerate(data["records"]) if r["observation_key"] in cache.index and r["next_observation_key"] in cache.index])
    if not len(covered):raise ValueError("No complete transitions covered by cache")
    s=cache.sample(data["records"][int(covered[0])]["observation_key"])
    net=NetworkConfig(s["k"].shape[-1]*2,max_tokens=s["k"].shape[0],architecture="raw_kv",shared_critic=True)
    results=[]
    for algorithm in ("iql","td3_bc"):
        for size in args.batch_sizes:
            learner=OfflineLearner(net,LearnerConfig(algorithm=algorithm,lr=1e-4,
                actor_delay=1 if algorithm=="iql" else 2),device="cuda")
            times=[];loads=[];torch.cuda.reset_peak_memory_stats()
            for step in range(args.steps+3):
                start=time.perf_counter();ids=covered[torch.randint(len(covered),(size,))]
                b=cache.batch(data,ids);loaded=time.perf_counter()-start
                metrics=learner.update(b);torch.cuda.synchronize();elapsed=time.perf_counter()-start
                if step>=3:times.append(elapsed);loads.append(loaded)
            row=dict(algorithm=algorithm,batch_size=size,steps=args.steps,
                seconds_per_update=sum(times)/len(times),seconds_per_load=sum(loads)/len(loads),memory=memory_stats(),
                last_metrics=metrics,covered_real_transitions=len(covered),
                parameters={name:sum(p.numel() for p in getattr(learner,name).parameters()) for name in ("actor","critic","value")})
            print(json.dumps(row),flush=True);results.append(row)
            del learner,b;torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.output,dict(device=torch.cuda.get_device_name(),results=results,
        note="Resource benchmark; batches can repeat a tiny real-data subset; no held-out policy evaluation"))


if __name__=="__main__":main()
