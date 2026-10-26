"""Prepare, audit, export features and train candidate offline RL adapters.

Run from the repository root with PYTHONPATH=src:. . No robot is contacted.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from ahawam.offline_rl.data import TransitionConfig, build_dataset, sha256
from ahawam.offline_rl.learner import LearnerConfig, OfflineLearner, masked_mse
from ahawam.offline_rl.networks import NetworkConfig


def audit(data):
    counts = Counter(r["end_reason"] for r in data["records"])
    return dict(format=data["format"], config=data["config"], conventions=data["conventions"],
                episodes=data["episodes"], transitions=len(data["records"]), end_reasons=dict(counts),
                note="Overlapping windows may share a success event; counts are not independent successes.")


def prepare(args):
    cfg = TransitionConfig(args.chunk_size,args.stride,args.gamma_tick,args.intervention_mode,args.temporal_mode)
    data = build_dataset(args.roots,cfg)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.save(data,args.output/"transitions.pt")
    (args.output/"audit.json").write_text(json.dumps(audit(data),ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(transitions=len(data["records"]),episodes=len(data["episodes"]),
                          pending=sum(e["status"]!="ready" for e in data["episodes"]),
                          end_reasons=dict(Counter(r["end_reason"] for r in data["records"])))))


def validate_cache(data,cache):
    spec = cache["feature_spec"]
    for key in ("base_checkpoint","normalization","token_source","cache_protocol"):
        if not spec.get(key):
            raise ValueError(f"Missing feature provenance: {key}")
    for episode in data["episodes"]:
        if episode["status"] == "ready" and spec["source_hashes"].get(episode["episode_uid"]) != episode["source"]["parquet_sha256"]:
            raise ValueError("Feature cache/source hash mismatch")
    if not data["records"]:
        raise ValueError("No trainable transitions; finish annotations or inspect the audit")
    keys = {r[k] for r in data["records"] for k in ("observation_key","next_observation_key")}
    for key in keys:
        sample = cache["samples"][key]
        if sample["reference"].shape != data["tensors"]["action"].shape[1:]:
            raise ValueError("Reference chunk shape does not match transitions")
        if sample["tokens"].ndim != 2 or sample["valid"].shape != sample["tokens"].shape[:1]:
            raise ValueError("Invalid feature token shape")
        if sample["valid"].dtype != torch.bool or not sample["valid"].any():
            raise ValueError("Invalid feature validity mask")
        for name in ("tokens","state","reference"):
            if not torch.isfinite(sample[name]).all():
                raise ValueError(f"Non-finite {name} in feature cache")


def make_batches(data,cache):
    validate_cache(data,cache)
    tensor = {k:v.clone() for k,v in data["tensors"].items()}
    records = data["records"]
    for prefix,key in (("","observation_key"),("next_","next_observation_key")):
        samples = [cache["samples"][r[key]] for r in records]
        tensor[prefix+"features"] = torch.stack([x["tokens"] for x in samples]).float()
        tensor[prefix+"feature_valid"] = torch.stack([x["valid"] for x in samples])
        tensor[prefix+"reference"] = torch.stack([x["reference"] for x in samples]).float()
        tensor[prefix+"state"] = torch.stack([x["state"] for x in samples]).float()
    tensor["action"] = (tensor["action"].float()*cache["action_scale"] + cache["action_offset"]).clamp(-5,5)
    for key in ("reward","discount"):
        tensor[key] = tensor[key].float()
    expected = tensor["action"].shape
    if tensor["reference"].shape != expected or tensor["next_reference"].shape != expected:
        raise ValueError("Reference chunk shape does not match transitions")
    for key,v in tensor.items():
        if v.is_floating_point() and not torch.isfinite(v).all():
            raise ValueError(f"Non-finite cached tensor: {key}")
    return tensor


def episode_split(records,seed,val_fraction=.2):
    episodes = sorted({r["episode_uid"] for r in records})
    if len(episodes)<2:
        raise ValueError("At least two episodes needed for a disjoint validation split")
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(episodes),generator=generator).tolist()
    validation = {episodes[i] for i in order[:max(1,round(len(episodes)*val_fraction))]}
    train = torch.tensor([i for i,r in enumerate(records) if r["episode_uid"] not in validation])
    val = torch.tensor([i for i,r in enumerate(records) if r["episode_uid"] in validation])
    return train,val,sorted(validation)


@torch.no_grad()
def validate(learner,tensors,indices):
    learner.actor.eval()
    learner.critic.eval()
    learner.value.eval()
    metrics = Counter()
    for group in indices.split(128):
        b = {k:v[group].to(learner.device) for k,v in tensors.items()}
        obs = b["features"],b["feature_valid"],b["state"]
        pi = learner.actor(*obs,b["reference"])
        q1,q2 = learner.critic(*obs,b["action"])
        next_obs = b["next_features"],b["next_feature_valid"],b["next_state"]
        if learner.config.algorithm == "td3_bc":
            next_a = learner.target_actor(*next_obs,b["next_reference"])
            next_v = torch.stack(learner.target_critic(*next_obs,next_a)).amin(0)
        else:
            next_v = learner.value(*next_obs)
        y = b["reward"] + b["discount"]*next_v
        metrics["bc_mse"] += float(masked_mse(pi,b["action"],b["action_valid"]).sum())
        metrics["q_data"] += float(torch.minimum(q1,q2).sum())
        if learner.config.algorithm != "bc":
            metrics["bellman_mse"] += float(((q1-y).square()+(q2-y).square()).sum())
    learner.actor.train()
    learner.critic.train()
    learner.value.train()
    return {k:v/len(indices) for k,v in metrics.items()}


def train(args):
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    data = torch.load(args.transitions,map_location="cpu",weights_only=True)
    cache = torch.load(args.features,map_location="cpu",weights_only=True)
    if cache["feature_spec"].get("synthetic") and not args.allow_synthetic_smoke:
        raise ValueError("Synthetic features require --allow-synthetic-smoke")
    tensors = make_batches(data,cache)
    train_ids,val_ids,val_episodes = episode_split(data["records"],args.seed)
    if cache["feature_spec"].get("reader_validation_episodes",val_episodes) != val_episodes:
        raise ValueError("Reader/learner validation split mismatch; reuse the readout seed")
    cfg = LearnerConfig(algorithm=args.algorithm,reference_penalty=args.reference_penalty)
    net = NetworkConfig(feature_dim=tensors["features"].shape[-1],chunk_size=data["config"]["chunk_size"],
                        width=args.width,max_tokens=tensors["features"].shape[1],architecture=args.architecture)
    learner = OfflineLearner(net,cfg,args.device)
    if args.resume:
        learner,provenance = OfflineLearner.load(args.resume,args.device)
        if learner.network_config != net or learner.config != cfg:
            raise ValueError("Resume configuration mismatch")
        if provenance["transitions_sha256"]!=sha256(args.transitions) or provenance["features_sha256"]!=sha256(args.features):
            raise ValueError("Resume data mismatch")
        if provenance.get("seed") != args.seed or provenance.get("validation_episodes") != val_episodes:
            raise ValueError("Resume training/validation split mismatch")
        if provenance.get("bc_warmup") != args.bc_warmup:
            raise ValueError("Resume bc-warmup schedule mismatch")
    args.output.mkdir(parents=True,exist_ok=True)
    provenance = dict(transitions_sha256=sha256(args.transitions),features_sha256=sha256(args.features),
                      feature_spec=cache["feature_spec"],data_config=data["config"],
                      validation_episodes=val_episodes,seed=args.seed,
                      bc_warmup=args.bc_warmup,
                      experimental_temporal_semantics=True,
                      warning="Offline loss/Q metrics do not measure robot success rate")
    with (args.output/"metrics.jsonl").open("a") as log:
        for step in range(args.steps):
            ids = train_ids[torch.randint(len(train_ids),(args.batch_size,))]
            actor_only = learner.steps < args.bc_warmup
            metrics = learner.update({k:v[ids] for k,v in tensors.items()},actor_only=actor_only)
            if learner.steps == args.bc_warmup and actor_only:
                learner.target_actor.load_state_dict(learner.actor.state_dict())
            if step==0 or (step+1)%args.log_every==0 or step+1==args.steps:
                row = dict(step=learner.steps,train=metrics,validation=validate(learner,tensors,val_ids))
                log.write(json.dumps(row)+"\n"); log.flush()
                print(json.dumps(row),flush=True)
        learner.save(args.output/"last.pt",provenance)
    (args.output/"provenance.json").write_text(json.dumps(provenance,indent=2)+"\n")


def readout(args):
    """Train on train episodes only; encode all frames with a frozen reader."""
    from ahawam.models.wan22.rl_token_readout import RLTokenReadout
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    data = torch.load(args.transitions,map_location="cpu",weights_only=True)
    cache = torch.load(args.features,map_location="cpu",weights_only=True)
    validate_cache(data,cache)
    train_ids,_,val_episodes = episode_split(data["records"],args.seed)
    keys = sorted({data["records"][i][k] for i in train_ids.tolist() for k in ("observation_key","next_observation_key")})
    tokens = torch.stack([cache["samples"][k]["tokens"] for k in keys]).float()
    valid = torch.stack([cache["samples"][k]["valid"] for k in keys])
    model = RLTokenReadout(tokens.shape[-1],args.width,max_tokens=tokens.shape[1]).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-4)
    for step in range(args.steps):
        ids = torch.randint(len(tokens),(args.batch_size,))
        loss,_,_ = model(tokens[ids].to(args.device),valid[ids].to(args.device))
        optimizer.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1,error_if_nonfinite=True);optimizer.step()
        if step==0 or (step+1)%args.log_every==0:
            print(json.dumps(dict(step=step+1,reconstruction_mse=float(loss.detach()))),flush=True)
    model.eval().requires_grad_(False)
    args.output.mkdir(parents=True,exist_ok=True)
    torch.save(dict(config=model.config,reader=model.state_dict(),feature_spec=cache["feature_spec"],
                    validation_episodes=val_episodes),args.output/"reader.pt")
    val_keys = sorted(k for k in cache["samples"] if k.rsplit(":",1)[0] in val_episodes)
    with torch.no_grad():
        errors = []
        for key in val_keys:
            sample = cache["samples"][key]
            loss,_,_ = model(sample["tokens"][None].float().to(args.device),sample["valid"][None].to(args.device))
            errors.append(float(loss))
    (args.output/"validation.json").write_text(json.dumps(dict(validation_episodes=val_episodes,
        frames=len(errors),reconstruction_mse=sum(errors)/len(errors),
        note="Reconstruction error alone does not establish a useful control representation"),indent=2)+"\n")
    with torch.no_grad():
        for sample in cache["samples"].values():
            z = model.encode(sample["tokens"][None].float().to(args.device),sample["valid"][None].to(args.device))[0].cpu()
            sample["tokens"],sample["valid"] = z[None],torch.ones(1,dtype=torch.bool)
    cache["feature_spec"] = dict(cache["feature_spec"],reader_sha256=sha256(args.output/"reader.pt"),
        reader_validation_episodes=val_episodes,token_source=cache["feature_spec"]["token_source"]+"/frozen_RLTokenReadout")
    torch.save(cache,args.output/"features.pt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--roots",type=Path,nargs="+",required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--chunk-size",type=int,default=8)
    p.add_argument("--stride",type=int,default=2)
    p.add_argument("--gamma-tick",type=float,default=.997)
    p.add_argument("--temporal-mode",choices=["tick","macro","skip_hold"],required=True)
    p.add_argument("--intervention-mode",choices=["mixed","censor"],default="mixed")
    e = sub.add_parser("export")
    e.add_argument("--transitions",type=Path,required=True)
    e.add_argument("--deploy-config",type=Path,required=True)
    e.add_argument("--output",type=Path,required=True)
    e.add_argument("--history-count",type=int,default=0)
    e.add_argument("--history-stride",type=int,default=8)
    e.add_argument("--num-tokens",type=int,default=64)
    e.add_argument("--layers",type=int,nargs="+",default=[-1])
    e.add_argument("--limit",type=int)
    for name in ("train","readout"):
        p = sub.add_parser(name)
        p.add_argument("--transitions",type=Path,required=True)
        p.add_argument("--features",type=Path,required=True)
        p.add_argument("--output",type=Path,required=True)
        p.add_argument("--steps",type=int,default=2000)
        p.add_argument("--batch-size",type=int,default=64)
        p.add_argument("--width",type=int,default=256)
        p.add_argument("--device",default="cpu")
        p.add_argument("--seed",type=int,default=42)
        p.add_argument("--cpu-threads",type=int,default=2)
        p.add_argument("--log-every",type=int,default=100)
        if name=="train":
            p.add_argument("--algorithm",choices=["iql","td3_bc","bc"],default="iql")
            p.add_argument("--architecture",choices=["mlp","attention"],default="mlp")
            p.add_argument("--bc-warmup",type=int,default=200)
            p.add_argument("--reference-penalty",type=float,default=0.0)
            p.add_argument("--allow-synthetic-smoke",action="store_true")
            p.add_argument("--resume",type=Path)
    args = parser.parse_args()
    if hasattr(args,"steps") and min(args.steps,args.batch_size,args.log_every,args.cpu_threads)<=0:
        parser.error("steps, batch-size, log-every and cpu-threads must be positive")
    if args.command=="prepare": prepare(args)
    elif args.command=="train": train(args)
    elif args.command=="readout": readout(args)
    else:
        from ahawam.offline_rl.features import export_features
        data = torch.load(args.transitions,map_location="cpu",weights_only=True)
        export_features(data,args.deploy_config,args.output,args.history_count,args.history_stride,args.num_tokens,tuple(args.layers),args.limit)


if __name__ == "__main__":
    main()
