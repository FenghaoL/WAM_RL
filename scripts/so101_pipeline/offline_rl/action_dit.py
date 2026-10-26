"""Frozen IQL Q/V -> detached dataset weights -> native Action DiT LoRA FM."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import torch

from ahawam.offline_rl.cache import ShardedCache, asset_fingerprint, atomic_json, memory_stats
from ahawam.offline_rl.data import sha256
from ahawam.offline_rl.flow_actor import (FrozenConditionProvider, adapter_state,
    install_action_lora, predict_native_flow, restore_adapter, weighted_flow_loss)
from ahawam.offline_rl.learner import OfflineLearner
from scripts.so101_pipeline.offline_rl.system import load_data, require_complete, split


def atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def weight_statistics(weights):
    w = weights.float()
    return dict(mean=float(w.mean()), min=float(w.min()), max=float(w.max()),
        quantiles=torch.quantile(w, torch.tensor([.01, .1, .5, .9, .99])).tolist(),
        ess=float(w.sum().square() / w.square().sum().clamp_min(1e-30)), count=len(w))


@torch.no_grad()
def export_weights(args, data, cache):
    """Do not use validation data to fit Q/V or normalize the actor objective."""
    identity = dict(critic_sha256=sha256(args.critic_checkpoint), transitions_sha256=sha256(args.transitions),
        feature_spec=cache.spec, observation_keys=[r["observation_key"] for r in data["records"]])
    path = args.output / "weights.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Existing weights have different provenance")
        return saved
    learner, previous = OfflineLearner.load(args.critic_checkpoint, "cuda")
    if learner.config.algorithm != "iql" or learner.steps <= previous["bc_warmup"]:
        raise ValueError("Need a trained IQL critic, beyond the BC-only warmup")
    train_ids, val_ids, val_episodes = split(data, previous["seed"])
    if (previous["transitions_sha256"] != identity["transitions_sha256"]
        or previous["feature_spec"] != cache.spec or previous["validation_episodes"] != val_episodes):
        raise ValueError("Critic/data/cache/split mismatch")
    learner.target_critic.eval().requires_grad_(False)
    learner.value.eval().requires_grad_(False)
    q_values, values = [], []
    for ids in torch.arange(len(data["records"])).split(64):
        b = {k: v.cuda() for k, v in cache.batch(data, ids).items()}
        obs = b["features"], b["feature_valid"], b["state"]
        q_values.append(torch.stack(learner.target_critic(*obs, b["action"])).amin(0).cpu())
        values.append(learner.value(*obs).cpu())
    q, value = torch.cat(q_values), torch.cat(values)
    advantage = q - value
    cfg = learner.config
    weights = (cfg.inverse_temperature * advantage).clamp(max=math.log(cfg.max_weight)).exp()
    if not torch.isfinite(weights).all() or weights[train_ids].sum() <= 0:
        raise FloatingPointError("Invalid IQL weights")
    by_episode = {}
    for i, record in enumerate(data["records"]):
        by_episode.setdefault(record["episode_uid"], []).append(i)
    report = dict(critic_steps=learner.steps, beta=cfg.inverse_temperature, max_weight=cfg.max_weight,
        train=weight_statistics(weights[train_ids]), validation=weight_statistics(weights[val_ids]),
        episodes={uid: weight_statistics(weights[ids]) for uid, ids in by_episode.items()})
    saved = dict(identity=identity, weights=weights, q=q, value=value, advantage=advantage,
        train_ids=train_ids, val_ids=val_ids, report=report)
    atomic_torch(path, saved)
    atomic_json(args.output / "weight_report.json", report)
    print(json.dumps(dict(stage="weights_exported", **report["train"])), flush=True)
    del learner
    torch.cuda.empty_cache()
    return saved


def checkpoint(path, model, optimizer, step, provenance, sampler):
    atomic_torch(path, dict(format="ahawam_iql_flow_lora/v1", adapter=adapter_state(model),
        optimizer=optimizer.state_dict(), step=step, provenance=provenance,
        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(), sampler_rng=sampler.get_state()))


@torch.no_grad()
def check_flow_equivalence(model, state, action):
    """The gradient-capable single-chunk route must agree with serving's flow."""
    timestep = torch.full((1,), 500., device=action.device, dtype=action.dtype)
    predicted = predict_native_flow(model, action, timestep, state)
    inference = model._inference_state
    deployed = model._predict_action_chunk_with_clean_cache(
        noisy_action_chunk=action, timestep_action=timestep,
        context=state["context"], context_mask=state["context_mask"],
        video_kv_cache=inference["_chunk_video_kv_cache"], video_seq_len=inference["video_seq_len"],
        video_tokens_per_frame=inference["video_tokens_per_frame"],
        action_history_kv_cache=None, action_history_seq_len=0, chunk_start=0,
        obs_context=state["proprio_context"], obs_context_mask=state["proprio_mask"],
        obs_chunk_offset=0, obs_proprio_tokens_per_chunk=model._require_obs_proprio_tokens_per_chunk())
    torch.testing.assert_close(predicted, deployed, rtol=0, atol=0)
    return float((predicted.float() - deployed.float()).abs().max())


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for native Action DiT")
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "metrics.jsonl").exists() and not args.resume:
        raise ValueError("Use --resume or a new output directory")
    data = load_data(args.transitions)
    cache = ShardedCache(args.cache)
    require_complete(data, cache)
    weights = export_weights(args, data, cache)
    if args.export_weights_only:
        return
    # OfflineLearner.load restores its own RNG. Do not let computing vs reusing
    # the weights cache change native-policy/LoRA initialization or FM noise.
    torch.manual_seed(args.seed)
    from deploy.server.ahawam_policy import AHAWAMPolicy
    started = time.perf_counter()
    print(json.dumps(dict(stage="loading_collector_sft")), flush=True)
    policy = AHAWAMPolicy(str(args.deploy_config))
    model = policy.model
    model.eval().requires_grad_(False)
    print(json.dumps(dict(stage="collector_sft_loaded", seconds=time.perf_counter() - started)), flush=True)
    if data["config"]["chunk_size"] != model.action_chunk_size or not data["tensors"]["action_valid"].all():
        raise ValueError("Native FM route requires matching, complete natural action chunks")
    deployment = policy._load_deploy_config(str(args.deploy_config))
    base = Path(deployment["checkpoint_path"])
    if not base.is_absolute():
        base = policy.project_root / base
    if (asset_fingerprint(base, cache.root.parent / "checkpoint_fingerprint.json") != cache.spec["base_checkpoint_sha256"]
        or sha256(args.deploy_config) != cache.spec["deploy_config_sha256"]):
        raise ValueError("Native policy is not the cached collector checkpoint/config")
    for path, digest in cache.spec["code_sha256"].items():
        if sha256(Path(path)) != digest:
            raise ValueError(f"Frozen conditioning code changed: {path}")
    stats = Path(deployment["dataset_stats_path"])
    if not stats.is_absolute():
        stats = policy.project_root / stats
    if sha256(stats) != cache.spec["stats_sha256"]:
        raise ValueError("Normalizer differs from cache")
    targets = install_action_lora(model, args.rank, args.alpha)
    parameters = [p for p in model.parameters() if p.requires_grad]
    trainable = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if any(not name.endswith(("lora_a", "lora_b")) for name in trainable):
        raise RuntimeError("Unexpected trainable frozen condition parameter")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=0.)
    sampler = torch.Generator().manual_seed(args.seed)
    provenance = dict(transitions_sha256=sha256(args.transitions), weights_sha256=sha256(args.output / "weights.pt"),
        base_checkpoint_sha256=cache.spec["base_checkpoint_sha256"], feature_spec=cache.spec,
        rank=args.rank, alpha=args.alpha, lr=args.lr, accumulate=args.accumulate, seed=args.seed,
        uniform_weights=args.uniform_weights, targets=targets,
        code_sha256={p: sha256(Path(p)) for p in (__file__, "src/ahawam/offline_rl/flow_actor.py")})
    initial_step = 0
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["format"] != "ahawam_iql_flow_lora/v1" or saved["provenance"] != provenance:
            raise ValueError("Action adapter resume provenance mismatch")
        restore_adapter(model, saved["adapter"])
        optimizer.load_state_dict(saved["optimizer"])
        initial_step = saved["step"]
        torch.set_rng_state(saved["rng"])
        torch.cuda.set_rng_state(saved["cuda_rng"])
        sampler.set_state(saved["sampler_rng"])
    if initial_step >= args.steps:
        raise ValueError("Requested total steps must exceed the restored step")
    # Eval disables dropout; autograd stays enabled for the adapters. No serving
    # sampler or no_grad wrapper is used in the actor's training forward.
    model.eval()
    provider = FrozenConditionProvider(policy, data, cache)
    action = (data["tensors"]["action"].float() * torch.tensor(cache.spec["action_scale"])
              + torch.tensor(cache.spec["action_offset"])).clamp(-5, 5)
    valid = data["tensors"]["action_valid"]
    train_ids, val_ids = weights["train_ids"], weights["val_ids"]
    val_ids = val_ids[torch.linspace(0, len(val_ids) - 1, min(args.val_samples, len(val_ids))).long()]
    count = sum(p.numel() for p in parameters)
    atomic_json(args.output / "config.json", dict(provenance=provenance, trainable_parameters=count,
        steps=args.steps, microbatch=1, effective_batch=args.accumulate, model_dtype="bf16", adapter_dtype="fp32",
        validation_ids=val_ids.tolist(), train_transitions=len(train_ids)))
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps(dict(stage="model_ready", load_seconds=time.perf_counter() - started,
        trainable_parameters=count, memory=memory_stats())), flush=True)

    def flow(index, verify=False):
        state = provider.prepare(data["records"][index], verify=verify)
        clean = action[index:index+1].cuda().to(model.torch_dtype)
        noise = torch.randn_like(clean)
        scheduler = model.train_action_scheduler
        timestep = scheduler.sample_training_t(1, clean.device, clean.dtype)
        noisy = scheduler.add_noise(clean, noise, timestep)
        target = scheduler.training_target(clean, noise, timestep)
        # A scalar timestep is returned for B=1 by training_weight.
        prediction = predict_native_flow(model, noisy, timestep, state)
        weight = torch.ones(1, device=clean.device) if args.uniform_weights else weights["weights"][index:index+1].cuda()
        loss, unweighted = weighted_flow_loss(prediction, target, valid[index:index+1].cuda(),
            scheduler.training_weight(timestep), weight)
        return loss, unweighted, weight, state

    def validation():
        # Fixed noise/timesteps and no disturbance to training RNG.
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.manual_seed(12345)
            result = []
            for index in val_ids.tolist():
                loss, plain, _, _ = flow(index)
                result.append([float(loss), float(plain[0])])
        return dict(zip(("weighted_fm", "unweighted_fm"), torch.tensor(result).mean(0).tolist()))

    timings = []
    try:
        first_index = int(train_ids[0])
        # Check both episode start and a nonzero offset with all six history
        # slots available. A zero-history/zero-offset check alone is insufficient.
        by_key = {data["records"][int(i)]["observation_key"]: int(i) for i in train_ids}
        extra = None
        for path in sorted(cache.root.glob("*_ep*.json")):
            extra = next((by_key[row["key"]] for row in json.loads(path.read_text())["rows"]
                if row["key"] in by_key and len(row["history_frames"]) == cache.spec["history_count"]
                and row["offset"] > 0), None)
            if extra is not None:
                break
        if extra is None:
            raise ValueError("No nonzero-offset/full-history sample available for equivalence check")
        checks = []
        for index in dict.fromkeys((first_index, extra)):
            state = provider.prepare(data["records"][index], verify=True)
            equivalence = check_flow_equivalence(model, state, action[index:index+1].cuda().to(model.torch_dtype))
            checks.append(dict(observation=data["records"][index]["observation_key"],
                kv=state["cache_verification"], flow_max_error=equivalence))
        atomic_json(args.output / "equivalence.json", dict(checks=checks))
        if not args.resume:
            baseline = validation()
            atomic_json(args.output / "baseline_validation.json", baseline)
            checkpoint(args.output / "step_000000.pt", model, optimizer, 0, provenance, sampler)
        last_snapshot = adapter_state(model) if args.smoke else None
        with (args.output / "metrics.jsonl").open("a") as log:
            for step in range(initial_step + 1, args.steps + 1):
                tick = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                ids = train_ids[torch.randint(len(train_ids), (args.accumulate,), generator=sampler)]
                metrics = []
                for index in ids.tolist():
                    loss, plain, weight, _ = flow(index)
                    (loss / args.accumulate).backward()
                    metrics.append([float(loss.detach()), float(plain[0]), float(weight[0])])
                grad = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                if not grad > 0:
                    raise RuntimeError("Action DiT adapters received no gradient")
                optimizer.step()
                if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
                    raise RuntimeError("Frozen parameter received a gradient")
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - tick)
                if step == initial_step + 1 or step % args.log_every == 0 or step == args.steps:
                    means = torch.tensor(metrics).mean(0)
                    row = dict(step=step, weighted_fm=float(means[0]), unweighted_fm=float(means[1]),
                        weight_mean=float(means[2]), grad_norm=float(grad),
                        seconds_per_update=sum(timings[-20:]) / len(timings[-20:]),
                        memory=memory_stats())
                    if step == initial_step + 1 or step % args.save_every == 0 or step == args.steps:
                        row["validation"] = validation()
                    log.write(json.dumps(row) + "\n")
                    log.flush()
                    print(json.dumps(row), flush=True)
                if step % args.save_every == 0 or step == args.steps:
                    checkpoint(args.output / "last.pt", model, optimizer, step, provenance, sampler)
                    checkpoint(args.output / f"step_{step:06d}.pt", model, optimizer, step, provenance, sampler)
        if args.smoke:
            final = adapter_state(model)
            changed = sum(not torch.equal(final[k], v) for k, v in last_snapshot.items())
            if not changed:
                raise RuntimeError("Adapters did not change after optimization")
            saved = torch.load(args.output / "last.pt", map_location="cpu", weights_only=True)
            restore_adapter(model, saved["adapter"])
            for k, v in adapter_state(model).items():
                torch.testing.assert_close(v, final[k], rtol=0, atol=0)
            atomic_json(args.output / "smoke_passed.json", dict(changed_adapter_tensors=changed,
                frozen_gradients_absent=True, checkpoint_roundtrip=True, equivalence=equivalence))
        atomic_json(args.output / "profile.json", dict(total_steps=args.steps, initial_step=initial_step,
            wall_seconds=time.perf_counter() - started, memory=memory_stats(),
            seconds_per_update=sum(timings) / max(1, len(timings)), effective_batch=args.accumulate))
    finally:
        provider.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transitions", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--critic-checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deploy-config", type=Path, default=Path("scripts/so101_pipeline/offline_rl/configs/cylinder_sft_cache.yaml"))
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--accumulate", type=int, default=8)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--alpha", type=float, default=16.)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--val-samples", type=int, default=16)
    p.add_argument("--resume", type=Path)
    p.add_argument("--uniform-weights", action="store_true", help="Same-model FM BC ablation, no advantage weighting")
    p.add_argument("--export-weights-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Also assert adapter updates and checkpoint roundtrip")
    args = p.parse_args()
    if min(args.steps, args.accumulate, args.rank, args.log_every, args.save_every, args.val_samples) < 1 or min(args.lr, args.alpha) <= 0:
        p.error("Use positive update counts, batch sizes and learning rates")
    train(args)
