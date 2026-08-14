#!/usr/bin/env python3
"""Throughput/memory tuning sweep for AHA-WAM SO101 full fine-tuning on 4x A100-40GB.

Each config is a short training run (ZeRO-2 + CPU optimizer offload) launched through
the existing partial_finetune_zero2/train_entry.py, so nothing in src/ is modified --
we only vary the DeepSpeed JSON, the accelerate YAML, the settings YAML and env vars.

Per config we record: per-step wall time, per-step history_len, peak GPU memory, and
whether it OOMed. Results are appended to RESULTS.md after every run so a long sweep
stays readable (and useful) even if it is interrupted.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BENCH_ROOT = REPO_ROOT / "runs" / "so101_g05_square_v1" / "tuning_bench"
BASE_SETTINGS = REPO_ROOT / "scripts" / "so101_pipeline" / "full_finetune_zero3" / "settings.yaml"
TRAIN_ENTRY = "scripts/so101_pipeline/partial_finetune_zero2/train_entry.py"
RESULTS_MD = BENCH_ROOT / "RESULTS.md"
RESULTS_JSON = BENCH_ROOT / "results.json"


@dataclass
class Config:
    tag: str
    note: str
    batch_size: int = 3
    grad_accum: int = 1
    num_workers: int = 4
    omp_threads: int = 16
    round_robin: bool = True
    pin_memory: bool = True
    overlap_comm: bool = False
    grad_checkpoint: bool = True
    steps: int = 40
    num_gpus: int = 4
    extra: dict = field(default_factory=dict)

    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum * self.num_gpus


def write_ds_json(path, cfg):
    payload = {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": cfg.pin_memory},
            "offload_param": {"device": "none"},
            "round_robin_gradients": cfg.round_robin,
            "overlap_comm": cfg.overlap_comm,
            "contiguous_gradients": True,
            "reduce_bucket_size": 2e8,
            "allgather_bucket_size": 2e8,
        },
        "steps_per_print": 2000000,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_accel_yaml(path, ds_json):
    path.write_text(
        "compute_environment: LOCAL_MACHINE\n"
        "debug: false\n"
        "distributed_type: DEEPSPEED\n"
        "deepspeed_config:\n"
        "  deepspeed_config_file: " + str(ds_json) + "\n"
        "  deepspeed_multinode_launcher: nossh\n"
        "  zero3_init_flag: false\n"
        "mixed_precision: null\n"
        "machine_rank: 0\n"
        "main_training_function: main\n"
        "num_machines: 1\n"
        "num_processes: 1\n"
        "rdzv_backend: static\n"
        "same_network: true\n"
        "use_cpu: false\n",
        encoding="utf-8",
    )


TOP_LEVEL_PATCHES = {
    "batch_size": lambda c: str(c.batch_size),
    "gradient_accumulation_steps": lambda c: str(c.grad_accum),
    "num_workers": lambda c: str(c.num_workers),
}


def write_settings(path, cfg):
    lines = BASE_SETTINGS.read_text(encoding="utf-8").splitlines()
    out, seen = [], set()
    for line in lines:
        m = re.match(r"^([a-z_]+):\s*(\S.*)$", line)
        if m and m.group(1) in TOP_LEVEL_PATCHES and m.group(1) not in seen:
            key = m.group(1)
            seen.add(key)
            out.append(key + ": " + TOP_LEVEL_PATCHES[key](cfg))
            continue
        m2 = re.match(r"^(\s+)mot_checkpoint_mixed_attn:\s*\S+$", line)
        if m2:
            out.append(m2.group(1) + "mot_checkpoint_mixed_attn: " + str(cfg.grad_checkpoint).lower())
            continue
        out.append(line)
    for key, fn in TOP_LEVEL_PATCHES.items():
        if key not in seen:
            out.append(key + ": " + fn(cfg))
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


TS_RE = re.compile(r"\d\d/\d\d \[(\d\d):(\d\d):(\d\d)\]")
STEP_RE = re.compile(r"step=(\d+)/")
HIST_RE = re.compile(r"history_len=([\d.]+)")
OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory")


def parse_train_log(text):
    steps = []
    for m in STEP_RE.finditer(text):
        idx = int(m.group(1))
        ts_match = None
        for tm in TS_RE.finditer(text, 0, m.start()):
            ts_match = tm
        ts = None
        if ts_match:
            h, mi, s = (int(x) for x in ts_match.groups())
            ts = h * 3600 + mi * 60 + s
        hm = HIST_RE.search(text, m.end())
        hist = float(hm.group(1)) if hm else None
        steps.append({"step": idx, "ts": ts, "history_len": hist})
    dedup = {}
    for s in steps:
        dedup.setdefault(s["step"], s)
    ordered = [dedup[k] for k in sorted(dedup)]
    deltas = []
    for a, b in zip(ordered, ordered[1:]):
        if a["ts"] is not None and b["ts"] is not None:
            d = b["ts"] - a["ts"]
            if d < 0:
                d += 86400
            deltas.append(d)
    return {"steps": ordered, "deltas": deltas, "oom": bool(OOM_RE.search(text))}


def parse_gpu_csv(path):
    if not path.exists():
        return {"peak_mib": None, "samples": []}
    peak, samples = 0, []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            mem = int(float(parts[2]))
        except ValueError:
            continue
        ts = None
        tm = re.search(r"(\d\d):(\d\d):(\d\d)", parts[0])
        if tm:
            h, mi, s = (int(x) for x in tm.groups())
            ts = h * 3600 + mi * 60 + s
        samples.append((ts, mem))
        peak = max(peak, mem)
    return {"peak_mib": peak or None, "samples": samples}


def mem_by_history(steps, samples):
    buckets = {}
    for a, b in zip(steps, steps[1:]):
        if a["ts"] is None or b["ts"] is None or a["history_len"] is None:
            continue
        window = [m for (t, m) in samples if t is not None and a["ts"] <= t <= b["ts"]]
        if not window:
            continue
        h = a["history_len"]
        buckets[h] = max(buckets.get(h, 0), max(window))
    return buckets


def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def run_one(cfg, port):
    run_dir = BENCH_ROOT / cfg.tag
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    ds_json = run_dir / "ds.json"
    accel_yaml = run_dir / "accelerate.yaml"
    settings = run_dir / "settings.yaml"
    write_ds_json(ds_json, cfg)
    write_accel_yaml(accel_yaml, ds_json)
    write_settings(settings, cfg)

    gpu_csv = run_dir / "gpu.csv"
    train_log = run_dir / "train.log"

    env = os.environ.copy()
    env.update({
        "CUDA_HOME": env.get("CONDA_PREFIX", ""),
        "CUDA_VISIBLE_DEVICES": ",".join(str(i) for i in range(cfg.num_gpus)),
        "PYTHONPATH": str(REPO_ROOT) + ((":" + env["PYTHONPATH"]) if env.get("PYTHONPATH") else ""),
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "OMP_NUM_THREADS": str(cfg.omp_threads),
        "HYDRA_FULL_ERROR": "1",
        "TRITON_CACHE_DIR": str(run_dir / "triton_cache"),
        "TORCH_EXTENSIONS_DIR": str(REPO_ROOT / "runs" / "cache" / "torch_extensions"),
        "WANDB_MODE": "disabled",
    })
    env["PATH"] = env["CUDA_HOME"] + "/bin:" + env["PATH"]

    gpu_fh = open(gpu_csv, "w")
    mon = subprocess.Popen(
        ["nvidia-smi",
         "--id=" + ",".join(str(i) for i in range(cfg.num_gpus)),
         "--query-gpu=timestamp,index,memory.used",
         "--format=csv,noheader,nounits", "--loop=1"],
        stdout=gpu_fh, stderr=subprocess.DEVNULL, env=env,
    )

    cmd = [
        "accelerate", "launch",
        "--config_file", str(accel_yaml),
        "--num_processes", str(cfg.num_gpus),
        "--num_machines", "1", "--machine_rank", "0",
        "--main_process_ip", "127.0.0.1", "--main_process_port", str(port),
        TRAIN_ENTRY,
        "--settings", str(settings),
        "--num-gpus", str(cfg.num_gpus),
        "--output-dir", str(run_dir / "model"),
        "--wandb-name", "bench-" + cfg.tag,
        "--smoke-steps", str(cfg.steps),
    ]

    t0 = time.time()
    with open(train_log, "w") as lf:
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=lf,
                                stderr=subprocess.STDOUT, env=env)
        try:
            rc = proc.wait(timeout=5400)
        except subprocess.TimeoutExpired:
            proc.kill()
            rc = -9
    wall = time.time() - t0

    mon.terminate()
    try:
        mon.wait(timeout=10)
    except subprocess.TimeoutExpired:
        mon.kill()
    gpu_fh.close()
    subprocess.run(["pkill", "-f", "train_entry.py --settings " + str(settings)],
                   capture_output=True)
    time.sleep(20)

    text = train_log.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_train_log(text)
    gpu = parse_gpu_csv(gpu_csv)
    steady = parsed["deltas"][3:] if len(parsed["deltas"]) > 6 else parsed["deltas"]
    med = median(steady)
    hists = [s["history_len"] for s in parsed["steps"] if s["history_len"] is not None]

    result = {
        "tag": cfg.tag, "note": cfg.note,
        "batch_size": cfg.batch_size, "grad_accum": cfg.grad_accum,
        "effective_batch": cfg.effective_batch(), "num_workers": cfg.num_workers,
        "omp_threads": cfg.omp_threads, "round_robin": cfg.round_robin,
        "pin_memory": cfg.pin_memory, "overlap_comm": cfg.overlap_comm,
        "grad_checkpoint": cfg.grad_checkpoint,
        "requested_steps": cfg.steps, "completed_steps": len(parsed["steps"]),
        "oom": parsed["oom"], "returncode": rc,
        "median_step_s": med,
        "samples_per_s": (cfg.effective_batch() / med) if med else None,
        "peak_mib": gpu["peak_mib"],
        "max_history_len": max(hists) if hists else None,
        "mem_by_history": dict((str(k), v) for k, v in
                               sorted(mem_by_history(parsed["steps"], gpu["samples"]).items())),
        "wall_s": round(wall, 1),
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def append_results(result):
    all_results = []
    if RESULTS_JSON.exists():
        all_results = json.loads(RESULTS_JSON.read_text(encoding="utf-8"))
    all_results = [r for r in all_results if r["tag"] != result["tag"]]
    all_results.append(result)
    RESULTS_JSON.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    render_markdown(all_results)


def render_markdown(results):
    ok = [r for r in results if not r["oom"] and r["samples_per_s"]]
    best = max(ok, key=lambda r: r["samples_per_s"]) if ok else None
    L = []
    L.append("# AHA-WAM SO101 full finetune - ZeRO-2 tuning results")
    L.append("")
    L.append("Updated: " + time.strftime("%Y-%m-%d %H:%M:%S") + "  |  runs: " + str(len(results)))
    L.append("")
    L.append("**Hardware**: 4 x A100-SXM4-40GB (39.49 GiB usable each), 96 CPU cores, 503 GB RAM  ")
    L.append("**Data**: `data/so101_g05_square_v1_ahawam_v21` (125 episodes / 20,756 frames; 19,570 train)  ")
    L.append("**Fixed**: ZeRO-2 + optimizer CPU offload, full finetune, 4 GPUs, bf16")
    L.append("")
    L.append("## Results (sorted by throughput)")
    L.append("")
    L.append("| tag | micro_bs | accum | eff.batch | OMP | RR | pin | overlap | grad_ckpt | steps | median s/step | **samples/s** | peak MiB | max hist | status |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (-(x["samples_per_s"] or 0))):
        status = "OOM" if r["oom"] else ("ok" if r["returncode"] == 0 else "rc=" + str(r["returncode"]))
        sps = ("**%.3f**" % r["samples_per_s"]) if r["samples_per_s"] else "-"
        med = ("%.1f" % r["median_step_s"]) if r["median_step_s"] else "-"
        L.append("| `%s` | %d | %d | %d | %d | %s | %s | %s | %s | %d/%d | %s | %s | %s | %s | %s |" % (
            r["tag"], r["batch_size"], r["grad_accum"], r["effective_batch"], r["omp_threads"],
            "Y" if r["round_robin"] else "N", "Y" if r["pin_memory"] else "N",
            "Y" if r["overlap_comm"] else "N", "Y" if r["grad_checkpoint"] else "N",
            r["completed_steps"], r["requested_steps"], med, sps,
            str(r["peak_mib"] or "-"),
            str(r["max_history_len"] if r["max_history_len"] is not None else "-"), status))
    if best:
        total_samples = 19570 * 4
        hours = total_samples / best["samples_per_s"] / 3600
        L.append("")
        L.append("**Fastest so far: `%s` - %.3f samples/s**  " % (best["tag"], best["samples_per_s"]))
        L.append("At that rate 4 epochs (%d samples) takes about **%.1f hours** "
                 "(measured ZeRO-3 baseline: 38.5 h)." % (total_samples, hours))
    L.append("")
    L.append("## Memory vs history_len (the OOM-safety data)")
    L.append("")
    L.append("`history_aware_batching=true` makes every sample in a batch share the same history")
    L.append("length. Longer history = more video tokens = larger activations, so memory jumps in")
    L.append("steps. **A safe batch size must fit at the longest history (`num_history_frames=6`)**,")
    L.append("otherwise you get the bs=8 failure mode: 9 clean steps, then OOM when a long-history")
    L.append("batch finally shows up.")
    L.append("")
    for r in sorted(results, key=lambda x: x["tag"]):
        if r["mem_by_history"]:
            detail = ", ".join("h=%s: %s MiB" % (k, v) for k, v in r["mem_by_history"].items())
            L.append("- `%s` (bs=%d): %s" % (r["tag"], r["batch_size"], detail))
    L.append("")
    L.append("## Config notes")
    L.append("")
    for r in sorted(results, key=lambda x: x["tag"]):
        L.append("- `%s`: %s" % (r["tag"], r["note"]))
    RESULTS_MD.write_text("\n".join(L) + "\n", encoding="utf-8")


def phase1_configs(steps):
    return [
        Config("p1-bs2", "batch sweep: micro_bs=2", batch_size=2, steps=steps),
        Config("p1-bs3", "batch sweep: micro_bs=3", batch_size=3, steps=steps),
        Config("p1-bs4", "batch sweep: micro_bs=4", batch_size=4, steps=steps),
        Config("p1-bs5", "batch sweep: micro_bs=5", batch_size=5, steps=steps),
        Config("p1-bs6", "batch sweep: micro_bs=6 (expected at/over the ceiling)",
               batch_size=6, steps=steps),
    ]


def phase2_configs(bs, steps):
    """One-factor-at-a-time from the phase-1 baseline (OMP=16, RR=on, pin=on)."""
    return [
        Config("p2-omp1", "OMP_NUM_THREADS=1 (torchrun default, control)",
               batch_size=bs, omp_threads=1, steps=steps),
        Config("p2-omp4", "OMP_NUM_THREADS=4", batch_size=bs, omp_threads=4, steps=steps),
        Config("p2-omp8", "OMP_NUM_THREADS=8", batch_size=bs, omp_threads=8, steps=steps),
        Config("p2-omp24", "OMP_NUM_THREADS=24 (96 cores / 4 ranks, fair share)",
               batch_size=bs, omp_threads=24, steps=steps),
        Config("p2-norr", "round_robin_gradients off (control)",
               batch_size=bs, round_robin=False, steps=steps),
        Config("p2-nopin", "offload pin_memory off (control)",
               batch_size=bs, pin_memory=False, steps=steps),
        Config("p2-overlap", "overlap_comm on (trades ~0.8GB VRAM for comm overlap)",
               batch_size=bs, overlap_comm=True, steps=steps),
    ]


def phase3_configs(bs, steps):
    """Effective-batch sweep: the main lever left, since the ~6 s step is dominated
    by the fixed per-optimizer-step cost (CPU Adam over the offloaded shard)."""
    return [
        Config("p3-acc2", "grad accum=2 (effective batch x2, amortises CPU Adam)",
               batch_size=bs, grad_accum=2, steps=max(20, steps // 2)),
        Config("p3-acc4", "grad accum=4 (x4)",
               batch_size=bs, grad_accum=4, steps=max(12, steps // 3)),
        Config("p3-acc8", "grad accum=8 (x8; diminishing returns expected here)",
               batch_size=bs, grad_accum=8, steps=max(8, steps // 5)),
        Config("p3-nockpt", "gradient checkpointing off + micro_bs=1 (memory for compute)",
               batch_size=1, grad_checkpoint=False, steps=steps),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--batch-size", type=int, default=3)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--port", type=int, default=29600)
    args = ap.parse_args()

    BENCH_ROOT.mkdir(parents=True, exist_ok=True)
    if args.phase == 1:
        configs = phase1_configs(args.steps)
    elif args.phase == 2:
        configs = phase2_configs(args.batch_size, args.steps)
    else:
        configs = phase3_configs(args.batch_size, args.steps)

    for i, cfg in enumerate(configs):
        print("\n" + "=" * 70)
        print("[%d/%d] %s - %s" % (i + 1, len(configs), cfg.tag, cfg.note))
        print("=" * 70, flush=True)
        res = run_one(cfg, args.port + i)
        append_results(res)
        print("  -> oom=%s steps=%s median_step=%s samples/s=%s peak=%s MiB max_hist=%s" % (
            res["oom"], res["completed_steps"], res["median_step_s"],
            res["samples_per_s"], res["peak_mib"], res["max_history_len"]), flush=True)
    print("\nPHASE DONE", flush=True)


if __name__ == "__main__":
    main()
