"""Persistent, fail-closed three-experiment queue (no robot connection)."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ahawam.offline_rl.cache import atomic_json


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve().parent
    common = ["--transitions", str(args.transitions.resolve()), "--cache", str(args.cache.resolve())]
    jobs = {}
    children = {}
    for name, gpu in (("iql", args.gpus[0]), ("td3_bc", args.gpus[1])):
        command = [sys.executable, str(script / "system.py"), "train", *common,
                   "--output", str(args.output / name), "--algorithm", name,
                   "--steps", str(args.steps), "--batch-size", "64", "--save-every", "1000"]
        jobs[name] = dict(gpu=gpu, command=command, status="pending")
    jobs["iql_action_dit"] = dict(gpu=args.gpus[2], status="waiting_for_iql_and_validation",
        command=[sys.executable, str(script / "action_dit.py"), *common,
                 "--critic-checkpoint", str(args.output / "iql/last.pt"),
                 "--output", str(args.output / "iql_action_dit"), "--steps", str(args.dit_steps)])

    def save():
        atomic_json(args.output / "queue.json", dict(supervisor_pid=os.getpid(), jobs=jobs))

    def start(name):
        job = jobs[name]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job["gpu"]), OMP_NUM_THREADS="2", PYTHONPATH="src:.")
        with (args.output / f"{name}.log").open("a") as log:
            child = subprocess.Popen(job["command"], env=env, stdout=log, stderr=subprocess.STDOUT)
        children[name] = child
        job.update(status="running", pid=child.pid, started_unix=time.time())
        save()

    save()
    start("iql")
    start("td3_bc")
    while True:
        for name, child in list(children.items()):
            code = child.poll()
            if code is not None:
                jobs[name].update(status="complete" if code == 0 else "failed", exit_code=code, ended_unix=time.time())
                del children[name]
                save()
        third = jobs["iql_action_dit"]
        if third["status"] == "waiting_for_iql_and_validation":
            if jobs["iql"]["status"] == "failed":
                third["status"] = "dependency_failed"
                save()
            elif jobs["iql"]["status"] == "complete" and (args.output / "action_dit_validated.json").exists():
                validation = json.loads((args.output / "action_dit_validated.json").read_text())
                from ahawam.offline_rl.data import sha256
                if any(sha256(Path(path)) != digest for path, digest in validation["code_sha256"].items()):
                    third["status"] = "validation_code_changed"
                    save()
                else:
                    start("iql_action_dit")
        if not children and third["status"] != "waiting_for_iql_and_validation":
            break
        time.sleep(5)
    if any(job["status"] != "complete" for job in jobs.values()):
        raise RuntimeError("An experiment failed; inspect queue.json and individual logs")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transitions", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpus", type=int, nargs=3, default=[0, 1, 2])
    p.add_argument("--steps", type=int, default=10000)
    p.add_argument("--dit-steps", type=int, default=2000)
    args = p.parse_args()
    if args.output.exists():
        p.error("Use a new output directory to avoid overwriting or duplicate supervisors")
    if len(set(args.gpus)) != 3 or min(args.gpus) < 0 or min(args.steps, args.dit_steps) < 1:
        p.error("Use three distinct GPUs and positive update counts")
    run(args)
