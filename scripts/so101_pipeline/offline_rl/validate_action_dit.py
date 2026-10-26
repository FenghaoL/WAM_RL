"""Run CPU contracts + real-data GPU smoke, then release the dependent queue."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ahawam.offline_rl.cache import atomic_json
from ahawam.offline_rl.data import sha256


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--queue-output", type=Path, required=True)
    p.add_argument("--smoke-output", type=Path, required=True)
    p.add_argument("--pilot-checkpoint", type=Path, required=True)
    p.add_argument("--transitions", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=3)
    args = p.parse_args()
    files = ["scripts/so101_pipeline/offline_rl/action_dit.py", "src/ahawam/offline_rl/flow_actor.py",
             "tests/test_offline_flow_actor.py", "tests/test_offline_rl.py", __file__]
    hashes = {path: sha256(Path(path)) for path in files}
    if args.smoke_output.exists():
        p.error("Use a fresh smoke output directory")
    args.queue_output.mkdir(parents=True, exist_ok=True)
    with (args.queue_output / "action_dit_cpu_tests.log").open("w") as log:
        subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-p", "test_offline*.py", "-q"],
            env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONPATH="src:."),
            stdout=log, stderr=subprocess.STDOUT, check=True)
    command = [sys.executable, "scripts/so101_pipeline/offline_rl/action_dit.py",
        "--transitions", str(args.transitions), "--cache", str(args.cache),
        "--critic-checkpoint", str(args.pilot_checkpoint), "--output", str(args.smoke_output),
        "--steps", "3", "--accumulate", "8", "--log-every", "1", "--save-every", "1", "--val-samples", "2", "--smoke"]
    with args.smoke_output.with_suffix(".log").open("w") as log:
        subprocess.run(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), OMP_NUM_THREADS="2", PYTHONPATH="src:."),
            stdout=log, stderr=subprocess.STDOUT, check=True)
    if any(sha256(Path(path)) != digest for path, digest in hashes.items()):
        raise RuntimeError("Code changed during validation; repeat on a fresh directory")
    smoke = json.loads((args.smoke_output / "smoke_passed.json").read_text())
    equivalence = json.loads((args.smoke_output / "equivalence.json").read_text())
    if (not smoke["frozen_gradients_absent"] or not smoke["checkpoint_roundtrip"]
        or smoke["changed_adapter_tensors"] <= 0 or len(equivalence["checks"]) < 2):
        raise RuntimeError("Incomplete native Action DiT validation")
    atomic_json(args.queue_output / "action_dit_validated.json", dict(code_sha256=hashes,
        smoke_output=str(args.smoke_output.resolve()), command=command, gpu=args.gpu,
        smoke=smoke, equivalence=equivalence,
        profile=json.loads((args.smoke_output / "profile.json").read_text())))
    print("Action DiT validation passed; dependent queue released.", flush=True)
