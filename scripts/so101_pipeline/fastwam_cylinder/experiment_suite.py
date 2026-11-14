"""Persistent four-experiment suite; smoke each scope before its formal run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from omegaconf import OmegaConf

from ahawam.offline_rl.cache import atomic_json


DEFAULT_ORDER=["franka_demo_sft","robocoin_demo_sft","franka_action_only","franka_full_vidext"]


def process_identity(pid):
    """PID + kernel start time prevents adopting a subsequently reused PID."""
    try:
        raw=(Path("/proc")/str(pid)/"stat").read_text().rsplit(")",1)[1].split()
        if raw[0]=="Z":return None
        return raw[19]
    except FileNotFoundError:return None


def wait_adopted(pid,identity):
    while process_identity(pid)==identity:time.sleep(5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("runs/fastwam_cylinder_20261105"))
    p.add_argument("--gpus",nargs=3,type=int,default=[0,1,2])
    p.add_argument("--port",type=int,default=29651)
    p.add_argument("--adopt-running",action="store_true",
                   help="Replace a stopped supervisor while preserving its running training process")
    args=p.parse_args();root=args.root.resolve()
    if len(set(args.gpus))!=3:raise ValueError("Use three different GPUs")
    stages=DEFAULT_ORDER
    adopted=None
    if args.adopt_running:
        state=json.loads((root/"queue.json").read_text())
        if process_identity(state["pid"]) is not None:raise ValueError("Old supervisor is still alive")
        name=state["current"];job=state["jobs"][name]
        if job["status"]!="running":raise ValueError("Only a formal running job can be adopted")
        identity=process_identity(job["pid"])
        if identity is None:raise ValueError("Training launcher is no longer running")
        command=(Path("/proc")/str(job["pid"])/"cmdline").read_bytes().split(b"\0")
        if b"scripts/so101_pipeline/fastwam_cylinder/run.py" not in command:
            raise ValueError("Process does not match the expected training launcher")
        adopted=(name,job["pid"],identity)
        state.update(previous_supervisor_pid=state["pid"],pid=os.getpid(),adopted_unix=time.time())
        state["jobs"]={name:state["jobs"][name] for name in stages}
    else:
        if (root/"queue.json").exists():raise ValueError("An experiment queue already exists")
        state=dict(pid=os.getpid(),status="starting",jobs={name:dict(status="pending") for name in stages})
    state["order"]=stages
    child=None
    def save():atomic_json(root/"queue.json",state)
    def stop(*_):raise InterruptedError("Queue interrupted")
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,args.gpus)),PYTHONPATH="src:.",
        OMP_NUM_THREADS="2",TOKENIZERS_PARALLELISM="false",WANDB_DISABLE_CODE="true",WANDB_DISABLE_GIT="true",
        DIFFSYNTH_MODEL_BASE_PATH=str(Path("checkpoints").resolve()),
        CUDA_HOME=str(Path(sys.executable).resolve().parents[1]),
        PATH=str(Path(sys.executable).resolve().parent)+os.pathsep+os.environ.get("PATH",""),
        TORCH_EXTENSIONS_DIR=str(Path("runs/cache/torch_extensions").resolve()),
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    env.pop("WANDB_DISABLED",None)
    try:
        if adopted is not None:
            name,pid,identity=adopted
            state["status"]="training";save()
            wait_adopted(pid,identity)
            cfg=OmegaConf.load(root/"configs"/f"{name}.yaml")
            result=json.loads((Path(cfg.output_dir)/"complete.json").read_text())
            if result["step"]!=int(cfg.max_steps):raise ValueError("Adopted training did not finish all steps")
            state["jobs"][name].update(status="complete",result=result,ended_unix=time.time());save()
        state["status"]="checking_toolchain";save()
        with (root/"toolchain.log").open("a") as log:
            subprocess.run([sys.executable,"-c",
                "from deepspeed.ops.op_builder import CPUAdamBuilder; CPUAdamBuilder().load(); print('CPU_ADAM_READY')"],
                env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        for name in stages:
            if state["jobs"][name]["status"]=="complete":continue
            cfg_path=root/"configs"/f"{name}.yaml";cfg=OmegaConf.load(cfg_path)
            checkpoint=Path(cfg.init_checkpoint)
            state.update(status="waiting_for_initialization",current=name);save()
            while not checkpoint.is_file():time.sleep(5)
            if checkpoint.stat().st_size<1_000_000_000:raise ValueError("Incomplete initialization checkpoint")
            from scripts.so101_pipeline.fastwam_cylinder.checkpoint_audit import audit_checkpoint
            state["jobs"][name]["checkpoint_audit"]=audit_checkpoint(cfg);save()
            command=[sys.executable,"-m","accelerate.commands.launch","--config_file",
                "scripts/so101_pipeline/full_finetune_zero2_offload/accelerate.yaml","--num_processes","3",
                "--main_process_port",str(args.port),"scripts/so101_pipeline/fastwam_cylinder/run.py","--config",str(cfg_path)]
            for smoke in (True,False):
                cmd=command+(["--smoke-steps","2"] if smoke else [])
                label=name+("_smoke" if smoke else "")
                state["jobs"][name].update(status="smoke_running" if smoke else "running",
                    command=cmd,started_unix=time.time());state["status"]="training";save()
                with (root/f"{label}.log").open("a") as log:
                    child=subprocess.Popen(cmd,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                state["jobs"][name]["pid"]=child.pid;save()
                code=child.wait();child=None
                if code:raise RuntimeError(f"{label} exited {code}, see its log")
                summary=Path(cfg.output_dir).parent/("smoke_model" if smoke else "model")/"complete.json"
                result=json.loads(summary.read_text())
                if result["step"]!=(2 if smoke else int(cfg.max_steps)):raise ValueError("Unexpected completed step count")
                state["jobs"][name]["smoke" if smoke else "result"]=result;save()
            state["jobs"][name].update(status="complete",ended_unix=time.time());save()
        state["status"]="complete";save()
    except BaseException as error:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        state.update(status="failed",error=repr(error));save();raise


if __name__=="__main__":main()
