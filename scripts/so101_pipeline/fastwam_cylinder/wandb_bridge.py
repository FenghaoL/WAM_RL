"""Backfill/follow existing Rich console metrics without touching the learner.

Historical console values have printed precision only. GPU memory is attached
only when a live progress snapshot matches the training step, never invented
for older steps. No model/data artifacts are uploaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import time

from ahawam.offline_rl.cache import atomic_json


NUMBER=r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
ENTRY=re.compile(r"^\d{2}/\d{2}\s+\[\d{2}:\d{2}:\d{2}\]")
ANSI=re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def training_records(text):
    text=ANSI.sub("",text)
    blocks=[];current=[]
    for line in text.splitlines():
        if ENTRY.match(line) and current:
            blocks.append(" ".join(current));current=[]
        current.append(line)
    if current:blocks.append(" ".join(current))
    records={}
    for block in blocks:
        match=re.search(r"\bepoch=(\d+)\s+step=(\d+)/(\d+)",block)
        if match is None or not re.search(r"\bopt="+NUMBER+r"s\)",block):continue
        values={key:float(value) for key,value in re.findall(r"\b([a-zA-Z_]\w*)=("+NUMBER+r")",block)}
        if not {"loss","loss_action","loss_video","lr","grad_norm","data","fwd","bwd","opt"}<=values.keys():continue
        payload={f"train/{key}":values[key] for key in ("loss","loss_action","loss_video","lr","grad_norm")}
        payload["train/epoch"]=int(match[1])
        for old,new in [("data","data_time_s"),("fwd","forward_time_s"),("bwd","backward_time_s"),("opt","optimizer_time_s")]:
            payload["performance/"+new]=values[old]
        if "speed" in values:payload["performance/steps_per_sec"]=values["speed"]
        samples=re.search(r"("+NUMBER+r")\s+samples/s",block)
        if samples:payload["performance/samples_per_sec"]=float(samples[1])
        records[int(match[2])]=payload
    return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path("runs/fastwam_cylinder_20261105"))
    p.add_argument("--name",default="franka_demo_sft")
    p.add_argument("--entity",default="fenghaoliu8-university-of-michigan")
    p.add_argument("--project",default="fastwam-so101-cylinder")
    args=p.parse_args();root=args.root.resolve();output=root/"wandb_bridge";output.mkdir(exist_ok=True)
    state_path=output/f"{args.name}.json"
    previous=json.loads(state_path.read_text()) if state_path.exists() else {}
    import wandb
    import yaml
    config=yaml.safe_load((root/args.name/"model/config.yaml").read_text())
    config["observability"]=dict(mode="external_console_bridge",historical_precision="printed console precision",
        training_process_restarted=False,model_artifacts_uploaded=False)
    run=wandb.init(entity=args.entity,project=args.project,name=args.name,group="cylinder-20261105",
        id="fwc1105frankademo",resume="allow",mode="online",dir=str(output),config=config,
        save_code=False,settings=wandb.Settings(x_disable_stats=True,disable_git=True))
    run.define_metric("global_step");run.define_metric("*",step_metric="global_step")
    last=int(previous.get("last_training_step",0));seen_val=set(previous.get("validation_steps",[]))
    def save(status):
        atomic_json(state_path,dict(status=status,url=run.url,run_id=run.id,last_training_step=last,
            validation_steps=sorted(seen_val),pid=__import__('os').getpid(),updated_unix=time.time(),
            historical_metrics="rounded console values",artifacts_uploaded=False))
    try:
        while True:
            records=training_records((root/f"{args.name}.log").read_text(errors="replace"))
            progress_path=root/args.name/"model/progress.json"
            progress=json.loads(progress_path.read_text()) if progress_path.exists() else None
            for step,metrics in sorted(records.items()):
                if step<=last:continue
                payload=dict(metrics,global_step=step)
                if progress is not None and progress["step"]==step:
                    payload.update({f"memory/{key}":value for key,value in progress["memory"].items()})
                run.log(payload);last=step
            val_file=root/args.name/"model/validation.jsonl"
            if val_file.exists():
                for line in val_file.read_text().splitlines():
                    row=json.loads(line);step=row.pop("step")
                    if step in seen_val:continue
                    run.log(dict(global_step=step,**{f"eval/{key}":value for key,value in row.items()}))
                    seen_val.add(step)
            save("following")
            if (root/args.name/"model/complete.json").exists():break
            queue=json.loads((root/"queue.json").read_text())
            if queue["status"]=="failed":raise RuntimeError("Training queue failed; bridge stopped")
            time.sleep(5)
        run.summary["final_training_step"]=last
        run.finish();save("complete")
    except BaseException:
        save("stopped");run.finish();raise


if __name__=="__main__":main()
