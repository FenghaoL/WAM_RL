"""Audit collector pauses and prepare corrected transitions, without GPU work.

Run with --original-transitions, --output, and --report. Original artifacts are
read only. A separate cache directory is required for subsequent GPU export.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ahawam.offline_rl.cache import atomic_json, conditioning_frames
from ahawam.offline_rl.data import TransitionConfig, build_dataset, sha256
from scripts.so101_pipeline.offline_rl.run import audit
from scripts.so101_pipeline.offline_rl.system import split


def coverage(counts):
    histogram=Counter(counts)
    n=len(counts)
    return dict(total=n,history_frame_counts=dict(sorted(histogram.items())),
        with_history_fraction=sum(c>0 for c in counts)/n if n else 0,
        full_six_fraction=sum(c==6 for c in counts)/n if n else 0,
        mean_history_frames=sum(counts)/n if n else 0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--original-transitions",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--report",type=Path,required=True)
    args=p.parse_args()
    if args.output.resolve() == args.original_transitions.parent.resolve():
        raise ValueError("Choose a new output directory")
    original=torch.load(args.original_transitions,map_location="cpu",weights_only=True)
    roots=sorted({Path(e["root"]) for e in original["episodes"]})
    legacy_cfg=replace(TransitionConfig(**original["config"]),timeline_mode="strict_ticks")
    legacy=build_dataset(roots,legacy_cfg)
    if legacy["records"] != original["records"]:
        raise ValueError("Legacy transition reproduction failed")
    for key in original["tensors"]:
        torch.testing.assert_close(legacy["tensors"][key],original["tensors"][key],rtol=0,atol=0)
    corrected=build_dataset(roots,replace(legacy_cfg,timeline_mode="episode_frames"))
    tables={};gaps=[];flags=[];settings={};all_hist=[];changed=0;categories=Counter()
    for root in roots:
        info=json.loads((root/"meta/info.json").read_text())
        events=[json.loads(s) for s in (root/"recording_context/events.jsonl").read_text().splitlines()]
        times=np.asarray([e["monotonic_ns"] for e in events])
        settings[root.name]=json.loads((root/"recording_context/collector_config.json").read_text())["control"]
        for ep in [e for e in corrected["episodes"] if Path(e["root"]) == root]:
            index=ep["episode_index"]
            table=pd.read_parquet(root/info["data_path"].format(episode_index=index,
                episode_chunk=index//int(info.get("chunks_size",1000))))
            tables[ep["episode_uid"]]=table
            for frame in range(len(table)):
                new=conditioning_frames(table,frame)
                old=conditioning_frames(table,frame,timeline_mode="strict_ticks")
                all_hist.append(len(new[1]));changed+=new!=old
            for before in np.flatnonzero(np.diff(table.tick_index.to_numpy())!=1):
                after=int(before)+1
                a,b=int(table.monotonic_ns.iloc[before]),int(table.monotonic_ns.iloc[after])
                inside=[events[j] for j in np.flatnonzero((times>=a)&(times<=b))]
                names=sorted({e["event"] for e in inside})
                pause=bool(set(names)&{"pause_requested","leader_alignment_started",
                    "leader_alignment_completed","human_control_started","human_control_stopped","policy_resume_requested"})
                categories["pause_or_handover" if pause else "unexplained"]+=1
                gaps.append(dict(episode_uid=ep["episode_uid"],before_frame=int(before),after_frame=after,
                    before_tick=int(table.tick_index.iloc[before]),after_tick=int(table.tick_index.iloc[after]),
                    wall_seconds=(b-a)/1e9,event_names=names,pause_or_handover=pause))
            for frame in np.flatnonzero(table.dropped_tick.to_numpy()):
                flags.append(dict(episode_uid=ep["episode_uid"],frame=int(frame)))
    groups={"all_cache":coverage(all_hist)}
    train,val,_=split(corrected)
    for name,indices in (("train",train),("validation",val)):
        groups[name]=coverage([len(conditioning_frames(tables[corrected["records"][i]["episode_uid"]],
            corrected["records"][i]["start_frame"])[1]) for i in indices.tolist()])
    args.output.mkdir(parents=True,exist_ok=True)
    torch.save(corrected,args.output/"transitions.pt")
    atomic_json(args.output/"audit.json",audit(corrected))
    oldkeys={r["observation_key"] for r in original["records"]}
    newkeys={r["observation_key"] for r in corrected["records"]}
    report=dict(original_transitions=str(args.original_transitions.resolve()),
        original_sha256=sha256(args.original_transitions),corrected_transitions=str((args.output/"transitions.pt").resolve()),
        corrected_sha256=sha256(args.output/"transitions.pt"),legacy_reproduction_exact=True,
        old_transitions=len(oldkeys),new_transitions=len(newkeys),added_windows=len(newkeys-oldkeys),
        removed_windows=len(oldkeys-newkeys),old_success_windows=sum(r["end_reason"]=="success" for r in original["records"]),
        new_success_windows=sum(r["end_reason"]=="success" for r in corrected["records"]),
        old_success_episodes=len({r["episode_uid"] for r in original["records"] if r["end_reason"]=="success"}),
        new_success_episodes=len({r["episode_uid"] for r in corrected["records"] if r["end_reason"]=="success"}),
        old_split=dict(train=len(split(original)[0]),validation=len(split(original)[1])),
        new_split=dict(train=len(train),validation=len(val)),
        conditioning_changed_observations=changed,observations=len(all_hist),
        history_coverage=groups,tick_jump_categories=dict(categories),tick_jumps=gaps,
        dropped_tick_rows=flags,collector_control_settings=settings,
        status="CPU preparation only; existing cache, checkpoints and videos unchanged; new cache not exported")
    args.report.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(args.report,report)
    print(json.dumps({k:v for k,v in report.items() if k not in {
        "tick_jumps","dropped_tick_rows","collector_control_settings"}},indent=2))


if __name__=="__main__":
    main()
