"""Prepare the 2026-10-26 episode recipe, derived video prefixes and transitions.

Raw recordings/annotations are read only. No reward is invented at recording
cutoffs. The fixed selection precedes any new critic scoring.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import av
import torch

from ahawam.offline_rl.cache import atomic_json
from ahawam.offline_rl.data import TransitionConfig, build_dataset, sha256
from scripts.so101_pipeline.offline_rl.run import audit
from scripts.so101_pipeline.offline_rl.system import split


FAILURE_HOLDOUT={("20260911_183057",0),("20260911_225809",1)}
HALF_PREFIX={("20260911_232927",7),("20260911_232927",15),
             ("20260912_001855",2),("20260912_001855",9)}


def prefix_video(source, target, count):
    target.parent.mkdir(parents=True,exist_ok=True)
    if target.exists():
        raise ValueError(f"Output already exists: {target}")
    temporary=target.with_suffix(".tmp.mp4")
    with av.open(str(source)) as incoming, av.open(str(temporary),"w") as outgoing:
        stream=incoming.streams.video[0]
        encoder=outgoing.add_stream("libx264",rate=stream.average_rate)
        encoder.width,encoder.height=stream.width,stream.height
        encoder.pix_fmt="yuv420p";encoder.options={"crf":"20","preset":"fast","threads":"2"}
        written=0
        for frame in incoming.decode(stream):
            frame.pts=None
            for packet in encoder.encode(frame):outgoing.mux(packet)
            written+=1
            if written==count:break
        for packet in encoder.encode():outgoing.mux(packet)
    if written!=count:raise ValueError("Video shorter than requested prefix")
    with av.open(str(temporary)) as video:
        if sum(1 for _ in video.decode(video=0))!=count:raise ValueError("Prefix encode frame count mismatch")
    temporary.replace(target)
    return dict(source=str(source.resolve()),output=str(target.resolve()),frames=count,sha256=sha256(target),
                purpose="viewable derived prefix; cache uses identical frame indices in original recording")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw-root",type=Path,default=Path("data/ahawam_hil_raw/so101_ahawam_cylinder_hil_v2"))
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():p.error("Choose a fresh output directory")
    roots=sorted(d for d in args.raw_root.iterdir() if (d/"meta/info.json").exists())
    cfg=TransitionConfig(chunk_size=8,stride=1,gamma_tick=.997,temporal_mode="skip_hold",timeline_mode="episode_frames")
    inspected=build_dataset(roots,cfg)
    episodes=inspected["episodes"]
    if len(episodes)!=103:raise ValueError("Expected 103 episodes; review changed dataset")
    unobserved=[e for e in episodes if e["outcome"]=="success" and e["first_success"] is None]
    failures=[e for e in episodes if e["outcome"]=="failure"]
    if len(unobserved)!=9 or len(failures)!=13:raise ValueError("Outcome/annotation inventory changed; review before training")
    rules={};special=[]
    for e in episodes:
        key=(Path(e["root"]).name,e["episode_index"])
        censored=e in unobserved
        half=key in HALF_PREFIX
        if half and not censored:raise ValueError("Prefix selected from wrong outcome group")
        holdout=key in FAILURE_HOLDOUT
        if holdout and e["outcome"]!="failure":raise ValueError("Failure holdout is not a failure")
        # All nine unobserved-success recordings enter TRAIN, including former
        # _val/ep0. Only the three originally labelled val successes stay val.
        part="val" if holdout or (e["root"].endswith("_val") and not censored) else "train"
        used=e["frames"]//2 if half else e["frames"]
        treatment="first_half_censored_success" if half else "full_censored_success" if censored else \
                  "failure_holdout" if holdout else "original_labelled_validation" if part=="val" else "ordinary_train"
        rule=dict(source=e["source"],split=part,retained_frames=used,
            reward_mode="observed_zero_nonterminal" if censored else "annotated_or_failure",treatment=treatment)
        rules[e["episode_uid"]]=rule
        if censored or holdout:special.append(dict(episode_uid=e["episode_uid"],run=key[0],episode_index=key[1],**rule))
    if sum(r["treatment"]=="failure_holdout" for r in rules.values())!=2 or \
       sum(r["treatment"]=="first_half_censored_success" for r in rules.values())!=4:
        raise ValueError("Selection count mismatch")
    recipe=dict(format="ahawam_episode_recipe/v1",date="2026-10-26",episodes=rules,
        decisions=dict(timeline="episode saved-frame order; pauses never reset history or transitions",
            no_success_label="observed reward=0; bootstrap remains gamma**8, even at cutoff; no fabricated +1 or failure terminal",
            half_definition="floor(original saved frames / 2); include observations [0, retained_frames), no action may cross cutoff",
            selection="fixed two source runs for half-prefixes; failure holdout includes one assisted and one policy-only failure; no critic-score selection",
            train_validation="explicit per-episode assignment; former _val/ep0 unobserved success moves to train"))
    corrected=build_dataset(roots,cfg,recipe)
    train,val,val_uids=split(corrected)
    assert set(corrected["records"][int(i)]["episode_uid"] for i in train).isdisjoint(val_uids)
    for i,r in enumerate(corrected["records"]):
        rule=rules[r["episode_uid"]]
        assert r["next_frame"]<rule["retained_frames"] and max(r["action_frames"])<rule["retained_frames"]-1
        if rule["reward_mode"]=="observed_zero_nonterminal":
            assert corrected["tensors"]["reward"][i]==0 and not corrected["tensors"]["terminated"][i]
            assert abs(float(corrected["tensors"]["discount"][i])-.997**8)<1e-8
    args.output.mkdir(parents=True)
    data_dir=args.output/"data";data_dir.mkdir()
    atomic_json(data_dir/"episode_recipe.json",recipe)
    torch.save(corrected,data_dir/"transitions.pt")
    atomic_json(data_dir/"audit.json",audit(corrected))
    clips=[]
    for e in episodes:
        rule=rules[e["episode_uid"]]
        if rule["treatment"]!="first_half_censored_success":continue
        root=Path(e["root"]);info=json.loads((root/"meta/info.json").read_text());ep=e["episode_index"]
        for cam in ("fixed","wrist"):
            source=root/info["video_path"].format(episode_index=ep,episode_chunk=ep//int(info.get("chunks_size",1000)),video_key=f"observation.images.{cam}")
            target=args.output/"prefix_videos"/root.name/f"episode_{ep:06d}_{cam}.mp4"
            clips.append(prefix_video(source,target,rule["retained_frames"]))
    report=dict(format="ahawam_rebuild_preparation/v1",special_episodes=special,prefix_videos=clips,
        episode_counts=dict(Counter(f"{e['split']}/{e['outcome']}" for e in corrected["episodes"])),
        total_transitions=len(corrected["records"]),train_transitions=len(train),validation_transitions=len(val),
        end_reasons=dict(Counter(r["end_reason"] for r in corrected["records"])),
        retained_observations=sum(r["retained_frames"] for r in rules.values()),
        excluded_tail_observations=sum(e["frames"]-e["retained_frames"] for e in corrected["episodes"]),
        recipe_sha256=sha256(data_dir/"episode_recipe.json"),transitions_sha256=sha256(data_dir/"transitions.pt"),
        original_media_unchanged=True)
    atomic_json(args.output/"preparation.json",report)
    print(json.dumps({k:v for k,v in report.items() if k not in {"special_episodes","prefix_videos"}},indent=2))


if __name__=="__main__":main()
