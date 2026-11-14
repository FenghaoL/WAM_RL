"""Pinned public weights, resumable range download with LFS SHA verification.

Only small metadata and one weight per selected initialization are downloaded.
Resume a caller-specified prefix; never touch an active downloader's files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import time

from huggingface_hub import get_hf_file_metadata,hf_hub_url
import requests


ASSETS={
    "franka":dict(repo="aabyaneh/fasterwam",revision="692b2912246655435f97801219f289c71bc6c417",
        file="joint/fastwam/train90_joint_job4415/checkpoints/weights/step_058620.pt",
        destination="checkpoints/FastWAM-Franka-Joint/joint/fastwam/train90_joint_job4415/checkpoints/weights/step_058620.pt"),
    "robocoin":dict(repo="SereneC/AHA-WAM-Pretrained",revision="dd11c34ce319c411b62317f0808729f343530eca",
        file="Fast-WAM-pretrained.pt",destination="checkpoints/FastWAM-RoboCOIN/Fast-WAM-pretrained.pt"),
}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("asset",choices=ASSETS)
    p.add_argument("--resume-prefix",type=Path);p.add_argument("--workers",type=int,default=6)
    args=p.parse_args();spec=ASSETS[args.asset];target=Path(spec["destination"])
    url=hf_hub_url(spec["repo"],spec["file"],revision=spec["revision"])
    meta=get_hf_file_metadata(url);expected=meta.etag.strip('"')
    if len(expected)!=64:raise ValueError("Expected an LFS SHA256")
    target.parent.mkdir(parents=True,exist_ok=True)
    partial=target.with_suffix(".ranges.incomplete")
    if target.exists():partial=target
    elif args.resume_prefix is not None:
        if partial.exists():raise ValueError("Partial already exists")
        args.resume_prefix.replace(partial)
    start=partial.stat().st_size if partial.exists() else 0
    if start>meta.size:raise ValueError("Oversized prefix")
    fd=os.open(partial,os.O_CREAT|os.O_RDWR,0o664)
    chunk=32*1024**2;done=0;begin=time.time()
    def download(a,b):
        for retry in range(4):
            try:
                response=requests.get(url,headers={"Range":f"bytes={a}-{b}","Cache-Control":"no-cache"},stream=True,timeout=120)
                if response.status_code!=206 or response.headers.get("Content-Range")!=f"bytes {a}-{b}/{meta.size}":
                    raise ValueError("Server returned wrong byte range")
                position=a
                for block in response.iter_content(1024*1024):
                    written=os.pwrite(fd,block,position)
                    if written!=len(block):raise IOError("Short positional write")
                    position+=written
                if position!=b+1:raise ValueError("Short download range")
                return b-a+1
            except Exception:
                if retry==3:raise
                time.sleep(1+retry)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(download,a,min(a+chunk-1,meta.size-1)) for a in range(start,meta.size,chunk)]
            for future in as_completed(futures):
                done+=future.result()
                if done%(256*1024**2)<chunk:
                    print(json.dumps(dict(asset=args.asset,downloaded_bytes=start+done,total=meta.size,
                        mb_per_second=done/max(time.time()-begin,1)/1e6)),flush=True)
        os.fsync(fd)
    finally:os.close(fd)
    digest=hashlib.sha256()
    with partial.open("rb") as stream:
        for block in iter(lambda:stream.read(16*1024**2),b""):digest.update(block)
    if digest.hexdigest()!=expected:raise ValueError("Downloaded checksum mismatch; do not use partial file")
    if partial!=target:partial.replace(target)
    report=dict(**spec,bytes=meta.size,sha256=expected,verified=True,seconds=time.time()-begin)
    target.with_suffix(".source.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__":main()
