"""Temporary real TCP server test on recorded images; never commands a robot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import numpy as np
import pandas as pd
import torch
import yaml

from ahawam.offline_rl.cache import ShardedCache, atomic_json
from ahawam.offline_rl.inference import FrozenKVActor
from scripts.frame_annotation.annotate import Video
from scripts.so101_pipeline.online_deploy.ahawam_tcp_client import AHAWAMTCPClient


def run(args):
    torch.set_num_threads(2)
    cfg = yaml.safe_load(args.config.read_text())
    mode = cfg["offline_rl"]["mode"]
    host, port = cfg["serving"]["host"], cfg["serving"]["port"]
    if host != "127.0.0.1":
        raise ValueError("Smoke server must remain on loopback")
    with socket.socket() as probe:
        if probe.connect_ex((host, port)) == 0:
            raise RuntimeError("Port already in use; will not touch an existing service")
    args.output.mkdir(parents=True, exist_ok=True)
    table = pd.read_parquet(args.episode_root / "data/chunk-000/episode_000001.parquet")
    if not (np.diff(table.tick_index.to_numpy()[:289]) == 1).all() or table.dropped_tick.iloc[:289].any():
        raise ValueError("Smoke cache comparison needs a fully observed 0..288 prefix")
    info = json.loads((args.episode_root / "meta/info.json").read_text())
    videos = {cam: Video(args.episode_root / info["video_path"].format(episode_index=1, episode_chunk=0,
        video_key=f"observation.images.{cam}"), len(table)) for cam in ("fixed", "wrist")}
    data = torch.load("runs/offline_rl/cylinder103_skip_hold/data/transitions.pt", map_location="cpu", weights_only=True)
    episode = next(e for e in data["episodes"] if e["root"] == str(args.episode_root.resolve()) and e["episode_index"] == 1)
    cache = ShardedCache("runs/offline_rl/cylinder103_skip_hold/cache")
    actor = FrozenKVActor(cfg["offline_rl"]["checkpoint_path"], "cpu") if mode != "iql_action_dit" else None
    started = time.perf_counter()
    server = None
    client = None
    report = dict(mode=mode, config=str(args.config), gpu=cfg["serving"]["gpu_id"], port=port, requests=[])
    try:
        command = ["bash", "scripts/so101_pipeline/online_deploy/run_action_server.sh", str(args.config)]
        with (args.output / "server.log").open("w") as log:
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                env=dict(os.environ, OMP_NUM_THREADS="2"))
        atomic_json(args.output / "launch.json", dict(pid=server.pid, command=command))
        while time.perf_counter()-started < 1800:
            if server.poll() is not None:
                raise RuntimeError(f"Server failed: exit={server.returncode}; inspect server.log")
            try:
                client = AHAWAMTCPClient(host, port)
                client.connect()
                break
            except OSError:
                time.sleep(.5)
        else:
            raise TimeoutError("Server did not become available")
        report["startup_seconds"] = time.perf_counter()-started
        report["health"] = client.health()
        assert report["health"]["offline_rl_mode"] == mode
        assert report["health"]["offline_rl_steps"] == cfg["offline_rl"]["expected_steps"]
        instruction = cfg["serving"]["instruction"]
        client.reset(instruction)
        first = None
        for frame in range(0, 289, 8):
            payload = dict(state_model=np.array(table["observation.state"].iloc[frame], copy=True), instruction=instruction,
                fixed_rgb=videos["fixed"].frame(frame), wrist_rgb=videos["wrist"].frame(frame))
            response = client.infer(**payload)
            action = np.asarray(response["action_chunk"])
            assert action.shape == (8, 6) and np.isfinite(action).all()
            if frame == 0:
                first = action.copy()
            row = dict(frame=frame, latency_ms=response["model_latency_ms"], memory=response.get("cuda_memory"))
            if actor is not None and frame in (0, 48, 288):
                sample = cache.sample(f"{episode['episode_uid']}:{frame}")
                expected = actor.action(sample["k"][None], sample["v"][None], sample["state"][None], sample["reference"][None])["absolute_targets"][0].cpu().numpy()
                row["offline_cache_max_abs_degrees"] = float(np.abs(action-expected).max())
                np.testing.assert_allclose(action, expected, atol=.005, rtol=2e-5)
            report["requests"].append(row)
            atomic_json(args.output / "progress.json", report)
        report["full_history_health"] = client.health()
        assert report["full_history_health"]["last_conditioning"] == dict(request_index=36, command_step=288,
            history_steps=[0, 48, 96, 144, 192, 240], anchor_step=288, offset=0, chunk_index=0)
        client.reset(instruction)
        again = client.infer(state_model=np.array(table["observation.state"].iloc[0], copy=True), instruction=instruction,
            fixed_rgb=videos["fixed"].frame(0), wrist_rgb=videos["wrist"].frame(0))
        np.testing.assert_allclose(again["action_chunk"], first, atol=1e-5, rtol=1e-6)
        report.update(reset_reproducible=True, total_seconds=time.perf_counter()-started,
            inference_shape=[8, 6], finite=True, full_history_verified=True)
        atomic_json(args.output / "passed.json", report)
        print(json.dumps(dict(mode=mode, result="passed", startup_seconds=report["startup_seconds"],
            last_latency_ms=report["requests"][-1]["latency_ms"], memory=report["requests"][-1]["memory"])), flush=True)
    finally:
        if client is not None:
            client.close()
        for video in videos.values():
            video.close()
        if server is not None:
            try:
                os.killpg(server.pid, signal.SIGTERM)
                server.wait(timeout=20)
            except ProcessLookupError:
                pass
            except subprocess.TimeoutExpired:
                os.killpg(server.pid, signal.SIGKILL)
                server.wait(timeout=20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episode-root", type=Path, default=Path("data/ahawam_hil_raw/so101_ahawam_cylinder_hil_v2/20260911_183057"))
    run(parser.parse_args())
