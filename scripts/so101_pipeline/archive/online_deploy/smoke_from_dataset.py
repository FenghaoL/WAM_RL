#!/usr/bin/env python3
"""Exercise a running deployment service with one real SO101 dataset frame."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ahawam.datasets.lerobot.lerobot.lerobot_dataset import LeRobotDataset
from scripts.so101_pipeline.online_deploy.ahawam_tcp_client import AHAWAMTCPClient


def _to_hwc_uint8(tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.shape[0] == 3:
        array = np.moveaxis(array, 0, -1)
    if np.issubdtype(array.dtype, np.floating) and float(array.max()) <= 1.0:
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument("--mode", choices=("action", "video", "both"), default="action")
    parser.add_argument("--dataset", default="data/so101_g05_square_v1_ahawam_v21")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=3)
    parser.add_argument(
        "--requests",
        type=int,
        default=1,
        help="Number of consecutive requests to send after one episode reset.",
    )
    parser.add_argument(
        "--save-video",
        default="runs/so101_g05_square_v1/online_previews/client_received.mp4",
    )
    args = parser.parse_args()

    root = Path(args.dataset).resolve()
    dataset = LeRobotDataset(
        repo_id=root.name,
        root=root,
        episodes=[args.episode],
        video_backend="pyav",
    )
    sample = dataset[args.frame]
    instruction = str(sample["task"])
    observation = {
        "fixed_rgb": _to_hwc_uint8(sample["observation.images.fixed"]),
        "wrist_rgb": _to_hwc_uint8(sample["observation.images.wrist"]),
        "state_model": sample["observation.state"].detach().cpu().numpy(),
        "instruction": instruction,
    }

    with AHAWAMTCPClient(args.host, args.port) as client:
        print("health:", client.health())
        print("reset:", client.reset(instruction))
        for request_index in range(args.requests):
            if args.mode == "action":
                response = client.infer(**observation)
            elif args.mode == "video":
                response = client.video_preview(**observation)
            else:
                response = client.infer_with_video(**observation)
            print(f"request_{request_index + 1}_type:", response["type"])
            print(f"request_{request_index + 1}_model_latency_ms:", response["model_latency_ms"])
            if "cuda_memory" in response:
                print(f"request_{request_index + 1}_cuda_memory:", response["cuda_memory"])
            if "action_chunk" in response:
                actions = np.asarray(response["action_chunk"])
                print(f"request_{request_index + 1}_action_shape:", actions.shape)
                print(f"request_{request_index + 1}_action_first:", np.round(actions[0], 4).tolist())
                print(
                    f"request_{request_index + 1}_action_finite:",
                    bool(np.isfinite(actions).all()),
                )
            if "future_video" in response:
                video = response["future_video"]
                output = Path(args.save_video)
                if args.requests > 1:
                    output = output.with_name(
                        f"{output.stem}_{request_index + 1:02d}{output.suffix}"
                    )
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(video["data"])
                print(f"request_{request_index + 1}_video_path:", output)
                print(f"request_{request_index + 1}_video_frames:", video["frame_count"])
                print(f"request_{request_index + 1}_video_fps:", video["fps"])
                print(
                    f"request_{request_index + 1}_video_generation_latency_ms:",
                    video["generation_latency_ms"],
                )
                print(
                    f"request_{request_index + 1}_video_history_frames_used:",
                    video.get("history_frames_used", 0),
                )
                print(
                    f"request_{request_index + 1}_video_current_frame_index:",
                    video.get("video_current_frame_index", 0),
                )


if __name__ == "__main__":
    main()
