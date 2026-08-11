#!/usr/bin/env python3
"""Standalone robot-side client for the AHA-WAM TCP service.

The protocol uses pickle and must only be exposed on a trusted private network.
"""

from __future__ import annotations

import argparse
import pickle
import socket
import struct
from pathlib import Path
from typing import Any, Optional

import numpy as np
from PIL import Image


SO101_SIGNS = np.asarray([1, -1, 1, 1, 1, 1], dtype=np.float32)
SO101_OFFSETS = np.asarray([0, 90, 90, 0, 0, 0], dtype=np.float32)


def arm_to_model(joints_arm: np.ndarray) -> np.ndarray:
    """LeRobot follower-arm degrees to the G05/model coordinate frame."""
    return SO101_SIGNS * np.asarray(joints_arm, dtype=np.float32) + SO101_OFFSETS


def model_to_arm(joints_model: np.ndarray) -> np.ndarray:
    """G05/model coordinates to LeRobot follower-arm degrees."""
    return (np.asarray(joints_model, dtype=np.float32) - SO101_OFFSETS) * SO101_SIGNS


def compose_so101_cameras(fixed_rgb: np.ndarray, wrist_rgb: np.ndarray) -> np.ndarray:
    """Build the RoboCOIN-style 384x320 image used during SO101 training.

    Layout: fixed camera on top, SO101 wrist at bottom-left, and a black
    bottom-right slot for RoboCOIN's unavailable right-wrist camera.
    """
    fixed = Image.fromarray(np.asarray(fixed_rgb, dtype=np.uint8), mode="RGB")
    wrist = Image.fromarray(np.asarray(wrist_rgb, dtype=np.uint8), mode="RGB")
    fixed = fixed.resize((320, 240), resample=Image.Resampling.BILINEAR)
    wrist = wrist.resize((320, 240), resample=Image.Resampling.BILINEAR)
    fixed = fixed.resize((320, 256), resample=Image.Resampling.BILINEAR)
    wrist = wrist.resize((160, 128), resample=Image.Resampling.BILINEAR)
    wrist_array = np.asarray(wrist, dtype=np.uint8)
    wrist_row = np.concatenate([wrist_array, np.zeros_like(wrist_array)], axis=1)
    return np.concatenate([np.asarray(fixed, dtype=np.uint8), wrist_row], axis=0)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            raise ConnectionError("AHA-WAM server closed the connection")
        data.extend(packet)
    return bytes(data)


class AHAWAMTCPClient:
    """One persistent connection to one stateful AHA-WAM policy session."""

    def __init__(self, host: str, port: int, timeout_s: float = 1800.0):
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        if self.sock is not None:
            return
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout_s)
        sock.settimeout(self.timeout_s)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock = sock

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self.sock is None:
            raise RuntimeError("Call connect() before request()")
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        self.sock.sendall(struct.pack("!I", len(payload)) + payload)
        response_size = struct.unpack("!I", _recv_exact(self.sock, 4))[0]
        response = pickle.loads(_recv_exact(self.sock, response_size))
        if not isinstance(response, dict):
            raise TypeError(f"Server returned {type(response).__name__}, expected dict")
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", repr(response)))
        return response

    def health(self) -> dict[str, Any]:
        return self.request({"type": "health"})

    def reset(self, instruction: str) -> dict[str, Any]:
        return self.request({"type": "reset", "instruction": instruction})

    @staticmethod
    def _observation(
        *,
        state_model: np.ndarray,
        instruction: str,
        fixed_rgb: Optional[np.ndarray] = None,
        wrist_rgb: Optional[np.ndarray] = None,
        front_rgb: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        state = np.asarray(state_model, dtype=np.float32).reshape(-1)
        if state.shape != (6,):
            raise ValueError(f"state_model must have shape [6], got {state.shape}")
        if front_rgb is not None:
            images = {"front": np.asarray(front_rgb, dtype=np.uint8)}
        else:
            if fixed_rgb is None or wrist_rgb is None:
                raise ValueError("Provide front_rgb or both fixed_rgb and wrist_rgb")
            images = {
                "fixed": np.asarray(fixed_rgb, dtype=np.uint8),
                "wrist": np.asarray(wrist_rgb, dtype=np.uint8),
            }
        return {"instruction": instruction, "state": state, "images": images}

    def infer(self, **observation: Any) -> dict[str, Any]:
        return self.request({"type": "infer", **self._observation(**observation)})

    def video_preview(self, **observation: Any) -> dict[str, Any]:
        return self.request(
            {"type": "video_preview", **self._observation(**observation)}
        )

    def infer_with_video(self, **observation: Any) -> dict[str, Any]:
        return self.request(
            {"type": "infer_with_video", **self._observation(**observation)}
        )

    def __enter__(self) -> "AHAWAMTCPClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _load_rgb(path: str) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument(
        "--mode", choices=("health", "action", "video", "both"), default="health"
    )
    parser.add_argument("--instruction", default="Pick up the white block.")
    parser.add_argument("--fixed")
    parser.add_argument("--wrist")
    parser.add_argument(
        "--state-model",
        default="30,110,110,75,0,25",
        help="Six comma-separated joint values in the model/G05 coordinate frame.",
    )
    parser.add_argument("--save-video", default="future_preview.mp4")
    args = parser.parse_args()

    state = np.asarray([float(value) for value in args.state_model.split(",")])
    with AHAWAMTCPClient(args.host, args.port) as client:
        print(client.health())
        if args.mode == "health":
            return
        if not args.fixed or not args.wrist:
            parser.error("--fixed and --wrist are required for action/video modes")
        observation = {
            "fixed_rgb": _load_rgb(args.fixed),
            "wrist_rgb": _load_rgb(args.wrist),
            "state_model": state,
            "instruction": args.instruction,
        }
        client.reset(args.instruction)
        if args.mode == "action":
            response = client.infer(**observation)
        elif args.mode == "video":
            response = client.video_preview(**observation)
        else:
            response = client.infer_with_video(**observation)
        if "action_chunk" in response:
            print("action_chunk:")
            print(np.asarray(response["action_chunk"]))
        if "future_video" in response:
            video = response["future_video"]
            Path(args.save_video).write_bytes(video["data"])
            print(
                f"saved {args.save_video}: {video['frame_count']} frames, "
                f"{video['fps']} FPS, generation {video['generation_latency_ms']:.1f} ms"
            )


if __name__ == "__main__":
    main()
