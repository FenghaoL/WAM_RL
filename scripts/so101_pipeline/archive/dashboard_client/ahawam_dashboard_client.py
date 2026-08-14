#!/usr/bin/env python
"""SO-101 dashboard client for the deployed AHA-WAM TCP services.

The worker thread is the only owner of the robot, cameras, and trusted pickle
socket. The Tk main thread only submits commands and paints immutable snapshots.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import pickle
import queue
import re
import socket
import struct
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image


LOGGER = logging.getLogger("ahawam-so101-dashboard")
MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
JOINT_COUNT = len(MOTORS)
HOME_ARM = np.asarray([3.1, -34.3, 31.5, 55.9, -12.3, 13.4], dtype=np.float32)
SUPPORTED_TASKS = (
    "Pick up the white block.",
    "Put the white block on the paper roll.",
)
MODES = ("action", "video_preview", "infer_with_video")


class RequestAborted(RuntimeError):
    pass


def state_for_server(arm: np.ndarray) -> np.ndarray:
    """Return the calibrated-degree frame used by this SO101 checkpoint."""
    return np.asarray(arm, dtype=np.float32).copy()


def action_for_robot(action: np.ndarray) -> np.ndarray:
    """The server denormalizes actions directly into calibrated degrees."""
    return np.asarray(action, dtype=np.float32).copy()


def get_state(observation: dict[str, Any]) -> np.ndarray:
    return np.asarray([observation[f"{motor}.pos"] for motor in MOTORS], dtype=np.float32)


def require_rgb(image: Any, name: str, height: int, width: int) -> np.ndarray:
    array = np.asarray(image)
    if array.shape != (height, width, 3):
        raise ValueError(f"{name} produced {array.shape}; expected {(height, width, 3)}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def compose_so101_cameras(
    fixed: np.ndarray,
    wrist: np.ndarray,
    *,
    fixed_crop_right_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the exact fixed-top/wrist-left/black-right model view."""
    if fixed_crop_right_px:
        fixed = fixed[:, : fixed.shape[1] - fixed_crop_right_px, :]
    if fixed.shape[:2] != (480, 480):
        raise ValueError(f"fixed crop must be 480x480, got {fixed.shape[:2]}")
    if wrist.shape[:2] != (480, 640):
        raise ValueError(f"wrist frame must be 480x640, got {wrist.shape[:2]}")
    # Match training and deploy/server/ahawam_policy.py: both cameras first
    # pass through 320x240, then occupy their final RoboCOIN-style slots.
    fixed_240 = Image.fromarray(fixed, mode="RGB").resize(
        (320, 240), resample=Image.Resampling.BILINEAR
    )
    wrist_240 = Image.fromarray(wrist, mode="RGB").resize(
        (320, 240), resample=Image.Resampling.BILINEAR
    )
    fixed_model = np.asarray(
        fixed_240.resize((320, 256), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    wrist_left = np.asarray(
        wrist_240.resize((160, 128), resample=Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )
    wrist_row = np.concatenate([wrist_left, np.zeros_like(wrist_left)], axis=1)
    front = np.concatenate([fixed_model, wrist_row], axis=0)
    return np.ascontiguousarray(fixed), np.ascontiguousarray(front)


def clip_target(target: np.ndarray, current: np.ndarray, max_step: float) -> np.ndarray:
    delta = np.asarray(target, dtype=np.float32) - np.asarray(current, dtype=np.float32)
    largest = float(np.max(np.abs(delta)))
    if largest == 0.0 or largest <= max_step:
        return np.asarray(target, dtype=np.float32)
    return np.asarray(current, dtype=np.float32) + delta * (max_step / largest)


def summarize_packet(packet: Any) -> str:
    def sanitize(value: Any) -> Any:
        if isinstance(value, bytes):
            return f"<{len(value)} bytes>"
        if isinstance(value, np.ndarray):
            if value.size <= 64:
                return value.tolist()
            return {"shape": list(value.shape), "dtype": str(value.dtype)}
        if isinstance(value, dict):
            return {str(key): sanitize(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value

    return json.dumps(sanitize(packet), ensure_ascii=False, indent=2, default=str)


class AHAWAMTCPClient:
    """Length-prefixed pickle client for a trusted localhost SSH tunnel."""

    def __init__(self, host: str, port: int, timeout_s: float):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sock: socket.socket | None = None

    def connect(self) -> None:
        self.close()
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(min(5.0, self.timeout_s))
        sock.connect((self.host, self.port))
        sock.settimeout(0.1)
        self.sock = sock

    def close(self) -> None:
        if self.sock is not None:
            with contextlib.suppress(Exception):
                self.sock.close()
        self.sock = None

    def _recv_exact(
        self,
        size: int,
        *,
        deadline: float,
        on_wait: Callable[[], None] | None,
    ) -> bytes:
        if self.sock is None:
            raise ConnectionError("policy socket is not connected")
        data = bytearray()
        while len(data) < size:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"policy response exceeded {self.timeout_s:g} seconds")
            try:
                packet = self.sock.recv(size - len(data))
            except socket.timeout:
                if on_wait is not None:
                    on_wait()
                continue
            if not packet:
                raise ConnectionError("policy server closed the connection")
            data.extend(packet)
        return bytes(data)

    def request(
        self,
        message: dict[str, Any],
        *,
        on_wait: Callable[[], None] | None = None,
    ) -> dict[str, Any]:
        if self.sock is None:
            raise ConnectionError("policy socket is not connected")
        payload = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        deadline = time.monotonic() + self.timeout_s
        self.sock.settimeout(min(5.0, self.timeout_s))
        self.sock.sendall(struct.pack("!I", len(payload)) + payload)
        self.sock.settimeout(0.1)
        header = self._recv_exact(4, deadline=deadline, on_wait=on_wait)
        (size,) = struct.unpack("!I", header)
        if size <= 0 or size > 512 * 1024 * 1024:
            raise ValueError(f"invalid policy message size: {size}")
        raw = self._recv_exact(size, deadline=deadline, on_wait=on_wait)
        response = pickle.loads(raw)  # noqa: S301 - trusted loopback SSH tunnel only
        if not isinstance(response, dict):
            raise TypeError(f"policy reply must be a dict, got {type(response).__name__}")
        return response


def build_robot(args: argparse.Namespace):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower

    cameras = {
        "fixed": OpenCVCameraConfig(
            index_or_path=args.fixed_camera,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            auto_exposure=args.fixed_auto_exposure,
            exposure=args.fixed_exposure,
        ),
        "wrist": OpenCVCameraConfig(
            index_or_path=args.wrist_camera,
            width=args.width,
            height=args.height,
            fps=args.camera_fps,
            auto_exposure=args.wrist_auto_exposure,
            exposure=args.wrist_exposure,
        ),
    }
    config = SO101FollowerConfig(
        port=args.robot_port,
        id=args.robot_id,
        cameras=cameras,
        use_degrees=True,
        max_relative_target=None,
    )
    return SO101Follower(config)


def calibration_limits(robot: Any, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """Derive client-side limits from the exact loaded LeRobot calibration."""
    low: list[float] = []
    high: list[float] = []
    for name in MOTORS:
        motor = robot.bus.motors[name]
        calibration = robot.bus.calibration[name]
        mode = str(motor.norm_mode.value)
        if mode == "degrees":
            midpoint = (calibration.range_min + calibration.range_max) / 2.0
            resolution = robot.bus.model_resolution_table[motor.model] - 1
            lower = (calibration.range_min - midpoint) * 360.0 / resolution + margin
            upper = (calibration.range_max - midpoint) * 360.0 / resolution - margin
        elif mode == "range_0_100":
            lower, upper = 0.0, 100.0
        elif mode == "range_m100_100":
            lower, upper = -100.0, 100.0
        else:
            raise ValueError(f"unsupported motor normalization mode: {mode}")
        low.append(lower)
        high.append(upper)
    return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)


def format_vector(value: np.ndarray | None) -> str:
    if value is None:
        return "waiting"
    return "  ".join(f"{name}={number:7.1f}" for name, number in zip(MOTORS, value, strict=True))


def write_jsonl(path: str | None, record: dict[str, Any]) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value),
            )
            + "\n"
        )


def save_future_video(
    future: dict[str, Any],
    output_dir: Path,
    task: str,
    request_id: int,
) -> tuple[Path, np.ndarray | None]:
    data = future.get("data")
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("future_video.data must contain MP4 bytes")
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "_", task.lower()).strip("_")[:48] or "task"
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"{stamp}_{request_id:04d}_{slug}.mp4"
    path.write_bytes(bytes(data))

    capture = cv2.VideoCapture(str(path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames: list[np.ndarray] = []
    if frame_count > 0:
        for index in sorted({0, frame_count // 2, max(0, frame_count - 1)}):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if ok and frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (320, 144), interpolation=cv2.INTER_AREA)
                cv2.putText(
                    frame,
                    f"frame {index}",
                    (8, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                frames.append(frame)
    capture.release()
    contact_sheet = np.concatenate(frames, axis=1) if frames else None
    return path, contact_sheet


@dataclass
class RuntimeState:
    args: argparse.Namespace
    commands: queue.Queue[tuple[str, Any]] = field(default_factory=queue.Queue)
    lock: threading.Lock = field(default_factory=threading.Lock)
    stop_event: threading.Event = field(default_factory=threading.Event)
    abort_event: threading.Event = field(default_factory=threading.Event)
    status: str = "Starting local robot worker..."
    detail: str = ""
    robot_connected: bool = False
    server_connected: bool = False
    active: bool = False
    torque_enabled: bool = True
    home_ready: bool = False
    task: str = ""
    mode: str = "action"
    health: dict[str, Any] = field(default_factory=dict)
    raw_arm: np.ndarray | None = None
    model_state: np.ndarray | None = None
    action_model: np.ndarray | None = None
    target_arm: np.ndarray | None = None
    joint_low: np.ndarray | None = None
    joint_high: np.ndarray | None = None
    fixed: np.ndarray | None = None
    fixed_crop: np.ndarray | None = None
    wrist: np.ndarray | None = None
    model_view: np.ndarray | None = None
    sent_view: np.ndarray | None = None
    image_seq: int = 0
    chunk_step: int = 0
    chunk_size: int = 0
    completed_chunks: int = 0
    request_count: int = 0
    last_round_trip_ms: float | None = None
    last_model_ms: float | None = None
    last_preview_path: str = ""
    preview_contact_sheet: np.ndarray | None = None
    cards: deque[dict[str, Any]] = field(default_factory=deque)
    next_card_id: int = 1

    def update(self, **values: Any) -> None:
        with self.lock:
            for key, value in values.items():
                setattr(self, key, value)

    def append_card(self, card: dict[str, Any]) -> None:
        with self.lock:
            card["id"] = self.next_card_id
            self.next_card_id += 1
            self.cards.append(card)
            while len(self.cards) > self.args.dashboard_history:
                self.cards.popleft()

    def snapshot(self) -> dict[str, Any]:
        with self.lock:
            return {
                "status": self.status,
                "detail": self.detail,
                "robot_connected": self.robot_connected,
                "server_connected": self.server_connected,
                "active": self.active,
                "torque_enabled": self.torque_enabled,
                "home_ready": self.home_ready,
                "task": self.task,
                "mode": self.mode,
                "health": dict(self.health),
                "raw_arm": None if self.raw_arm is None else self.raw_arm.copy(),
                "model_state": None if self.model_state is None else self.model_state.copy(),
                "action_model": None if self.action_model is None else self.action_model.copy(),
                "target_arm": None if self.target_arm is None else self.target_arm.copy(),
                "joint_low": None if self.joint_low is None else self.joint_low.copy(),
                "joint_high": None if self.joint_high is None else self.joint_high.copy(),
                "fixed": self.fixed,
                "fixed_crop": self.fixed_crop,
                "wrist": self.wrist,
                "model_view": self.model_view,
                "sent_view": self.sent_view,
                "image_seq": self.image_seq,
                "chunk_step": self.chunk_step,
                "chunk_size": self.chunk_size,
                "completed_chunks": self.completed_chunks,
                "request_count": self.request_count,
                "last_round_trip_ms": self.last_round_trip_ms,
                "last_model_ms": self.last_model_ms,
                "last_preview_path": self.last_preview_path,
                "preview_contact_sheet": self.preview_contact_sheet,
                "cards": list(self.cards),
            }


class RobotPolicyWorker(threading.Thread):
    """Own all robot, camera, and socket operations."""

    def __init__(self, runtime: RuntimeState):
        super().__init__(daemon=True, name="AHAWAMRobotPolicyWorker")
        self.runtime = runtime
        self.args = runtime.args
        self.robot = None
        self.client: AHAWAMTCPClient | None = None
        self.action_chunk_model = np.empty((0, JOINT_COUNT), dtype=np.float32)
        self.chunk_index = 0
        self.last_preview_update = 0.0

    def run(self) -> None:
        try:
            self.robot = build_robot(self.args)
            self.runtime.update(status="Connecting follower and cameras...")
            self.robot.connect()
            low, high = calibration_limits(self.robot, self.args.joint_limit_margin)
            self.runtime.update(
                robot_connected=True,
                joint_low=low,
                joint_high=high,
                status="Robot and cameras connected; connecting policy server...",
                detail=f"limits low={np.round(low, 1).tolist()} high={np.round(high, 1).tolist()}",
            )
            home_ready = False
            home_failure_detail = ""
            if self.args.home_on_startup and not self.args.dry_run:
                home_ready = self._move_to_home()
                if not home_ready:
                    home_failure_detail = self.runtime.snapshot()["detail"]
            self.runtime.update(home_ready=home_ready)
            try:
                self._connect_server()
            except Exception as exc:
                self._close_server()
                self.runtime.update(
                    status="Policy server unavailable; cameras remain available.",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            if not home_ready:
                self.runtime.update(
                    status="Ready. Home is optional; Run uses the current arm pose.",
                    detail=home_failure_detail,
                )

            while not self.runtime.stop_event.is_set():
                self._drain_commands()
                if self.runtime.snapshot()["active"]:
                    self._action_tick()
                else:
                    self._idle_tick()
                    time.sleep(max(0.01, 1.0 / self.args.dashboard_camera_fps))
        except Exception as exc:
            LOGGER.exception("robot policy worker exited")
            self.runtime.update(status=f"Worker error: {type(exc).__name__}", detail=str(exc), active=False)
        finally:
            self._close_server()
            if self.robot is not None:
                with contextlib.suppress(Exception):
                    self.robot.disconnect()
            self.runtime.update(robot_connected=False, server_connected=False, active=False)

    def _close_server(self) -> None:
        if self.client is not None:
            self.client.close()
        self.client = None
        self.runtime.update(server_connected=False)

    def _connect_server(self) -> None:
        self._close_server()
        self.runtime.update(status="Connecting AHA-WAM server...", detail=f"{self.args.host}:{self.args.port}")
        client = AHAWAMTCPClient(self.args.host, self.args.port, self.args.timeout_s)
        client.connect()
        self.client = client
        try:
            health = self._request({"type": "health"}, allow_abort=False)
            self._validate_health(health)
        except Exception:
            self._close_server()
            raise
        self.runtime.update(
            server_connected=True,
            health=health,
            status="Ready. Select a cached task and mode, then press Run.",
            detail=self._health_summary(health),
        )

    def _health_summary(self, health: dict[str, Any]) -> str:
        return (
            f"chunk={health.get('action_chunk_size')} horizon={health.get('action_horizon')} "
            f"image={health.get('model_image_shape_hwc')} video={health.get('video_preview_enabled')} "
            f"text={health.get('text_mode')}"
        )

    def _validate_health(self, health: dict[str, Any]) -> None:
        if not health.get("ok") or health.get("type") != "health":
            raise RuntimeError(f"invalid health response: {summarize_packet(health)}")
        expected = {
            "action_dim": JOINT_COUNT,
            "proprio_dim": JOINT_COUNT,
            "action_chunk_size": self.args.expected_chunk_size,
            "action_horizon": self.args.expected_action_horizon,
        }
        for key, value in expected.items():
            if int(health.get(key, -1)) != value:
                raise RuntimeError(f"server {key}={health.get(key)!r}; client requires {value}")
        if list(health.get("model_image_shape_hwc") or []) != [384, 320, 3]:
            raise RuntimeError(f"server model image shape mismatch: {health.get('model_image_shape_hwc')!r}")
        accepted = health.get("accepted_image_keys") or []
        required_images = ["front"] if self.args.image_transport == "front" else ["fixed", "wrist"]
        if required_images not in accepted and tuple(required_images) not in accepted:
            raise RuntimeError(f"server does not advertise images {required_images!r} support: {accepted!r}")

    def _request(self, message: dict[str, Any], *, allow_abort: bool = True) -> dict[str, Any]:
        if self.client is None:
            raise ConnectionError("policy server is not connected")
        if self.args.print_server_responses:
            LOGGER.info("request type=%s", message.get("type"))
        response = self.client.request(
            message,
            on_wait=(lambda: self._wait_tick(allow_abort=allow_abort)),
        )
        if self.args.print_server_responses:
            print(f"\n[AHA-WAM response]\n{summarize_packet(response)}", flush=True)
        return response

    def _wait_tick(self, *, allow_abort: bool) -> None:
        if self.runtime.stop_event.is_set() or (allow_abort and self.runtime.abort_event.is_set()):
            raise RequestAborted("operator aborted the pending policy request")
        now = time.monotonic()
        if now - self.last_preview_update >= 1.0 / self.args.dashboard_camera_fps:
            self._capture_observation()

    def _ensure_server(self) -> None:
        if self.client is None or not self.runtime.snapshot()["server_connected"]:
            self._connect_server()

    def _reset_server(self, task: str) -> None:
        self._ensure_server()
        response = self._request({"type": "reset", "instruction": task}, allow_abort=False)
        if not response.get("ok") or response.get("type") != "reset_ack":
            raise RuntimeError(f"full reset failed: {summarize_packet(response)}")
        self.action_chunk_model = np.empty((0, JOINT_COUNT), dtype=np.float32)
        self.chunk_index = 0
        self.runtime.update(chunk_step=0, chunk_size=0, action_model=None, target_arm=None)

    def _best_effort_reset(self, task: str) -> None:
        with contextlib.suppress(Exception):
            self.runtime.abort_event.clear()
            self._reset_server(task)

    def _drain_commands(self) -> None:
        while True:
            try:
                command, value = self.runtime.commands.get_nowait()
            except queue.Empty:
                return
            self.runtime.abort_event.clear()
            try:
                self._handle_command(command, value)
            except Exception as exc:
                LOGGER.exception("dashboard command failed: %s", command)
                self.runtime.update(active=False, status=f"Command failed: {command}", detail=str(exc))
                if isinstance(exc, (ConnectionError, TimeoutError, OSError, RequestAborted)):
                    self._close_server()

    def _handle_command(self, command: str, value: Any) -> None:
        snap = self.runtime.snapshot()
        task = snap["task"] or self.args.task
        if command == "close":
            self.runtime.stop_event.set()
            return
        if command == "reconnect":
            self.runtime.update(active=False)
            self._connect_server()
            return
        if command == "health":
            self._ensure_server()
            health = self._request({"type": "health"}, allow_abort=False)
            self._validate_health(health)
            self.runtime.update(health=health, status="Health check passed.", detail=self._health_summary(health))
            return
        if command == "start":
            task = str(value["task"]).strip()
            mode = str(value["mode"])
            if task not in SUPPORTED_TASKS:
                raise ValueError("task is not one of the two exact cached prompts")
            if mode not in MODES:
                raise ValueError(f"unsupported mode: {mode}")
            if not snap["torque_enabled"] and mode == "action" and not self.args.dry_run:
                raise RuntimeError("torque is off; enable torque before live action mode")
            self.runtime.update(active=False, task=task, mode=mode, completed_chunks=0)
            self._reset_server(task)
            if mode == "action":
                self.runtime.update(active=True, status=f"Action control running: {task}", detail="requesting chunk 1")
            else:
                self._diagnostic_once(mode, task)
            return
        if command == "stop":
            self.runtime.update(active=False, status="Stopped by operator; resetting complete model session...")
            self._reset_server(task)
            self.runtime.update(status="Stopped; full model reset acknowledged.")
            return
        if command == "reset":
            self.runtime.update(active=False, status="Performing full AHA-WAM reset...")
            self._reset_server(task)
            self.runtime.update(status="Full reset complete.", detail="rolling K/V, inference state, observation, and chunk index cleared")
            return
        if command == "home":
            if self.args.dry_run:
                self.runtime.update(status="Home blocked in dry-run. Enable live motion to move the arm.")
                return
            if not snap["torque_enabled"]:
                self.runtime.update(status="Home blocked: torque is off.")
                return
            self.runtime.update(active=False, home_ready=False)
            self._best_effort_reset(task)
            if self._move_to_home():
                self._best_effort_reset(task)
                self.runtime.update(home_ready=True, status="Home complete; policy session reset.", detail="Press Run when ready.")
            return
        if command == "torque_off":
            self.runtime.update(active=False, status="Disabling torque...")
            self.robot.bus.disable_torque()
            self.runtime.update(torque_enabled=False, home_ready=False, status="Torque OFF; resetting policy session...")
            self._best_effort_reset(task)
            self.runtime.update(status="Torque OFF. The arm may move freely.")
            return
        if command == "torque_on":
            self._best_effort_reset(task)
            self.robot.bus.enable_torque()
            self.runtime.update(torque_enabled=True, status="Torque ON. Press Run when ready.")
            return
        raise ValueError(f"unknown command: {command}")

    def _capture_observation(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        observation = self.robot.get_observation()
        raw_arm = get_state(observation)
        if not np.all(np.isfinite(raw_arm)):
            raise ValueError("robot state contains NaN or Inf")
        fixed = require_rgb(observation["fixed"], "fixed", self.args.height, self.args.width)
        wrist = require_rgb(observation["wrist"], "wrist", self.args.height, self.args.width)
        fixed_crop, model_view = compose_so101_cameras(
            fixed,
            wrist,
            fixed_crop_right_px=self.args.fixed_crop_right_px,
        )
        model_state = state_for_server(raw_arm)
        self.last_preview_update = time.monotonic()
        self.runtime.update(
            raw_arm=raw_arm,
            model_state=model_state,
            fixed=fixed,
            fixed_crop=fixed_crop,
            wrist=wrist,
            model_view=model_view,
            image_seq=self.runtime.snapshot()["image_seq"] + 1,
        )
        return raw_arm, model_state, fixed_crop, wrist, model_view

    def _validate_action_chunk(self, chunk: Any) -> tuple[np.ndarray, np.ndarray]:
        model = np.asarray(chunk, dtype=np.float32)
        expected = (self.args.expected_chunk_size, JOINT_COUNT)
        if model.shape != expected:
            raise ValueError(f"action chunk shape must be {expected}, got {model.shape}")
        if not np.all(np.isfinite(model)):
            raise ValueError("action chunk contains NaN or Inf")
        arm = action_for_robot(model)
        return model, arm

    def _build_request(self, kind: str, task: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
        raw_arm, model_state, fixed_crop, wrist, model_view = self._capture_observation()
        if self.args.image_transport == "front":
            images = {"front": np.ascontiguousarray(model_view, dtype=np.uint8)}
        else:
            images = {
                "fixed": np.ascontiguousarray(fixed_crop, dtype=np.uint8),
                "wrist": np.ascontiguousarray(wrist, dtype=np.uint8),
            }
        request = {
            "type": kind,
            "instruction": task,
            "state": np.ascontiguousarray(model_state, dtype=np.float32),
            "images": images,
        }
        self.runtime.update(sent_view=model_view.copy())
        return request, raw_arm, model_state, model_view

    def _infer(self, kind: str, task: str) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, float]:
        request, raw_arm, model_state, model_view = self._build_request(kind, task)
        image_summary = {
            key: {"shape": list(value.shape), "dtype": str(value.dtype), "min": int(value.min()), "max": int(value.max())}
            for key, value in request["images"].items()
        }
        LOGGER.info(
            "outbound %s instruction=%r state=%s images=%s",
            kind,
            task,
            np.round(model_state, 2).tolist(),
            image_summary,
        )
        started = time.monotonic()
        try:
            response = self._request(request)
        except ConnectionError as exc:
            raise ConnectionError(
                f"server disconnected while processing {kind}; tunnel was already proven by health/reset. "
                f"Inspect the GPU-server traceback. state={np.round(model_state, 2).tolist()} "
                f"images={image_summary}"
            ) from exc
        round_trip_ms = (time.monotonic() - started) * 1000.0
        if not response.get("ok"):
            raise RuntimeError(f"server rejected {kind}: {summarize_packet(response)}")
        request_count = self.runtime.snapshot()["request_count"] + 1
        self.runtime.update(
            request_count=request_count,
            last_round_trip_ms=round_trip_ms,
            last_model_ms=float(response.get("model_latency_ms", float("nan"))),
        )
        return response, raw_arm, model_state, model_view, round_trip_ms

    def _move_to_home(self) -> bool:
        home = np.asarray(self.args.home_arm, dtype=np.float32)
        checked_indices = np.arange(JOINT_COUNT - 1 if self.args.home_ignore_gripper else JOINT_COUNT)
        self.runtime.abort_event.clear()
        self.runtime.update(status="Moving to the optional saved Home pose...", detail=np.round(home, 1).tolist())
        deadline = time.monotonic() + self.args.home_timeout_s
        next_log = 0.0
        while time.monotonic() < deadline and not self.runtime.stop_event.is_set():
            if self.runtime.abort_event.is_set():
                self.runtime.update(status="Home stopped by operator.")
                return False
            raw_arm, _model_state, _fixed_crop, _wrist, _model_view = self._capture_observation()
            errors = np.abs(home - raw_arm)
            error = float(np.max(errors[checked_indices]))
            worst_index = int(checked_indices[np.argmax(errors[checked_indices])])
            error_summary = "  ".join(f"{name}={value:.1f}" for name, value in zip(MOTORS, errors, strict=True))
            detail = f"home max={error:.1f} ({MOTORS[worst_index]}); errors: {error_summary}"
            self.runtime.update(target_arm=home.copy(), detail=detail)
            if time.monotonic() >= next_log:
                LOGGER.info("%s", detail)
                next_log = time.monotonic() + 1.0
            if error <= self.args.home_tolerance:
                suffix = "; gripper excluded from completion check" if self.args.home_ignore_gripper else ""
                self.runtime.update(status="Optional Home pose reached.", detail=f"{detail}{suffix}")
                return True
            target = raw_arm + np.clip(home - raw_arm, -self.args.home_step, self.args.home_step)
            self.robot.send_action({f"{motor}.pos": float(target[index]) for index, motor in enumerate(MOTORS)})
            time.sleep(self.args.home_step_interval_s)
        raw_arm, _model_state, _fixed_crop, _wrist, _model_view = self._capture_observation()
        errors = np.abs(home - raw_arm)
        error = float(np.max(errors[checked_indices]))
        worst_index = int(checked_indices[np.argmax(errors[checked_indices])])
        error_summary = "  ".join(f"{name}={value:.1f}" for name, value in zip(MOTORS, errors, strict=True))
        self.runtime.update(
            status="Home timed out; arm motion stopped.",
            detail=(
                f"max={error:.1f} at {MOTORS[worst_index]} after {self.args.home_timeout_s:.0f}s; "
                f"errors: {error_summary}"
            ),
        )
        return False

    def _request_action_chunk(self, task: str) -> None:
        response, raw_arm, model_state, model_view, round_trip_ms = self._infer("infer", task)
        if response.get("type") != "action_chunk":
            raise RuntimeError(f"unexpected infer response type: {response.get('type')!r}")
        model, _arm = self._validate_action_chunk(response.get("action_chunk"))
        self.action_chunk_model = model
        self.chunk_index = 0
        model_ms = float(response.get("model_latency_ms", float("nan")))
        cuda = dict(response.get("cuda_memory") or {})
        self.runtime.update(chunk_step=0, chunk_size=len(model), last_model_ms=model_ms)
        self.runtime.append_card(
            {
                "kind": "action",
                "task": task,
                "image": model_view.copy(),
                "raw_arm": raw_arm.copy(),
                "model_state": model_state.copy(),
                "action_first": model[0].copy(),
                "action_last": model[-1].copy(),
                "round_trip_ms": round_trip_ms,
                "model_ms": model_ms,
                "server_step": response.get("server_inference_step"),
                "cuda": cuda,
            }
        )
        write_jsonl(
            self.args.timing_log,
            {
                "timestamp": time.time(),
                "type": "infer",
                "task": task,
                "round_trip_ms": round_trip_ms,
                "model_latency_ms": model_ms,
                "state_arm": raw_arm,
                "state_model": model_state,
                "action_chunk_model": model,
                "cuda_memory": cuda,
            },
        )

    def _action_tick(self) -> None:
        started = time.monotonic()
        try:
            task = self.runtime.snapshot()["task"]
            if self.chunk_index >= len(self.action_chunk_model):
                self._request_action_chunk(task)
                # Chunk-boundary inference time is a hold period, not part of
                # the following 15 Hz action execution schedule.
                started = time.monotonic()
            raw_arm, _model_state, _fixed_crop, _wrist, _model_view = self._capture_observation()
            action_model = self.action_chunk_model[self.chunk_index]
            target_arm_unclipped = action_for_robot(action_model)
            # Calibration bounds are displayed for operator context only. The
            # user intentionally disabled range-based rejection and clamping;
            # retain only the per-tick delta limit and finite/shape checks.
            target_arm = clip_target(target_arm_unclipped, raw_arm, self.args.max_step)
            if self.runtime.abort_event.is_set():
                raise RequestAborted("operator aborted before the next robot action")
            if not self.args.dry_run:
                self.robot.send_action(
                    {f"{motor}.pos": float(target_arm[index]) for index, motor in enumerate(MOTORS)}
                )
            self.chunk_index += 1
            completed = self.runtime.snapshot()["completed_chunks"]
            if self.chunk_index == len(self.action_chunk_model):
                completed += 1
            self.runtime.update(
                action_model=action_model.copy(),
                target_arm=target_arm.copy(),
                chunk_step=self.chunk_index,
                completed_chunks=completed,
                status="DRY RUN action stream" if self.args.dry_run else "LIVE action stream",
                detail=f"chunk {self.chunk_index}/{len(self.action_chunk_model)}; completed={completed}",
            )
            if self.args.max_chunks > 0 and completed >= self.args.max_chunks:
                self.runtime.commands.put(("stop", None))
        except RequestAborted as exc:
            self._close_server()
            self.runtime.update(active=False, status="Pending inference aborted.", detail=str(exc))
        except Exception as exc:
            LOGGER.exception("action stream stopped")
            self.runtime.update(active=False, status=f"ACTION STOPPED: {type(exc).__name__}", detail=str(exc))
            if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
                self._close_server()
            else:
                self._best_effort_reset(self.runtime.snapshot()["task"])
        elapsed = time.monotonic() - started
        time.sleep(max(0.0, 1.0 / self.args.action_fps - elapsed))

    def _diagnostic_once(self, mode: str, task: str) -> None:
        health = self.runtime.snapshot()["health"]
        if not health.get("video_preview_enabled", False):
            raise RuntimeError("connected server does not enable video preview; use the 12001 diagnostic tunnel")
        self.runtime.update(status=f"Running one {mode} request...", detail="no robot motion will be executed")
        response, raw_arm, model_state, model_view, round_trip_ms = self._infer(mode, task)
        if mode == "video_preview" and response.get("type") != "future_video":
            raise RuntimeError(f"unexpected video response type: {response.get('type')!r}")
        if mode == "infer_with_video" and response.get("type") != "action_chunk_with_video":
            raise RuntimeError(f"unexpected joint response type: {response.get('type')!r}")
        action_model = None
        if mode == "infer_with_video":
            action_model, _action_arm = self._validate_action_chunk(response.get("action_chunk"))
            self.runtime.update(action_model=action_model[0].copy())
        future = dict(response.get("future_video") or {})
        request_id = self.runtime.snapshot()["request_count"]
        video_path, contact_sheet = save_future_video(
            future,
            Path(self.args.preview_output_dir),
            task,
            request_id,
        )
        generation_ms = float(future.get("generation_latency_ms", float("nan")))
        self.runtime.update(
            active=False,
            last_preview_path=str(video_path),
            preview_contact_sheet=contact_sheet,
            status=f"{mode} complete; robot was not moved.",
            detail=f"saved {video_path.name}; RTT={round_trip_ms:.0f} ms generation={generation_ms:.0f} ms",
        )
        self.runtime.append_card(
            {
                "kind": mode,
                "task": task,
                "image": model_view.copy(),
                "raw_arm": raw_arm.copy(),
                "model_state": model_state.copy(),
                "action_first": None if action_model is None else action_model[0].copy(),
                "action_last": None if action_model is None else action_model[-1].copy(),
                "round_trip_ms": round_trip_ms,
                "model_ms": generation_ms,
                "server_step": response.get("server_inference_step"),
                "cuda": dict(response.get("cuda_memory") or {}),
                "video_path": str(video_path),
            }
        )
        write_jsonl(
            self.args.timing_log,
            {
                "timestamp": time.time(),
                "type": mode,
                "task": task,
                "round_trip_ms": round_trip_ms,
                "generation_latency_ms": generation_ms,
                "video_path": str(video_path),
                "action_chunk_model": action_model,
            },
        )

    def _idle_tick(self) -> None:
        try:
            self._capture_observation()
        except Exception as exc:
            self.runtime.update(status=f"Camera/state error: {type(exc).__name__}", detail=str(exc), active=False)


class Dashboard:
    def __init__(self, runtime: RuntimeState, worker: RobotPolicyWorker):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.runtime = runtime
        self.worker = worker
        self.root = tk.Tk()
        self.root.title("AHA-WAM | SO-101 deployment dashboard")
        self.root.geometry("1460x980")
        self.root.minsize(1120, 760)
        self.photos: dict[str, Any] = {}
        self.image_labels: dict[str, Any] = {}
        self.card_widgets: dict[int, Any] = {}
        self.last_image_seq = -1
        self.task_var = tk.StringVar(value=runtime.args.task)
        self.mode_var = tk.StringVar(value=runtime.args.mode)
        self.status_var = tk.StringVar(value="Starting...")
        self.detail_var = tk.StringVar(value="")
        self.health_var = tk.StringVar(value="server health: waiting")
        self.preview_path_var = tk.StringVar(value="future video: none")
        self.vector_vars = {name: tk.StringVar(value="waiting") for name in ("arm", "model", "action", "target", "limits")}
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build(self) -> None:
        tk, ttk = self.tk, self.ttk
        style = ttk.Style(self.root)
        with contextlib.suppress(Exception):
            style.theme_use("clam")
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")
        motion = "DRY RUN" if self.runtime.args.dry_run else "LIVE MOTION"
        ttk.Label(top, text=f"AHA-WAM | SO-101 | {motion}", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(
            top,
            text="15 Hz control | calibrated degrees | 8x6 chunks | full reset | trusted SSH pickle",
            font=("Consolas", 10),
        ).pack(anchor="w", pady=(2, 8))

        controls = ttk.Frame(top)
        controls.pack(fill="x")
        ttk.Label(controls, text="Task:").pack(side="left")
        task_box = ttk.Combobox(
            controls,
            textvariable=self.task_var,
            values=SUPPORTED_TASKS,
            state="readonly",
            width=43,
            font=("Segoe UI", 11),
        )
        task_box.pack(side="left", padx=(5, 10))
        ttk.Label(controls, text="Mode:").pack(side="left")
        ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=MODES,
            state="readonly",
            width=18,
        ).pack(side="left", padx=(5, 10))
        buttons = ttk.Frame(top)
        buttons.pack(fill="x", pady=(6, 0))
        for label, command in (
            ("Run", self._start),
            ("Stop + reset", lambda: self._command("stop", urgent=True)),
            ("Full reset", lambda: self._command("reset", urgent=True)),
            ("Home", lambda: self._command("home", urgent=True)),
            ("Health", lambda: self._command("health")),
            ("Reconnect", lambda: self._command("reconnect", urgent=True)),
            ("Torque off", lambda: self._command("torque_off", urgent=True)),
            ("Torque on", lambda: self._command("torque_on")),
            ("Open previews", self._open_previews),
            ("Close", self._close),
        ):
            ttk.Button(buttons, text=label, command=command).pack(side="left", padx=(0, 5))

        ttk.Label(top, textvariable=self.status_var, foreground="#a14f00", font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(7, 0))
        ttk.Label(top, textvariable=self.detail_var, foreground="#59636e", font=("Consolas", 9)).pack(anchor="w")
        ttk.Label(top, textvariable=self.health_var, foreground="#304b67", font=("Consolas", 9)).pack(anchor="w")

        joints = ttk.LabelFrame(top, text="STATE / ACTION SAFETY", padding=6)
        joints.pack(fill="x", pady=(7, 0))
        for label, key in (
            ("Robot state (calibrated deg)", "arm"),
            ("State sent to server", "model"),
            ("Current server action", "action"),
            ("Target after per-tick delta limit", "target"),
            ("Calibration reference (display only)", "limits"),
        ):
            line = ttk.Frame(joints)
            line.pack(fill="x")
            ttk.Label(line, text=f"{label:<31}", font=("Consolas", 9, "bold")).pack(side="left")
            ttk.Label(line, textvariable=self.vector_vars[key], font=("Consolas", 9)).pack(side="left")

        camera_board = ttk.LabelFrame(top, text="LIVE CAMERA / EXACT MODEL INPUT", padding=6)
        camera_board.pack(fill="x", pady=(7, 0))
        image_row = ttk.Frame(camera_board)
        image_row.pack(fill="x")
        for key, title in (
            ("fixed_crop", "fixed crop | 480x480"),
            ("wrist", "wrist | 640x480"),
            ("model_view", "model input | fixed top + wrist left + black right"),
        ):
            box = ttk.LabelFrame(image_row, text=title, padding=4)
            box.pack(side="left", fill="both", expand=True, padx=(0, 7))
            label = ttk.Label(box, text="Waiting for camera...", anchor="center")
            label.pack(fill="both", expand=True)
            self.image_labels[key] = label

        future = ttk.LabelFrame(top, text="LATEST FUTURE VIDEO CONTACT SHEET", padding=5)
        future.pack(fill="x", pady=(7, 0))
        self.future_label = ttk.Label(future, text="Use video_preview or infer_with_video on port 12001.", anchor="center")
        self.future_label.pack(fill="x")
        ttk.Label(future, textvariable=self.preview_path_var, font=("Consolas", 8)).pack(anchor="w")

        outer = ttk.Frame(self.root)
        outer.pack(fill="both", expand=True, pady=(4, 0))
        self.canvas = tk.Canvas(outer, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.cards_frame = ttk.Frame(self.canvas, padding=(10, 4, 10, 12))
        self.canvas_window = self.canvas.create_window((0, 0), window=self.cards_frame, anchor="nw")
        self.cards_frame.bind("<Configure>", lambda _event: self.canvas.configure(scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda event: self.canvas.itemconfigure(self.canvas_window, width=event.width))
        self.root.bind_all("<MouseWheel>", lambda event: self.canvas.yview_scroll(int(-event.delta / 120), "units"), add="+")

    def _command(self, name: str, *, urgent: bool = False) -> None:
        if urgent:
            self.runtime.abort_event.set()
        self.runtime.commands.put((name, None))

    def _start(self) -> None:
        self.runtime.abort_event.set()
        self.runtime.commands.put(("start", {"task": self.task_var.get().strip(), "mode": self.mode_var.get()}))

    def _close(self) -> None:
        self.runtime.abort_event.set()
        self.runtime.commands.put(("close", None))
        self.runtime.stop_event.set()
        self.root.after(50, self.root.destroy)

    def _open_previews(self) -> None:
        output = Path(self.runtime.args.preview_output_dir)
        output.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(output)  # type: ignore[attr-defined]

    def _to_photo(self, rgb: np.ndarray, width: int, height: int):
        from PIL import Image, ImageTk

        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        return ImageTk.PhotoImage(Image.fromarray(resized, mode="RGB"))

    def _set_image(self, key: str, rgb: np.ndarray | None, width: int, height: int) -> None:
        if rgb is None:
            return
        photo = self._to_photo(rgb, width, height)
        self.image_labels[key].configure(image=photo, text="")
        self.photos[key] = photo

    def _add_card(self, card: dict[str, Any]) -> None:
        card_id = int(card["id"])
        frame = self.ttk.LabelFrame(
            self.cards_frame,
            text=f"REQUEST #{card_id} | {card['kind']} | {card['task']}",
            padding=7,
        )
        frame.pack(fill="x", pady=(0, 7))
        left = self.ttk.Frame(frame)
        left.pack(side="left", padx=(0, 10))
        label = self.ttk.Label(left, text="No sent image")
        label.pack()
        if card.get("image") is not None:
            photo = self._to_photo(card["image"], 240, 288)
            label.configure(image=photo, text="")
            self.photos[f"card-{card_id}"] = photo
        right = self.ttk.Frame(frame)
        right.pack(side="left", fill="both", expand=True)
        cuda = card.get("cuda") or {}
        lines = [
            f"RTT={card.get('round_trip_ms', float('nan')):.0f} ms | model={card.get('model_ms', float('nan')):.0f} ms | server_step={card.get('server_step')}",
            f"arm state:   {np.round(card['raw_arm'], 1).tolist()}",
            f"server state: {np.round(card['model_state'], 1).tolist()}",
            f"action first: {None if card.get('action_first') is None else np.round(card['action_first'], 1).tolist()}",
            f"action last:  {None if card.get('action_last') is None else np.round(card['action_last'], 1).tolist()}",
            f"CUDA peak allocated/reserved MiB: {cuda.get('peak_allocated_mib')} / {cuda.get('peak_reserved_mib')}",
        ]
        if card.get("video_path"):
            lines.append(f"video: {card['video_path']}")
        self.ttk.Label(right, text="\n".join(lines), justify="left", font=("Consolas", 9)).pack(anchor="w")
        self.card_widgets[card_id] = frame

    def _refresh(self) -> None:
        snap = self.runtime.snapshot()
        active = "ACTIVE" if snap["active"] else "IDLE"
        torque = "TORQUE ON" if snap["torque_enabled"] else "TORQUE OFF"
        home = "HOME DONE" if snap["home_ready"] else "HOME OPTIONAL"
        links = f"robot={'OK' if snap['robot_connected'] else 'DOWN'} server={'OK' if snap['server_connected'] else 'DOWN'}"
        self.status_var.set(f"{active} | {torque} | {home} | {links} | {snap['status']}")
        self.detail_var.set(
            f"mode={snap['mode']} chunk={snap['chunk_step']}/{snap['chunk_size']} completed={snap['completed_chunks']} "
            f"requests={snap['request_count']} RTT={snap['last_round_trip_ms']} ms | {snap['detail']}"
        )
        self.health_var.set(f"server health: {summarize_packet(snap['health']).replace(chr(10), ' ')[:360]}")
        self.vector_vars["arm"].set(format_vector(snap["raw_arm"]))
        self.vector_vars["model"].set(format_vector(snap["model_state"]))
        self.vector_vars["action"].set(format_vector(snap["action_model"]))
        self.vector_vars["target"].set(format_vector(snap["target_arm"]))
        if snap["joint_low"] is not None and snap["joint_high"] is not None:
            self.vector_vars["limits"].set(
                "  ".join(
                    f"{name}=[{low:.0f},{high:.0f}]"
                    for name, low, high in zip(MOTORS, snap["joint_low"], snap["joint_high"], strict=True)
                )
            )

        if snap["image_seq"] != self.last_image_seq:
            self._set_image("fixed_crop", snap["fixed_crop"], 280, 280)
            self._set_image("wrist", snap["wrist"], 320, 240)
            self._set_image("model_view", snap["model_view"], 250, 300)
            self.last_image_seq = snap["image_seq"]
        if snap["preview_contact_sheet"] is not None:
            photo = self._to_photo(snap["preview_contact_sheet"], 960, 144)
            self.future_label.configure(image=photo, text="")
            self.photos["future"] = photo
        self.preview_path_var.set(f"future video: {snap['last_preview_path'] or 'none'}")

        card_ids = {int(card["id"]) for card in snap["cards"]}
        for card_id in list(self.card_widgets):
            if card_id not in card_ids:
                self.card_widgets.pop(card_id).destroy()
        for card in snap["cards"]:
            if int(card["id"]) not in self.card_widgets:
                self._add_card(card)

        if self.runtime.stop_event.is_set() and not self.worker.is_alive():
            self.root.after(100, self.root.destroy)
            return
        self.root.after(max(30, int(1000 / self.runtime.args.dashboard_fps)), self._refresh)

    def run(self) -> None:
        self.worker.start()
        self.root.after(50, self._refresh)
        self.root.mainloop()
        self.runtime.stop_event.set()
        self.worker.join(timeout=3.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12000)
    parser.add_argument("--task", default=SUPPORTED_TASKS[0])
    parser.add_argument("--mode", choices=MODES, default="action")
    parser.add_argument("--robot-port", default="COM4")
    parser.add_argument("--robot-id", default="fenghao_so101_follower")
    parser.add_argument("--fixed-camera", type=int, default=2)
    parser.add_argument("--wrist-camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fixed-crop-right-px", type=int, default=160)
    parser.add_argument("--image-transport", choices=("front", "raw"), default="front")
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--fixed-auto-exposure", type=float, default=0.25)
    parser.add_argument("--wrist-auto-exposure", type=float, default=0.25)
    parser.add_argument("--fixed-exposure", type=float, default=-6.0)
    parser.add_argument("--wrist-exposure", type=float, default=-6.0)
    parser.add_argument("--action-fps", type=float, default=15.0)
    parser.add_argument("--expected-chunk-size", type=int, default=8)
    parser.add_argument("--expected-action-horizon", type=int, default=40)
    parser.add_argument("--max-step", type=float, default=3.0)
    parser.add_argument("--joint-limit-margin", type=float, default=0.0)
    parser.add_argument("--home-arm", type=float, nargs=JOINT_COUNT, default=HOME_ARM.tolist())
    parser.add_argument("--home-on-startup", action="store_true")
    parser.add_argument("--home-ignore-gripper", action="store_true")
    parser.add_argument("--home-timeout-s", type=float, default=15.0)
    parser.add_argument("--home-tolerance", type=float, default=2.0)
    parser.add_argument("--home-step", type=float, default=5.0)
    parser.add_argument("--home-step-interval-s", type=float, default=0.02)
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--dashboard-fps", type=float, default=10.0)
    parser.add_argument("--dashboard-camera-fps", type=float, default=8.0)
    parser.add_argument("--dashboard-history", type=int, default=50)
    parser.add_argument("--preview-output-dir", default="outputs/ahawam_previews")
    parser.add_argument("--timing-log", default="outputs/ahawam_client_timing.jsonl")
    parser.add_argument("--print-server-responses", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("pickle transport is restricted to a trusted local SSH tunnel")
    if args.width - args.fixed_crop_right_px != 480 or args.height != 480:
        parser.error("deployed model requires fixed crop 480x480 from native 640x480")
    if args.action_fps != 15.0:
        parser.error("deployed checkpoint requires action-fps=15")
    if args.expected_chunk_size != 8 or args.expected_action_horizon != 40:
        parser.error("deployed checkpoint requires chunk-size=8 and action-horizon=40")
    if args.camera_fps <= 0 or args.max_step <= 0 or args.timeout_s <= 0:
        parser.error("camera-fps, max-step, and timeout-s must be positive")
    if args.home_timeout_s <= 0 or args.home_tolerance <= 0 or args.home_step <= 0 or args.home_step_interval_s <= 0:
        parser.error("home timing, tolerance, and step values must be positive")
    if args.dashboard_fps <= 0 or args.dashboard_camera_fps <= 0 or args.dashboard_history <= 0:
        parser.error("dashboard rates and history must be positive")
    if args.joint_limit_margin < 0 or args.max_chunks < 0:
        parser.error("joint-limit-margin and max-chunks cannot be negative")
    return args


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    runtime = RuntimeState(args=args, task=args.task, mode=args.mode)
    worker = RobotPolicyWorker(runtime)
    Dashboard(runtime, worker).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
