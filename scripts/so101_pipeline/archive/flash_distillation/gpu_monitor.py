#!/usr/bin/env python3
"""Sample physical GPU utilization and memory until interrupted."""

from __future__ import annotations

import argparse
import csv
import os
import signal
import subprocess
import time
from pathlib import Path


running = True


def stop(_signum, _frame):
    global running
    running = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument(
        "--watch-pid",
        type=int,
        default=None,
        help="Exit automatically when this training launcher PID no longer exists.",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "index",
        "memory.used",
        "memory.total",
        "utilization.gpu",
        "power.draw",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", *fields])
        while running:
            if args.watch_pid is not None:
                try:
                    os.kill(args.watch_pid, 0)
                except ProcessLookupError:
                    break
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={','.join(fields)}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                now = time.time()
                for line in result.stdout.splitlines():
                    writer.writerow(
                        [f"{now:.3f}", *[part.strip() for part in line.split(",")]]
                    )
                handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
