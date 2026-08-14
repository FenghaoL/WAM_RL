#!/usr/bin/env python3
"""Sample GPU utilization and memory until interrupted."""

from __future__ import annotations

import argparse
import csv
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
    parser.add_argument("--interval", type=float, default=0.5)
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
    query = ",".join(fields)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp", *fields])
        while running:
            now = time.time()
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--query-gpu={query}",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    writer.writerow([f"{now:.3f}", *[value.strip() for value in line.split(",")]])
                handle.flush()
            time.sleep(args.interval)


if __name__ == "__main__":
    main()
