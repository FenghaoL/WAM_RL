#!/usr/bin/env python3
"""Temporarily reserve free GPU memory while a training launch is prepared."""

from __future__ import annotations

import argparse
import os
import signal
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--free-mib", type=int, default=1536)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    args = parser.parse_args()

    physical_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>")
    torch.cuda.set_device(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    reserve_bytes = max(0, free_bytes - args.free_mib * 1024**2)
    element_count = reserve_bytes // torch.empty((), dtype=torch.uint8).element_size()
    allocation = torch.empty(element_count, dtype=torch.uint8, device="cuda:0")
    print(
        f"reserved={allocation.numel() / 1024**2:.0f} MiB "
        f"physical_gpu={physical_gpu} total={total_bytes / 1024**2:.0f} MiB",
        flush=True,
    )

    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while not stopping:
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
