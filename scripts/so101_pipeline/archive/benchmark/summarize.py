#!/usr/bin/env python3
"""Create a compact Markdown summary from one benchmark log and GPU CSV."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


SPEED_RE = re.compile(r"speed=([0-9.]+) step/s, ([0-9.]+) samples/s")
TIMING_RE = re.compile(
    r"timing\(data=([0-9.]+)s fwd=([0-9.]+)s bwd=([0-9.]+)s opt=([0-9.]+)s\)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--gpu-csv", required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    log_text = Path(args.log).read_text(encoding="utf-8", errors="replace")
    # Rich logging wraps long progress lines to the terminal width. Collapse
    # whitespace so a metric remains parseable even when it spans many lines.
    flat_log = re.sub(r"\s+", " ", log_text)
    speeds = [(float(a), float(b)) for a, b in SPEED_RE.findall(flat_log)]
    timings = [tuple(float(value) for value in row) for row in TIMING_RE.findall(flat_log)]
    peak_memory = {}
    peak_util = {}
    with Path(args.gpu_csv).open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            gpu = int(row["index"])
            peak_memory[gpu] = max(peak_memory.get(gpu, 0.0), float(row["memory.used"]))
            peak_util[gpu] = max(peak_util.get(gpu, 0.0), float(row["utilization.gpu"]))

    result = {
        "name": args.name,
        "exit_code": args.exit_code,
        "oom": "out of memory" in log_text.lower(),
        "last_steps_per_sec": speeds[-1][0] if speeds else None,
        "last_samples_per_sec": speeds[-1][1] if speeds else None,
        "last_timing_seconds": timings[-1] if timings else None,
        "peak_memory_mib": peak_memory,
        "peak_gpu_util_percent": peak_util,
    }
    output = Path(args.output)
    output.write_text(
        "# " + args.name + "\n\n```json\n" + json.dumps(result, indent=2) + "\n```\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
