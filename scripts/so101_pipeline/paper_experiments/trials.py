"""Create a randomized manual robot test sheet and summarize audited outcomes.

This tool never controls the robot. Pending rows never count as failures/successes.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
from collections import defaultdict

FIELDS = ["task", "protocol", "condition_id", "order", "model", "status",
          "physical_success", "interventions", "duration_s", "checkpoint", "camera_id", "notes"]


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denom = 1 + z*z/total
    mid = (p + z*z/(2*total))/denom
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total))/denom
    return [max(0., mid-half), min(1., mid+half)]


def summarize(rows):
    groups, pending, seen = defaultdict(list), defaultdict(int), set()
    for row in rows:
        key = tuple(row[k] for k in ("task", "protocol", "model"))
        if not all(key) or not row["condition_id"]:
            raise ValueError("Task, protocol, model and condition ID are required")
        unique = (*key, row["condition_id"])
        if unique in seen:
            raise ValueError(f"Duplicate trial condition: {unique}")
        seen.add(unique)
        if row["status"] == "pending":
            pending[key] += 1
            continue
        if row["status"] != "completed" or row["physical_success"] not in ("0", "1"):
            raise ValueError("Completed trials require a binary physical_success; otherwise use pending")
        interventions = int(row["interventions"])
        duration = float(row["duration_s"])
        if interventions < 0 or duration < 0 or not math.isfinite(duration):
            raise ValueError("Invalid intervention count/duration")
        if not row["checkpoint"].strip() or not row["camera_id"].strip():
            raise ValueError("Completed trials must identify checkpoint and camera setup")
        groups[key].append(dict(condition=row["condition_id"], autonomous=int(row["physical_success"] == "1" and interventions == 0),
                                physical=int(row["physical_success"]), interventions=interventions,
                                duration=duration, checkpoint=row["checkpoint"], camera=row["camera_id"]))
    output = {"groups": [], "paired": [], "note": "Manual trial outcomes, not training loss. CI assumes independent reset conditions; repeated sessions may require cluster analysis."}
    for key in sorted(groups.keys() | pending.keys()):
        values = groups[key]
        checkpoints = {v["checkpoint"] for v in values}
        cameras = {v["camera"] for v in values}
        if len(checkpoints) > 1 or len(cameras) > 1:
            raise ValueError(f"Mixed checkpoint/camera within one model group: {key}; use distinct labels")
        n = len(values)
        successes = sum(v["autonomous"] for v in values)
        output["groups"].append(dict(task=key[0], protocol=key[1], model=key[2], completed=n, pending=pending[key],
            autonomous_successes=successes, autonomous_rate=successes/n if n else None, ci95=wilson(successes,n),
            physical_successes=sum(v["physical"] for v in values),
            trials_with_intervention=sum(v["interventions"] > 0 for v in values),
            mean_duration_all_completed=sum(v["duration"] for v in values)/n if n else None,
            checkpoint=next(iter(checkpoints), None), camera_id=next(iter(cameras), None)))
    for a, b in itertools.combinations(sorted(groups), 2):
        if a[:2] != b[:2]:
            continue
        av = {v["condition"]: v for v in groups[a]}
        bv = {v["condition"]: v for v in groups[b]}
        shared = av.keys() & bv.keys()
        if any(av[c]["camera"] != bv[c]["camera"] for c in shared):
            raise ValueError("Paired models used different camera setups")
        wins = sum(av[c]["autonomous"] > bv[c]["autonomous"] for c in shared)
        losses = sum(av[c]["autonomous"] < bv[c]["autonomous"] for c in shared)
        output["paired"].append(dict(task=a[0], protocol=a[1], model_a=a[2], model_b=b[2],
                                     matched_conditions=len(shared), a_only_success=wins, b_only_success=losses,
                                     unpaired_a=len(av)-len(shared), unpaired_b=len(bv)-len(shared)))
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("--task", required=True)
    plan.add_argument("--protocol", choices=["full_task", "local_recovery", "development"], required=True)
    plan.add_argument("--models", nargs="+", required=True)
    plan.add_argument("--conditions", type=int, default=20)
    plan.add_argument("--seed", type=int, default=20260913)
    plan.add_argument("--output", required=True)
    report = sub.add_parser("summarize")
    report.add_argument("input")
    args = p.parse_args()
    if args.command == "summarize":
        with open(args.input, newline="") as f:
            print(json.dumps(summarize(list(csv.DictReader(f))), ensure_ascii=False, indent=2))
        return
    if args.conditions <= 0 or len(set(args.models)) != len(args.models):
        raise ValueError("Positive number of conditions and unique model labels required")
    rng = random.Random(args.seed)
    with Path(args.output).open("x", newline="") as f:
        writer = csv.DictWriter(f, FIELDS)
        writer.writeheader()
        for i in range(args.conditions):
            models = list(args.models)
            rng.shuffle(models)
            for j, model in enumerate(models):
                writer.writerow(dict(task=args.task, protocol=args.protocol, condition_id=f"{i+1:03d}",
                                     order=j+1, model=model, status="pending"))


if __name__ == "__main__":
    main()
