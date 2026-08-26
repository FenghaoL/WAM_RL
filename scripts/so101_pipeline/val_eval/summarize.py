#!/usr/bin/env python3
"""Merge per-checkpoint val_eval JSONs into one readable markdown table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HEADLINE = ["val_loss", "action_l2", "action_l1", "history_len"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)

    rows = []
    for f in sorted(d.glob("*.json")):
        # The dataset drops a dataset_stats.json into the work dir, and we write
        # SUMMARY.json here too; accept only real per-checkpoint results.
        if f.name in {"SUMMARY.json", "dataset_stats.json"}:
            continue
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"skipping unreadable {f}")
            continue
        if isinstance(payload, dict) and "checkpoint" in payload and "summary" in payload:
            rows.append(payload)
    if not rows:
        print("no results found")
        return
    rows.sort(key=lambda r: Path(r["checkpoint"]).stem)

    keys = [k for k in HEADLINE if any(k in r["summary"] for r in rows)]
    extra = sorted({k for r in rows for k in r["summary"]} - set(keys))

    L = ["# Validation scores for saved checkpoints", ""]
    r0 = rows[0]
    L += [
        f"Val set: {r0['val_size']} samples from " +
        ", ".join(f"`{Path(p).name}`" for p in r0["val_dataset_dirs"]),
        f"Scored on a fixed random subset of {r0['num_scored']} samples "
        f"(identical for every checkpoint), {r0['num_inference_steps']} inference steps.",
        "",
        "`action_l1`/`action_l2` are in **de-normalised units** (real robot units) and are",
        "the meaningful policy-quality numbers. AHAWAM's `infer` returns actions only, so",
        "there are no video PSNR/SSIM metrics for this model.",
        "",
        "| checkpoint | " + " | ".join(keys) + " | scored |",
        "|---" * (len(keys) + 2) + "|",
    ]
    for r in rows:
        cells = []
        for k in keys:
            s = r["summary"].get(k)
            cells.append(f"{s['mean']:.4f} ± {s['std']:.3f}" if s else "—")
        L.append(f"| `{Path(r['checkpoint']).stem}` | " + " | ".join(cells)
                 + f" | {r['num_scored']} |")

    best = {}
    for k in keys:
        cand = [(r["summary"][k]["mean"], Path(r["checkpoint"]).stem)
                for r in rows if k in r["summary"]]
        if cand and k != "history_len":
            best[k] = min(cand)
    if best:
        L += ["", "**Lowest (better) per metric:**", ""]
        for k, (v, tag) in best.items():
            L.append(f"- `{k}`: **{tag}** ({v:.4f})")

    if extra:
        L += ["", "## Other metrics", "",
              "| checkpoint | " + " | ".join(extra) + " |",
              "|---" * (len(extra) + 1) + "|"]
        for r in rows:
            cells = [f"{r['summary'][k]['mean']:.4f}" if k in r["summary"] else "—"
                     for k in extra]
            L.append(f"| `{Path(r['checkpoint']).stem}` | " + " | ".join(cells) + " |")

    out = d / "SUMMARY.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    (d / "SUMMARY.json").write_text(
        json.dumps([{"checkpoint": r["checkpoint"], "summary": r["summary"]} for r in rows],
                   indent=2), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
