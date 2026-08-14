#!/usr/bin/env python3
"""Re-derive metrics from the raw bench logs with better timing precision.

The live driver uses a median of per-step deltas, but log timestamps only have
1-second resolution, so at ~6 s/step that quantises to about +/-8%. Averaging the
whole span from step 5 to the last step instead pushes the error under 1%.
We also pull the trainer's own reported samples/s as an independent check.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
BENCH = REPO / "runs" / "so101_g05_square_v1" / "tuning_bench"
TRAIN_FRAMES, EPOCHS = 19570, 4

TS_RE = re.compile(r"\d\d/\d\d \[(\d\d):(\d\d):(\d\d)\]")
STEP_RE = re.compile(r"step=(\d+)/")
HIST_RE = re.compile(r"history_len=([\d.]+)")
SPEED_RE = re.compile(r"speed=([\d.]+) step/s, ([\d.]+) samples/s")
OOM_RE = re.compile(r"OutOfMemoryError|CUDA out of memory")


def steps_with_ts(text):
    out = {}
    for m in STEP_RE.finditer(text):
        idx = int(m.group(1))
        if idx in out:
            continue
        last = None
        for tm in TS_RE.finditer(text, 0, m.start()):
            last = tm
        if not last:
            continue
        h, mi, s = (int(x) for x in last.groups())
        hm = HIST_RE.search(text, m.end())
        out[idx] = {"ts": h * 3600 + mi * 60 + s,
                    "hist": float(hm.group(1)) if hm else None}
    return out


def steady_mem(gpu_csv, t_start, t_end):
    """Median + max GPU memory on device 0 inside the steady training window."""
    vals = []
    if not gpu_csv.exists():
        return None, None
    for line in gpu_csv.read_text(errors="ignore").splitlines():
        p = [x.strip() for x in line.split(",")]
        if len(p) < 3 or p[1] != "0":
            continue
        tm = re.search(r"(\d\d):(\d\d):(\d\d)", p[0])
        if not tm:
            continue
        h, mi, s = (int(x) for x in tm.groups())
        t = h * 3600 + mi * 60 + s
        if t_start <= t <= t_end:
            try:
                vals.append(int(float(p[2])))
            except ValueError:
                pass
    if not vals:
        return None, None
    vals.sort()
    return vals[len(vals) // 2], vals[-1]


def main():
    rows = []
    for d in sorted(BENCH.iterdir()):
        rj = d / "result.json"
        log = d / "train.log"
        if not rj.exists() or not log.exists():
            continue
        r = json.loads(rj.read_text())
        text = log.read_text(errors="ignore")
        st = steps_with_ts(text)
        keys = sorted(st)
        warm = 5
        span_sps = None
        if len(keys) > warm + 3:
            a, b = keys[warm], keys[-1]
            dt = st[b]["ts"] - st[a]["ts"]
            if dt < 0:
                dt += 86400
            n = b - a
            if n > 0 and dt > 0:
                r["span_step_s"] = dt / n
                span_sps = r["effective_batch"] / (dt / n)
        sp = SPEED_RE.findall(text)
        r["trainer_samples_per_s"] = float(sp[-1][1]) if sp else None
        r["span_samples_per_s"] = span_sps
        # best estimate: prefer the span average, fall back to trainer average
        r["best_sps"] = span_sps or r["trainer_samples_per_s"]
        if keys:
            t0 = st[keys[min(warm, len(keys) - 1)]]["ts"]
            t1 = st[keys[-1]]["ts"]
            med, mx = steady_mem(d / "gpu.csv", t0, t1)
            r["steady_mem_med"], r["steady_mem_max"] = med, mx
        r["oom"] = bool(OOM_RE.search(text))
        hs = [v["hist"] for v in st.values() if v["hist"] is not None]
        r["max_history_len"] = max(hs) if hs else None
        rows.append(r)

    rows.sort(key=lambda x: -(x["best_sps"] or 0))
    (BENCH / "results_refined.json").write_text(json.dumps(rows, indent=2))

    L = ["# AHA-WAM SO101 全量微调 — ZeRO-2 调参实验（精修版）", "",
         "计时改用「第5步→末步的总跨度 ÷ 步数」，把 1 秒时间戳量化误差从 ±8% 压到 <1%；",
         "并列出 trainer 自报的 samples/s 作为独立交叉验证（该值含预热，会略微偏低）。", "",
         "**硬件**：4 × A100-SXM4-40GB（每卡可用 39.49 GiB）、96 核、503 GB 内存  ",
         "**数据**：SO101 `so101_g05_square_v1_ahawam_v21`，训练集 19,570 帧  ",
         "**固定条件**：ZeRO-2 + 优化器 CPU offload、全量微调、4 卡、bf16", "",
         "| 配置 | micro_bs | accum | 有效batch | OMP | RR | pin | ovlp | ckpt | 步数 | 步时(s) | **样本/秒** | trainer自报 | 稳态显存 | 峰值 | 最大hist | 4ep估时 | 结果 |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        sps = r["best_sps"]
        eta = ("%.1f h" % (TRAIN_FRAMES * EPOCHS / sps / 3600)) if sps else "—"
        L.append("| `%s` | %d | %d | %d | %d | %s | %s | %s | %s | %d/%d | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["tag"], r["batch_size"], r["grad_accum"], r["effective_batch"], r["omp_threads"],
            "Y" if r["round_robin"] else "N", "Y" if r["pin_memory"] else "N",
            "Y" if r["overlap_comm"] else "N", "Y" if r["grad_checkpoint"] else "N",
            r["completed_steps"], r["requested_steps"],
            ("%.2f" % r["span_step_s"]) if r.get("span_step_s") else "—",
            ("**%.2f**" % sps) if sps else "—",
            ("%.2f" % r["trainer_samples_per_s"]) if r.get("trainer_samples_per_s") else "—",
            str(r.get("steady_mem_med") or "—"), str(r.get("peak_mib") or "—"),
            str(r["max_history_len"] if r["max_history_len"] is not None else "—"),
            eta, "OOM" if r["oom"] else "ok"))
    ok = [r for r in rows if not r["oom"] and r["best_sps"]]
    if ok:
        b = ok[0]
        L += ["", "**最快：`%s` — %.2f 样本/秒，4 epoch 约 %.1f 小时**" % (
            b["tag"], b["best_sps"], TRAIN_FRAMES * EPOCHS / b["best_sps"] / 3600),
            "（对照基线：你实测的 ZeRO-3 + 35%% offload 为 0.565 样本/秒 / 38.5 小时）"]
    L += ["", "## 各组说明", ""]
    for r in sorted(rows, key=lambda x: x["tag"]):
        L.append("- `%s`：%s" % (r["tag"], r["note"]))
    (BENCH / "RESULTS_REFINED.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("wrote", BENCH / "RESULTS_REFINED.md")
    for r in rows:
        print("  %-12s bs=%d acc=%d sps=%s trainer=%s steady=%s oom=%s" % (
            r["tag"], r["batch_size"], r["grad_accum"],
            ("%.2f" % r["best_sps"]) if r["best_sps"] else "-",
            r.get("trainer_samples_per_s"), r.get("steady_mem_med"), r["oom"]))


if __name__ == "__main__":
    main()
