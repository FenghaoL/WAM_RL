"""All-episode value diagnostics: score frozen KV, render MP4s/CSV and HTML gallery."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import html
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from ahawam.offline_rl.cache import ShardedCache, atomic_json, memory_stats
from ahawam.offline_rl.data import effective_success_frame, sha256
from ahawam.offline_rl.networks import NetworkConfig, StateValue, TwinChunkQ
from ahawam.offline_rl.value_diagnostics import diagnostic_windows, episode_group, recorded_returns
from scripts.so101_pipeline.offline_rl.system import split
from scripts.so101_pipeline.offline_rl.value_video_render import SERIES, render_episode


def read_table(episode):
    root = Path(episode["root"])
    info = json.loads((root / "meta/info.json").read_text())
    ep = episode["episode_index"]
    fields = dict(episode_index=ep, episode_chunk=ep // int(info.get("chunks_size", 1000)))
    path = root / info["data_path"].format(**fields)
    if sha256(path) != episode["source"]["parquet_sha256"]:
        raise ValueError("Parquet changed since training")
    videos = {cam: str(root / info["video_path"].format(**fields, video_key=f"observation.images.{cam}"))
        for cam in ("fixed", "wrist")}
    return pd.read_parquet(path), info, videos


def score_inputs(cache, uid, frames, device):
    samples = [cache.sample(f"{uid}:{int(frame)}") for frame in frames]
    features = torch.stack([torch.cat((s["k"], s["v"]), -1) for s in samples]).to(device)
    return features, torch.ones(features.shape[:2], dtype=torch.bool, device=device), torch.stack([s["state"] for s in samples]).to(device)


@torch.no_grad()
def score_episode(e, cache, models, data, record_index, val_uids, steps, batch_size, output, video_hashes):
    table, info, videos = read_table(e)
    group, outcome_label = episode_group(e, val_uids)
    stem = f"{group.replace('/', '__')}__{outcome_label}__{Path(e['root']).name}__ep{e['episode_index']:06d}"
    scores = output / "scores" / f"{stem}.json"
    if scores.exists():
        return scores
    if e["episode_uid"] not in video_hashes or any(sha256(Path(path)) != video_hashes[e["episode_uid"]][cam] for cam,path in videos.items()):
        raise ValueError("Raw videos differ from the frozen feature cache's videos")
    n, uid = len(table), e["episode_uid"]
    if n != e["frames"]:
        raise ValueError("Episode frame count changed")
    windows, reasons = diagnostic_windows(table, data["config"]["chunk_size"])
    reference = recorded_returns(table, e, windows, data["config"]["gamma_tick"])
    values = {key: np.full(n, np.nan) for key, _, _ in SERIES}
    values["recorded_return"] = reference
    device = next(models["iql_q"].parameters()).device
    raw_actions = np.stack(table.executed_action_arm).astype(np.float32)
    scale = torch.tensor(cache.spec["action_scale"], dtype=torch.float32)
    offset = torch.tensor(cache.spec["action_offset"], dtype=torch.float32)
    for first in range(0, n, batch_size):
        ids = list(range(first, min(first+batch_size, n)))
        obs = score_inputs(cache, uid, ids, device)
        values["iql_v"][ids] = models["iql_v"](*obs).float().cpu().numpy()
        use = [i for i, frame in enumerate(ids) if frame in windows]
        if use:
            q_obs = tuple(x[use] for x in obs)
            actions = np.stack([raw_actions[windows[ids[i]]] for i in use])
            # Preserve the training loader's FP32 multiply, then FP32 add and
            # clamp; a NumPy float64 detour subtly changes the action inputs.
            action = (torch.from_numpy(actions) * scale + offset).clamp(-5, 5).to(device)
            q1, q2 = models["iql_q"](*q_obs, action)
            t1, t2 = models["td3_q"](*q_obs, action)
            for key, val in zip(("iql_q1", "iql_q2", "td3_q1", "td3_q2"), (q1, q2, t1, t2)):
                values[key][[ids[i] for i in use]] = val.float().cpu().numpy()
    if any(not np.isfinite(v).all() for key, v in values.items() if key == "iql_v"):
        raise FloatingPointError("Nonfinite V")
    for key in ("iql_q1", "iql_q2", "td3_q1", "td3_q2"):
        if not np.isfinite(values[key][list(windows)]).all():
            raise FloatingPointError("Nonfinite Q")
    success = effective_success_frame(table, e["first_success"])
    rows = []
    for frame in range(n):
        key = f"{uid}:{frame}"
        record_id = record_index.get(key)
        if e["status"] != "ready":
            scope = "EXCLUDED_EPISODE_PROBE"
        elif record_id is not None:
            scope = "VAL_WINDOW" if uid in val_uids else "TRAIN_WINDOW"
        elif success is not None and frame >= success:
            scope = "POST_SUCCESS_PROBE"
        else:
            scope = "NONTRAINING_OBSERVATION"
        row = dict(frame=frame, elapsed_seconds=float((table.monotonic_ns.iloc[frame]-table.monotonic_ns.iloc[0])/1e9),
            tick_index=int(table.tick_index.iloc[frame]), control_source=int(table.control_source.iloc[frame]),
            q_reason=reasons[frame], scope=scope, action_frames=windows.get(frame),
            **{k: float(v[frame]) if np.isfinite(v[frame]) else None for k, v in values.items()})
        row["iql_q_min"] = min(row["iql_q1"], row["iql_q2"]) if row["iql_q1"] is not None else None
        row["td3_q_min"] = min(row["td3_q1"], row["td3_q2"]) if row["td3_q1"] is not None else None
        row["reward"] = row["bootstrap_discount"] = row["iql_td_target"] = None
        if record_id is not None:
            reward = float(data["tensors"]["reward"][record_id])
            discount = float(data["tensors"]["discount"][record_id])
            next_frame = data["records"][record_id]["next_frame"]
            row.update(reward=reward, bootstrap_discount=discount,
                iql_td_target=reward+discount*float(values["iql_v"][next_frame]))
            # Regression check: displayed Q uses the EXACT actions used by training.
            indices = windows[frame]
            torch.testing.assert_close(torch.from_numpy(raw_actions[indices]), data["tensors"]["action"][record_id], rtol=0, atol=0)
            if indices != data["records"][record_id]["action_frames"]:
                raise ValueError("Diagnostic/training action window differs")
        rows.append(row)
    preview_candidates = [r["frame"] for r in rows if r["q_reason"] == "valid_chunk" and (success is None or r["frame"] < success)]
    payload = dict(episode=e, group=group, outcome_label=outcome_label, stem=stem, rows=rows, videos=videos,
        video_sha256=video_hashes[uid],
        fps=info["fps"], effective_success=success, checkpoint_steps=steps,
        preview_frame=preview_candidates[-1] if preview_candidates else n-1)
    atomic_json(scores, payload)
    with (output / "scores" / f"{stem}.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return scores


def describe(values):
    a = np.array([v for v in values if v is not None], dtype=float)
    if not len(a):
        return dict(count=0)
    return dict(count=len(a), min=float(a.min()), mean=float(a.mean()), max=float(a.max()),
        std=float(a.std()), p05=float(np.quantile(a, .05)), p95=float(np.quantile(a, .95)))


def summarize(paths):
    groups = defaultdict(list)
    per_episode = []
    y_values = [0., 1.]
    for path in paths:
        payload = json.loads(path.read_text())
        rows = payload["rows"]
        group = payload["group"]
        # Use valid original training/validation windows for their primary stats.
        eligible = [r for r in rows if r["scope"] in {"TRAIN_WINDOW", "VAL_WINDOW", "EXCLUDED_EPISODE_PROBE"} and r["q_reason"] == "valid_chunk"]
        groups[group].extend(eligible)
        for key, _, _ in SERIES:
            y_values.extend(r[key] for r in rows if r[key] is not None)
        valid = [r for r in rows if r["q_reason"] == "valid_chunk"]
        per_episode.append(dict(stem=payload["stem"], episode_uid=payload["episode"]["episode_uid"],
            group=group, frames=len(rows), q_windows=len(valid), included_windows=len(eligible),
            iql_v=describe([r["iql_v"] for r in eligible]),
            iql_q_min=describe([r["iql_q_min"] for r in eligible]),
            td3_q_min=describe([r["td3_q_min"] for r in eligible]),
            v_first=rows[0]["iql_v"], v_last=rows[-1]["iql_v"]))
    result = {}
    for group, rows in groups.items():
        result[group] = dict(windows=len(rows), **{key: describe([r[key] for r in rows]) for key in ("iql_v", "iql_q_min", "td3_q_min", "recorded_return")})
        result[group]["iql_disagreement"] = describe([abs(r["iql_q1"]-r["iql_q2"]) for r in rows])
        result[group]["td3_disagreement"] = describe([abs(r["td3_q1"]-r["td3_q2"]) for r in rows])
        result[group]["terminal_success_iql_abs_error"] = describe([abs(r["iql_q_min"]-r["reward"]) for r in rows if r["reward"] is not None and r["reward"] > 0])
        result[group]["terminal_success_td3_abs_error"] = describe([abs(r["td3_q_min"]-r["reward"]) for r in rows if r["reward"] is not None and r["reward"] > 0])
        result[group]["terminal_failure_iql_abs_error"] = describe([abs(r["iql_q_min"]) for r in rows if r["reward"] == 0 and r["bootstrap_discount"] == 0])
        result[group]["terminal_failure_td3_abs_error"] = describe([abs(r["td3_q_min"]) for r in rows if r["reward"] == 0 and r["bootstrap_discount"] == 0])
        result[group]["iql_bellman_abs_error"] = describe([abs(r["iql_q_min"]-r["iql_td_target"]) for r in rows if r["iql_td_target"] is not None])
        for algo in ("iql", "td3"):
            pairs = [(r[f"{algo}_q_min"], r["recorded_return"]) for r in rows if r["recorded_return"] is not None]
            result[group][f"{algo}_recorded_return_abs_difference"] = describe([abs(q-g) for q, g in pairs])
    lo, hi = min(y_values), max(y_values)
    return dict(groups=result, episodes=per_episode, ylimits=[lo-.05*(hi-lo), hi+.05*(hi-lo)],
        note="Recorded behavior return is a diagnostic reference, not ground truth learned-policy Q. No negative/positive label invented for excluded episodes.")


def gallery(output, paths, summary):
    cards = []
    counts = Counter()
    for path in paths:
        p = json.loads(path.read_text())
        group, stem, e = p["group"], p["stem"], p["episode"]
        counts[group] += 1
        video = f"videos/{group}/{stem}.mp4"
        cards.append(f'''<article data-group="{html.escape(group)}" data-name="{html.escape(stem)}">
<h3>{html.escape(group)} · {Path(e['root']).name} · ep {e['episode_index']:03d}</h3>
<p>{html.escape(p['outcome_label'])} · {e['frames']} frames · success={e['first_success']} / reward frame={p['effective_success']}</p>
<video controls preload="none" poster="{video[:-4]}.jpg" src="{video}"></video>
<p><a href="{video}">下载 / 打开 MP4</a> · <a href="scores/{stem}.csv">逐帧 CSV</a> · <a href="scores/{stem}.json">完整数据与窗口</a></p></article>''')
    options = ''.join(f'<option value="{html.escape(g)}">{html.escape(g)} ({n})</option>' for g,n in sorted(counts.items()))
    text = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>价值函数视频检查</title>
<style>body{background:#0b1220;color:#e2e8f0;font:16px system-ui;max-width:1500px;margin:30px auto;padding:0 20px}a{color:#93c5fd}article{background:#172033;padding:16px;border-radius:12px;margin:20px 0}video{width:100%;max-width:1280px}select,input{font-size:18px;padding:8px;margin-right:12px}.banner{background:#273449;padding:18px;line-height:1.8}h3{margin:4px 0}p{line-height:1.6}</style>
<h1>IQL / TD3+BC：逐帧价值函数检查</h1><div class="banner">
使用两项 10,000 步 checkpoint 的当前 Q1/Q2，及 IQL V；不是 target Q。Q 输入后续 8 个真实 POLICY/HUMAN 动作，V 输入冻结 KV＋关节状态，不是仅看当前一张图。<br>
理论回报尺度为 0–1（单次成功 +1，γ=0.997/保留动作），预测不做裁剪，也不是校准的成功概率。REC return 是已记录混合行为的折扣回报，不能当成新 actor 的 Q 真值。<br>
HOLD、缺动作和丢帧处 Q 显示 N/A；V 仍会显示，但 HOLD、成功后等未拟合观察属于外推。灰色区=HOLD，绿竖线=奖励对齐成功帧，橙竖线=原成功标注帧。<br>
9 条提前停止录像的 episode 标为 excluded_early_stop，未参与 Q/V 训练，没有臆造为失败或给它们画零回报。验证集只有 3 条标注成功＋1 条提前停止，没有失败样本。
</div><p><a href="summary.json">分组统计与误差</a> · <a href="provenance.json">checkpoint / cache 来源</a> · <a href="README.md">使用说明</a></p>
<select id="group"><option value="">全部视频</option>OPTIONS</select><input id="query" placeholder="搜索 run / episode / early_stop"><span id="count"></span>
CARDS
<script>function filter(){let n=0;for(const a of document.querySelectorAll('article')){const ok=(!group.value||a.dataset.group===group.value)&&a.dataset.name.toLowerCase().includes(query.value.toLowerCase());a.hidden=!ok;if(ok)n++;}document.getElementById('count').textContent=n+' 个视频';}group.onchange=filter;query.oninput=filter;filter();</script></html>'''
    (output / "index.html").write_text(text.replace("OPTIONS", options).replace("CARDS", '\n'.join(cards)))


def main(args):
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "scores").mkdir(exist_ok=True)
    started = time.perf_counter()
    data = torch.load(args.transitions, map_location="cpu", weights_only=True)
    cache = ShardedCache(args.cache)
    val_uids = set(split(data)[2])
    provenance = dict(format="ahawam_value_diagnostic/v1", transitions_sha256=sha256(args.transitions), feature_spec=cache.spec,
        checkpoints={name: dict(path=str(path.resolve()), sha256=sha256(path)) for name,path in (("iql",args.iql),("td3",args.td3))},
        heads="current critic Q1/Q2 and current IQL V; no score clipping", batch_size=args.batch_size,
        code_sha256={p:sha256(Path(p)) for p in (__file__, "src/ahawam/offline_rl/value_diagnostics.py", "scripts/so101_pipeline/offline_rl/value_video_render.py")})
    identity = args.output / "provenance.json"
    if identity.exists() and json.loads(identity.read_text()) != provenance:
        raise ValueError("Diagnostic output provenance changed; use a new output directory")
    atomic_json(identity, provenance)
    models, steps = {}, []
    for name,path in (("iql",args.iql),("td3",args.td3)):
        blob = torch.load(path, map_location="cpu", weights_only=True)
        previous = blob["provenance"]
        if previous["feature_spec"] != cache.spec or previous["transitions_sha256"] != provenance["transitions_sha256"] or set(previous["validation_episodes"]) != val_uids:
            raise ValueError("Critic/data/cache/split provenance mismatch")
        if blob["config"]["algorithm"] != ("iql" if name == "iql" else "td3_bc"):
            raise ValueError("Wrong algorithm checkpoint")
        cfg = NetworkConfig(**blob["network"])
        models[name+"_q"] = TwinChunkQ(cfg)
        models[name+"_q"].load_state_dict(blob["models"]["critic"])
        if name == "iql":
            models["iql_v"] = StateValue(cfg)
            models["iql_v"].load_state_dict(blob["models"]["value"])
        steps.append(blob["steps"])
        del blob
    for model in models.values():
        model.eval().requires_grad_(False).to(args.device)
    record_index = {r["observation_key"]: i for i,r in enumerate(data["records"])}
    video_hashes = {}
    for path in args.cache.glob("done_rank*.json"):
        video_hashes.update(json.loads(path.read_text())["video_sha256"])
    paths = []
    episodes = data["episodes"][:args.limit] if args.limit else data["episodes"]
    for i,e in enumerate(episodes):
        path = score_episode(e, cache, models, data, record_index, val_uids, steps, args.batch_size, args.output, video_hashes)
        paths.append(path)
        print(json.dumps(dict(stage="scored", episode=i+1, total=len(episodes), uid=e["episode_uid"], seconds=time.perf_counter()-started)), flush=True)
    summary = summarize(paths)
    atomic_json(args.output / "summary.json", summary)
    del models, cache
    if torch.cuda.is_available():
        scoring_memory = memory_stats()
        torch.cuda.empty_cache()
    else:
        scoring_memory = {}
    jobs = [dict(scores=str(path), video=str(args.output / "videos" / json.loads(path.read_text())["group"] / (path.stem+".mp4")),
        ylimits=summary["ylimits"], window=args.window) for path in paths]
    # Spawn after GPU scoring, never fork an initialized CUDA runtime.
    rendered = []
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = [pool.submit(render_episode, job) for job in jobs]
        for future in as_completed(futures):
            rendered.append(future.result())
            atomic_json(args.output / "render_progress.json", dict(episodes=len(rendered), total=len(jobs), results=rendered))
            print(json.dumps(dict(stage="rendered", episode=len(rendered), total=len(jobs), result=rendered[-1])), flush=True)
    gallery(args.output, paths, summary)
    (args.output / "README.md").write_text("# 价值函数视频检查\n\n打开 index.html 筛选并播放全部视频。videos/ 是 H.264 MP4，scores/ 为逐帧 CSV/JSON。\n\n"
        "训练集与验证集保留原划分；excluded_early_stop/ 是未参与训练的 9 条提前结束录像。\n\n"
        "Q 是当前状态加后续 8 个真实非 HOLD 动作的值；V 使用 KV 加关节状态。Q 不可构造时显示 N/A，未补零或延用上一帧。\n\n"
        "单次 +1 的折扣回报理论在 [0,1]；模型输出不裁剪，不是成功概率。绿色 REC return 为记录行为回报，非新策略的 Q 真值。\n\n"
        "灰色区=HOLD；绿竖线=奖励对齐成功帧，橙竖线=原标注帧。成功后仍播放原视频并标 POST_SUCCESS_PROBE。\n\n"
        "完整实现与解释见 docs/offline_rl/15_VALUE_VIDEO_DIAGNOSTICS_ZH.md。\n")
    atomic_json(args.output / "complete.json", dict(episodes=len(rendered), frames=sum(r["frames"] for r in rendered),
        video_bytes=sum(r["bytes"] for r in rendered), total_seconds=time.perf_counter()-started,
        scoring_memory=scoring_memory, checkpoint_steps=steps, groups=dict(Counter(r["group"] for r in rendered))))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--transitions", type=Path, default=Path("runs/offline_rl/cylinder103_skip_hold/data/transitions.pt"))
    p.add_argument("--cache", type=Path, default=Path("runs/offline_rl/cylinder103_skip_hold/cache"))
    p.add_argument("--iql", type=Path, default=Path("runs/offline_rl/cylinder103_experiments_20261025/iql/last.pt"))
    p.add_argument("--td3", type=Path, default=Path("runs/offline_rl/cylinder103_experiments_20261025/td3_bc/last.pt"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--window", type=int, default=300)
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    if min(args.batch_size, args.workers, args.window) < 1 or (args.limit is not None and args.limit < 1):
        p.error("Positive batch size, worker count, window and limit required")
    main(args)
