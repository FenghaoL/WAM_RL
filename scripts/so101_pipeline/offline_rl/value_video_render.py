"""CPU-only PIL/PyAV renderer: synchronized cameras, scores and rolling traces."""
from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


SERIES = [("iql_v", "IQL V", "#60a5fa"), ("iql_q1", "IQL Q1", "#fb923c"),
          ("iql_q2", "IQL Q2", "#facc15"), ("td3_q1", "TD3 Q1", "#c084fc"),
          ("td3_q2", "TD3 Q2", "#f472b6"), ("recorded_return", "REC return", "#4ade80")]
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"


def fit(image, size):
    image = Image.fromarray(image)
    image.thumbnail(size, Image.Resampling.BILINEAR)
    out = Image.new("RGB", size, "#030712")
    out.paste(image, ((size[0]-image.width)//2, (size[1]-image.height)//2))
    return out


def render_episode(job):
    import av
    from scripts.frame_annotation.annotate import Video
    payload = json.loads(Path(job["scores"]).read_text())
    if "video_sha256" in payload:
        for cam, source in payload["videos"].items():
            digest = hashlib.sha256()
            with open(source, "rb") as f:
                for block in iter(lambda: f.read(1024*1024), b""):
                    digest.update(block)
            if digest.hexdigest() != payload["video_sha256"][cam]:
                raise ValueError("Video changed between scoring and rendering")
    e, rows = payload["episode"], payload["rows"]
    n = len(rows)
    path = Path(job["video"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.mp4")
    fonts = {size: ImageFont.truetype(CJK, size) for size in (14, 17, 20, 24)}
    numeric = {size: ImageFont.truetype(MONO, size) for size in (13, 16, 28)}
    canvas = Image.new("RGB", (1280, 960), "#111827")
    draw = ImageDraw.Draw(canvas)
    groups = {"train": "训练集", "val": "验证集", "excluded_early_stop": "未参与训练：提前结束、未录到成功"}
    group = payload["group"]
    outcome = "失败" if e["outcome"] == "failure" else "成功（有成功帧）" if e["status"] == "ready" else "外部报告成功，录像未到终点"
    title = f"{groups[group.split('/')[0]]} | {outcome} | {Path(e['root']).name} / ep {e['episode_index']:03d}"
    draw.text((18, 6), title, font=fonts[20], fill="white")
    success = payload["effective_success"]
    draw.text((18, 35), f"IQL / TD3 step {payload['checkpoint_steps'][0]} / {payload['checkpoint_steps'][1]}   "
        f"成功原标注={e['first_success']}，奖励对齐帧={success}   Q 输入后续 8 个真实 POLICY/HUMAN 动作", font=fonts[14], fill="#cbd5e1")
    for x, label in ((18, "FIXED"), (658, "WRIST")):
        draw.text((x, 78), label, font=numeric[16], fill="white")
    for j, (_, label, color) in enumerate(SERIES):
        draw.text((18+j*211, 590), label, font=numeric[16], fill=color)
    draw.text((18, 685), f"最近 {job['window']} 原始帧 · 未平滑 · 全部统一纵轴 · 灰=HOLD · 绿竖线=奖励帧 / 橙=原标注", font=fonts[17], fill="#cbd5e1")
    plot = (66, 728, 1250, 902)
    ymin, ymax = job["ylimits"]
    def py(value):
        return plot[3] - (value-ymin)/(ymax-ymin)*(plot[3]-plot[1])
    for value in np.linspace(ymin, ymax, 5):
        y = py(value)
        draw.line((plot[0], y, plot[2], y), fill="#334155")
        draw.text((4, y-8), f"{value:.2f}", font=numeric[13], fill="#94a3b8")
    for value in (0., 1.):
        y = py(value)
        for x in range(plot[0], plot[2], 12):
            draw.line((x, y, x+5, y), fill="#94a3b8")
    draw.text((18, 934), "参考0/1是奖励尺度，不是校准成功率。REC return=已记录行为折扣回报，非 learned-policy Q 真值。", font=fonts[14], fill="#94a3b8")
    all_values = {key: np.array([np.nan if r[key] is None else r[key] for r in rows]) for key, _, _ in SERIES}
    sources = np.array([r["control_source"] for r in rows])
    videos = {cam: Video(source, n) for cam, source in payload["videos"].items()}
    writer = None
    preview = payload["preview_frame"]
    try:
        writer = av.open(str(temporary), "w", format="mp4", options={"movflags": "+faststart"})
        stream = writer.add_stream("libx264", rate=Fraction(str(payload["fps"])).limit_denominator(1000))
        stream.width, stream.height, stream.pix_fmt = 1280, 960, "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "23"}
        stream.codec_context.thread_count = 2
        for t, row in enumerate(rows):
            image = canvas.copy()
            d = ImageDraw.Draw(image)
            mode = ("POLICY", "HUMAN", "HOLD")[row["control_source"]]
            d.text((18, 57), f"frame {t:04d}/{n-1:04d}  elapsed {row['elapsed_seconds']:6.2f}s  {mode}  |  {row['scope']}",
                font=numeric[16], fill="#fcd34d" if mode == "HOLD" else "#e2e8f0")
            image.paste(fit(videos["fixed"].frame(t), (640, 480)), (0, 100))
            image.paste(fit(videos["wrist"].frame(t), (640, 480)), (640, 100))
            for j, (key, _, color) in enumerate(SERIES):
                value = row[key]
                d.text((18+j*211, 615), "N/A" if value is None else f"{value:+.4f}", font=numeric[28], fill=color)
            message = dict(hold="Q N/A: HOLD removed from action clock", incomplete_chunk="Q N/A: fewer than 8 recorded commands",
                tick_gap_or_drop="Q N/A: missing tick / dropped frame", valid_chunk="Q(s, data A[8]); current trained heads, no score clipping")[row["q_reason"]]
            d.text((18, 655), message, font=numeric[16], fill="#94a3b8")
            left, right = max(0, t-job["window"]+1), max(job["window"]-1, t)
            def px(frame):
                return plot[0]+(frame-left)/(right-left)*(plot[2]-plot[0])
            # Shade HOLD spans, without carrying forward or interpolating a Q.
            span = None
            for i in range(left, t+2):
                held = i <= t and sources[i] == 2
                if held and span is None:
                    span = i
                if not held and span is not None:
                    d.rectangle((px(span), plot[1], px(i), plot[3]), fill="#1e293b")
                    span = None
            for key, _, color in SERIES:
                segment = []
                for i in range(left, t+1):
                    value = all_values[key][i]
                    if np.isfinite(value):
                        segment.append((px(i), py(value)))
                    else:
                        if len(segment) > 1:
                            d.line(segment, fill=color, width=2)
                        segment = []
                if len(segment) > 1:
                    d.line(segment, fill=color, width=2)
                if np.isfinite(all_values[key][t]):
                    x, y = px(t), py(all_values[key][t])
                    d.ellipse((x-2, y-2, x+2, y+2), fill=color)
            for marker, color in ((success, "#4ade80"), (e["first_success"], "#f59e0b")):
                if marker is not None and left <= marker <= t:
                    x = px(marker)
                    d.line((x, plot[1], x, plot[3]), fill=color, width=2)
            d.line((px(t), plot[1], px(t), plot[3]), fill="white")
            for x, text in ((plot[0], str(left)), (plot[2]-60, str(right))):
                d.text((x, 908), text, font=numeric[13], fill="#94a3b8")
            if t == preview:
                image.save(path.with_suffix(".jpg"), quality=85)
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                writer.mux(packet)
        for packet in stream.encode():
            writer.mux(packet)
        writer.close()
        writer = None
        with av.open(str(temporary)) as check:
            if check.streams.video[0].frames != n:
                raise ValueError("Encoded frame count differs from source")
        os.replace(temporary, path)
    finally:
        if writer is not None:
            writer.close()
        for video in videos.values():
            video.close()
    return dict(video=str(path), frames=n, bytes=path.stat().st_size, group=group, episode_uid=e["episode_uid"])
