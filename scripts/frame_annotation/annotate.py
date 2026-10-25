"""Frame-indexed, non-destructive annotation of LeRobot v2.1 episodes."""
from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Episode:
    def __init__(self, root, info, row, manifest):
        self.root, self.info = root, info
        self.index, self.length = int(row["episode_index"]), int(row["length"])
        self.manifest = manifest
        self.logical_id = manifest.get("logical_episode_id")
        self.fields = dict(episode_index=self.index,
                           episode_chunk=self.index // int(info.get("chunks_size", 1000)))
        self.path = root / info["data_path"].format(**self.fields)
        self.label_path = root / "recording_context/frame_annotations" / f"episode_{self.index:06d}.json"

    def video_path(self, camera):
        return self.root / self.info["video_path"].format(
            **self.fields, video_key=f"observation.images.{camera}")

    def identity(self):
        return {"episode_index": self.index, "logical_episode_id": self.logical_id,
                "num_frames": self.length, "parquet_path": self.path.relative_to(self.root).as_posix(),
                "parquet_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}

    def table(self):
        table = pq.read_table(self.path)
        if len(table) != self.length:
            raise ValueError(f"Episode length mismatch: {self.path}")
        frames = table["frame_index"].to_pylist()
        if frames != list(range(self.length)):
            raise ValueError(f"Non-contiguous frame_index: {self.path}")
        if set(table["episode_index"].to_pylist()) != {self.index}:
            raise ValueError(f"Episode index mismatch: {self.path}")
        return table

    def load(self):
        identity = self.identity()
        if not self.label_path.exists():
            return {"format": "so101_frame_annotations/v1", "source": identity,
                    "reviewed_labels": [], "intervals": [], "notes": ""}
        doc = json.loads(self.label_path.read_text(encoding="utf-8"))
        if doc.get("format") != "so101_frame_annotations/v1" or doc["source"] != identity:
            raise ValueError(f"Annotation source changed; inspect {self.label_path}")
        validate(doc, self.length)
        return doc


def validate(doc, length):
    for label in doc["reviewed_labels"]:
        check_label(label)
    for item in doc["intervals"]:
        check_label(item["label"])
        a, b = item["start_frame"], item["end_frame"]
        if type(a) is not int or type(b) is not int or not 0 <= a <= b < length:
            raise ValueError(f"Invalid inclusive interval: {item}")


def check_label(label):
    if not re.fullmatch(r"[a-z][a-z0-9_]*", label):
        raise ValueError("Label must use lowercase English letters, digits and underscores")


def save(ep, doc):
    validate(doc, ep.length)
    if ep.identity() != doc["source"]:
        raise ValueError("Source changed during annotation")
    doc["updated_utc"] = datetime.now(timezone.utc).isoformat()
    ep.label_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ep.label_path.with_suffix(".json.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(doc, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, ep.label_path)


def discover(roots):
    episodes = []
    for root in roots:
        root = Path(root).resolve()
        info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        if info["codebase_version"] != "v2.1":
            raise ValueError(f"Only LeRobot v2.1 is supported: {root}")
        manifests = {int(m["data_episode_index"]): m for m in
                     read_jsonl(root / "recording_context/episodes_manifest.jsonl")}
        for row in read_jsonl(root / "meta/episodes.jsonl"):
            episodes.append(Episode(root, info, row, manifests.get(int(row["episode_index"]), {})))
    if not episodes:
        raise ValueError("No saved episodes found")
    return episodes


def dense_labels(doc, length, label):
    positive = np.zeros(length, dtype=bool)
    for item in doc["intervals"]:
        if item["label"] == label:
            positive[item["start_frame"]:item["end_frame"] + 1] = True
    valid = np.ones(length, dtype=bool) if label in doc["reviewed_labels"] else positive.copy()
    # An onset is only known when its preceding observation is reviewed negative.
    onset_valid = valid.copy()
    onset_valid[1:] &= valid[:-1]
    onset = positive & np.r_[True, ~positive[:-1]] & onset_valid
    return positive, valid, onset, onset_valid


def export(episodes):
    count = 0
    for ep in episodes:
        if not ep.label_path.exists():
            continue
        doc, table = ep.load(), ep.table()
        data = {k: table[k] for k in ("episode_index", "frame_index", "timestamp", "index") if k in table.column_names}
        for label in sorted({"success", *doc["reviewed_labels"], *(i["label"] for i in doc["intervals"])}):
            positive, valid, onset, onset_valid = dense_labels(doc, ep.length, label)
            data[f"label.{label}"] = pa.array(positive)
            data[f"label.{label}.valid"] = pa.array(valid)
            if label == "success":
                data["success_event"] = pa.array(onset)
                data["success_event.valid"] = pa.array(onset_valid)
        output = ep.root / "recording_context/frame_labels" / ep.path.relative_to(ep.root / "data")
        output.parent.mkdir(parents=True, exist_ok=True)
        result = pa.table(data).replace_schema_metadata({
            b"annotation_source": json.dumps(doc, ensure_ascii=False).encode("utf-8"),
            b"semantics": b"Labels describe observation at frame_index; not transition reward or done."})
        temp = output.with_suffix(".parquet.tmp")
        pq.write_table(result, temp)
        os.replace(temp, output)
        count += 1
    print(f"Exported {count} annotated episodes; original data unchanged.")


class Video:
    """Exact sequential decoding; bounded RGB cache, independent of seek PTS."""
    def __init__(self, path, length):
        import av
        self.container = av.open(str(path))
        self.stream = self.container.streams.video[0]
        decoded_count = sum(1 for _ in self.container.decode(self.stream))
        if decoded_count != length:
            self.container.close()
            raise ValueError(f"Video has {decoded_count} frames; table has {length}: {path}")
        self.container.seek(0, stream=self.stream)
        self.decoder = iter(self.container.decode(self.stream))
        self.position, self.cache = -1, OrderedDict()

    def frame(self, index):
        if index not in self.cache:
            if index <= self.position:
                self.container.seek(0, stream=self.stream)
                self.decoder = iter(self.container.decode(self.stream))
                self.position = -1
            while self.position < index:
                frame = next(self.decoder)
                self.position += 1
                if self.position == index:
                    self.cache[index] = frame.to_ndarray(format="rgb24")
            while len(self.cache) > 24:
                self.cache.popitem(last=False)
        return self.cache[index]

    def close(self):
        self.container.close()


class App:
    def __init__(self, episodes, labels):
        import tkinter as tk
        from tkinter import ttk, messagebox
        from PIL import Image, ImageTk
        self.tk, self.ttk, self.msg = tk, ttk, messagebox
        self.Image, self.ImageTk = Image, ImageTk
        self.episodes, self.videos, self.photos = episodes, [], []
        self.selected, self.frame, self.start, self.playing = 0, 0, None, False
        self.play_job = None
        self.dirty, self.pending = False, None
        self.win = tk.Tk()
        self.win.report_callback_exception = lambda kind, value, tb: self.msg.showerror("Annotation error", str(value))
        self.win.title("SO-101 frame annotation")
        self.win.geometry("1450x880")
        self.label = tk.StringVar(value=labels[0])
        self.status, self.detail = tk.StringVar(), tk.StringVar()
        left = ttk.Frame(self.win, padding=8)
        left.pack(side="left", fill="y")
        self.list = tk.Listbox(left, width=43, height=32, exportselection=False)
        self.list.pack(fill="both", expand=True)
        for ep in episodes:
            self.list.insert("end", f"{ep.root.name} / {ep.index:03d} / {ep.manifest.get('outcome', '?')}")
        self.list.bind("<<ListboxSelect>>", self.choose)
        panel = ttk.Frame(self.win, padding=8)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, textvariable=self.status, font=("Segoe UI", 13, "bold"), wraplength=1000).pack(anchor="w")
        views = ttk.Frame(panel)
        views.pack(fill="both", expand=True)
        self.views = []
        for camera in ("fixed", "wrist"):
            box = ttk.Frame(views)
            box.pack(side="left", fill="both", expand=True)
            ttk.Label(box, text=camera, font=("Segoe UI", 14)).pack()
            view = ttk.Label(box)
            view.pack(expand=True)
            self.views.append(view)
        self.slider = tk.Scale(panel, orient="horizontal", showvalue=False, command=self.slide)
        self.slider.pack(fill="x")
        ttk.Label(panel, textvariable=self.detail, font=("Consolas", 11)).pack(anchor="w")
        controls = ttk.Frame(panel)
        controls.pack(fill="x", pady=5)
        for text, command in [("Play / pause [Space]", self.play), ("-1 [Left]", lambda: self.step(-1)),
                              ("+1 [Right]", lambda: self.step(1)), ("-15", lambda: self.step(-15)),
                              ("+15", lambda: self.step(15))]:
            ttk.Button(controls, text=text, command=command).pack(side="left")
        row = ttk.Frame(panel)
        row.pack(fill="x", pady=5)
        ttk.Label(row, text="Label:").pack(side="left")
        ttk.Combobox(row, textvariable=self.label, values=labels, width=17).pack(side="left")
        for text, command in [("Start [I]", self.mark_start), ("End + add [O]", self.mark_end),
                              ("Point [K]", lambda: self.add(self.frame, self.frame)),
                              ("To end [E]", lambda: self.add(self.frame, self.ep.length - 1))]:
            ttk.Button(row, text=text, command=command).pack(side="left")
        self.ranges = tk.Listbox(panel, height=5, exportselection=False)
        self.ranges.pack(fill="x")
        ttk.Button(panel, text="Delete selected interval", command=self.delete).pack(anchor="w")
        self.notes = tk.StringVar()
        note_row = ttk.Frame(panel)
        note_row.pack(fill="x")
        ttk.Label(note_row, text="Notes:").pack(side="left")
        ttk.Entry(note_row, textvariable=self.notes).pack(side="left", fill="x", expand=True)
        self.notes.trace_add("write", lambda *_: setattr(self, "dirty", True))
        footer = ttk.Frame(panel)
        footer.pack(fill="x", pady=8)
        for text, command in [("Save draft [S]", self.persist), ("Review label + save [R]", self.review),
                              ("Previous", lambda: self.move(-1)), ("Next [N]", lambda: self.move(1)),
                              ("Export saved labels", lambda: self.safe_export())]:
            ttk.Button(footer, text=text, command=command).pack(side="left")
        ttk.Label(panel, text="Review applies to current label: unmarked frames become negative.\n"
                  "Draft: unmarked frames remain UNKNOWN. Existing episode outcome is reference only.").pack(anchor="w")
        self.win.bind("<KeyPress>", self.key)
        # Stop native Space activation from also invoking the focused button.
        def bind_buttons(widget):
            if isinstance(widget, (ttk.Button, tk.Listbox)):
                widget.bind("<space>", lambda event: self.key(event))
            for child in widget.winfo_children():
                bind_buttons(child)
        bind_buttons(self.win)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self.load(0)

    def load(self, index):
        self.playing = False
        if self.pending:
            self.win.after_cancel(self.pending)
            self.pending = None
        for v in self.videos:
            v.close()
        self.videos = []
        self.selected, self.ep = index, self.episodes[index]
        self.doc, self.table = self.ep.load(), self.ep.table()
        for camera in ("fixed", "wrist"):
            self.videos.append(Video(self.ep.video_path(camera), self.ep.length))
        self.frame, self.start = 0, None
        self.notes.set(self.doc.get("notes", ""))
        self.dirty = False
        self.slider.configure(to=self.ep.length - 1)
        self.list.selection_clear(0, "end")
        self.list.selection_set(index)
        self.list.see(index)
        self.refresh_ranges()
        self.show(0)

    def show(self, index):
        self.frame = max(0, min(int(index), self.ep.length - 1))
        self.slider.set(self.frame)
        self.photos = []
        for v, view in zip(self.videos, self.views):
            image = self.Image.fromarray(v.frame(self.frame))
            image.thumbnail((520, 390))
            photo = self.ImageTk.PhotoImage(image)
            view.configure(image=photo)
            self.photos.append(photo)
        timestamp = self.table["timestamp"][self.frame].as_py()
        active = [i["label"] for i in self.doc["intervals"] if i["start_frame"] <= self.frame <= i["end_frame"]]
        self.detail.set(f"frame {self.frame}/{self.ep.length - 1} | dataset time {timestamp:.3f}s | labels: {active} | I={self.start}")

    def slide(self, value):
        if int(float(value)) == self.frame:
            return
        if self.pending:
            self.win.after_cancel(self.pending)
        self.pending = self.win.after(90, lambda: self.seek(int(float(value))))

    def seek(self, value):
        self.pending = None
        self.playing = False
        self.show(value)

    def step(self, delta):
        self.playing = False
        self.show(self.frame + delta)

    def play(self):
        was_playing = self.playing
        if self.play_job is not None:
            self.win.after_cancel(self.play_job)
            self.play_job = None
        self.playing = not was_playing
        if self.playing:
            self.tick()

    def tick(self):
        self.play_job = None
        if self.playing and self.frame < self.ep.length - 1:
            self.show(self.frame + 1)
            self.play_job = self.win.after(max(1, round(1000 / self.ep.info["fps"])), self.tick)
        else:
            self.playing = False

    def mark_start(self):
        self.playing, self.start = False, self.frame
        self.show(self.frame)

    def mark_end(self):
        if self.start is None:
            self.msg.showinfo("Select start", "Press I on the first frame of the interval.")
            return
        self.add(self.start, self.frame)
        self.start = None

    def add(self, start, end):
        label = self.label.get().strip()
        try:
            check_label(label)
            if end < start:
                raise ValueError("End precedes start")
        except ValueError as exc:
            self.msg.showerror("Invalid annotation", str(exc))
            return
        self.playing = False
        self.doc["intervals"].append(dict(label=label, start_frame=start, end_frame=end))
        self.dirty = True
        self.refresh_ranges()
        self.show(self.frame)

    def delete(self):
        selected = self.ranges.curselection()
        if selected:
            self.doc["intervals"].pop(selected[0])
            self.dirty = True
            self.refresh_ranges()
            self.show(self.frame)

    def refresh_ranges(self):
        self.ranges.delete(0, "end")
        for item in self.doc["intervals"]:
            self.ranges.insert("end", f"{item['label']}: [{item['start_frame']}, {item['end_frame']}] inclusive")
        reviewed = self.doc["reviewed_labels"]
        self.status.set(f"{self.ep.root.name} / episode {self.ep.index:06d} | outcome: {self.ep.manifest.get('outcome', '?')}\nReviewed labels: {reviewed}")

    def persist(self):
        self.doc["notes"] = self.notes.get()
        save(self.ep, self.doc)
        self.dirty = False
        self.refresh_ranges()

    def review(self):
        label = self.label.get().strip()
        try:
            check_label(label)
        except ValueError as exc:
            self.msg.showerror("Invalid label", str(exc))
            return
        if not self.msg.askyesno("Review complete", f"Mark all unmarked frames NEGATIVE for '{label}'?\nOnly confirm after reviewing the entire episode."):
            return
        if label not in self.doc["reviewed_labels"]:
            self.doc["reviewed_labels"].append(label)
        self.persist()

    def leave(self):
        if self.dirty:
            answer = self.msg.askyesnocancel("Unsaved annotations", "Save changes before leaving?")
            if answer is None:
                return False
            if answer:
                self.persist()
        return True

    def move(self, delta):
        target = self.selected + delta
        if 0 <= target < len(self.episodes) and self.leave():
            self.load(target)

    def choose(self, _event):
        selected = self.list.curselection()
        if selected and selected[0] != self.selected:
            if self.leave():
                self.load(selected[0])
            else:
                self.list.selection_clear(0, "end")
                self.list.selection_set(self.selected)

    def safe_export(self):
        if self.leave():
            try:
                export(self.episodes)
                self.msg.showinfo("Export", "Saved annotations exported to recording_context/frame_labels.")
            except Exception as exc:
                self.msg.showerror("Export failed", str(exc))

    def key(self, event):
        if isinstance(event.widget, (self.ttk.Entry, self.ttk.Combobox, self.tk.Entry)):
            return
        actions = {"space": self.play, "left": lambda: self.step(-1), "right": lambda: self.step(1),
                   "i": self.mark_start, "o": self.mark_end, "k": lambda: self.add(self.frame, self.frame),
                   "e": lambda: self.add(self.frame, self.ep.length - 1), "s": self.persist,
                   "r": self.review, "n": lambda: self.move(1)}
        action = actions.get(event.keysym.lower())
        if action:
            action()
            return "break"

    def close(self):
        if self.leave():
            self.playing = False
            for v in self.videos:
                v.close()
            self.win.destroy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", action="append", required=True, type=Path)
    parser.add_argument("--labels", nargs="+", default=["success", "insert", "release", "recovery"])
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    episodes = discover(args.dataset)
    if args.export:
        export(episodes)
    elif args.audit:
        for root in args.dataset:
            selected = [ep for ep in episodes if ep.root == root.resolve()]
            for ep in selected:
                ep.table()
                ep.load()
            print(f"{root.name}: {len(selected)} episodes; tables and annotation identities valid")
        print("Columns:", episodes[0].table().column_names)
    else:
        for label in args.labels:
            check_label(label)
        App(episodes, args.labels).win.mainloop()


if __name__ == "__main__":
    main()
