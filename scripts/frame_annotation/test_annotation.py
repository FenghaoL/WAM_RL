import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from annotate import discover, save, export, dense_labels, Video


class Tests(unittest.TestCase):
    def test_unknown_and_reviewed_and_events(self):
        doc = {"intervals": [dict(label="success", start_frame=2, end_frame=4)], "reviewed_labels": []}
        positive, valid, event, event_valid = dense_labels(doc, 6, "success")
        self.assertEqual(positive.tolist(), [False, False, True, True, True, False])
        self.assertEqual(valid.tolist(), positive.tolist())
        self.assertFalse(event.any())
        doc["reviewed_labels"] = ["success"]
        _, valid, event, _ = dense_labels(doc, 6, "success")
        self.assertTrue(valid.all())
        self.assertEqual(np.flatnonzero(event).tolist(), [2])
        self.assertFalse(dense_labels(doc, 6, "release")[1].any())

    def test_roundtrip_export_and_source_change(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "meta").mkdir()
            (root / "data/chunk-000").mkdir(parents=True)
            (root / "meta/info.json").write_text(json.dumps(dict(
                codebase_version="v2.1", chunks_size=1000,
                data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")))
            (root / "meta/episodes.jsonl").write_text(json.dumps(dict(episode_index=0, length=6)))
            path = root / "data/chunk-000/episode_000000.parquet"
            pq.write_table(pa.table(dict(episode_index=[0]*6, frame_index=list(range(6)), timestamp=[i/15 for i in range(6)])), path)
            ep = discover([root])[0]
            original = path.read_bytes()
            doc = ep.load()
            doc["intervals"] = [dict(label="success", start_frame=3, end_frame=5)]
            doc["reviewed_labels"] = ["success"]
            save(ep, doc)
            self.assertEqual(ep.load(), doc)
            export([ep])
            table = pq.read_table(root / "recording_context/frame_labels/chunk-000/episode_000000.parquet")
            self.assertEqual(table["success_event"].to_pylist(), [False, False, False, True, False, False])
            self.assertEqual(path.read_bytes(), original)
            pq.write_table(pa.table(dict(episode_index=[0]*6, frame_index=list(range(6)), timestamp=[i/30 for i in range(6)])), path)
            with self.assertRaisesRegex(ValueError, "source changed"):
                ep.load()

    def test_video_exact_seek(self):
        import av
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "test.mp4"
            with av.open(str(path), "w") as output:
                stream = output.add_stream("mpeg4", rate=15)
                stream.width, stream.height = 64, 48
                stream.pix_fmt = "yuv420p"
                for i in range(12):
                    frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), i*20, np.uint8), format="rgb24")
                    for packet in stream.encode(frame):
                        output.mux(packet)
                for packet in stream.encode():
                    output.mux(packet)
            video = Video(path, 12)
            try:
                for i in [0, 10, 3, 11, 0]:
                    self.assertLess(abs(float(video.frame(i).mean()) - i*20), 6)
            finally:
                video.close()
            with self.assertRaisesRegex(ValueError, "table has"):
                Video(path, 13)


if __name__ == "__main__":
    unittest.main()
