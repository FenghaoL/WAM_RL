import unittest
from argparse import Namespace
from omegaconf import OmegaConf
from scripts.so101_pipeline.paper_experiments.prepare import apply_control, power
from scripts.so101_pipeline.paper_experiments.trials import summarize, wilson


class PaperExperimentsTests(unittest.TestCase):
    def test_context_control_preserves_validation_and_backbone(self):
        cfg = OmegaConf.create({"data": {"train": {"correction_history_crosses_segment": True}},
                               "model": {"num_history_frames": 6},
                               "correction_val": {"history_crosses_segment": True}})
        out = apply_control(cfg, "context_truncated")
        self.assertTrue(cfg.data.train.correction_history_crosses_segment)
        self.assertFalse(out.data.train.correction_history_crosses_segment)
        self.assertEqual(out.model.num_history_frames, 6)
        self.assertTrue(out.correction_val.history_crosses_segment)
        with self.assertRaises(ValueError):
            apply_control(out, "context_truncated")

    @staticmethod
    def row(**kwargs):
        return dict(task="cylinder", protocol="full_task", model="a", condition_id="1", status="completed",
                    physical_success="1", interventions="0", duration_s="10", checkpoint="a.pt", camera_id="calib1", **kwargs)

    def test_assisted_success_is_not_autonomous_and_pending_is_not_denominator(self):
        a = self.row()
        b = dict(a, model="b", checkpoint="b.pt", interventions="1")
        pending = dict(a, condition_id="2", status="pending", physical_success="")
        report = summarize([a, b, pending])
        self.assertEqual(report["groups"][0]["completed"], 1)
        self.assertEqual(report["groups"][0]["pending"], 1)
        self.assertEqual(report["groups"][1]["autonomous_rate"], 0)
        self.assertEqual(report["paired"][0]["a_only_success"], 1)

    def test_reject_duplicate_and_checkpoint_mixing(self):
        a = self.row()
        with self.assertRaises(ValueError): summarize([a,a])
        with self.assertRaises(ValueError): summarize([a,dict(a,condition_id="2",checkpoint="changed.pt")])
        with self.assertRaises(ValueError): summarize([dict(a, duration_s="nan")])

    def test_interval_and_empty(self):
        self.assertIsNone(wilson(0,0))
        lo, hi = wilson(0,10)
        self.assertAlmostEqual(lo,0)
        self.assertAlmostEqual(hi,0.2775328,places=6)
        self.assertEqual(summarize([])["groups"], [])

    def test_legacy_power_data_cannot_be_selected(self):
        with self.assertRaisesRegex(ValueError, "Legacy power"):
            power(Namespace(hil_root="data/ahawam_hil_raw/so101_ahawam_power_adapter_hil_v2",
                            camera_note="new setup", run_root="runs/should_not_exist"))


if __name__ == "__main__": unittest.main()
