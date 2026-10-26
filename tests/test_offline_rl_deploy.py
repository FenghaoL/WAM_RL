"""Deployment history, reset, actor normalization and backward-compatible config tests."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
import yaml

from deploy.server.offline_rl_policy import OfflineRLPolicy, history_steps
from ahawam.offline_rl.inference import FrozenKVActor
from ahawam.offline_rl.learner import LearnerConfig, OfflineLearner
from ahawam.offline_rl.networks import NetworkConfig


def fake_policy(external=False):
    policy = object.__new__(OfflineRLPolicy)
    policy.device = "cpu"
    policy.model = SimpleNamespace(torch_dtype=torch.bfloat16, video_rope_frame_stride=4, _inference_state=None)
    policy.model.reset_history = lambda: None
    policy.action_chunk_size = 8
    policy.history_count = 6
    policy.history_stride_commands = 48
    policy._rl_requests = 0
    policy._rl_observations = {}
    policy.last_conditioning = None
    policy.current_instruction = "insert"
    policy._DEFAULT_PROMPT = "{task}"
    policy._observation = dict(image_tensor=torch.zeros(1, 3, 4, 4), state_vector=np.zeros(6, dtype=np.float32))
    policy._current_observation_hash = b"image"
    policy._resolve_text_condition = lambda prompt: (None, None)
    policy._normalize_state = lambda state: torch.from_numpy(state).unsqueeze(0)
    policy._denormalize_action = lambda action: action.float().cpu().numpy() * 2 + 10
    policy.prefills = []
    def prefill(**kwargs):
        policy.prefills.append(policy.model._observed_frame_index*4)
        policy.model._inference_state = {}
    policy._prefill_episode = prefill
    def infer(**kwargs):
        policy.model._inference_state["_chunk_video_kv_cache"] = [dict(k=torch.ones(1, 2, 4), v=torch.ones(1, 2, 4))]
        return dict(action_chunk=torch.full((8, 6), .25), chunk_index=0)
    policy._infer_action_chunk = infer
    policy.external_actor = None
    if external:
        def actor(k, v, state, reference):
            assert k.dtype == v.dtype == torch.bfloat16
            torch.testing.assert_close(reference, torch.full((1, 8, 6), .25))
            return dict(normalized=torch.full((1, 8, 6), .5))
        policy.external_actor = SimpleNamespace(action=actor)
    return policy


class DeploymentTests(unittest.TestCase):
    def test_history_clock_and_bounded_observation_bank(self):
        policy = fake_policy()
        self.assertEqual(history_steps(0), [])
        for _ in range(39):
            output = policy.get_action()
            self.assertEqual(output.shape, (8, 6))
            self.assertTrue((output == 10.5).all())
        self.assertEqual(policy.last_conditioning["command_step"], 304)
        self.assertEqual(policy.last_conditioning["history_steps"], [16, 64, 112, 160, 208, 256])
        self.assertEqual(policy.prefills[-7:], [16, 64, 112, 160, 208, 256, 304])
        self.assertLessEqual(len(policy._rl_observations), 37)

    def test_external_actor_uses_bf16_kv_and_denormalizes_once(self):
        policy = fake_policy(external=True)
        result = policy.get_action()
        self.assertTrue((result == 11).all())
        self.assertEqual(policy.last_conditioning["chunk_index"], 0)
        self.assertEqual(policy._rl_requests, 1)

    def test_failed_inference_does_not_advance_clock(self):
        policy = fake_policy()
        policy._infer_action_chunk = lambda **kwargs: (_ for _ in ()).throw(ValueError("inference failed"))
        with self.assertRaises(ValueError):
            policy.get_action()
        self.assertEqual(policy._rl_requests, 0)
        self.assertEqual(policy._rl_observations, {})

    def test_reset_clears_all_episode_history(self):
        policy = fake_policy()
        policy.get_action()
        policy.reset_obsrvationwindows()
        self.assertIsNone(policy._observation)
        self.assertEqual(policy._rl_observations, {})
        self.assertEqual(policy._rl_requests, 0)
        self.assertIsNone(policy.last_conditioning)

    def test_actor_only_checkpoint_and_absolute_target_conversion(self):
        cfg = NetworkConfig(feature_dim=8, architecture="raw_kv", width=8, num_heads=2, max_tokens=2)
        learner = OfflineLearner(cfg, LearnerConfig(algorithm="iql"))
        spec = dict(base_checkpoint_sha256="base", action_scale=[2.]*6, action_offset=[-1.]*6)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"actor.pt"
            learner.save(path, dict(feature_spec=spec))
            with patch.object(OfflineLearner, "load", side_effect=RuntimeError("Must not load critics/optimizers")):
                actor = FrozenKVActor(path, "cpu", "base")
            k = torch.randn(1, 2, 4).bfloat16()
            state, ref = torch.zeros(1, 6), torch.ones(1, 8, 6) * .5
            result = actor.action(k, k, state, ref)
            with torch.no_grad():
                expected = learner.actor(k.repeat(1, 1, 2), torch.ones(1, 2, dtype=torch.bool), state, ref)
            torch.testing.assert_close(result["normalized"], expected)
            torch.testing.assert_close(result["absolute_targets"], (expected+1)/2)

    def test_three_configs_have_distinct_gpus_ports_and_collector_base(self):
        root = Path("scripts/so101_pipeline/online_deploy/deployments")
        configs = [yaml.safe_load((root/f"cylinder_offline_{name}.yaml").read_text()) for name in ("iql", "td3_bc", "iql_action_dit")]
        self.assertEqual({c["serving"]["gpu_id"] for c in configs}, {0, 1, 2})
        self.assertEqual({c["serving"]["port"] for c in configs}, {12130, 12131, 12132})
        self.assertTrue(all(c["checkpoint_path"].endswith("step_006572.pt") for c in configs))
        self.assertTrue(all(c["serving"]["max_return_actions"] == 8 for c in configs))


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
