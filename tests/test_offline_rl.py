"""Temporal/reward contract tests and actual offline learner updates on CPU."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from ahawam.offline_rl.data import TransitionConfig, build_dataset, episode_transitions, inspect_episode
from ahawam.offline_rl.features import causal_history_frames, pool_routed_values
from ahawam.offline_rl.learner import LearnerConfig, OfflineLearner, expectile_loss
from ahawam.offline_rl.networks import NetworkConfig, TwinChunkQ
from scripts.so101_pipeline.offline_rl.run import episode_split, make_batches


def table(sources=(0,0,2,2,0,0,0,0,0),gaps=False):
    n = len(sources)
    a = np.arange(n,dtype=np.float32)[:,None].repeat(6,1)
    for i,s in enumerate(sources):
        if s==2: a[i]=a[i-1]
    ticks = np.arange(n)
    if gaps: ticks[3:] += 4
    return pd.DataFrame(dict(frame_index=np.arange(n),episode_index=np.zeros(n,dtype=int),
        action=list(a),executed_action_arm=list(a),**{"observation.state":list(a*.5)},
        control_source=list(sources),tick_index=ticks,monotonic_ns=np.arange(n)*100_000_000,
        dropped_tick=False,policy_chunk_id=[1,1,-1,-1,2,2,2,2,3][:n],
        action_index_in_chunk=[0,1,-1,-1,0,1,2,3,0][:n]))


def report(success=4):
    return dict(status="ready",first_success=success,episode_uid="run/ep",root="/tmp/run",
                episode_index=0,outcome_class="SUCCESS_WITH_INTERVENTION")


def batch(n=16,c=2):
    success = torch.arange(n)%2 == 0
    features = success.float()[:,None,None].repeat(1,3,4)
    return dict(features=features,feature_valid=torch.ones(n,3,dtype=torch.bool),state=torch.zeros(n,6),
        next_features=torch.zeros_like(features),next_feature_valid=torch.ones(n,3,dtype=torch.bool),next_state=torch.zeros(n,6),
        action=success.float()[:,None,None].repeat(1,c,6),action_valid=torch.ones(n,c,dtype=torch.bool),
        reference=torch.zeros(n,c,6),next_reference=torch.zeros(n,c,6),reward=success.float(),discount=torch.zeros(n))


class TemporalTests(unittest.TestCase):
    def test_success_observation_rewards_incoming_action_and_truncates(self):
        rows = episode_transitions(table(),report(4),TransitionConfig(8,1,.9))
        self.assertEqual(rows[0]["next_frame"],4)
        self.assertAlmostEqual(rows[0]["reward"],.9**3)
        self.assertEqual(rows[0]["discount"],0)
        self.assertEqual(rows[0]["action_valid"].sum(),4)
        np.testing.assert_array_equal(rows[0]["action"][4:],np.repeat(rows[0]["action"][3:4],4,axis=0))
        self.assertTrue(all(r["start_frame"]<4 and r["next_frame"]<=4 for r in rows))

    def test_failure_all_zero_and_no_last_action_without_observation(self):
        rows = episode_transitions(table(),report(None),TransitionConfig(2,1,.9))
        self.assertTrue(all(r["reward"]==0 for r in rows))
        self.assertEqual(rows[-1]["next_frame"],8)
        self.assertTrue(rows[-1]["terminated"])

    def test_gaps_never_spliced(self):
        rows = episode_transitions(table(gaps=True),report(None),TransitionConfig(2,1,.9))
        self.assertTrue(all(not(r["start_frame"]<3<=r["next_frame"]) for r in rows))

    def test_pending_success_never_becomes_negative(self):
        r = dict(report(),status="pending_success_annotation")
        self.assertEqual(episode_transitions(table(),r,TransitionConfig()),[])

    def test_macro_includes_hold_reward_and_duration(self):
        rows = episode_transitions(table(),report(4),TransitionConfig(2,1,.9,temporal_mode="macro"))
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["duration_ticks"],4)
        self.assertEqual(rows[0]["action_valid"].sum(),2)
        self.assertAlmostEqual(rows[0]["reward"],.9**3)
        self.assertEqual(rows[0]["control_source"].tolist(),[0,0])

    def test_macro_continuing_discount_counts_hold_ticks(self):
        rows = episode_transitions(table(),report(None),TransitionConfig(2,1,.9,temporal_mode="macro"))
        self.assertAlmostEqual(rows[0]["discount"],.9**4)
        self.assertFalse(rows[0]["terminated"])

    def test_censor_is_explicit_and_blocks_assisted_bootstrap(self):
        d = table((0,0,1,1,0,0,0,0,0))
        mixed = episode_transitions(d,report(4),TransitionConfig(4,1,.9))
        censor = episode_transitions(d,report(4),TransitionConfig(4,1,.9,"censor"))
        self.assertGreater(mixed[0]["reward"],0)
        self.assertEqual(censor[0]["reward"],0)
        self.assertEqual(censor[0]["next_frame"],2)
        self.assertTrue(censor[0]["truncated"])
        self.assertEqual(censor[0]["discount"],0)

    def test_success_observation_precedes_first_human_command(self):
        d = table((0,0,1,1,0,0,0,0,0))
        for mode in ("tick","macro"):
            row = episode_transitions(d,report(2),TransitionConfig(2,1,.9,"censor",mode))[0]
            self.assertTrue(row["terminated"])
            self.assertFalse(row["truncated"])
            self.assertEqual(row["end_reason"],"success")
            self.assertAlmostEqual(row["reward"],.9)

    def test_annotation_identity_is_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp);(root/"data").mkdir();(root/"recording_context/frame_annotations").mkdir(parents=True)
            d = table();d.to_parquet(root/"data/episode_000000.parquet")
            (root/"recording_context/frame_annotations/episode_000000.json").write_text(json.dumps(dict(
                format="so101_frame_annotations/v1",source={},reviewed_labels=[],intervals=[])))
            with self.assertRaisesRegex(ValueError,"Stale annotation"):
                inspect_episode(root,dict(episode_index=0,length=len(d)),{},dict(data_path="data/episode_{episode_index:06d}.parquet"))

    def test_history_never_uses_future_or_crosses_gap(self):
        self.assertEqual(causal_history_frames(table(gaps=True),8,6,2),[4,6,8])

    def test_episode_split_has_no_neighbor_frame_leakage(self):
        records = [dict(episode_uid=f"ep{i//5}") for i in range(20)]
        train,val,_ = episode_split(records,42)
        self.assertFalse({records[i]["episode_uid"] for i in train}.intersection({records[i]["episode_uid"] for i in val}))


class LearnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_expectile_sign_and_terminal_target(self):
        self.assertAlmostEqual(float(expectile_loss(torch.tensor([2.,-2.]),.7)),2.)
        learner = OfflineLearner(NetworkConfig(4,chunk_size=2,width=16,max_tokens=3),LearnerConfig())
        m = learner.update(batch())
        self.assertAlmostEqual(m["target_mean"],.5)
        self.assertTrue(all(p.grad is None for p in learner.target_critic.parameters()))

    def test_ordered_independent_qs_and_frozen_features(self):
        torch.manual_seed(42)
        q = TwinChunkQ(NetworkConfig(4,chunk_size=2,width=16,max_tokens=3,architecture="attention"))
        b = batch();b["features"].requires_grad_(True)
        action = torch.randn_like(b["action"])
        q1,q2 = q(b["features"],b["feature_valid"],b["state"],action)
        rev = q(b["features"],b["feature_valid"],b["state"],action.flip(1))[0]
        self.assertGreater(float((q1-rev).abs().max()),1e-5)
        self.assertNotEqual(next(q.q1.parameters()).data_ptr(),next(q.q2.parameters()).data_ptr())
        (q1+q2).sum().backward()
        self.assertIsNone(b["features"].grad)

    def test_actual_iql_learns_success_value_and_roundtrips(self):
        torch.manual_seed(4)
        learner = OfflineLearner(NetworkConfig(4,chunk_size=2,width=16,max_tokens=3),LearnerConfig(lr=.003))
        b = batch()
        for _ in range(80): learner.update(b)
        with torch.no_grad():
            q = learner.critic(b["features"],b["feature_valid"],b["state"],b["action"])[0]
        self.assertGreater(float(q[::2].mean()-q[1::2].mean()),.5)
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"learner.pt";learner.save(path,dict(test=True))
            restored,prov=OfflineLearner.load(path)
            self.assertTrue(prov["test"])
            self.assertEqual(restored.steps,80)
            for key,value in learner.actor.state_dict().items():
                torch.testing.assert_close(value,restored.actor.state_dict()[key])

    def test_td3_bc_uses_action_gradient_and_delayed_actor(self):
        learner=OfflineLearner(NetworkConfig(4,chunk_size=2,width=16,max_tokens=3),LearnerConfig(algorithm="td3_bc"))
        before=copy.deepcopy(learner.actor.state_dict())
        self.assertNotIn("actor_loss",learner.update(batch()))
        self.assertIn("actor_loss",learner.update(batch()))
        self.assertTrue(any(not torch.equal(before[k],v) for k,v in learner.actor.state_dict().items()))

    def test_routed_pool_preserves_layer_and_token_order(self):
        v=torch.arange(32.).reshape(1,8,4)
        out=pool_routed_values([dict(v=v),dict(v=v+100)],(0,1),4)
        self.assertEqual(tuple(out.shape),(1,4,8))
        torch.testing.assert_close(out[0,0,:4],v[0,:2].mean(0))
        torch.testing.assert_close(out[0,0,4:],v[0,:2].mean(0)+100)


if __name__=="__main__": unittest.main()
