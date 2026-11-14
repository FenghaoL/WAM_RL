"""Native FastWAM adaptation, supervision boundaries and storage contract."""
from pathlib import Path
import tempfile
import unittest

import torch
from omegaconf import OmegaConf

from scripts.so101_pipeline.fastwam_cylinder.model import adapted_state,masked_video_losses
from scripts.so101_pipeline.fastwam_cylinder.trainer import FastWAMTrainer


class FastWAMCylinderTests(unittest.TestCase):
    def test_wandb_bridge_recovers_wrapped_logs_and_skips_partial_records(self):
        from scripts.so101_pipeline.fastwam_cylinder.wandb_bridge import training_records
        log='''11/05 [17:00:00] INFO epoch=0 step=1/6572 trainer.py:1248
          loss=1.2500 loss_action=0.7500 loss_video=0.5000
          grad_norm=2.0000 lr=6.09e-08 speed=0.14 step/s, 1.73 samples/s eta=11:00:00
          timing(data=0.010s fwd=0.500s bwd=4.700s opt=0.000s)
11/05 [17:00:07] INFO epoch=0 step=2/6572 loss=1.0 loss_action=0.5 loss_video=0.5
          grad_norm=1.0 lr=1.00e-5 speed=0.14 step/s
'''
        records=training_records(log)
        self.assertEqual(list(records),[1])
        self.assertEqual(records[1]['train/loss_action'],.75)
        self.assertAlmostEqual(records[1]['train/lr'],6.09e-8)
        self.assertEqual(records[1]['performance/backward_time_s'],4.7)

    def test_wandb_native_logs_train_and_val_at_same_custom_step(self):
        from types import SimpleNamespace
        entries=[]
        obj=SimpleNamespace(global_step=42,wandb_run=SimpleNamespace(log=lambda payload:entries.append(payload)))
        FastWAMTrainer._wandb_log(obj,{'train/loss':1.})
        FastWAMTrainer._wandb_log(obj,{'eval/val_loss':2.})
        self.assertEqual([row['global_step'] for row in entries],[42,42])
        self.assertIn('train/loss',entries[0]);self.assertIn('eval/val_loss',entries[1])

    def test_action_only_freezes_video_and_retains_proprio_updates(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dit=torch.nn.Module()
                self.dit.mixtures=torch.nn.ModuleDict(dict(video=torch.nn.Linear(4,4),action=torch.nn.Linear(4,4)))
                self.dit.action_branch_embedding=torch.nn.Parameter(torch.zeros(2,4))
                self.proprio_encoder=torch.nn.Linear(6,4)
            @property
            def mot(self):return self.dit
        model=Model()
        FastWAMTrainer._apply_dit_only_train_mode(model,freeze_video_dit=True)
        self.assertTrue(all(not p.requires_grad for p in model.dit.mixtures['video'].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.dit.mixtures['action'].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.proprio_encoder.parameters()))
        self.assertFalse(model.dit.action_branch_embedding.requires_grad)

    def test_cross_robot_adaptation_only_allows_action_interfaces(self):
        target={"mixtures.action.action_encoder.weight":torch.zeros(4,6),"backbone":torch.zeros(4,4),
            "action_branch_embedding":torch.zeros(2,4)}
        source={"mixtures.action.action_encoder.weight":torch.zeros(4,7),"backbone":torch.ones(4,4)}
        state,skipped=adapted_state(source,target,True)
        self.assertEqual(set(skipped),{"mixtures.action.action_encoder.weight"})
        self.assertEqual(set(state),{"backbone"})
        with self.assertRaises(ValueError):adapted_state(source,target,False)
        with self.assertRaises(ValueError):adapted_state(dict(source,backbone=torch.ones(5,4)),target,True)
        with self.assertRaises(ValueError):adapted_state({"mixtures.action.action_encoder.weight":source["mixtures.action.action_encoder.weight"]},target,True)

    def test_masked_video_loss_excludes_partial_labels_and_preserves_scale(self):
        loss=torch.tensor([1.,9.],requires_grad=True)
        out=masked_video_losses(loss,torch.tensor([True,False])).mean()
        self.assertEqual(float(out),1.)
        out.backward();torch.testing.assert_close(loss.grad,torch.tensor([1.,0.]))
        self.assertEqual(float(masked_video_losses(loss,[False,False]).mean()),0.)
        with self.assertRaises(ValueError):masked_video_losses(loss,[True])

    def test_weights_only_latest_retention_is_bounded(self):
        from types import SimpleNamespace
        class Model:
            def save_checkpoint(self,path,optimizer=None,step=None):
                if optimizer is not None:raise AssertionError("optimizer must never be saved")
                torch.save(dict(mot={"x":torch.ones(1)},step=step),path)
        with tempfile.TemporaryDirectory() as temp:
            m=Model();obj=SimpleNamespace(weights_dir=temp,global_step=1,
                accelerator=SimpleNamespace(unwrap_model=lambda x:x),model=m)
            FastWAMTrainer._save_weights_checkpoint(obj,"step_000001")
            obj.global_step=2
            path=FastWAMTrainer._save_weights_checkpoint(obj,"step_000002")
            self.assertEqual([p.name for p in Path(temp).iterdir()],["step_000002.pt"])
            self.assertNotIn("optimizer",torch.load(path,weights_only=True))

    def test_four_configs_share_data_and_branch_only_from_franka_sft(self):
        root=Path("runs/fastwam_cylinder_20261105/configs")
        if not root.exists():self.skipTest("Generated run configurations not present")
        configs={p.stem:OmegaConf.load(p) for p in root.glob("*.yaml")}
        self.assertEqual(len(configs),4)
        a,b=configs["franka_action_only"],configs["franka_full_vidext"]
        self.assertEqual(OmegaConf.to_container(a.data),OmegaConf.to_container(b.data))
        self.assertEqual(a.init_checkpoint,b.init_checkpoint)
        self.assertTrue(a.freeze_video_dit);self.assertFalse(b.freeze_video_dit)
        self.assertEqual(a.model.loss.lambda_video,0.)
        for c in configs.values():
            self.assertIsNone(c.resume);self.assertFalse(c.zero2_save_training_state)
            self.assertEqual(c.data.train.max_action_offset,0);self.assertEqual(c.data.train.num_history_frames,0)
            self.assertEqual(c.data.train.action_horizon,48)


if __name__=="__main__":unittest.main()
