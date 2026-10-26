"""Loss/gradient contracts for native Action DiT policy extraction."""
from pathlib import Path
import tempfile
import unittest

import torch
from torch import nn

from ahawam.offline_rl.flow_actor import (LoRALinear, adapter_state, install_action_lora,
    action_editor_cache, load_action_lora, restore_adapter, weighted_flow_loss)


class FlowActorTests(unittest.TestCase):
    def test_editor_fp32_outputs_are_quantized_like_serving_before_action_attention(self):
        original = torch.randn(1, 120, 12, requires_grad=True)
        result = action_editor_cache([dict(k=original, v=original + .1)], torch.bfloat16)
        self.assertEqual(result[0]["k"].shape, (1, 1, 120, 12))
        self.assertEqual(result[0]["k"].dtype, torch.bfloat16)
        self.assertFalse(result[0]["k"].requires_grad)
        torch.testing.assert_close(result[0]["k"][:, 0], original.detach().bfloat16(), rtol=0, atol=0)

    def test_weighted_loss_is_per_chunk_detached_and_masks_padding(self):
        pred = torch.tensor([[[1.], [100.]], [[3.], [3.]]], requires_grad=True)
        weight = torch.tensor([2., 4.], requires_grad=True)
        loss, unweighted = weighted_flow_loss(pred, torch.zeros_like(pred),
            torch.tensor([[True, False], [True, True]]), torch.tensor([3., 5.]), weight)
        torch.testing.assert_close(unweighted, torch.tensor([3., 45.]))
        self.assertEqual(float(loss), 93.)
        loss.backward()
        self.assertIsNone(weight.grad)
        self.assertEqual(float(pred.grad[0, 1]), 0.)
        torch.testing.assert_close(pred.grad[:, 0, 0], torch.tensor([6., 30.]))

    def test_scalar_timestep_for_microbatch_one_and_invalid_shapes(self):
        pred = torch.ones(1, 8, 6)
        loss, _ = weighted_flow_loss(pred, pred * 0, torch.ones(1, 8, dtype=torch.bool),
            torch.tensor(2.), torch.tensor([3.]))
        self.assertEqual(float(loss), 6.)
        with self.assertRaises(ValueError):
            weighted_flow_loss(pred, pred * 0, torch.ones(1, 8, dtype=torch.bool),
                torch.ones(1), torch.ones(1, 1))

    def test_lora_initially_exact_then_changes_only_adapter_and_roundtrips(self):
        torch.manual_seed(9)
        base = nn.Linear(4, 3)
        frozen = base.weight.detach().clone()
        x = torch.randn(2, 4)
        expected = base(x).detach()
        layer = LoRALinear(base, 2, 2.)
        torch.testing.assert_close(layer(x), expected, rtol=0, atol=0)
        optimizer = torch.optim.AdamW([p for p in layer.parameters() if p.requires_grad], lr=.1)
        for _ in range(2):
            optimizer.zero_grad()
            layer(x).square().mean().backward()
            optimizer.step()
        torch.testing.assert_close(base.weight, frozen, rtol=0, atol=0)
        self.assertIsNone(base.weight.grad)
        self.assertFalse(torch.equal(layer(x), expected))
        state = adapter_state(layer)
        output = layer(x).detach()
        with torch.no_grad():
            layer.lora_b.zero_()
        restore_adapter(layer, state)
        torch.testing.assert_close(layer(x), output, rtol=0, atol=0)

    def test_install_freezes_planner_and_all_original_action_parameters(self):
        class Dummy(nn.Module):
            def __init__(self):
                super().__init__()
                self.planner = nn.Linear(4, 4)
                self.action_expert = nn.Module()
                self.action_expert.blocks = nn.ModuleList([nn.Module()])
                self.action_expert.blocks[0].self_attn = nn.ModuleDict(
                    {k: nn.Linear(4, 4) for k in ("q", "k", "v", "o")})
        model = Dummy()
        names = install_action_lora(model, 2, 2.)
        self.assertEqual(len(names), 4)
        self.assertEqual(sum(p.requires_grad for p in model.parameters()), 8)
        self.assertTrue(all(not p.requires_grad for p in model.planner.parameters()))
        with self.assertRaises(ValueError):
            install_action_lora(model, 2, 2.)

    def test_adapter_shape_validation_does_not_broadcast(self):
        layer = LoRALinear(nn.Linear(4, 3), 2, 2.)
        saved = adapter_state(layer)
        saved["lora_b"] = torch.zeros(1, 2)
        with self.assertRaises(ValueError):
            restore_adapter(layer, saved)

    def test_loading_for_serving_checks_base_and_reproduces_adapter(self):
        def make():
            model = nn.Module()
            model.action_expert = nn.Module()
            model.action_expert.blocks = nn.ModuleList([nn.Module()])
            model.action_expert.blocks[0].self_attn = nn.ModuleDict(
                {k: nn.Linear(4, 4) for k in ("q", "k", "v", "o")})
            return model
        model = make()
        baseline = {k: v.clone() for k, v in model.state_dict().items()}
        targets = install_action_lora(model, 2, 2.)
        with torch.no_grad():
            model.action_expert.blocks[0].self_attn.q.lora_b.normal_()
        x = torch.randn(2, 4)
        expected = model.action_expert.blocks[0].self_attn.q(x).detach()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            torch.save(dict(format="ahawam_iql_flow_lora/v1", step=2, adapter=adapter_state(model),
                provenance=dict(base_checkpoint_sha256="expected", rank=2, alpha=2., targets=targets)), path)
            other = make()
            other.load_state_dict(baseline)
            with self.assertRaises(ValueError):
                load_action_lora(other, path, "wrong")
            load_action_lora(other, path, "expected")
            torch.testing.assert_close(other.action_expert.blocks[0].self_attn.q(x), expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
