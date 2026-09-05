import copy
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from ahawam.models.wan22.mot import LayerwiseChunkKVCacheEditor, MoT
from ahawam.models.wan22.action_dit import ActionDiT
from ahawam.models.wan22.wan_video_dit import WanVideoDiT
from ahawam.models.wan22.ovcr_adapter import (
    OVCRAdapter, configure_adapter, training_loss_with_anchor, capture_anchor_flow,
)
from ahawam.models.wan22.rl_token_readout import RLTokenReadout


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(1)

    def editor(self, mode=None):
        editor = LayerwiseChunkKVCacheEditor(query_dim=12, num_layers=2, num_heads=2, attn_hidden_dim=8)
        # Emulate a trained P0. A freshly zeroed base editor has no query Jacobian.
        for projection in editor.layer_delta_proj:
            nn.init.normal_(projection[-1].weight, std=.1)
        editor.requires_grad_(False)
        if mode:
            configure_adapter(editor, dict(mode=mode, rank=4, num_tokens=2))
        return editor

    def args(self, layer=0):
        return dict(layer_idx=layer, chunk_queries=torch.randn(2, 3, 4, 12),
                    first_frame_keys=torch.randn(2, 5, 8), first_frame_values=torch.randn(2, 5, 8))

    def test_identity_and_no_cache_mutation(self):
        for mode in OVCRAdapter.MODES:
            editor = self.editor(mode)
            args = self.args()
            originals = {key: value.clone() for key, value in args.items() if torch.is_tensor(value)}
            output = editor.build_layer_updated_cache(**args)
            with editor.adapter.disabled():
                base = editor.build_layer_updated_cache(**args)
            for key in ("k", "v"):
                self.assertTrue(torch.equal(output[key], base[key]), mode)
            for key, value in originals.items():
                self.assertTrue(torch.equal(value, args[key]))
            self.assertEqual(output["v"].shape, (2, 3, 5, 8))

    def test_adapter_learns_through_frozen_attention(self):
        for mode in OVCRAdapter.MODES:
            editor = self.editor(mode)
            args = self.args()
            output = editor.build_layer_updated_cache(**args)
            frozen_action_query = torch.randn(2, 3, 2, 8)
            attention = torch.softmax(frozen_action_query @ output["k"].transpose(-1, -2) / 8**.5, -1)
            action = attention @ output["v"]
            loss = (action - torch.randn_like(action)).square().mean()
            loss.backward()
            grads = [p.grad for p in editor.adapter.parameters() if p.grad is not None]
            self.assertTrue(any(g.abs().sum() > 0 for g in grads), mode)
            self.assertTrue(all(p.grad is None for name, p in editor.named_parameters() if not name.startswith("adapter.")))
            torch.optim.SGD(editor.adapter.parameters(), lr=.3).step()
            changed = editor.build_layer_updated_cache(**args)
            self.assertFalse(torch.equal(changed["v"], output["v"]), mode)
            if mode == "value":
                self.assertTrue(torch.equal(changed["k"], output["k"]))
            editor.adapter.scale = 0
            bypassed = editor.build_layer_updated_cache(**args)
            with editor.adapter.disabled():
                base = editor.build_layer_updated_cache(**args)
            self.assertTrue(torch.equal(bypassed["v"], base["v"]))

    def test_native_cache_and_real_action_dit_gradient(self):
        for mode in OVCRAdapter.MODES:
            def expert():
                return ActionDiT(hidden_dim=16, action_dim=6, ffn_dim=32, text_dim=12,
                                 freq_dim=8, eps=1e-6, num_heads=2, attn_head_dim=4,
                                 num_layers=2, action_chunk_size=2, use_gradient_checkpointing=True)
            mot = MoT({"video": expert(), "action": expert()})
            mot.chunk_kv_cache_editor = self.editor(mode)
            mot.requires_grad_(False)
            mot.chunk_kv_cache_editor.adapter.requires_grad_(True)
            mot.train()
            cache = [dict(k=torch.randn(2, 9, 8), v=torch.randn(2, 9, 8)) for _ in range(2)]
            saved = copy.deepcopy(cache)
            q = torch.randn(2, 3, 4, 12)
            updated = mot.build_chunk_updated_video_kv_cache(video_kv_cache=cache, chunk_queries=q,
                                                              video_tokens_per_frame=5, chunk_index=1)
            direct = mot.chunk_kv_cache_editor.build_layer_updated_cache(
                layer_idx=0, chunk_queries=q, first_frame_keys=cache[0]["k"][:, :5],
                first_frame_values=cache[0]["v"][:, :5])
            torch.testing.assert_close(updated[0]["v"], direct["v"][:, 1])
            action = mot.mixtures["action"]
            pre = action.pre_dit(torch.randn(2, 2, 6), torch.ones(2), torch.randn(2, 3, 12), chunk_size=2)
            out = mot.forward_action_with_video_cache(
                action_tokens=pre["tokens"], action_freqs=pre["freqs"], action_t_mod=pre["t_mod"],
                action_context_payload=dict(context=pre["context"], mask=pre["context_mask"]),
                video_kv_cache=updated, video_seq_len=5, attention_mask=torch.ones(7, 7, dtype=torch.bool))
            pred = action.post_dit(out, pre)
            pred.square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in mot.chunk_kv_cache_editor.adapter.parameters()), mode)
            self.assertTrue(all(p.grad is None for p in action.parameters()))
            for a, b in zip(cache, saved):
                self.assertTrue(torch.equal(a["k"], b["k"]))
                self.assertTrue(torch.equal(a["v"], b["v"]))

    def test_state_dict_roundtrip_and_layer_selection(self):
        editor = self.editor()
        original = copy.deepcopy(editor.state_dict())
        adapter = configure_adapter(editor, dict(mode="value", rank=3, layers=[1]))
        nn.init.normal_(adapter.blocks["1"].up.weight)
        args = self.args(1)
        expected = editor.build_layer_updated_cache(**args)
        clone = self.editor()
        configure_adapter(clone, adapter.config)
        clone.load_state_dict(editor.state_dict(), strict=True)
        self.assertTrue(torch.equal(expected["v"], clone.build_layer_updated_cache(**args)["v"]))
        self.assertTrue(all(torch.equal(value, editor.state_dict()[key]) for key, value in original.items()))
        for invalid in ([2], [-1], [], [0, 0]):
            with self.assertRaises(ValueError):
                configure_adapter(self.editor(), dict(layers=invalid))

    def test_native_joint_training_path_with_frozen_video(self):
        for mode in OVCRAdapter.MODES:
            video = WanVideoDiT(hidden_dim=24, in_dim=4, ffn_dim=32, out_dim=4,
                                text_dim=12, freq_dim=8, eps=1e-6, patch_size=(1, 1, 1),
                                num_heads=2, attn_head_dim=12, num_layers=2, has_image_input=False,
                                seperated_timestep=True)
            action = ActionDiT(hidden_dim=16, action_dim=6, ffn_dim=32, text_dim=12,
                               freq_dim=8, eps=1e-6, num_heads=2, attn_head_dim=12,
                               num_layers=2, action_chunk_size=2, use_gradient_checkpointing=True)
            mot = MoT({"video": video, "action": action})
            mot.configure_chunk_kv_cache_editor(query_dim=12)
            for projection in mot.chunk_kv_cache_editor.layer_delta_proj:
                nn.init.normal_(projection[-1].weight, std=.1)
            mot.eval().requires_grad_(False)
            mot.training = True
            adapter = configure_adapter(mot.chunk_kv_cache_editor, dict(mode=mode, rank=4))
            context = torch.randn(2, 3, 12)
            vp = video.pre_dit(torch.randn(2, 4, 2, 2, 2), torch.ones(2), context,
                               fuse_vae_embedding_in_latents=True)
            ap = action.pre_dit(torch.randn(2, 4, 6), torch.ones(2), context, chunk_size=2)
            out = mot.forward_prior_action_with_chunk_updated_kv(
                video_tokens=vp["tokens"], video_freqs=vp["freqs"], video_t_mod=vp["t_mod"],
                video_context_payload=dict(context=vp["context"], mask=vp["context_mask"]),
                video_attention_mask=torch.ones(8, 8, dtype=torch.bool).tril(),
                action_tokens=ap["tokens"], action_freqs=ap["freqs"], action_t_mod=ap["t_mod"],
                action_context_payload=dict(context=ap["context"], mask=ap["context_mask"]),
                chunk_queries=torch.randn(2, 2, 4, 12), video_tokens_per_frame=4, action_chunk_size=2)
            action.post_dit(out["action_prior"], ap).square().mean().backward()
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters()), mode)
            self.assertTrue(all(p.grad is None for p in video.parameters()))
            self.assertTrue(all(p.grad is None for p in action.parameters()))

    def test_native_checkpoint_metadata_auto_installs_adapter(self):
        from ahawam.models.wan22.ahawam_chunk_base import AHAWAMChunkBase
        # Exercise the actual serializer/loader with tiny modules, no VAE assets.
        class Harness(nn.Module):
            save_checkpoint = AHAWAMChunkBase.save_checkpoint
            load_checkpoint = AHAWAMChunkBase.load_checkpoint
            _adapt_action_branch_embedding_state = AHAWAMChunkBase._adapt_action_branch_embedding_state
            def __init__(self, editor):
                super().__init__()
                self.mot = nn.Module()
                self.mot.chunk_kv_cache_editor = editor
                self.action_obs_visual_proj = nn.Linear(2, 2)
                self.chunk_obs_query_encoder = nn.Linear(2, 2)
                self.proprio_encoder = None
                self.torch_dtype = torch.float32
        for mode in OVCRAdapter.MODES:
            src = Harness(self.editor(mode))
            for block in src.mot.chunk_kv_cache_editor.adapter.blocks.values():
                nn.init.normal_(block.up.weight)
            dest = Harness(self.editor())
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "weights.pt"
                src.save_checkpoint(path, step=12)
                dest.load_checkpoint(path)
                for key, value in src.state_dict().items():
                    self.assertTrue(torch.equal(value, dest.state_dict()[key]))
                payload = torch.load(path, weights_only=True)
                key = next(k for k in payload["mot"] if ".adapter." in k)
                del payload["mot"][key]
                torch.save(payload, path)
                with self.assertRaises(RuntimeError):
                    dest.load_checkpoint(path)
                plain = Harness(self.editor())
                plain.save_checkpoint(path)
                dest.load_checkpoint(path)
                args = self.args()
                torch.testing.assert_close(
                    plain.mot.chunk_kv_cache_editor.build_layer_updated_cache(**args)["v"],
                    dest.mot.chunk_kv_cache_editor.build_layer_updated_cache(**args)["v"],
                    rtol=0, atol=0)

    def test_anchor_uses_same_noise_masks_demo_and_restores_rng(self):
        editor = self.editor("value")
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.mot = nn.Module()
                self.mot.chunk_kv_cache_editor = editor
                self.ovcr_anchor_weight = 2.0
        model = Model()
        noises = []
        def base(sample, tiled=False):
            pred = torch.rand(2, 3, 6)
            noises.append(pred.clone())
            if editor.adapter.enabled:
                # Known deviation: demo valid positions +2; human positions +9.
                pred = pred + torch.tensor([2., 9.])[:, None, None]
            capture_anchor_flow(model, pred, torch.tensor([[False, False, True], [False, False, False]]))
            return pred.sum() * 0, {}
        state = torch.get_rng_state()
        expected_random = torch.rand(2, 3, 6)
        after_one = torch.get_rng_state()
        torch.set_rng_state(state)
        loss, metrics = training_loss_with_anchor(model, dict(data_source=torch.tensor([0, 1])), base)
        self.assertTrue(torch.equal(noises[0], expected_random))
        self.assertTrue(torch.equal(noises[0], noises[1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), after_one))
        self.assertEqual(loss.item(), 8.)
        self.assertEqual(metrics["loss_anchor"], 4.)
        self.assertFalse(hasattr(model, "_ovcr_anchor_context"))
        self.assertTrue(editor.adapter.enabled)

    def test_readout_padding_detach_and_learning(self):
        model = RLTokenReadout(12, hidden_dim=8, num_heads=2, num_layers=1, max_tokens=4)
        inputs = torch.randn(2, 4, 12, requires_grad=True)
        valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
        loss, z, pred = model(inputs, valid)
        loss.backward()
        self.assertIsNone(inputs.grad)
        self.assertGreater(model.query.grad.abs().sum().item(), 0)
        self.assertEqual(z.shape, (2, 8))
        altered = inputs.detach().clone()
        altered[~valid] = float("nan")
        loss2, z2, _ = model(altered, valid)
        torch.testing.assert_close(z, z2)
        torch.testing.assert_close(loss, loss2)
        with self.assertRaises(ValueError):
            model(inputs, torch.zeros_like(valid))


if __name__ == "__main__":
    unittest.main()
