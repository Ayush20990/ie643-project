"""Checks for observed-history boundaries and exact adaptation loss."""
import io
import unittest

import numpy as np
import torch

from src.data import make_windows
from src.refinement import reconstruct_inputs
from src.refined_models import (ChannelRidge, NORMALIZATIONS, RefinedForecaster,
                                channel_features, convex_weights, ridge_path,
                                sufficient_statistics)


class RefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_stitch_uses_only_inputs_and_keeps_stocks_separate(self):
        for stride in (1, 2):
            arrays = [np.arange(100, dtype=np.float32)[:, None],
                      1000 + np.arange(100, dtype=np.float32)[:, None]]
            x, original_future = make_windows(arrays, 20, 5, stride)
            count = len(x) // 2
            observed = reconstruct_inputs(x, [count, count], stride)
            for i, a in enumerate(observed):
                np.testing.assert_array_equal(a, arrays[i][:len(a)])
                self.assertLess(float(a.max()), float(original_future[i * count:(i + 1) * count].max()))
            hx, hy = make_windows(observed, 20, 5, stride)
            self.assertFalse(bool(((hy < 1000).any((1, 2)) & (hy >= 1000).any((1, 2))).any()))
            changed = x.clone()
            changed[count + 1, 0, 0] += 1
            with self.assertRaises(ValueError):
                reconstruct_inputs(changed, [count, count], stride)

    def test_normal_equations_minimize_actual_forecast_loss(self):
        torch.manual_seed(2)
        x = torch.randn(90, 24, 2) * torch.rand(90, 1, 2).add(.2)
        y = torch.randn(90, 4, 2)
        penalty = .13
        for mode in NORMALIZATIONS:
            with self.subTest(mode=mode):
                g, r = sufficient_statistics(x, y, mode)
                ref = torch.randn_like(r).float() * .03
                weight = next(ridge_path(g, r, [penalty], ref))[1].requires_grad_()
                f, offset, scale = channel_features(x, mode)
                p = (torch.einsum("bcf,cfh->bch", f, weight) * scale[:, :, None] + offset[:, :, None]).transpose(1, 2)
                # Normal equations average over examples; sum over output horizons.
                loss = (p - y).square().sum((1, 2)).mean() + penalty * (weight - ref).square().sum()
                loss.backward()
                self.assertLess(float(weight.grad.abs().max()), 2e-5)

    def test_hybrid_roundtrip_and_convex_baseline_selection(self):
        cfg = {"L": 48, "H": 6, "columns": ["a", "b"]}
        model = RefinedForecaster(cfg, {"architecture": "patch_mlp"}).eval()
        model.linear.mode.copy_(torch.tensor([1, 3]))
        model.linear.weight.normal_(std=.01)
        model.mix.copy_(torch.tensor([[.5, .25, .25, 0], [0, 0, 0, 1.]]))
        x = torch.randn(4, 48, 2)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        loaded = RefinedForecaster(cfg, {"architecture": "patch_mlp"}).eval()
        loaded.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(loaded(x), model(x))
        np.testing.assert_array_equal(model(x).detach().numpy()[..., 1], 0)
        truth = np.zeros((20, 6, 2), np.float32)
        w = convex_weights([np.ones_like(truth), truth], truth)
        self.assertTrue((w >= 0).all())
        np.testing.assert_allclose(w.sum(-1), 1, atol=1e-6)
        self.assertGreater(float(w[:, 1].min()), .99)


if __name__ == "__main__":
    unittest.main()
