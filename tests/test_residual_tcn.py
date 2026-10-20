"""Preserve the strong baseline exactly and verify residual-head fitting units."""
import unittest

import torch

from src.model import make_forecaster
from src.residual_tcn import ResidualLiteTCN


class ResidualTCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_folded_skip_is_exact_dlinear_including_boundary_padding(self):
        cfg = {"L": 48, "H": 6, "columns": ["a", "b"]}
        tcn = make_forecaster(cfg).eval()
        linear = make_forecaster(cfg, {"architecture": "dlinear"}).eval()
        with torch.no_grad():
            for head in (linear.head, linear.trend):
                head.weight.normal_(std=.05)
                head.bias.normal_(std=.02)
            linear.revin.w.uniform_(.8, 1.2)
            linear.revin.b.normal_(std=.03)
        model = ResidualLiteTCN(cfg).initialize(tcn, linear)
        x = torch.randn(5, 48, 2)*2+4
        with torch.no_grad():
            torch.testing.assert_close(model(x), linear(x), rtol=1e-5, atol=2e-6)
        self.assertFalse(any(p.requires_grad for p in model.enc.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.head.parameters()))
        self.assertFalse(hasattr(model, "recon"))

    def test_residual_design_matches_inference_head_order_and_scaling(self):
        cfg = {"L": 48, "H": 6, "columns": ["a", "b"]}
        model = ResidualLiteTCN(cfg).initialize(make_forecaster(cfg), make_forecaster(cfg, {"architecture": "dlinear"}))
        x = torch.randn(5, 48, 2)
        weight = torch.randn_like(model.head_weights())*.05
        model.set_head(weight)
        torch.testing.assert_close(model.head_weights(), weight)
        with torch.no_grad():
            base, design = model.baseline_and_design(x)
            torch.testing.assert_close(model(x), base+torch.einsum("bcf,cfh->bhc", design, weight), rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
