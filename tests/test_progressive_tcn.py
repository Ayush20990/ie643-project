"""Unfreezing must update the encoder without changing the strong linear skip."""
import inspect
import unittest

import torch

from src.model import make_forecaster
from src.progressive_tcn import ProgressiveLiteTCN, input_pair, forecast_objective
from src.progressive_train import adapt_fit


class ProgressiveTCNTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def model(self):
        cfg = {"L": 48, "H": 6, "columns": ["a", "b"]}
        return ProgressiveLiteTCN(cfg).initialize(make_forecaster(cfg), make_forecaster(cfg, {"architecture": "dlinear"}))

    def test_staged_updates_preserve_skip_and_normalizer(self):
        torch.manual_seed(43)
        model = self.model()
        initial = {n: p.detach().clone() for n, p in model.named_parameters()}
        opt = torch.optim.AdamW(model.optimizer_groups(), weight_decay=1e-4)
        x, y = torch.randn(12, 48, 2), torch.randn(12, 6, 2)
        for r in (1, 2, 5, 9, 10):
            model.train()
            stage = model.set_stage("progressive", r)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            opt.zero_grad(set_to_none=True)
            forecast_objective(model(x), y, 1).backward()
            opt.step()
            if stage == "head":
                self.assertTrue(all(torch.equal(p, before[n]) for n, p in model.named_parameters() if n.startswith(("enc.", "proj."))))
            if stage == "projection":
                self.assertTrue(all(torch.equal(p, before[n]) for n, p in model.named_parameters() if n.startswith("enc.")))
        self.assertTrue(any(not torch.equal(p, initial[n]) for n, p in model.named_parameters() if n.startswith("enc.")))
        self.assertTrue(any(not torch.equal(p, initial[n]) for n, p in model.named_parameters() if n.startswith("proj.")))
        self.assertTrue(all(torch.equal(p, initial[n]) for n, p in model.named_parameters() if n.startswith(("linear.", "revin."))))
        model.set_stage("head_only", 24)
        self.assertFalse(any(p.requires_grad for p in model.enc.parameters()))

    def test_pretext_excludes_tail_and_other_windows(self):
        model = self.model().eval()
        x = torch.randn(2, 48, 2, requires_grad=True)
        prefix, truth = input_pair(x, 6)
        pred = model(prefix)
        altered = x.detach().clone()
        altered[0, -6:] += 100
        altered[1] -= 100
        p2, _ = input_pair(altered, 6)
        torch.testing.assert_close(pred[0], model(p2)[0])
        pred[0].sum().backward()
        self.assertEqual(x.grad[0, -6:].abs().sum().item(), 0.)
        self.assertEqual(x.grad[1].abs().sum().item(), 0.)
        self.assertFalse(truth.requires_grad)

    def test_adapter_api_excludes_target_forecasting_labels(self):
        names = list(inspect.signature(adapt_fit).parameters)
        self.assertEqual([n for n in names if n.startswith("target")], ["target_train_x", "target_val_x"])


if __name__ == "__main__":
    unittest.main()
