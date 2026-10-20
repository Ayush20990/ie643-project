"""Regression checks for leakage, gradient flow, loading and input validation."""
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.improve import CANDIDATES, adapt_history
from src.inference import read_forecast_csv
from src.losses import spectral_align
from src.metrics import gap_closure
from src.model import make_forecaster
from src.data import chrono_split, make_windows, Scaler


class ForecastingRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = {"L": 48, "H": 6, "columns": ["a", "b"]}

    def test_architectures_roundtrip_and_input_pretext_is_causal(self):
        for name, spec in CANDIDATES.items():
            with self.subTest(architecture=name):
                torch.manual_seed(0)
                model = make_forecaster(self.cfg, spec).eval()
                x = torch.randn(3, 48, 2, requires_grad=True)
                self.assertEqual(tuple(model(x).shape), (3, 6, 2))
                model.inner_forecast_loss(x).backward()
                self.assertEqual(float(x.grad[:, -6:].abs().sum()), 0)
                self.assertGreater(float(x.grad[:, :-6].abs().sum()), 0)
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                loaded = make_forecaster(self.cfg, spec).eval()
                loaded.load_state_dict(torch.load(buffer, weights_only=True))
                torch.testing.assert_close(model(x), loaded(x))

    def test_spectral_loss_updates_encoder(self):
        model = make_forecaster(self.cfg)
        xs, xt = torch.randn(8, 48, 2), 2 * torch.randn(8, 48, 2)
        fs = model.enc(model.revin.norm(xs)[0].transpose(1, 2)).transpose(1, 2)
        ft = model.enc(model.revin.norm(xt)[0].transpose(1, 2)).transpose(1, 2)
        loss = spectral_align(fs, ft)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.enc.parameters()), 0)

    def test_adaptation_falls_back_when_source_validation_deteriorates(self):
        model = make_forecaster(self.cfg)
        x, y = torch.randn(8, 48, 2), torch.randn(8, 6, 2)
        src = {"train": (x, y), "val": (x, y)}
        with patch("src.improve.validation_mse", side_effect=[1.0, 1.2]), \
                patch("src.improve.inner_score", side_effect=[1.0, 0.1]):
            adapted, history, score, epoch = adapt_history(model, src, x, x,
                {"inner": 0.5, "anchor": 0.001}, 1, 0, lambda _: None)
        self.assertFalse(history[-1]["eligible"])
        self.assertEqual((score, epoch), (1.0, 0))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, adapted.state_dict()[key], rtol=0, atol=0)

    def test_csv_rejects_lookahead_imputation_and_mixed_series(self):
        bad_files = [
            "date,a,b\n2020-01-01,1,2\n2020-01-02,,3\n",
            "date,a,b\n2020-01-02,1,2\n2020-01-01,2,3\n",
            "date,a,b\n2020-01-01,1,2\n2020-01-01,2,3\n",
            "date,series,a,b\n2020-01-01,A,1,2\n2020-01-02,B,2,3\n",
            "a,b\ninf,2\n",
        ]
        for value in bad_files:
            with self.subTest(csv=value), self.assertRaises(ValueError):
                read_forecast_csv(io.StringIO(value), "energy", ["a", "b"])
        df = read_forecast_csv(io.StringIO("date,a,b\n2020-01-01,1,2\n2020-01-02,2,3\n"), "energy", ["a", "b"])
        self.assertEqual(df.shape, (2, 2))

    def test_gap_closure_undefined_when_reference_does_not_improve(self):
        self.assertTrue(np.isnan(gap_closure(1, .8, 1.1)))
        self.assertAlmostEqual(gap_closure(1, .8, .6), .5)

    def test_split_futures_do_not_cross_chronological_boundaries(self):
        array = np.arange(100, dtype=np.float32)[:, None]
        parts = chrono_split([array], (.6, .2, .2), L=10)
        windows = {key: make_windows(value, 10, 5) for key, value in parts.items()}
        self.assertEqual(float(windows["train"][1].max()), 59)
        self.assertEqual(float(windows["val"][1].min()), 60)
        self.assertEqual(float(windows["val"][1].max()), 79)
        self.assertEqual(float(windows["test"][1].min()), 80)
        self.assertEqual(float(windows["test"][1].max()), 99)
        scaler = Scaler.fit(parts["train"])
        self.assertAlmostEqual(float(scaler.mean[0]), 29.5)


if __name__ == "__main__":
    unittest.main()
