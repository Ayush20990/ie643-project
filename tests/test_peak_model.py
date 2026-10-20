"""Clock alignment, within-input isolation, causal convolutions and peak gradients."""
import io
import unittest

import numpy as np
import pandas as pd
import torch

from src.peak_model import (CausalResidualBlock, PeakLiteTCN, calendar_features,
                            forecast_loss, pack_calendar, pretext)
from src.inference import forecast_batches
from src.peak_calibration import CalibratedLiteTCN, correct_prediction


class PeakModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = {"L": 72, "H": 24, "season": 24, "columns": ["temperature", "load"]}

    def test_clock_rotation_matches_known_future_dates(self):
        dates = pd.date_range("2024-12-29 09:00", periods=96, freq="h")
        clock = calendar_features(dates)
        model = PeakLiteTCN(self.cfg)
        computed = model.future_clock(torch.from_numpy(clock[71:72]))[0].detach().numpy()
        np.testing.assert_allclose(computed, clock[72:96], atol=2e-6)
        np.testing.assert_allclose(calendar_features(dates.as_unit("us")), clock, atol=1e-7)

    def test_pretext_excludes_withheld_values_and_supports_real_prefix(self):
        dates = pd.date_range("2024-01-01", periods=72, freq="h").to_numpy()[None].repeat(2, 0)
        values = np.random.default_rng(0).normal(size=(2, 72, 2)).astype(np.float32)
        packed = torch.tensor(pack_calendar(values, dates), requires_grad=True)
        model = PeakLiteTCN(self.cfg).eval()
        prefix, observed = pretext(packed, 24, 2)
        self.assertEqual(prefix.shape[1], 48)
        self.assertFalse(observed.requires_grad)
        prediction = model(prefix)
        self.assertEqual(tuple(prediction.shape), (2, 24, 2))
        prediction[0].sum().backward()
        self.assertEqual(float(packed.grad[0, -24:].abs().sum()), 0.)
        self.assertEqual(float(packed.grad[1].abs().sum()), 0.)
        self.assertGreater(float(packed.grad[0, :-24].abs().sum()), 0.)

    def test_convolution_preserves_resolution_and_is_causal(self):
        block = CausalResidualBlock(8, 4, 0).eval()
        x = torch.randn(2, 8, 48)
        changed = x.clone()
        changed[:, :, 25:] = 100
        self.assertEqual(block(x).shape, x.shape)
        torch.testing.assert_close(block(x)[:, :, :25], block(changed)[:, :, :25])

    def test_shape_loss_penalizes_flat_forecast_and_roundtrip(self):
        truth = torch.zeros(1, 24, 2)
        truth[:, 10, 0] = 4
        truth[:, 16, 0] = -3
        flat = torch.zeros_like(truth, requires_grad=True)
        loss = forecast_loss(flat, truth, key=0, shape=True)
        self.assertGreater(float(loss.detach()), float(forecast_loss(flat, truth, 0, False).detach()))
        loss.backward()
        self.assertGreater(float(flat.grad.abs().sum()), 0.)
        model = PeakLiteTCN(self.cfg, calendar=False).eval()
        x = torch.randn(2, 72, 2)
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        loaded = PeakLiteTCN(self.cfg, calendar=False).eval()
        loaded.load_state_dict(torch.load(buffer, weights_only=True))
        torch.testing.assert_close(model(x), loaded(x))

    def test_inference_uses_supplied_clock_and_rejects_missing_hours(self):
        model = PeakLiteTCN(self.cfg).eval()
        with torch.no_grad():
            model.temporal_decoder[-1].weight.normal_(std=.02)
        times = pd.date_range("2024-01-01", periods=72, freq="h").to_numpy()[None]
        x = np.random.default_rng(4).normal(size=(1, 72, 2)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(pack_calendar(x, times))).numpy()
        np.testing.assert_allclose(forecast_batches(model, x, dates=times), expected, atol=1e-6)
        with self.assertRaises(ValueError):
            forecast_batches(model, x)
        bad_times = times.copy()
        bad_times[:, -1] += np.timedelta64(1, "h")
        with self.assertRaises(ValueError):
            forecast_batches(model, x, dates=bad_times)

    def test_encoder_receives_forecast_gradient_after_residual_head_initialization(self):
        model = PeakLiteTCN(self.cfg, calendar=False)
        x, y = torch.randn(4, 72, 2), torch.randn(4, 24, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        for _ in range(2):
            optimizer.zero_grad()
            forecast_loss(model(x), y, 0, shape=True).backward()
            encoder_gradient = sum(float(p.grad.abs().sum()) for p in model.encoder.parameters() if p.grad is not None)
            optimizer.step()
        self.assertGreater(encoder_gradient, 0.)

    def test_recent_residual_correction_is_inside_input_and_decays(self):
        prediction = torch.zeros(2, 24, 2)
        history_prediction, history_observed = torch.zeros(2, 12, 2), torch.ones(2, 12, 2)
        corrected = correct_prediction(prediction, history_prediction, history_observed,
                                       kind="level", strength=.5, decay=6)
        torch.testing.assert_close(corrected[:, 0], torch.full((2, 2), .5))
        self.assertLess(float(corrected[:, -1].max()), .02)
        # L=96 => base context72, leaving two disjoint real 12h blocks for
        # nested input-only calibration/validation.
        cfg = {**self.cfg, "L": 96}
        base = PeakLiteTCN(cfg, calendar=False).eval()
        wrapped = CalibratedLiteTCN(base, strength=.5).eval()
        x = torch.randn(2, 96, 2, requires_grad=True)
        p = wrapped(x[:, :-12])[:, :12]
        p[0].sum().backward()
        self.assertEqual(float(x.grad[0, -12:].abs().sum()), 0.)
        self.assertEqual(float(x.grad[1].abs().sum()), 0.)
        # Vectorized selection and deployed wrapper must apply the same rule.
        with torch.no_grad():
            expected = correct_prediction(base(x), base(x[:, :-12])[:, :12], x[:, -12:], strength=.5)
            torch.testing.assert_close(expected, wrapped(x))


if __name__ == "__main__":
    unittest.main()
