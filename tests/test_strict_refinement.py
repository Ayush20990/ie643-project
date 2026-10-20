"""Strict target-input isolation, source retention and ensemble persistence."""
import io
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.strict_models import COMPONENTS, StrictEnsemble, input_pretext
from src.strict_refinement import strict_calibration
from src.context_models import ContextRidge, ContextEnsemble, real_context_pretext


class StrictRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_pretext_cannot_read_withheld_values_or_other_windows(self):
        inputs = torch.randn(5, 48, 2, requires_grad=True)
        context, observed = input_pretext(inputs, 6)
        self.assertFalse(observed.requires_grad)
        changed = inputs.detach().clone()
        changed[0, -6:] = 10000
        changed[1] = -999
        context2, _ = input_pretext(changed, 6)
        torch.testing.assert_close(context[0], context2[0])
        context[0].sum().backward()
        self.assertEqual(float(inputs.grad[0, -6:].abs().sum()), 0.)
        self.assertEqual(float(inputs.grad[1:].abs().sum()), 0.)
        self.assertGreater(float(inputs.grad[0, :-6].abs().sum()), 0.)

    def test_calibration_enforces_source_guard(self):
        cfg = {"L": 48, "H": 6, "columns": ["a", "b"]}
        model = StrictEnsemble(cfg, {"architecture": "patch_mlp"})
        truth = torch.ones(12, 6, 2)
        components = np.full((12, 6, 2, len(COMPONENTS)), .7, np.float32)
        components[..., 3] = -10  # Appealing to the target pretext, disastrous on source.
        pretext = np.ones_like(components)
        pretext[..., 3] = -10
        target_inputs = torch.full((12, 48, 2), -10.)
        with patch("src.strict_refinement.component_predictions", return_value=pretext):
            record = strict_calibration(model, components, truth, target_inputs)
        for item in record:
            self.assertLessEqual(item["source_val_MSE"], 1.1 * item["source_reference_MSE"] + 1e-8)
            self.assertLessEqual(item["selection_score"], 1.00001)
        np.testing.assert_allclose(model.mix.sum(-1), 1, atol=1e-6)
        self.assertTrue(bool((model.mix >= 0).all()))

    def test_sparse_forward_matches_all_components_and_roundtrips(self):
        cfg = {"L": 48, "H": 6, "columns": ["a", "b"]}
        model = StrictEnsemble(cfg, {"architecture": "patch_mlp"}).eval()
        model.mix.copy_(torch.tensor([[.25, 0, .2, .1, .1, .1, .25, 0],
                                      [0, 0, 0, 0, 0, 0, .5, .5]]))
        x = torch.randn(3, 48, 2)
        expected = (model.components(x) * model.mix[None, None]).sum(-1)
        torch.testing.assert_close(model(x), expected)
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        buf.seek(0)
        loaded = StrictEnsemble(cfg, {"architecture": "patch_mlp"}).eval()
        loaded.load_state_dict(torch.load(buf, weights_only=True))
        torch.testing.assert_close(loaded(x), expected)

    def test_short_context_branch_uses_only_real_unmasked_prefix(self):
        x = torch.randn(3, 48, 2)
        prefix, truth = real_context_pretext(x, 6)
        self.assertEqual(prefix.shape[1], 42)
        torch.testing.assert_close(truth, x[:, -6:])
        model = ContextRidge(48, 6, 2)
        model.weight.normal_(std=.01)
        padded, _ = input_pretext(x, 6)
        torch.testing.assert_close(model(padded), model(prefix))
        # The first six artificial positions cannot influence this branch.
        padded[:, :6] = 1e6
        torch.testing.assert_close(model(padded), model(prefix))
        cfg = {"L": 48, "H": 6, "columns": ["a", "b"]}
        ensemble = ContextEnsemble(cfg, {"architecture": "patch_mlp"}).eval()
        ensemble.models["linear_source"].load_state_dict(model.state_dict())
        ensemble.mix.zero_()
        ensemble.mix[:, 0] = 1
        torch.testing.assert_close(ensemble(x), model(x))


if __name__ == "__main__":
    unittest.main()
