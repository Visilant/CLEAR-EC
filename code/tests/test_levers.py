"""C1 levers (2026-09-13): the density-map CD head and the trimmed relative loss.

Both sit behind new RegressionConfig fields (cd_head, loss_trim) whose defaults must leave the
trainer bit-identical to the recorded runs; tests/test_parity.py checks that against golden files.
"""
import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from experiments import REPO
from experiments.score import load_predictions
from experiments.spec import load_spec
from src.training.config import RegressionConfig
from src.training.convnext_regression import DensityCDHead, TimmWholeImageRegression, _feature_grid
from src.training.losses import RelativeAbsoluteErrorLoss, _build_loss
from src.training.models import build_regression_model

TINY = "timm:convnextv2_atto"  # 3.7M params; pretrained=False so nothing is downloaded
CONTEXT = (64, 96)
UNIT_STATS = {m: {"mean": 0.0, "std": 1.0} for m in ("CD", "CV", "HEX")}


def _tiny(cd_head: str, seed: int = 0) -> torch.nn.Module:
    torch.manual_seed(seed)
    cfg = RegressionConfig(model=TINY, pretrained=False, context_height=CONTEXT[0], context_width=CONTEXT[1],
                           cd_head=cd_head)
    return build_regression_model(cfg, load_pretrained=False).eval()


class DensityHeadTests(unittest.TestCase):
    def test_forward_shape_and_map(self):
        model = _tiny("density")
        x = torch.randint(0, 256, (2, 1, 128, 160), dtype=torch.uint8)
        with torch.no_grad():
            out = model(x)
            feat = model.backbone.forward_features(model._prepare(x))
            density = model.density.density_map(feat)
        self.assertEqual(tuple(out.shape), (2, 3))
        self.assertTrue(torch.isfinite(out).all())
        self.assertEqual(tuple(density.shape), (2, 1, *_feature_grid(CONTEXT)))
        self.assertTrue((density >= 0).all(), "softplus map is non-negative")
        # CD is the summed map times the learned scale plus the learned bias, nothing else.
        expected = model.density.scale * density.sum(dim=(1, 2, 3)) + model.density.bias
        torch.testing.assert_close(out[:, 0], expected)

    def test_feature_grid_matches_backbone_stride(self):
        model = _tiny("density")
        with torch.no_grad():
            feat = model.backbone.forward_features(model._prepare(torch.zeros(1, 1, *CONTEXT)))
        self.assertEqual(tuple(feat.shape[-2:]), _feature_grid(CONTEXT))
        self.assertEqual(_feature_grid((486, 648)), (15, 20))

    def test_initial_scale_is_one_over_grid(self):
        head = DensityCDHead(8, (3, 5))
        self.assertAlmostEqual(float(head.scale), 1.0 / 15)
        self.assertEqual(float(head.bias), 0.0)
        feat = torch.randn(4, 8, 3, 5)
        # A summed softplus map divided by its area sits at the same order as a GAP head output.
        self.assertLess(float(head(feat).abs().mean()), 2.0)

    def test_gap_default_adds_no_parameters_and_density_keeps_cv_hex(self):
        gap = _tiny("gap", seed=1)
        density = _tiny("density", seed=2)
        self.assertNotIn("density.conv.weight", gap.state_dict(), "cd_head='gap' must keep the frozen layout")
        self.assertIn("density.conv.weight", density.state_dict())
        missing, unexpected = density.load_state_dict(gap.state_dict(), strict=False)
        self.assertFalse(unexpected)
        self.assertTrue(all(k.startswith("density.") for k in missing), missing)
        x = torch.randint(0, 256, (2, 1, 64, 96), dtype=torch.uint8)
        with torch.no_grad():
            a, b = gap(x), density(x)
        torch.testing.assert_close(a[:, 1:], b[:, 1:])
        self.assertFalse(torch.allclose(a[:, 0], b[:, 0]))

    def test_same_seed_same_outputs(self):
        x = torch.randint(0, 256, (2, 1, 64, 96), dtype=torch.uint8)
        with torch.no_grad():
            a, b = _tiny("gap", seed=3)(x), _tiny("gap", seed=3)(x)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_density_head_rejected_where_unsupported(self):
        with self.assertRaises(ValueError):
            build_regression_model(RegressionConfig(model="small", cd_head="density"))
        with self.assertRaises(ValueError):
            build_regression_model(RegressionConfig(model="convnext_tiny", input_mode="fixed", cd_head="density",
                                                    pretrained=False), load_pretrained=False)
        with self.assertRaises(ValueError):
            TimmWholeImageRegression("convnextv2_atto", pretrained=False, cd_head="mean")


class TrimmedLossTests(unittest.TestCase):
    def _batch(self, errors):
        target = torch.ones(len(errors), 3)
        pred = target + torch.tensor(errors)[:, None]
        return pred, target

    def test_trim_drops_ceil_fraction_of_largest_samples(self):
        errors = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        pred, target = self._batch(errors)
        plain = RelativeAbsoluteErrorLoss(UNIT_STATS)
        trimmed = RelativeAbsoluteErrorLoss(UNIT_STATS, trim=0.125)
        self.assertAlmostEqual(float(plain(pred, target, target)), 0.45, places=6)
        self.assertAlmostEqual(float(trimmed(pred, target, target)), 0.4, places=6)  # ceil(0.125*8) = 1 dropped
        self.assertEqual(math.ceil(0.02 * 8), 1)
        two = RelativeAbsoluteErrorLoss(UNIT_STATS, trim=0.2)  # ceil(1.6) = 2 dropped
        self.assertAlmostEqual(float(two(pred, target, target)), 0.35, places=6)

    def test_small_batches_are_not_trimmed(self):
        pred, target = self._batch([0.1, 0.2, 1.0])  # batch 3 < MIN_TRIM_BATCH (4): untouched
        plain = RelativeAbsoluteErrorLoss(UNIT_STATS)
        trimmed = RelativeAbsoluteErrorLoss(UNIT_STATS, trim=0.125)
        torch.testing.assert_close(trimmed(pred, target, target), plain(pred, target, target), rtol=0, atol=0)
        self.assertEqual(RelativeAbsoluteErrorLoss.MIN_TRIM_BATCH, 4)

    def test_batch_of_four_is_trimmed(self):
        pred, target = self._batch([0.1, 0.2, 0.3, 1.0])  # ceil(0.125 * 4) = 1 dropped
        trimmed = RelativeAbsoluteErrorLoss(UNIT_STATS, trim=0.125)
        self.assertAlmostEqual(float(trimmed(pred, target, target)), 0.2, places=6)

    def test_zero_trim_is_the_existing_expression(self):
        pred, target = self._batch([0.1, 0.5, 0.2, 0.9, 0.3, 0.4, 0.6, 0.7])
        target[2, 2] = 0.0  # excluded by the MAPE definition
        pred.requires_grad_(True)
        loss = RelativeAbsoluteErrorLoss(UNIT_STATS)(pred, target, target)
        valid = target.abs() > 1e-5
        relative = (pred - target).abs() / target.abs().clamp_min(1e-5)
        torch.testing.assert_close(loss, relative[valid].mean(), rtol=0, atol=0)

    def test_trim_uses_per_sample_mean_over_valid_metrics_and_backpropagates(self):
        pred, target = self._batch([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        target[0, 2] = 0.0  # sample 0 keeps two valid metrics, both at 0.1 relative error
        pred = pred.clone().requires_grad_(True)
        loss = RelativeAbsoluteErrorLoss(UNIT_STATS, trim=0.125)(pred, target, target)
        self.assertAlmostEqual(float(loss), 0.4, places=6)
        loss.backward()
        self.assertEqual(float(pred.grad[7].abs().sum()), 0.0, "dropped sample gets no gradient")
        self.assertGreater(float(pred.grad[0].abs().sum()), 0.0)

    def test_build_loss_threads_trim(self):
        self.assertEqual(_build_loss("relative", UNIT_STATS).trim, 0.0)
        self.assertEqual(_build_loss("relative", UNIT_STATS, trim=0.125).trim, 0.125)
        with self.assertRaises(ValueError):
            RelativeAbsoluteErrorLoss(UNIT_STATS, trim=1.0)


class ExternalReferenceTests(unittest.TestCase):
    def test_compare_reference_glob_resolves_against_repo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested.yaml"
            path.write_text("name: nested\nresults_dir: results/x/y/z\narms: [{name: a}]\n")
            spec = load_spec(path)
        frame = load_predictions(spec, [], "code/tests/fixtures/parity_v2fold0_val20_cpu.csv", "last", "flips")
        self.assertEqual(len(frame), 20)
        self.assertEqual(set(frame.columns), {"idx", "ID", "CD", "CV", "HEX"})
        fixture = pd.read_csv(REPO / "code/tests/fixtures/parity_v2fold0_val20_cpu.csv")
        self.assertEqual(list(frame["ID"]), list(fixture["ID"]))


if __name__ == "__main__":
    unittest.main()
