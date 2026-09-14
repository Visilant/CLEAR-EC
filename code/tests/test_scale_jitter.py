"""Label-transforming scale augmentation (RegressionConfig.scale_jitter, 2026-09-13 line C2).

Physics: zooming a frame by s at fixed pixel spacing multiplies the true cell density by 1/s^2 and
leaves CV and HEX unchanged. The zoom resamples straight from the full-resolution cache frame onto one
shared canvas (min(1, min s) of the model's input size), so no padding is ever fabricated."""
import unittest

import numpy as np
import pandas as pd
import torch

from src.training.common import METRICS
from src.training.data import _rescale_cd_targets, _scale_jitter_batch, _zoom_batch
from src.training.targets import _forward_target_space


def _stripes(height: int, width: int, period: float, batch: int = 1) -> torch.Tensor:
    """uint8 vertical stripes (cosine along x) of a known period in native pixels."""
    xs = torch.arange(width, dtype=torch.float32)
    row = 127.5 + 100.0 * torch.cos(2 * np.pi * xs / period)
    img = row.expand(height, width)
    return img.round().to(torch.uint8)[None, None].expand(batch, 1, height, width).contiguous()


def _period_along_x(img: torch.Tensor) -> float:
    """Dominant period (pixels) of a (H, W) float image along x, from the row-averaged FFT."""
    row = img.float().mean(dim=0)
    row = row - row.mean()
    spectrum = torch.fft.rfft(row).abs()
    spectrum[0] = 0.0
    k = int(torch.argmax(spectrum))
    return row.numel() / k


class ZoomTests(unittest.TestCase):
    H, W = 256, 320          # native frame
    CANVAS = (128, 160)      # the model input: a 2x downsample, like 972x1296 -> 486x648
    PERIOD = 16.0            # native pixels; 8 px on the unzoomed canvas

    def _check_scale(self, s: float, antialias: bool = False):
        x = _stripes(self.H, self.W, self.PERIOD)
        out, cd_mult = _zoom_batch(x, torch.tensor([s]), self.CANVAS, antialias=antialias)
        expected_period = self.PERIOD * (self.CANVAS[1] / self.W) * s
        self.assertAlmostEqual(_period_along_x(out[0, 0]), expected_period, delta=0.25 * expected_period / 8)
        self.assertAlmostEqual(float(cd_mult[0]), 1.0 / s ** 2, delta=0.02 / s ** 2)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(0.0 <= float(out.min()) and float(out.max()) <= 1.0)
        return out

    def test_zoom_in_crops_centre_to_the_canvas(self):
        out = self._check_scale(1.25)
        self.assertEqual(tuple(out.shape), (1, 1, *self.CANVAS))

    def test_zoom_out_shrinks_the_canvas_instead_of_padding(self):
        out = self._check_scale(0.8)
        self.assertEqual(tuple(out.shape), (1, 1, round(0.8 * self.CANVAS[0]), round(0.8 * self.CANVAS[1])))

    def test_antialias_path(self):
        self._check_scale(0.85, antialias=True)

    def test_scale_one_matches_the_model_resample(self):
        x = _stripes(self.H, self.W, self.PERIOD)
        out, cd_mult = _zoom_batch(x, torch.tensor([1.0]), self.CANVAS)
        ref = torch.nn.functional.interpolate(x.float() / 255.0, size=self.CANVAS, mode="bilinear", align_corners=False)
        torch.testing.assert_close(out, ref)
        self.assertEqual(float(cd_mult[0]), 1.0)

    def test_mixed_batch_shares_one_canvas_and_per_sample_scale(self):
        x = _stripes(self.H, self.W, self.PERIOD, batch=2)
        s = torch.tensor([1.2, 0.8])
        out, cd_mult = _zoom_batch(x, s, self.CANVAS)
        self.assertEqual(tuple(out.shape[-2:]), (round(0.8 * self.CANVAS[0]), round(0.8 * self.CANVAS[1])))
        for i in range(2):
            expected_period = self.PERIOD * (self.CANVAS[1] / self.W) * float(s[i])
            self.assertAlmostEqual(_period_along_x(out[i, 0]), expected_period, delta=0.25 * expected_period / 8)
            self.assertAlmostEqual(float(cd_mult[i]), 1.0 / float(s[i]) ** 2, delta=0.03)

    def test_scale_jitter_batch_draws_within_range(self):
        torch.manual_seed(0)
        x = _stripes(self.H, self.W, self.PERIOD, batch=16)
        out, cd_mult = _scale_jitter_batch(x, 0.2, self.CANVAS)
        s = cd_mult.rsqrt()
        self.assertTrue(bool((s >= np.exp(-0.2) * 0.98).all()) and bool((s <= np.exp(0.2) * 1.02).all()))
        self.assertGreater(float(s.std()), 0.02)
        self.assertLessEqual(out.shape[-2], self.CANVAS[0])
        self.assertLessEqual(out.shape[-1], self.CANVAS[1])


class TargetTests(unittest.TestCase):
    def _stats_and_targets(self, target_space: str):
        raw = np.array([[2500.0, 0.30, 0.60], [1800.0, 0.40, 0.50], [3200.0, 0.25, 0.70]])
        transformed = _forward_target_space(raw, target_space)
        stats = {m: {"mean": float(transformed[:, j].mean()), "std": float(transformed[:, j].std() + 1e-8)}
                 for j, m in enumerate(METRICS)}
        y = torch.tensor((transformed - [stats[m]["mean"] for m in METRICS]) / [stats[m]["std"] for m in METRICS],
                         dtype=torch.float32)
        return raw, stats, y, torch.tensor(raw, dtype=torch.float32)

    def _check(self, target_space: str):
        raw, stats, y, y_raw = self._stats_and_targets(target_space)
        cd_mult = torch.tensor([1.0 / 1.2 ** 2, 1.0 / 0.9 ** 2, 1.0])
        y_new, y_raw_new = _rescale_cd_targets(y, y_raw, cd_mult, stats, target_space)
        # CD scales by the multiplier in raw units; CV and HEX are untouched in both spaces.
        torch.testing.assert_close(y_raw_new[:, 0], y_raw[:, 0] * cd_mult)
        torch.testing.assert_close(y_raw_new[:, 1:], y_raw[:, 1:])
        torch.testing.assert_close(y_new[:, 1:], y[:, 1:])
        # The normalised CD is what the dataset would have produced for the scaled raw label.
        expected = _forward_target_space(raw * np.array([cd_mult.numpy(), [1, 1, 1], [1, 1, 1]]).T, target_space)
        expected_cd = (expected[:, 0] - stats["CD"]["mean"]) / stats["CD"]["std"]
        torch.testing.assert_close(y_new[:, 0], torch.tensor(expected_cd, dtype=torch.float32), rtol=1e-5, atol=1e-5)
        # Inputs are not modified in place.
        torch.testing.assert_close(y_raw[:, 0], torch.tensor(raw[:, 0], dtype=torch.float32))

    def test_linear_space(self):
        self._check("linear")

    def test_log_space(self):
        self._check("log")


class EpochTests(unittest.TestCase):
    """scale_jitter=0 must leave _run_epoch exactly as it was (no draw, no resample, no target change)."""

    def _run(self, scale_jitter: float, seed: int = 0):
        from torch.utils.data import DataLoader
        from src.training.config import RegressionConfig
        from src.training.data import MetricRegressionDataset
        from src.training.loop import _run_epoch
        from src.training.losses import _build_loss
        from src.training.models import build_regression_model
        from src.training.targets import target_space_stats

        torch.manual_seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)
        memmap = rng.integers(0, 255, size=(6, 64, 96), dtype=np.uint8)
        frame = pd.DataFrame({"idx": np.arange(6), "ID": [f"i{i}" for i in range(6)],
                              "CD": rng.uniform(1500, 3500, 6), "CV": rng.uniform(0.2, 0.5, 6), "HEX": rng.uniform(0.4, 0.7, 6)})
        stats = target_space_stats(frame, "linear")
        ds = MetricRegressionDataset(memmap, frame, stats)
        loader = DataLoader(ds, batch_size=3, shuffle=False, num_workers=0)
        cfg = RegressionConfig(model="small", downsample=2, normalization="group", loss="relative")
        model = build_regression_model(cfg)
        criterion = _build_loss("relative", stats)
        opt = torch.optim.SGD(model.parameters(), lr=1e-3)
        kwargs = {} if scale_jitter is None else {"scale_jitter": scale_jitter}
        loss, mape = _run_epoch(model, loader, criterion, torch.device("cpu"), stats, opt, **kwargs)
        return loss, mape, [p.detach().clone() for p in model.parameters()]

    def test_zero_jitter_is_bit_identical_to_the_default_call(self):
        loss_a, mape_a, params_a = self._run(None)
        loss_b, mape_b, params_b = self._run(0.0)
        self.assertEqual(loss_a, loss_b)
        self.assertEqual(mape_a, mape_b)
        for a, b in zip(params_a, params_b):
            self.assertTrue(torch.equal(a, b))

    def test_zero_jitter_never_resamples(self):
        from unittest import mock
        with mock.patch("src.training.loop._scale_jitter_batch", side_effect=AssertionError("must not be called")):
            self._run(0.0)

    def test_positive_jitter_changes_the_epoch(self):
        loss_a, _, params_a = self._run(0.0)
        loss_b, _, params_b = self._run(0.2)
        self.assertNotEqual(loss_a, loss_b)
        self.assertFalse(all(torch.equal(a, b) for a, b in zip(params_a, params_b)))


class ScaleTTATests(unittest.TestCase):
    """predict_tta scales: default (1.0,) is the untouched flip path; extra scales zoom, predict, undo on CD."""

    def _setup(self):
        from src.training.config import RegressionConfig
        from src.training.models import build_regression_model
        from src.training.targets import target_space_stats
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        cfg = RegressionConfig(model="convnext_tiny", context_height=64, context_width=96, loss="relative")
        model = build_regression_model(cfg, load_pretrained=False).eval()
        memmap = rng.integers(0, 255, size=(5, 128, 192), dtype=np.uint8)
        frame = pd.DataFrame({"idx": np.arange(5), "ID": [f"i{i}" for i in range(5)],
                              "CD": rng.uniform(1500, 3500, 5), "CV": rng.uniform(0.2, 0.5, 5), "HEX": rng.uniform(0.4, 0.7, 5)})
        return model, target_space_stats(frame, "linear"), cfg, memmap, frame

    def test_parse_tta(self):
        from src.training.predict import VIEWS, parse_tta
        self.assertEqual(parse_tta("flips"), (VIEWS["flips"], (1.0,)))
        self.assertEqual(parse_tta("flips_s3"), (VIEWS["flips"], (0.9, 1.0, 1.1)))
        self.assertEqual(parse_tta("none"), (VIEWS["none"], (1.0,)))

    def test_scales_default_is_the_plain_path_and_s3_restores_the_model(self):
        from src.training.predict import VIEWS, predict_frame_tta
        model, stats, cfg, memmap, frame = self._setup()
        device = torch.device("cpu")
        plain = predict_frame_tta(model, stats, cfg, memmap, frame, device, views=VIEWS["flips"], batch_size=2)
        again = predict_frame_tta(model, stats, cfg, memmap, frame, device, views=VIEWS["flips"], batch_size=2, scales=(1.0,))
        pd.testing.assert_frame_equal(plain, again)
        scaled = predict_frame_tta(model, stats, cfg, memmap, frame, device, views=VIEWS["none"], batch_size=2, scales=(0.9, 1.0, 1.1))
        self.assertEqual(model.context_size, (64, 96))
        self.assertEqual(list(scaled["ID"]), list(frame["ID"]))
        self.assertTrue(np.isfinite(scaled[list(METRICS)].to_numpy()).all())
        self.assertFalse(np.allclose(scaled["CD"], plain["CD"]))


if __name__ == "__main__":
    unittest.main()
