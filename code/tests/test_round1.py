"""Checkpoint compatibility and paired validation comparison safeguards."""
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from experiments.stats import paired_cluster_ci
from src.training.regression_cnn import (
    RegressionConfig, RelativeAbsoluteErrorLoss, SmallRegressionCNN, _load_checkpoint,
)
from src.training.convnext_regression import ConvNeXtTinyRegression


class RoundOneTests(unittest.TestCase):
    def test_legacy_and_group_checkpoints_reload_with_prediction_parity(self):
        torch.set_num_threads(1)
        for normalization in ('batch', 'group'):
            with self.subTest(normalization=normalization), tempfile.TemporaryDirectory() as tmp:
                config = asdict(RegressionConfig(normalization=normalization))
                if normalization == 'batch':
                    config.pop('normalization')  # Existing submission schema.
                model = SmallRegressionCNN(normalization=normalization).eval()
                x = torch.randint(0, 256, (2, 1, 128, 160), dtype=torch.uint8)
                torch.save({'config': config, 'target_stats': {}, 'model_state': model.state_dict()},
                           Path(tmp)/'best_model.pt')
                loaded, _, _ = _load_checkpoint(Path(tmp), torch.device('cpu'))
                loaded.eval()
                with torch.no_grad():
                    torch.testing.assert_close(model(x), loaded(x))

    def test_cluster_delta_masks_zeros_per_metric(self):
        targets = np.array([[0., 2., 4.], [10., 0., 4.], [10., 2., 0.], [10., 2., 4.]])
        baseline = targets * 1.2
        candidate = targets * 1.1
        candidate[targets == 0] = 1e9  # Must have no effect on any scored metric.
        groups = np.array(['a', 'a', 'b', 'b'])
        interval = paired_cluster_ci(candidate, baseline, targets, groups, n_boot=100)
        np.testing.assert_allclose(interval, [-10., -10.], atol=1e-10)
        np.testing.assert_allclose(paired_cluster_ci(baseline, baseline, targets, groups, n_boot=100), [0., 0.])

    def test_relative_loss_masks_zero_targets(self):
        stats = {name: {'mean': 10., 'std': 2.} for name in ('CD', 'CV', 'HEX')}
        loss = RelativeAbsoluteErrorLoss(stats)
        target = torch.tensor([[0., 10., 20.]])
        normalized = (target - 10.) / 2.
        prediction = normalized.clone()
        prediction[0, 0] = 1e8
        self.assertEqual(float(loss(prediction, normalized)), 0.)

    def test_convnext_patch_selection_is_deterministic(self):
        model = ConvNeXtTinyRegression(
            input_mode='quality', pretrained=False, context_size=(64, 80), patch_size=48
        ).eval()
        x = torch.randint(0, 256, (1, 1, 96, 128), dtype=torch.uint8)
        with torch.no_grad():
            first = model._select_patches(x)
            second = model._select_patches(x)
        torch.testing.assert_close(first, second)
        self.assertEqual(tuple(first.shape), (1, 4, 1, 48, 48))


if __name__ == '__main__':
    unittest.main()
