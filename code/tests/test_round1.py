"""Checkpoint compatibility and paired validation comparison safeguards."""
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from scripts.summarize_round1 import paired_cluster_ci
from src.training.regression_cnn import RegressionConfig, SmallRegressionCNN, _load_checkpoint


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


if __name__ == '__main__':
    unittest.main()
