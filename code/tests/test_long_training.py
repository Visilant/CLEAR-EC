"""Check that extended training preserves the successful schedule exactly."""
import unittest

import torch

from scripts.run_training import build_parser
from src.training.regression_cnn import _build_scheduler


class LongTrainingTests(unittest.TestCase):
    def test_tail_preserves_prefix_and_holds_floor(self):
        def rates(steps):
            parameter = torch.nn.Parameter(torch.zeros(1))
            optimizer = torch.optim.AdamW([parameter], lr=1e-4)
            scheduler = _build_scheduler(optimizer, "cosine", 1, 8, 10)
            values = []
            for _ in range(steps):
                values.append(optimizer.param_groups[0]["lr"])
                optimizer.step()
                scheduler.step()
            return values

        control, long = rates(80), rates(400)
        self.assertEqual(control, long[:80])
        for value in long[80:]:
            self.assertAlmostEqual(value, 1e-6, places=14)

    def test_cli_accepts_independent_budgets(self):
        args = build_parser().parse_args([
            "--epochs", "40", "--sched_epochs", "8", "--save_epochs", "8,12,20,32,40"
        ])
        self.assertEqual(args.epochs, 40)
        self.assertEqual(args.sched_epochs, 8)
        self.assertEqual(args.save_epochs, "8,12,20,32,40")


if __name__ == "__main__":
    unittest.main()
