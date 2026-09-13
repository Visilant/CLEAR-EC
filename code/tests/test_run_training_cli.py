"""The training CLI must expose every RegressionConfig field and accept a JSON config."""
import json
import tempfile
import unittest
from dataclasses import asdict, fields
from pathlib import Path

from scripts import run_training
from src.training.config import RegressionConfig


class TrainingCliTests(unittest.TestCase):
    def test_every_config_field_is_reachable(self):
        dests = {a.dest for a in run_training.build_parser()._actions}
        for f in fields(RegressionConfig):
            if f.name in run_training.SKIPPED_FIELDS:
                continue
            flag = run_training.INVERTED_FLAGS.get(f.name, f.name)
            self.assertIn(flag, dests, f"RegressionConfig.{f.name} has no CLI flag")

    def test_flags_round_trip_into_config(self):
        args = run_training.build_parser().parse_args([
            "--model", "timm:convnextv2_tiny.fcmae_ft_in22k_in1k", "--epochs", "8", "--ema", "0.999",
            "--sched", "cosine", "--warmup_epochs", "1", "--patience", "0", "--amp", "--channels_last",
            "--augment_flips", "--clip_grad", "1.0", "--fold", "0", "--n_folds", "5", "--no_pretrained",
            "--float_inputs", "--save_epochs", "8,12", "--loss", "relative", "--weight_decay", "1e-4",
        ])
        cfg = run_training.config_from_args(args, seed=123)
        self.assertEqual(cfg.seed, 123)
        self.assertEqual(cfg.model, "timm:convnextv2_tiny.fcmae_ft_in22k_in1k")
        self.assertEqual((cfg.epochs, cfg.ema, cfg.sched, cfg.warmup_epochs, cfg.patience), (8, 0.999, "cosine", 1, 0))
        self.assertTrue(cfg.amp and cfg.channels_last and cfg.augment_flips)
        self.assertEqual((cfg.clip_grad, cfg.fold, cfg.n_folds), (1.0, 0, 5))
        self.assertFalse(cfg.pretrained)
        self.assertFalse(cfg.uint8_inputs)
        self.assertEqual(cfg.save_epochs, (8, 12))
        self.assertEqual(cfg.loss, "relative")
        self.assertEqual(cfg.weight_decay, 1e-4)

    def test_defaults_match_dataclass(self):
        args = run_training.build_parser().parse_args([])
        self.assertEqual(asdict(run_training.config_from_args(args, seed=42)), asdict(RegressionConfig()))

    def test_config_json_round_trips(self):
        cfg = RegressionConfig(model="convnext_tiny", epochs=3, seed=7, fold=2, n_folds=5)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"config": asdict(cfg), "job": "x"}))
            loaded = RegressionConfig(**json.loads(path.read_text())["config"])
        self.assertEqual(loaded, cfg)

    def test_removed_methods_are_rejected(self):
        parser = run_training.build_parser()
        self.assertEqual(run_training.parse_methods("regression"), ["regression"])
        for bad in ("calibration", "regression,calibration", "pseudo", "all"):
            with self.assertRaises(SystemExit):
                parser.parse_args(["--method", bad])


if __name__ == "__main__":
    unittest.main()
