"""RegressionConfig defaults and field order are frozen: saved checkpoints rely on them."""
import unittest
from dataclasses import asdict, fields

from src.training.config import RegressionConfig, config_from_checkpoint, recipe_hash

FROZEN_DEFAULTS = {
    "epochs": 30, "batch_size": 8, "lr": 1e-4, "weight_decay": 1e-5, "patience": 5, "seed": 42,
    "loss": "huber", "downsample": 4, "num_workers": 4, "amp": False, "uint8_inputs": True,
    "cpu_threads": 4, "channels_last": False, "normalization": "batch", "architecture": None,
    "model": "small", "input_mode": "whole", "pretrained": True, "context_height": 486,
    "context_width": 648, "patch_size": 384, "num_patches": 4, "augment_flips": False,
    "ema": 0.0, "sched": "none", "sched_epochs": 0, "save_epochs": (), "warmup_epochs": 0,
    "target_space": "linear", "photometric": False, "fold": -1, "n_folds": 0, "all_data": False,
    "clip_grad": 0.0, "drop_path": 0.1, "antialias": False, "train_fraction": 1.0,
    "exclude_idx_file": "", "crop_scale": 1.0,
    # Added 2026-09-13 (line C1): density-map CD head and trimmed relative loss; defaults keep the old behaviour.
    "cd_head": "gap", "loss_trim": 0.0,
    # Added 2026-09-13 (line C2): label-transforming scale augmentation; 0 keeps the old behaviour.
    "scale_jitter": 0.0,
}


class ConfigTests(unittest.TestCase):
    def test_defaults_frozen(self):
        self.assertEqual(asdict(RegressionConfig()), FROZEN_DEFAULTS)
        self.assertEqual([f.name for f in fields(RegressionConfig)], list(FROZEN_DEFAULTS))

    def test_old_checkpoint_config_without_new_fields_loads(self):
        old = {k: v for k, v in FROZEN_DEFAULTS.items()
               if k not in ("clip_grad", "uint8_inputs", "crop_scale", "cd_head", "loss_trim", "scale_jitter")}
        cfg = config_from_checkpoint(old)
        self.assertEqual(cfg.clip_grad, 0.0)
        self.assertEqual((cfg.cd_head, cfg.loss_trim), ("gap", 0.0))
        self.assertEqual(cfg.scale_jitter, 0.0)
        self.assertFalse(cfg.uint8_inputs)  # historical checkpoints used CPU float conversion

    def test_recipe_hash_ignores_run_identity(self):
        a = RegressionConfig(model="convnext_tiny", seed=1, fold=0, n_folds=5)
        b = RegressionConfig(model="convnext_tiny", seed=2, fold=3, n_folds=5, num_workers=0)
        c = RegressionConfig(model="convnext_tiny", seed=1, fold=0, n_folds=5, lr=2e-4)
        self.assertEqual(recipe_hash(a), recipe_hash(b))
        self.assertNotEqual(recipe_hash(a), recipe_hash(c))
        self.assertEqual(len(recipe_hash(a)), 12)


if __name__ == "__main__":
    unittest.main()
