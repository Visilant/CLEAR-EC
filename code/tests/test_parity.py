"""Checkpoint parity: the refactored predict path must reproduce the pre-refactor predictions.

Fixtures in tests/fixtures/ were generated on 2026-09-13 with the unmodified night worktree
predictor (scripts/night_predict.py at commit faa3d0b) on CPU with one thread and flip TTA over
the 20 indices in parity_idx20.txt. The GPU check compares against the night's golden file,
which was produced on an A5000 with TF32 convolutions, hence the looser tolerance there.

Set CLEAR_EC_FULL_PARITY=1 to run the slow ConvNeXt CPU checks (about 2.5 minutes).
"""
from __future__ import annotations

import hashlib
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
FIXTURES = CODE / "tests" / "fixtures"
RESULTS = REPO / "results"
CACHE = REPO / "data" / "cache"
METRICS = ["CD", "CV", "HEX"]

CASES = {
    # name: (checkpoint dir, which, fixture, sha256[:16] of the checkpoint, slow)
    "smallcnn": (RESULTS / "audit_20260911/selection_fix/regression_cnn/seed_42", "best",
                 "parity_smallcnn_val20_cpu.csv", None, False),
    "fold0": (RESULTS / "night_20260912/fold0/regression_cnn/seed_123", "last",
              "parity_fold0_val20_cpu.csv", None, True),
    "v2fold0": (RESULTS / "night_20260912/v2fold0/regression_cnn/seed_123", "last",
                "parity_v2fold0_val20_cpu.csv", "408ae5df5570ee82", True),
}


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _predict(ckpt_dir: Path, which: str, indices: list[int], device: torch.device) -> pd.DataFrame:
    from src.data.cache import open_image_cache
    from src.training.common import labels_for_indices
    from src.training.models import load_checkpoint_file
    from src.training.predict import VIEWS, predict_frame_tta

    memmap, _ = open_image_cache(CACHE)
    frame = labels_for_indices(CACHE, REPO / "data/final_train_ids.csv", indices)
    model, stats, cfg, _epoch = load_checkpoint_file(
        ckpt_dir / ("best_model.pt" if which == "best" else "last.pt"), device)
    return predict_frame_tta(model, stats, cfg, memmap, frame, device, views=VIEWS["flips"], batch_size=16)


@unittest.skipUnless(CACHE.exists() and (RESULTS / "night_20260912").exists(), "local results/data absent")
class ParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.indices = [int(t) for t in (FIXTURES / "parity_idx20.txt").read_text().split()]
        assert len(cls.indices) == 20, cls.indices

    def _check_cpu(self, name: str):
        ckpt_dir, which, fixture, sha, slow = CASES[name]
        if slow and os.environ.get("CLEAR_EC_FULL_PARITY") != "1":
            self.skipTest("slow CPU parity; set CLEAR_EC_FULL_PARITY=1")
        ckpt = ckpt_dir / ("best_model.pt" if which == "best" else "last.pt")
        if sha is not None:
            self.assertEqual(_sha16(ckpt), sha, "checkpoint fixture drifted")
        torch.set_num_threads(1)
        got = _predict(ckpt_dir, which, self.indices, torch.device("cpu"))
        want = pd.read_csv(FIXTURES / fixture)
        self.assertEqual(list(got.idx), list(want.idx))
        np.testing.assert_allclose(got[METRICS].to_numpy(), want[METRICS].to_numpy(), rtol=1e-6, atol=0)

    def test_smallcnn_cpu_exact(self):
        self._check_cpu("smallcnn")

    def test_fold0_cpu_exact(self):
        self._check_cpu("fold0")

    def test_v2fold0_cpu_exact(self):
        self._check_cpu("v2fold0")

    @unittest.skipUnless(torch.cuda.is_available(), "GPU parity needs CUDA")
    def test_v2fold0_gpu_matches_night_golden(self):
        ckpt_dir, which, _fixture, sha, _slow = CASES["v2fold0"]
        self.assertEqual(_sha16(ckpt_dir / "last.pt"), sha)
        got = _predict(ckpt_dir, which, self.indices, torch.device("cuda:0"))
        golden = pd.read_csv(RESULTS / "night_20260912/golden/v2fold0_val_flips.csv").set_index("idx")
        want = golden.loc[self.indices, METRICS].to_numpy()
        np.testing.assert_allclose(got[METRICS].to_numpy(), want, rtol=1e-5, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "GPU parity needs CUDA")
    def test_convnext_gpu_close_to_cpu_fixture(self):
        # TF32 on Ampere: GPU vs CPU fp32 differ at the 1e-3 level; guards gross regressions cheaply.
        for name in ("fold0", "v2fold0"):
            ckpt_dir, which, fixture, _sha, _slow = CASES[name]
            got = _predict(ckpt_dir, which, self.indices, torch.device("cuda:0"))
            want = pd.read_csv(FIXTURES / fixture)
            np.testing.assert_allclose(got[METRICS].to_numpy(), want[METRICS].to_numpy(), rtol=5e-3, atol=0)


if __name__ == "__main__":
    unittest.main()
