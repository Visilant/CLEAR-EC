"""The slide-grouped fold definition must stay identical to the one the night's models used."""
import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data" / "cache"
NIGHT = REPO / "results" / "night_20260912"


@unittest.skipUnless((CACHE / "splits.json").exists(), "local cache absent")
class FoldTests(unittest.TestCase):
    def test_folds_match_night_runs_and_are_slide_disjoint(self):
        from src.training.folds import _slide_group_folds, all_labelled_indices, resolve_indices

        folds = _slide_group_folds(CACHE, 5, seed=42)
        everything = sorted(i for k in folds for i in folds[k])
        self.assertEqual(everything, all_labelled_indices(CACHE))
        self.assertEqual(len(everything), len(set(everything)))
        index = pd.read_csv(CACHE / "index.csv")
        slide_of = dict(zip(index["idx"].astype(int), index["slide_id"].astype(str)))
        slide_sets = [{slide_of[i] for i in folds[k]} for k in range(5)]
        for a in range(5):
            for b in range(a + 1, 5):
                self.assertTrue(slide_sets[a].isdisjoint(slide_sets[b]))
        for k in range(5):
            metrics = NIGHT / f"v2fold{k}" / "regression_cnn" / "seed_123" / "metrics.json"
            if metrics.exists():
                self.assertEqual(len(folds[k]), json.loads(metrics.read_text())["n_val"], f"fold {k}")
            self.assertEqual(resolve_indices(CACHE, f"fold:{k}/5"), folds[k])
        digest = hashlib.sha256(json.dumps(folds, sort_keys=True).encode()).hexdigest()[:16]
        self.assertEqual(digest, "60db43af95acb672")


if __name__ == "__main__":
    unittest.main()
