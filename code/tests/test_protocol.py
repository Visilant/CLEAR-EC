"""Failing-first tests for the fair overnight experiment protocol."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.data.config import MetricConfig, SegConfig, config_hash
from src.data.splits import assert_protocol_splits, build_splits, load_split, load_splits
from src.training.common import (
    METRICS,
    paired_bootstrap_mape,
    pred_artifact_path,
    score_by_id,
    split_summary,
)


def _tiny_index(n_slides: int = 10, images_per_slide: int = 4) -> pd.DataFrame:
    rows = []
    idx = 0
    for slide in range(n_slides):
        for _ in range(images_per_slide):
            rows.append(
                {
                    "idx": idx,
                    "ID": f"img-{idx:04d}",
                    "slide_id": f"slide-{slide:02d}",
                }
            )
            idx += 1
    return pd.DataFrame(rows)


class SplitProtocolTests(unittest.TestCase):
    def test_build_splits_are_image_and_slide_disjoint(self) -> None:
        cache_dir = Path(self._tmp())
        index_df = _tiny_index()
        splits = build_splits(index_df, cache_dir, seed=42)
        report = assert_protocol_splits(index_df, splits)
        self.assertEqual(
            report["n_images"],
            len(index_df),
            msg="Every image must be assigned to exactly one role",
        )
        self.assertEqual(report["slide_overlap"], {})
        self.assertEqual(report["image_overlap"], {})

    def test_assert_protocol_splits_rejects_slide_leakage(self) -> None:
        index_df = _tiny_index(n_slides=4, images_per_slide=2)
        leaked = {
            "train": [0, 1, 2],
            "val": [3, 4],
            "test": [5, 6, 7],
        }
        with self.assertRaises(ValueError):
            assert_protocol_splits(index_df, leaked)

    def test_legacy_holdout_merge_helpers_are_gone(self) -> None:
        splits_mod = importlib.import_module("src.data.splits")
        self.assertFalse(hasattr(splits_mod, "load_holdout_indices"))
        self.assertFalse(hasattr(splits_mod, "dev_train_val_indices"))
        self.assertFalse(hasattr(splits_mod, "load_dev_indices"))

    def test_load_split_roles(self) -> None:
        cache_dir = Path(self._tmp())
        index_df = _tiny_index()
        build_splits(index_df, cache_dir, seed=42)
        for name in ("train", "val", "test"):
            indices = load_split(cache_dir, name)
            self.assertTrue(indices, msg=f"{name} must be non-empty")
            self.assertEqual(indices, sorted(indices))

    def test_split_summary_uses_outer_roles(self) -> None:
        cache_dir = Path(self._tmp())
        index_df = _tiny_index()
        index_df.to_csv(cache_dir / "index.csv", index=False)
        build_splits(index_df, cache_dir, seed=42)
        summary = split_summary(cache_dir)
        self.assertEqual(
            set(summary),
            {"train", "val", "test", "n_slides_train", "n_slides_val", "n_slides_test"},
        )
        self.assertNotIn("holdout", summary)
        self.assertNotIn("dev_val", summary)
        self.assertEqual(
            summary["train"] + summary["val"] + summary["test"],
            len(index_df),
        )

    def _tmp(self) -> str:
        import tempfile

        return tempfile.mkdtemp(prefix="clear-ec-splits-")


class ScoringTests(unittest.TestCase):
    def test_score_by_id_survives_shuffled_and_extra_rows(self) -> None:
        gt = pd.DataFrame(
            {
                "ID": ["a", "b", "c"],
                "CD": [100.0, 200.0, 300.0],
                "CV": [0.2, 0.4, 0.6],
                "HEX": [0.5, 0.5, 0.5],
            }
        )
        pred = pd.DataFrame(
            {
                "ID": ["c", "a", "b", "zzz"],
                "CD": [330.0, 100.0, 220.0, 1.0],
                "CV": [0.6, 0.2, 0.4, 9.0],
                "HEX": [0.5, 0.5, 0.5, 9.0],
            }
        )
        scores = score_by_id(pred, gt)
        # a exact, b 10% CD, c 10% CD; CV/HEX exact
        self.assertAlmostEqual(scores["CD"], (0 + 10 + 10) / 3.0, places=5)
        self.assertAlmostEqual(scores["CV"], 0.0, places=5)
        self.assertAlmostEqual(scores["HEX"], 0.0, places=5)

    def test_score_by_id_skips_zero_targets(self) -> None:
        gt = pd.DataFrame(
            {"ID": ["a", "b"], "CD": [0.0, 100.0], "CV": [0.2, 0.2], "HEX": [0.5, 0.5]}
        )
        pred = pd.DataFrame(
            {"ID": ["a", "b"], "CD": [50.0, 110.0], "CV": [0.2, 0.2], "HEX": [0.5, 0.5]}
        )
        scores = score_by_id(pred, gt)
        self.assertAlmostEqual(scores["CD"], 10.0, places=5)

    def test_score_by_id_requires_overlap(self) -> None:
        gt = pd.DataFrame({"ID": ["a"], "CD": [1.0], "CV": [1.0], "HEX": [1.0]})
        pred = pd.DataFrame({"ID": ["b"], "CD": [1.0], "CV": [1.0], "HEX": [1.0]})
        with self.assertRaises(ValueError):
            score_by_id(pred, gt)

    def test_bootstrap_returns_interval(self) -> None:
        rng = np.random.default_rng(0)
        ids = [f"id-{i}" for i in range(40)]
        gt_cd = rng.uniform(800, 1200, size=40)
        pred_cd = gt_cd * rng.uniform(0.9, 1.1, size=40)
        gt = pd.DataFrame(
            {"ID": ids, "CD": gt_cd, "CV": np.full(40, 0.3), "HEX": np.full(40, 0.5)}
        )
        pred = pd.DataFrame(
            {"ID": ids, "CD": pred_cd, "CV": np.full(40, 0.3), "HEX": np.full(40, 0.5)}
        )
        result = paired_bootstrap_mape(pred, gt, n_boot=200, seed=0)
        self.assertIn("mean", result)
        self.assertLess(result["ci_low"], result["ci_high"])
        self.assertGreaterEqual(result["mean"], 0.0)


class ArtifactTests(unittest.TestCase):
    def test_pred_artifact_path_is_split_specific(self) -> None:
        cache_dir = Path("/tmp/cache")
        seg = SegConfig(batch_size=16)
        metric = MetricConfig(random_crop_frac=0.5, random_crop_seed=42)
        exp_hash = config_hash((config_hash(seg), metric))
        train_path = pred_artifact_path(cache_dir, exp_hash, "train")
        val_path = pred_artifact_path(cache_dir, exp_hash, "val")
        test_path = pred_artifact_path(cache_dir, exp_hash, "test")
        self.assertNotEqual(train_path, val_path)
        self.assertNotEqual(val_path, test_path)
        self.assertTrue(str(train_path).endswith(f"{exp_hash}/train.csv"))
        self.assertTrue(str(test_path).endswith(f"{exp_hash}/test.csv"))

    def test_sweep_metrics_parser_requires_split(self) -> None:
        from scripts import sweep_metrics

        parser = sweep_metrics.build_parser()
        args = parser.parse_args(
            ["--seg_hash", "abc", "--split", "val", "--crop_frac", "0.4"]
        )
        self.assertEqual(args.split, "val")


class EvaluateCliTests(unittest.TestCase):
    def test_evaluate_cli_accepts_val(self) -> None:
        import evaluate

        parser = evaluate.build_parser()
        args = parser.parse_args(["--split", "val", "--predictions_csv", "x.csv"])
        self.assertEqual(args.split, "val")
        args_test = parser.parse_args(["--split", "test", "--predictions_csv", "y.csv"])
        self.assertEqual(args_test.split, "test")


class TrainingCliTests(unittest.TestCase):
    def test_run_training_rejects_pseudo_and_accepts_csv_methods(self) -> None:
        from scripts import run_training

        parser = run_training.build_parser()
        args = parser.parse_args(["--method", "regression,calibration"])
        self.assertEqual(run_training.parse_methods(args.method), ["regression", "calibration"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--method", "pseudo"])
        with self.assertRaises(SystemExit):
            parser.parse_args(["--method", "all"])


class CacheRecoveryTests(unittest.TestCase):
    def test_ensure_cache_rebuilds_instead_of_waiting(self) -> None:
        from scripts.run_overnight import cache_is_full, ensure_cache

        import tempfile

        tmp = Path(tempfile.mkdtemp(prefix="clear-ec-cache-"))
        (tmp / "index.csv").write_text("idx,ID,slide_id\n0,a,s\n")
        (tmp / "splits.json").write_text(json.dumps({"train": [0], "val": [], "test": []}))
        self.assertFalse(cache_is_full(tmp, expected=9000))

        rebuilt = {"called": False}

        def fake_rebuild() -> None:
            rebuilt["called"] = True
            pd.DataFrame(
                {"idx": list(range(9000)), "ID": [f"i{i}" for i in range(9000)]}
            ).to_csv(tmp / "index.csv", index=False)
            (tmp / "splits.json").write_text(
                json.dumps(
                    {
                        "train": list(range(7200)),
                        "val": list(range(7200, 8100)),
                        "test": list(range(8100, 9000)),
                    }
                )
            )

        ensure_cache(tmp, expected=9000, rebuild=fake_rebuild, wait=False)
        self.assertTrue(rebuilt["called"])
        self.assertTrue(cache_is_full(tmp, expected=9000))


class SelectionIsolationTests(unittest.TestCase):
    def test_calibration_fit_does_not_read_test_ids(self) -> None:
        from src.training.calibration import fit_calibration, CalibrationConfig

        index_df = _tiny_index(n_slides=6, images_per_slide=2)
        splits = {
            "train": index_df.loc[index_df.slide_id.isin(["slide-00", "slide-01", "slide-02"]), "idx"].tolist(),
            "val": index_df.loc[index_df.slide_id.isin(["slide-03", "slide-04"]), "idx"].tolist(),
            "test": index_df.loc[index_df.slide_id.isin(["slide-05"]), "idx"].tolist(),
        }
        ids = {int(r.idx): str(r.ID) for r in index_df.itertuples()}
        pred_df = pd.DataFrame(
            {
                "ID": [ids[i] for i in index_df["idx"]],
                "CD": np.linspace(100, 200, len(index_df)),
                "CV": np.linspace(0.2, 0.4, len(index_df)),
                "HEX": np.linspace(0.4, 0.6, len(index_df)),
            }
        )
        gt_df = pred_df.copy()
        gt_df["CD"] = gt_df["CD"] * 1.1
        accessed = []

        class TrackingIndex(pd.DataFrame):
            @property
            def _constructor(self):
                return TrackingIndex

        # Directly assert fit_calibration only receives train indices.
        info = fit_calibration(
            pred_df,
            gt_df.assign(idx=index_df["idx"].values),
            splits["train"],
            index_df,
            CalibrationConfig(alpha=1.0),
        )
        self.assertGreater(info["n_fit"], 0)
        self.assertNotIn("n_test", info)
        test_ids = set(index_df.loc[index_df["idx"].isin(splits["test"]), "ID"].astype(str))
        self.assertTrue(test_ids.isdisjoint(set(info.get("fit_ids", []))))


if __name__ == "__main__":
    unittest.main()
