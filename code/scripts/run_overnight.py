#!/usr/bin/env python3
"""Dependency-aware overnight CLEAR-EC experiment runner.

Protocol: fit on train, select on val, evaluate test once after every arm is frozen.
Arms: Cellpose baseline, ridge calibration, three-seed regression CNN.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import MetricConfig, SegConfig, config_hash
from src.data.splits import assert_protocol_splits, load_split, load_splits
from src.training.calibration import evaluate_calibration_split, train_calibration
from src.training.common import (
    experiment_hash,
    paired_bootstrap_mape,
    pred_artifact_path,
    score_by_id,
    split_summary,
)
from src.training.regression_cnn import evaluate_regression_split

DEFAULT_SEG = SegConfig(
    model_type="cyto",
    diameter=None,
    flow_threshold=0.4,
    cellprob_threshold=0.0,
    min_size=15,
    batch_size=16,
)

CROP_SWEEP = [0.30, 0.35, 0.40, 0.45, 0.50]

ABLATION_CONFIGS: list[tuple[str, SegConfig]] = [
    ("cyto2", SegConfig(model_type="cyto2", batch_size=16)),
    ("diameter_40", SegConfig(model_type="cyto", diameter=40.0, batch_size=16)),
    ("flow_0.3", SegConfig(model_type="cyto", flow_threshold=0.3, batch_size=16)),
    ("flow_0.5", SegConfig(model_type="cyto", flow_threshold=0.5, batch_size=16)),
]

REGRESSION_SEEDS = (42, 43, 44)
SEED = 42
EXPECTED_COUNT = 9000


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the fair overnight CLEAR-EC experiment suite."
    )
    parser.add_argument("--cache_dir", type=str, default="../data/cache")
    parser.add_argument("--data_dir", type=str, default="../data/train_mha")
    parser.add_argument("--labels_csv", type=str, default="../data/final_train_ids.csv")
    parser.add_argument("--results_dir", type=str, default="../results/overnight")
    parser.add_argument("--seg_gpu", type=int, default=0)
    parser.add_argument("--train_gpu", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate cache/splits, write resolved manifest, and exit before GPU work.",
    )
    parser.add_argument(
        "--skip_regression",
        action="store_true",
        help="Skip the CNN arm (still runs Cellpose + calibration).",
    )
    return parser


def cache_is_full(
    cache_dir: Path,
    expected: int = EXPECTED_COUNT,
    require_memmap: bool = False,
) -> bool:
    index_path = cache_dir / "index.csv"
    splits_path = cache_dir / "splits.json"
    if not index_path.exists() or not splits_path.exists():
        return False
    try:
        n = len(pd.read_csv(index_path))
    except (pd.errors.EmptyDataError, OSError):
        return False
    if n < expected:
        return False
    try:
        splits = json.loads(splits_path.read_text())
    except json.JSONDecodeError:
        return False
    total = sum(len(v) for v in splits.values())
    if total < expected:
        return False
    if require_memmap and not (cache_dir / "images_u8.npy").exists():
        return False
    return True


def ensure_cache(
    cache_dir: Path,
    *,
    expected: int = EXPECTED_COUNT,
    rebuild=None,
    wait: bool = False,
    require_memmap: bool = False,
) -> None:
    """Rebuild an incomplete cache immediately. Never wait-loop first."""
    del wait  # waiting before rebuild is the deadlock we are preventing
    if cache_is_full(cache_dir, expected=expected, require_memmap=require_memmap):
        return
    if rebuild is None:
        raise RuntimeError(
            f"Cache at {cache_dir} is incomplete and no rebuild callback was provided."
        )
    rebuild()
    if not cache_is_full(cache_dir, expected=expected, require_memmap=require_memmap):
        raise RuntimeError(f"Cache at {cache_dir} is still incomplete after rebuild.")


def detect_gpus() -> list[str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ["0"]
    gpu_ids = []
    for line in result.stdout.splitlines():
        match = re.match(r"GPU (\d+):", line)
        if match:
            gpu_ids.append(match.group(1))
    return gpu_ids or ["0"]


def git_state(repo: Path) -> dict:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "branch": _run(["git", "branch", "--show-current"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
        "diff_stat": _run(["git", "diff", "--stat"]),
    }


def _seg_config_cli(seg_config: SegConfig, batch_size: int) -> list[str]:
    cmd = [
        "--model_type",
        seg_config.model_type,
        "--flow_threshold",
        str(seg_config.flow_threshold),
        "--cellprob_threshold",
        str(seg_config.cellprob_threshold),
        "--min_size",
        str(seg_config.min_size),
        "--batch_size",
        str(batch_size),
    ]
    if seg_config.diameter is not None:
        cmd.extend(["--diameter", str(seg_config.diameter)])
    if not seg_config.tile:
        cmd.append("--no_tile")
    return cmd


class OvernightRunner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.code_root = Path(__file__).resolve().parents[1]
        self.repo_root = self.code_root.parent
        self.cache_dir = (self.code_root / args.cache_dir).resolve()
        self.data_dir = (self.code_root / args.data_dir).resolve()
        self.labels_csv = (self.code_root / args.labels_csv).resolve()
        self.results_dir = (self.code_root / args.results_dir).resolve()
        self.scripts_dir = self.code_root / "scripts"
        self.seg_gpu = args.seg_gpu
        self.train_gpu = args.train_gpu
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.dry_run = args.dry_run
        self.skip_regression = args.skip_regression
        self.log_path = self.results_dir / "overnight.log"
        self.manifest_path = self.results_dir / "manifest.json"
        self._log_file = None
        self.python = sys.executable
        self.best_crop_frac = 0.4
        self.best_seg_config = DEFAULT_SEG
        self.best_seg_name = "cyto_default"
        self.regression_proc: subprocess.Popen | None = None
        self.manifest: dict = {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "protocol": {
                "train": "fit",
                "val": "selection",
                "test": "one-shot after freeze",
            },
            "arms": ["cellpose_baseline", "ridge_calibration", "regression_cnn"],
            "regression_seeds": list(REGRESSION_SEEDS),
            "seg_gpu": self.seg_gpu,
            "train_gpu": self.train_gpu,
            "git": git_state(self.repo_root),
            "phases": {},
            "configs": [],
            "test_access_at": None,
        }

    def log(self, message: str) -> None:
        line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
        print(line, flush=True)
        if self._log_file is not None:
            self._log_file.write(line + "\n")
            self._log_file.flush()

    def _save_manifest(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _run_cmd(self, cmd: list[str], *, phase: str, label: str, env=None) -> None:
        self.log(f"{phase}: {label}")
        self.log(f"  cmd: {' '.join(str(c) for c in cmd)}")
        t0 = time.time()
        subprocess.run(cmd, check=True, cwd=self.code_root, env=env)
        elapsed = time.time() - t0
        self.manifest["phases"].setdefault(phase, []).append(
            {
                "label": label,
                "elapsed_sec": round(elapsed, 1),
                "cmd": [str(c) for c in cmd],
            }
        )
        self._save_manifest()

    def _rebuild_cache(self) -> None:
        self._run_cmd(
            [
                self.python,
                str(self.scripts_dir / "build_cache.py"),
                "--data_dir",
                str(self.data_dir),
                "--labels_csv",
                str(self.labels_csv),
                "--cache_dir",
                str(self.cache_dir),
                "--force_cache",
            ],
            phase="cache",
            label="rebuild full memmap cache",
        )

    def ensure_ready(self) -> None:
        ensure_cache(
            self.cache_dir,
            expected=EXPECTED_COUNT,
            rebuild=self._rebuild_cache,
            wait=False,
            require_memmap=True,
        )
        index_df = pd.read_csv(self.cache_dir / "index.csv")
        splits = load_splits(self.cache_dir)
        report = assert_protocol_splits(index_df, splits)
        summary = split_summary(self.cache_dir)
        self.manifest["split_summary"] = summary
        self.manifest["split_report"] = {
            "n_images": report["n_images"],
            "counts": report["counts"],
            "n_slides": report["n_slides"],
        }
        self._save_manifest()
        self.log(f"Splits OK: {summary}")

    def _segment(self, seg_config: SegConfig, split: str, label: str) -> None:
        cmd = [
            self.python,
            str(self.scripts_dir / "run_segmentation_cache.py"),
            "--cache_dir",
            str(self.cache_dir),
            "--split",
            split,
            "--gpu",
            str(self.seg_gpu),
            *_seg_config_cli(seg_config, self.batch_size),
        ]
        self._run_cmd(cmd, phase="segment", label=label)

    def _sweep(self, seg_hash: str, crop_fracs: list[float], split: str, label: str) -> None:
        cmd = [
            self.python,
            str(self.scripts_dir / "sweep_metrics.py"),
            "--cache_dir",
            str(self.cache_dir),
            "--seg_hash",
            seg_hash,
            "--crop_frac",
            ",".join(str(x) for x in crop_fracs),
            "--seed",
            str(SEED),
            "--split",
            split,
            "--gt_csv",
            str(self.labels_csv),
        ]
        self._run_cmd(cmd, phase="sweep", label=label)

    def _mean_error(self, seg_config: SegConfig, crop_frac: float, split: str) -> float | None:
        metric = MetricConfig(random_crop_frac=crop_frac, random_crop_seed=SEED)
        exp_hash = experiment_hash(seg_config, metric)
        pred_csv = pred_artifact_path(self.cache_dir, exp_hash, split)
        if not pred_csv.exists():
            return None
        pred_df = pd.read_csv(pred_csv)
        gt_df = pd.read_csv(self.labels_csv)
        return float(score_by_id(pred_df, gt_df)["mean"])

    def start_regression(self) -> None:
        if self.skip_regression:
            self.log("Skipping regression arm.")
            return
        log_path = self.results_dir / "logs" / "regression.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            self.python,
            str(self.scripts_dir / "run_training.py"),
            "--method",
            "regression",
            "--gpu",
            "0",
            "--cache_dir",
            str(self.cache_dir),
            "--labels_csv",
            str(self.labels_csv),
            "--results_dir",
            str(self.results_dir / "training"),
            "--epochs",
            str(self.epochs),
            "--seeds",
            ",".join(str(s) for s in REGRESSION_SEEDS),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(self.train_gpu)
        self.log(f"Starting regression seeds {REGRESSION_SEEDS} on GPU {self.train_gpu}")
        handle = open(log_path, "w")
        self.regression_proc = subprocess.Popen(
            cmd,
            cwd=self.code_root,
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
        self.manifest["regression_pid"] = self.regression_proc.pid
        self._save_manifest()

    def wait_regression(self) -> None:
        if self.regression_proc is None:
            return
        code = self.regression_proc.wait()
        if code != 0:
            raise RuntimeError(f"Regression subprocess exited {code}")
        self.log("Regression arm finished.")

    def tune_baseline(self) -> None:
        self._segment(DEFAULT_SEG, "val", "default cyto masks on val")
        self._sweep(
            config_hash(DEFAULT_SEG),
            CROP_SWEEP,
            "val",
            "crop sweep on val",
        )
        best_crop = 0.4
        best_err = float("inf")
        for crop in CROP_SWEEP:
            err = self._mean_error(DEFAULT_SEG, crop, "val")
            self.manifest["configs"].append(
                {
                    "name": f"crop_{crop:.2f}",
                    "seg": "cyto_default",
                    "crop_frac": crop,
                    "split": "val",
                    "mean_error_pct": err,
                }
            )
            if err is not None and err < best_err:
                best_err = err
                best_crop = crop
        self.best_crop_frac = best_crop
        self.log(f"Best crop on val: {best_crop} (mean {best_err:.2f}%)")

        candidates: list[tuple[str, SegConfig, float]] = []
        default_err = self._mean_error(DEFAULT_SEG, best_crop, "val")
        if default_err is not None:
            candidates.append(("cyto_default", DEFAULT_SEG, default_err))

        for name, seg_config in ABLATION_CONFIGS:
            self._segment(seg_config, "val", f"ablation {name} masks on val")
            self._sweep(
                config_hash(seg_config),
                [best_crop],
                "val",
                f"score {name} on val",
            )
            err = self._mean_error(seg_config, best_crop, "val")
            self.manifest["configs"].append(
                {
                    "name": name,
                    "crop_frac": best_crop,
                    "split": "val",
                    "mean_error_pct": err,
                    "seg_hash": config_hash(seg_config),
                }
            )
            if err is not None:
                candidates.append((name, seg_config, err))

        candidates.sort(key=lambda item: item[2])
        name, seg_config, err = candidates[0]
        default = next((c for c in candidates if c[0] == "cyto_default"), None)
        if default is not None and abs(default[2] - err) < 0.5:
            name, seg_config, err = default
            self.log("Top ablation within 0.5% of default; keeping default.")
        self.best_seg_name = name
        self.best_seg_config = seg_config
        self.manifest["best_seg_name"] = name
        self.manifest["best_seg_hash"] = config_hash(seg_config)
        self.manifest["best_crop_frac"] = best_crop
        self.manifest["best_val_mean_error_pct"] = err
        self._save_manifest()
        self.log(f"Frozen Cellpose config: {name} crop={best_crop} val_mean={err:.2f}%")

    def materialize_frozen_preds(self, splits: tuple[str, ...]) -> dict[str, Path]:
        metric = MetricConfig(
            random_crop_frac=self.best_crop_frac, random_crop_seed=SEED
        )
        exp_hash = experiment_hash(self.best_seg_config, metric)
        paths = {}
        for split in splits:
            self._segment(
                self.best_seg_config,
                split,
                f"frozen {self.best_seg_name} masks on {split}",
            )
            self._sweep(
                config_hash(self.best_seg_config),
                [self.best_crop_frac],
                split,
                f"frozen preds on {split}",
            )
            paths[split] = pred_artifact_path(self.cache_dir, exp_hash, split)
        self.manifest["frozen_exp_hash"] = exp_hash
        self._save_manifest()
        return paths

    def run_calibration(self, pred_paths: dict[str, Path]) -> None:
        out_dir = self.results_dir / "training" / "calibration"
        train_calibration(
            self.cache_dir,
            self.labels_csv,
            out_dir,
            pred_csv_train=pred_paths["train"],
            pred_csv_val=pred_paths["val"],
        )
        self.log("Calibration frozen on val.")

    def evaluate_test(self, pred_paths: dict[str, Path]) -> None:
        stamp = datetime.now(timezone.utc).isoformat()
        self.manifest["test_access_at"] = stamp
        self._save_manifest()
        self.log(f"Opening test split at {stamp}")

        gt = pd.read_csv(self.labels_csv)
        test_scores: dict[str, dict] = {}

        baseline_pred = pd.read_csv(pred_paths["test"])
        baseline_scores = paired_bootstrap_mape(baseline_pred, gt)
        test_scores["cellpose_baseline"] = baseline_scores
        baseline_pred.to_csv(self.results_dir / "predictions_test_baseline.csv", index=False)

        cal_dir = self.results_dir / "training" / "calibration"
        cal_scores = evaluate_calibration_split(
            self.cache_dir,
            self.labels_csv,
            cal_dir,
            pred_csv=pred_paths["test"],
            split="test",
        )
        cal_pred = pd.read_csv(cal_dir / "predictions_test.csv")
        test_scores["ridge_calibration"] = {
            **cal_scores,
            **{
                k: v
                for k, v in paired_bootstrap_mape(cal_pred, gt).items()
                if k in {"ci_low", "ci_high", "n", "n_boot"}
            },
        }

        if not self.skip_regression:
            seed_scores = []
            for seed in REGRESSION_SEEDS:
                seed_dir = self.results_dir / "training" / "regression_cnn" / f"seed_{seed}"
                scores = evaluate_regression_split(
                    self.cache_dir,
                    self.labels_csv,
                    seed_dir,
                    split="test",
                    gpu=self.train_gpu,
                )
                pred = pd.read_csv(seed_dir / "predictions_test.csv")
                boot = paired_bootstrap_mape(pred, gt)
                seed_scores.append({"seed": seed, **boot})
            test_scores["regression_cnn"] = seed_scores

        self.manifest["test_scores"] = test_scores
        self.manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        self._save_manifest()
        (self.results_dir / "test_scores.json").write_text(json.dumps(test_scores, indent=2))
        self.log(f"Test scores: {json.dumps(test_scores, indent=2)}")

    def write_report(self) -> None:
        cmd = [
            self.python,
            str(self.scripts_dir / "summarize_baseline.py"),
            "--manifest",
            str(self.manifest_path),
            "--results_dir",
            str(self.results_dir),
            "--cache_dir",
            str(self.cache_dir),
            "--gt_csv",
            str(self.labels_csv),
        ]
        self._run_cmd(cmd, phase="report", label="write REPORT.md")

    def run(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        (self.results_dir / "logs").mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as log_file:
            self._log_file = log_file
            self.log(
                f"Starting fair overnight run (seg_gpu={self.seg_gpu}, "
                f"train_gpu={self.train_gpu}, dry_run={self.dry_run})"
            )
            self.ensure_ready()
            if self.dry_run:
                self.manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
                self.manifest["dry_run"] = True
                self._save_manifest()
                self.write_report()
                self.log("Dry run complete.")
                return
            self.start_regression()
            try:
                self.tune_baseline()
                pred_paths = self.materialize_frozen_preds(("train", "val"))
                self.run_calibration(pred_paths)
                self.wait_regression()
                pred_paths.update(self.materialize_frozen_preds(("test",)))
                self.evaluate_test(pred_paths)
                self.write_report()
                self.log("Overnight run complete.")
            except Exception as exc:
                (self.results_dir / "FAILED").write_text(
                    f"{datetime.now(timezone.utc).isoformat()}\n{type(exc).__name__}: {exc}\n"
                )
                if self.regression_proc is not None and self.regression_proc.poll() is None:
                    self.regression_proc.terminate()
                raise


def main() -> None:
    args = build_parser().parse_args()
    OvernightRunner(args).run()


if __name__ == "__main__":
    main()
