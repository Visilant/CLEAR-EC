#!/usr/bin/env python3
"""Run CLEAR-EC training arms: regression CNN and ridge calibration."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache import open_image_cache
from src.training.calibration import CalibrationConfig, train_calibration
from src.training.common import resolve_path, split_summary, update_manifest
from src.training.regression_cnn import RegressionConfig, train_regression_cnn

VALID_METHODS = ("regression", "calibration")


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    if not methods:
        raise SystemExit("No training methods specified.")
    unknown = [item for item in methods if item not in VALID_METHODS]
    if unknown:
        raise SystemExit(
            f"Unknown or removed method(s): {unknown}. "
            f"Valid: {', '.join(VALID_METHODS)}"
        )
    return methods


def _method_arg(value: str) -> str:
    parse_methods(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLEAR-EC training pipeline CLI.")
    parser.add_argument(
        "--method",
        type=_method_arg,
        default="regression",
        help="Comma-separated methods: regression,calibration",
    )
    parser.add_argument("--gpu", type=int, default=1, help="GPU for regression.")
    parser.add_argument("--cache_dir", type=str, default="../data/cache")
    parser.add_argument("--labels_csv", type=str, default="../data/final_train_ids.csv")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results/training",
        help="Root for per-method outputs and manifest.json.",
    )
    parser.add_argument(
        "--pred_csv_train",
        type=str,
        default="",
        help="Frozen Cellpose train-split predictions for calibration.",
    )
    parser.add_argument(
        "--pred_csv_val",
        type=str,
        default="",
        help="Frozen Cellpose val-split predictions for calibration.",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cpu_threads", type=int, default=4)
    parser.add_argument("--loss", choices=["huber", "mse"], default="huber")
    parser.add_argument("--normalization", choices=["batch", "group"], default="batch")
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels_last", action="store_true")
    parser.add_argument("--float_inputs", action="store_true", help="Legacy FP32 CPU preprocessing")
    parser.add_argument(
        "--seeds",
        type=str,
        default="42",
        help="Comma-separated regression seeds (default: 42).",
    )
    parser.add_argument(
        "--min_cache_count",
        type=int,
        default=9000,
        help="Require at least this many cached images to start training.",
    )
    return parser


def _cache_count(cache_dir: Path) -> int:
    _, index_df = open_image_cache(cache_dir)
    return len(index_df)


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise SystemExit("No regression seeds specified.")
    return seeds


def main() -> None:
    args = build_parser().parse_args()
    code_root = Path(__file__).resolve().parents[1]
    cache_dir = resolve_path(code_root, args.cache_dir)
    labels_csv = resolve_path(code_root, args.labels_csv)
    results_root = resolve_path(code_root, args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    n_cached = _cache_count(cache_dir)
    summary = split_summary(cache_dir)
    print(f"Cache: {n_cached} images | splits: {summary}")

    if n_cached < args.min_cache_count:
        print(
            f"WARNING: cache has {n_cached} images (< {args.min_cache_count}). "
            "Training will still run for smoke testing."
        )

    methods = parse_methods(args.method)
    seeds = _parse_seeds(args.seeds)

    manifest_path = results_root / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "cache_dir": str(cache_dir),
            "labels_csv": str(labels_csv),
            "split_summary": summary,
            "methods": {},
        }

    errors: dict[str, str] = {}
    for method in methods:
        try:
            if method == "regression":
                for seed in seeds:
                    cfg = RegressionConfig(
                        epochs=args.epochs,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        patience=args.patience,
                        seed=seed,
                        num_workers=args.num_workers,
                        cpu_threads=args.cpu_threads,
                        loss=args.loss,
                        normalization=args.normalization,
                        downsample=args.downsample,
                        weight_decay=args.weight_decay,
                        amp=args.amp,
                        channels_last=args.channels_last,
                        uint8_inputs=not args.float_inputs,
                    )
                    train_regression_cnn(
                        cache_dir,
                        labels_csv,
                        results_root / "regression_cnn" / f"seed_{seed}",
                        gpu=args.gpu,
                        config=cfg,
                    )
            elif method == "calibration":
                if not args.pred_csv_train or not args.pred_csv_val:
                    raise FileNotFoundError(
                        "Calibration requires --pred_csv_train and --pred_csv_val "
                        "from the frozen Cellpose config."
                    )
                train_calibration(
                    cache_dir,
                    labels_csv,
                    results_root / "calibration",
                    pred_csv_train=resolve_path(code_root, args.pred_csv_train),
                    pred_csv_val=resolve_path(code_root, args.pred_csv_val),
                    config=CalibrationConfig(),
                )
        except Exception as exc:
            errors[method] = str(exc)
            print(f"[{method}] ERROR: {exc}")

    update_manifest(results_root, {"errors": errors, "n_cached": n_cached,
                    "split_summary": summary, "cache_dir": str(cache_dir),
                    "labels_csv": str(labels_csv)})

    if errors:
        raise SystemExit(f"Training finished with errors: {errors}")
    print(f"Training complete -> {results_root}")


if __name__ == "__main__":
    main()
