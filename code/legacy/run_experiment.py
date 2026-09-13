#!/usr/bin/env python3
"""Orchestrate cache build, mask caching, metric sweep, and evaluation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache import build_image_cache, open_image_cache
from src.data.config import MetricConfig, SegConfig, config_hash
from src.data.splits import load_split
from src.training.common import pred_artifact_path, score_by_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a full CLEAR-EC experiment loop.")
    parser.add_argument("--cache_dir", type=str, default="../data/cache")
    parser.add_argument("--data_dir", type=str, default="../data/train_mha")
    parser.add_argument("--labels_csv", type=str, default="../data/final_train_ids.csv")
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument(
        "--seg_config",
        type=str,
        default="",
        help="Optional JSON file with SegConfig fields.",
    )
    parser.add_argument(
        "--metric_sweep",
        type=str,
        default="crop_frac=0.3,0.4,0.45",
        help="Metric sweep spec, e.g. crop_frac=0.3,0.4,0.45",
    )
    parser.add_argument("--gpus", type=str, default="0")
    parser.add_argument("--gt_csv", type=str, default="../data/final_train_ids.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force_cache", action="store_true")
    return parser


def _load_seg_config(path: Path) -> SegConfig:
    with open(path) as f:
        data = json.load(f)
    return SegConfig(**data)


def _parse_metric_sweep(spec: str) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        key, values = part.split("=", 1)
        out[key.strip()] = [float(v.strip()) for v in values.split(",") if v.strip()]
    return out


def _seg_config_cli(seg_config: SegConfig) -> list[str]:
    if seg_config.net_avg:
        raise ValueError("This runner currently supports net_avg=False only")
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
        str(seg_config.batch_size),
    ]
    if seg_config.diameter is not None:
        cmd.extend(["--diameter", str(seg_config.diameter)])
    if not seg_config.tile:
        cmd.append("--no_tile")
    return cmd


def main() -> None:
    args = build_parser().parse_args()
    code_root = Path(__file__).resolve().parents[1]
    cache_dir = (code_root / args.cache_dir).resolve()
    data_dir = (code_root / args.data_dir).resolve()
    labels_csv = (code_root / args.labels_csv).resolve()
    scripts_dir = Path(__file__).resolve().parent  # legacy/ holds the segmentation and sweep scripts

    limit = args.limit if args.limit > 0 else None

    build_image_cache(
        data_dir=data_dir,
        labels_csv=labels_csv,
        cache_dir=cache_dir,
        seed=args.seed,
        force=args.force_cache,
        # --limit restricts the experiment, never the shared dataset cache.
        limit=None,
        expected_count=9000,
    )

    if args.seg_config:
        seg_config = _load_seg_config((code_root / args.seg_config).resolve())
    else:
        seg_config = SegConfig()
    seg_hash = config_hash(seg_config)
    print(f"Using seg_hash={seg_hash}")

    seg_cmd = [
        sys.executable,
        str(scripts_dir / "run_segmentation_cache.py"),
        "--cache_dir",
        str(cache_dir),
        "--split",
        args.split,
        "--gpus",
        args.gpus,
        *_seg_config_cli(seg_config),
    ]
    if limit is not None:
        seg_cmd.extend(["--limit", str(limit)])
    subprocess.run(seg_cmd, check=True)

    sweep = _parse_metric_sweep(args.metric_sweep)
    crop_fracs = sweep.get("crop_frac", [0.4])
    crop_spec = ",".join(str(x) for x in crop_fracs)

    sweep_cmd = [
        sys.executable,
        str(scripts_dir / "sweep_metrics.py"),
        "--cache_dir",
        str(cache_dir),
        "--seg_hash",
        seg_hash,
        "--crop_frac",
        crop_spec,
        "--seed",
        str(args.seed),
        "--split",
        args.split,
    ]
    if limit is not None:
        sweep_cmd.extend(["--limit", str(limit)])
    if args.gt_csv:
        sweep_cmd.extend(["--gt_csv", str((code_root / args.gt_csv).resolve())])
    subprocess.run(sweep_cmd, check=True)

    if not args.gt_csv:
        print("No --gt_csv provided; skipping best-config selection.")
        return

    gt_df = pd.read_csv((code_root / args.gt_csv).resolve())
    _, index_df = open_image_cache(cache_dir)
    if args.split == "all":
        indices = index_df["idx"].astype(int).tolist()
    else:
        indices = load_split(cache_dir, args.split)
    if limit is not None:
        indices = indices[:limit]

    best = None
    for crop_frac in crop_fracs:
        metric_config = MetricConfig(random_crop_frac=crop_frac, random_crop_seed=args.seed)
        exp_hash = config_hash((seg_hash, metric_config))
        pred_csv = pred_artifact_path(cache_dir, exp_hash, args.split, limit=args.limit)
        if not pred_csv.exists():
            continue
        pred_df = pd.read_csv(pred_csv)
        expected_ids = index_df.loc[index_df.idx.isin(indices), "ID"]
        mean_err = score_by_id(pred_df, gt_df, expected_ids=expected_ids)["mean"]
        candidate = (mean_err, crop_frac, exp_hash, pred_csv)
        if best is None or candidate[0] < best[0]:
            best = candidate

    if best is None:
        print("Could not determine best metric config.")
        return

    mean_err, crop_frac, exp_hash, pred_csv = best
    print(
        f"\nBest config: crop_frac={crop_frac}, exp_hash={exp_hash}, "
        f"mean error={mean_err:.2f}% -> {pred_csv}"
    )


if __name__ == "__main__":
    main()
