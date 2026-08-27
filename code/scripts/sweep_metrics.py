#!/usr/bin/env python3
"""Fast metric-only sweeps over cached segmentation masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache import open_image_cache
from src.data.config import MetricConfig, config_hash
from src.data.mask_cache import load_mask
from src.data.metrics import metrics_from_mask
from src.data.splits import load_split
from src.training.common import pred_artifact_path, score_by_id


def _parse_float_list(value: str) -> list[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sweep metric configs on cached masks.")
    parser.add_argument("--cache_dir", type=str, default="../data/cache")
    parser.add_argument("--seg_hash", type=str, required=True)
    parser.add_argument(
        "--crop_frac",
        type=str,
        default="0.4",
        help="Comma-separated random crop fractions.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--gt_csv",
        type=str,
        default="",
        help="Optional ground-truth CSV for summary evaluation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    code_root = Path(__file__).resolve().parents[1]
    cache_dir = (code_root / args.cache_dir).resolve()
    limit = args.limit if args.limit > 0 else None

    memmap, index_df = open_image_cache(cache_dir)
    if args.split == "all":
        indices = index_df["idx"].astype(int).tolist()
    else:
        indices = load_split(cache_dir, args.split)
    if limit is not None:
        indices = indices[:limit]

    crop_fracs = _parse_float_list(args.crop_frac)
    gt_df = pd.read_csv(args.gt_csv) if args.gt_csv else None

    summary_rows = []
    for crop_frac in crop_fracs:
        metric_config = MetricConfig(
            random_crop_frac=crop_frac,
            random_crop_seed=args.seed,
        )
        exp_hash = config_hash((args.seg_hash, metric_config))
        out_csv = pred_artifact_path(cache_dir, exp_hash, args.split)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for idx in indices:
            row = index_df.loc[index_df["idx"] == idx].iloc[0]
            image_id = str(row["ID"])
            masks_full = load_mask(cache_dir, args.seg_hash, idx)
            image_gray = memmap[idx]
            pred = metrics_from_mask(
                masks_full,
                image_gray,
                image_id,
                metric_config,
            )
            if pred is not None:
                rows.append({k: pred[k] for k in ("ID", "CD", "CV", "HEX")})

        pred_df = pd.DataFrame(rows)
        pred_df.to_csv(out_csv, index=False)
        print(f"Wrote {len(pred_df)} predictions -> {out_csv} (crop_frac={crop_frac})")

        if gt_df is not None and not pred_df.empty:
            scores = score_by_id(pred_df, gt_df)
            summary_rows.append(
                {
                    "crop_frac": crop_frac,
                    "exp_hash": exp_hash,
                    "split": args.split,
                    "n_preds": len(pred_df),
                    "CD Error (%)": scores["CD"],
                    "CV Error (%)": scores["CV"],
                    "HEX Error (%)": scores["HEX"],
                    "mean": scores["mean"],
                }
            )

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        print("\n=== Mean percent error by crop_frac ===")
        print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
