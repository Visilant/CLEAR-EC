#!/usr/bin/env python3
"""Build the shared memmap image cache and slide-level splits."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache import build_image_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build CLEAR-EC memmap image cache.")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/train_mha",
        help="Directory of .mha training images.",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="../data/final_train_ids.csv",
        help="CSV with ID and slide_id columns.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../data/cache",
        help="Output cache directory.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split RNG seed.")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, build cache for only the first N labeled images.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even if cache already exists.",
    )
    parser.add_argument(
        "--force_cache",
        action="store_true",
        help="Alias for --force.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    code_root = Path(__file__).resolve().parents[1]
    data_dir = (code_root / args.data_dir).resolve()
    labels_csv = (code_root / args.labels_csv).resolve()
    cache_dir = (code_root / args.cache_dir).resolve()
    limit = args.limit if args.limit > 0 else None

    build_image_cache(
        data_dir=data_dir,
        labels_csv=labels_csv,
        cache_dir=cache_dir,
        seed=args.seed,
        force=args.force or args.force_cache,
        limit=limit,
        expected_count=9000 if limit is None else 0,
    )


if __name__ == "__main__":
    main()
