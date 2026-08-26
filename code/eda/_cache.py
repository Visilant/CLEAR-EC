"""Shared cache loading and CLI helpers for EDA phases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from src.data.cache import build_image_cache, open_image_cache
from src.data.segment import resolve_indices as _resolve_indices
from src.data.splits import load_splits


def resolve_path(path: str | Path) -> Path:
    """Resolve a path relative to code/ unless already absolute."""
    path = Path(path)
    if path.is_absolute():
        return path
    return (CODE_ROOT / path).resolve()


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Register CLI arguments shared across EDA scripts."""
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="../data/cache",
        help="Memmap cache directory (images_u8.npy, index.csv, splits.json).",
    )
    parser.add_argument(
        "--labels_csv",
        type=str,
        default="../data/final_train_ids.csv",
        help="Ground-truth labels CSV (ID, CD, CV, HEX, slide_id).",
    )
    parser.add_argument(
        "--green_dir",
        type=str,
        default="../data/train_green",
        help="Directory of green ROI overlay TIFFs (phase 3).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../results/eda",
        help="Root directory for EDA figures and CSVs.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
        help="Slide-level split to analyze.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, cap indices within the chosen split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for subsampling and crop reproducibility.",
    )


def resolve_indices(
    cache_dir: Path,
    split: str,
    limit: int | None = None,
) -> list[int]:
    """Resolve cache indices for a split, optionally capped."""
    return _resolve_indices(cache_dir, split, limit=limit)


def _attach_split_names(index_df: pd.DataFrame, splits: dict[str, list[int]]) -> pd.DataFrame:
    idx_to_split: dict[int, str] = {}
    for split_name, idx_list in splits.items():
        for idx in idx_list:
            idx_to_split[int(idx)] = split_name

    out = index_df.copy()
    out["split_name"] = out["idx"].astype(int).map(idx_to_split)
    return out


def load_eda_frame(
    cache_dir: Path,
    labels_csv: Path,
    split: str = "all",
) -> tuple[np.memmap, pd.DataFrame, dict[str, list[int]]]:
    """Return memmap, merged DataFrame (idx, ID, slide_id, CD, CV, HEX, split_name), splits dict."""
    memmap, index_df = open_image_cache(cache_dir)
    labels_df = pd.read_csv(labels_csv)
    labels_df["ID"] = labels_df["ID"].astype(str).str.strip()

    merged = index_df.merge(labels_df, on="ID", how="left", suffixes=("", "_label"))
    splits = load_splits(cache_dir)
    merged = _attach_split_names(merged, splits)

    if split != "all":
        split_indices = set(splits[split])
        merged = merged[merged["idx"].astype(int).isin(split_indices)].reset_index(drop=True)

    return memmap, merged, splits


def ensure_cache_exists(
    cache_dir: Path,
    labels_csv: Path,
    *,
    data_dir: Path | None = None,
    seed: int = 42,
    limit: int | None = None,
) -> None:
    """Build the memmap cache if images_u8.npy or index.csv is missing."""
    cache_dir = Path(cache_dir)
    memmap_path = cache_dir / "images_u8.npy"
    index_path = cache_dir / "index.csv"

    if memmap_path.exists() and index_path.exists():
        return

    if data_dir is None:
        data_dir = CODE_ROOT / "../data/train_mha"

    print(f"Cache not found under {cache_dir}; building from {data_dir} ...")
    build_image_cache(
        data_dir=Path(data_dir),
        labels_csv=Path(labels_csv),
        cache_dir=cache_dir,
        seed=seed,
        limit=limit,
        expected_count=9000 if limit is None else 0,
    )
