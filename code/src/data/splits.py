"""Slide-level grouped train/val/test splits (NumPy only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def build_splits(
    index_df: pd.DataFrame,
    cache_dir: Path,
    *,
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict[str, list[int]]:
    """Assign images to splits by slide_id to avoid leakage."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    slide_ids = index_df["slide_id"].astype(str).unique()
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(slide_ids)

    n_slides = len(shuffled)
    n_train = int(n_slides * train_frac)
    n_val = int(n_slides * val_frac)

    train_slides = set(shuffled[:n_train])
    val_slides = set(shuffled[n_train : n_train + n_val])
    test_slides = set(shuffled[n_train + n_val :])

    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for _, row in index_df.iterrows():
        idx = int(row["idx"])
        slide = str(row["slide_id"])
        if slide in train_slides:
            splits["train"].append(idx)
        elif slide in val_slides:
            splits["val"].append(idx)
        else:
            splits["test"].append(idx)

    for name in splits:
        splits[name].sort()

    splits_path = cache_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump(splits, f, indent=2)

    print(
        f"Wrote splits -> {splits_path}  "
        f"(train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])})"
    )
    return splits


def load_splits(cache_dir: Path) -> dict[str, list[int]]:
    """Load splits.json from the cache directory."""
    splits_path = Path(cache_dir) / "splits.json"
    with open(splits_path) as f:
        return json.load(f)


def load_split(cache_dir: Path, name: str) -> list[int]:
    """Return the list of image indices for split `name`."""
    splits = load_splits(cache_dir)
    if name not in splits:
        raise KeyError(f"Unknown split {name!r}; available: {list(splits)}")
    return splits[name]
