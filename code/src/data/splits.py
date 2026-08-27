"""Slide-level grouped train/val/test splits (NumPy only)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROLES = ("train", "val", "test")


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

    assert_protocol_splits(index_df, splits)

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
    return [int(i) for i in splits[name]]


def assert_protocol_splits(
    index_df: pd.DataFrame,
    splits: dict[str, list[int]],
) -> dict:
    """Require image- and slide-disjoint train/val/test roles.

    Raises ValueError if any role is missing, empty, or leaks images/slides.
    """
    missing = [role for role in ROLES if role not in splits]
    if missing:
        raise ValueError(f"Missing split roles: {missing}")

    sets = {role: {int(i) for i in splits[role]} for role in ROLES}
    empty = [role for role, values in sets.items() if not values]
    if empty:
        raise ValueError(f"Empty split roles: {empty}")

    image_overlap: dict[str, list[int]] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sets[left] & sets[right]
        if overlap:
            image_overlap[f"{left}&{right}"] = sorted(overlap)[:20]
    if image_overlap:
        raise ValueError(f"Image overlap between splits: {image_overlap}")

    assigned = sets["train"] | sets["val"] | sets["test"]
    all_idx = set(index_df["idx"].astype(int))
    if assigned != all_idx:
        raise ValueError(
            "Split assignment does not cover the index exactly "
            f"(assigned={len(assigned)}, index={len(all_idx)}, "
            f"missing={len(all_idx - assigned)}, extra={len(assigned - all_idx)})"
        )

    slide_overlap: dict[str, list[str]] = {}
    n_slides: dict[str, int] = {}
    if "slide_id" in index_df.columns:
        idx_to_slide = {
            int(idx): str(slide)
            for idx, slide in zip(index_df["idx"], index_df["slide_id"])
        }
        slide_sets = {
            role: {idx_to_slide[i] for i in values if i in idx_to_slide}
            for role, values in sets.items()
        }
        n_slides = {role: len(slides) for role, slides in slide_sets.items()}
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = slide_sets[left] & slide_sets[right]
            if overlap:
                slide_overlap[f"{left}&{right}"] = sorted(overlap)[:20]
        if slide_overlap:
            raise ValueError(f"Slide overlap between splits: {slide_overlap}")

    return {
        "n_images": len(assigned),
        "counts": {role: len(sets[role]) for role in ROLES},
        "n_slides": n_slides,
        "image_overlap": {},
        "slide_overlap": {},
    }
