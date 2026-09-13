"""Slide-grouped cross-validation folds over all labelled images, and index-set resolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.cache import open_image_cache
from src.data.splits import load_split


def all_labelled_indices(cache_dir: Path) -> list[int]:
    """Sorted union of the train, val and test splits (all 9,000 labelled images)."""
    return sorted(
        set(load_split(cache_dir, "train"))
        | set(load_split(cache_dir, "val"))
        | set(load_split(cache_dir, "test"))
    )


def _slide_group_folds(cache_dir: Path, n_folds: int, seed: int = 42) -> dict[int, list[int]]:
    """Deterministic slide-grouped K-fold split over all labelled indices (train+val+test)."""
    _, index_df = open_image_cache(cache_dir)
    combined = sorted(
        set(load_split(cache_dir, "train"))
        | set(load_split(cache_dir, "val"))
        | set(load_split(cache_dir, "test"))
    )
    idx_to_slide = {int(idx): str(slide) for idx, slide in zip(index_df["idx"], index_df["slide_id"])}
    slides = sorted({idx_to_slide[i] for i in combined})
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(slides))
    fold_of_slide = {slide: i % n_folds for i, slide in enumerate(shuffled)}
    folds: dict[int, list[int]] = {k: [] for k in range(n_folds)}
    for i in combined:
        folds[fold_of_slide[idx_to_slide[i]]].append(i)
    for k in folds:
        folds[k].sort()
    return folds


def resolve_indices(cache_dir: Path, spec: str) -> list[int]:
    """train | val | test | all | fold:k/n | <path to a whitespace-separated idx file>."""
    if spec in {"train", "val", "test"}:
        return load_split(cache_dir, spec)
    if spec == "all":
        return all_labelled_indices(cache_dir)
    if spec.startswith("fold:"):
        k, n = spec[5:].split("/")
        return _slide_group_folds(cache_dir, int(n), seed=42)[int(k)]
    return [int(x) for x in Path(spec).read_text().split()]
