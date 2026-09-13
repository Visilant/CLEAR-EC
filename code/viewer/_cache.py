"""Cache + label merge helpers for the training data viewer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cache import open_image_cache
from src.data.splits import load_split, load_splits

METRIC_COLS = ["CD", "CV", "HEX"]


def resolve_repo_data_dir(code_root: Path | None = None) -> Path:
    """Find shared data/ whether running from main checkout or a worktree."""
    code_root = (code_root or Path(__file__).resolve().parents[1]).resolve()

    for rel in ("../data", "../../../data"):
        candidate = (code_root / rel).resolve()
        if candidate.is_dir():
            return candidate

    for parent in code_root.parents:
        candidate = parent / "data"
        if candidate.is_dir() and (candidate / "final_train_ids.csv").exists():
            return candidate

    return (code_root / "../data").resolve()


def _attach_split_names(index_df: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    splits = load_splits(cache_dir)
    idx_to_split: dict[int, str] = {}
    for split_name, indices in splits.items():
        for idx in indices:
            idx_to_split[int(idx)] = split_name
    out = index_df.copy()
    out["split_name"] = out["idx"].astype(int).map(idx_to_split)
    return out


def load_viewer_frame(
    cache_dir: Path | str,
    labels_csv: Path | str,
    split: str = "all",
) -> tuple[np.memmap, pd.DataFrame]:
    """Return memmap and merged DataFrame filtered by split.

    Columns include idx, ID, slide_id, CD, CV, HEX, split_name.
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    labels_csv = Path(labels_csv).expanduser().resolve()

    memmap, index_df = open_image_cache(cache_dir)
    labels_df = pd.read_csv(labels_csv)
    labels_df["ID"] = labels_df["ID"].astype(str).str.strip()

    merged = index_df.merge(labels_df, on="ID", how="left", suffixes=("", "_label"))
    if "slide_id_label" in merged.columns:
        merged["slide_id"] = merged["slide_id"].fillna(merged["slide_id_label"])
        merged = merged.drop(columns=["slide_id_label"])

    merged = _attach_split_names(merged, cache_dir)

    if split != "all":
        split_indices = set(load_split(cache_dir, split))
        merged = merged[merged["idx"].astype(int).isin(split_indices)].reset_index(drop=True)

    return memmap, merged


def split_summary(cache_dir: Path | str) -> dict[str, int]:
    """Return image counts per split from splits.json."""
    splits = load_splits(Path(cache_dir))
    return {name: len(indices) for name, indices in splits.items()}
