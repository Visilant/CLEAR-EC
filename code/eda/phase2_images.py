"""Phase 2 — per-image intensity and blur statistics from memmap."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import laplace

from eda._cache import (
    add_common_args,
    load_eda_frame,
    resolve_indices,
    resolve_path,
)

METRICS = ["CD", "CV", "HEX"]


def _compute_image_stats(gray: np.ndarray) -> dict:
    flat = gray.ravel().astype(np.float64)
    p1, p99 = np.percentile(flat, [1, 99])
    lap_var = float(laplace(gray.astype(np.float64)).var())
    return {
        "height": gray.shape[0],
        "width": gray.shape[1],
        "mean_intensity": float(flat.mean()),
        "std_intensity": float(flat.std()),
        "min_intensity": int(flat.min()),
        "max_intensity": int(flat.max()),
        "dynamic_range_p99_p1": float(p99 - p1),
        "laplacian_var": lap_var,
        "pct_near_0": float((flat < 10).mean() * 100),
        "pct_near_255": float((flat > 245).mean() * 100),
    }


def _stratified_subsample(
    df: pd.DataFrame,
    indices: list[int],
    limit: int,
    seed: int,
) -> list[int]:
    """Sample evenly across CD quartiles within the chosen split."""
    sub = df[df["idx"].astype(int).isin(indices)].copy()
    sub = sub.dropna(subset=["CD"])
    if len(sub) <= limit:
        return indices[:limit] if len(indices) > limit else indices

    rng = np.random.default_rng(seed)
    sub["cd_quartile"] = pd.qcut(sub["CD"], q=4, labels=False, duplicates="drop")
    per_q = max(1, limit // sub["cd_quartile"].nunique())
    chosen: list[int] = []
    for q in sorted(sub["cd_quartile"].unique()):
        q_idx = sub.loc[sub["cd_quartile"] == q, "idx"].astype(int).tolist()
        rng.shuffle(q_idx)
        chosen.extend(q_idx[:per_q])

    if len(chosen) < limit:
        remaining = [i for i in indices if i not in chosen]
        rng.shuffle(remaining)
        chosen.extend(remaining[: limit - len(chosen)])
    return chosen[:limit]


def _plot_intensity_histogram(stats_df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(stats_df["mean_intensity"], bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel("mean intensity")
    ax.set_ylabel("count")
    ax.set_title("Per-image mean intensity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_blur_vs_cd(stats_df: pd.DataFrame, out_path: Path) -> None:
    sub = stats_df.dropna(subset=["laplacian_var", "CD"])
    if sub.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(sub["laplacian_var"], sub["CD"], s=12, alpha=0.6)
    ax.set_xlabel("Laplacian variance (blur proxy)")
    ax.set_ylabel("CD")
    ax.set_title("Blur vs cell density")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_montage(
    memmap: np.memmap,
    stats_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """3x3 montage: low/median/high CD, CV, HEX."""
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    for row, metric in enumerate(METRICS):
        sub = stats_df.dropna(subset=[metric]).sort_values(metric)
        if len(sub) < 3:
            picks = sub
        else:
            picks = pd.concat([sub.iloc[[0]], sub.iloc[[len(sub) // 2]], sub.iloc[[-1]]])
        labels = ["low", "median", "high"]
        for col, (_, row_data) in enumerate(picks.iterrows()):
            idx = int(row_data["idx"])
            axes[row, col].imshow(memmap[idx], cmap="gray")
            axes[row, col].set_title(f"{metric} {labels[col]}: {row_data[metric]:.3g}")
            axes[row, col].axis("off")
    fig.suptitle("Example images by metric tertiles")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    cache_dir = resolve_path(args.cache_dir)
    labels_csv = resolve_path(args.labels_csv)
    out_dir = resolve_path(args.output_dir) / "phase2_images"
    out_dir.mkdir(parents=True, exist_ok=True)

    memmap, merged, _ = load_eda_frame(cache_dir, labels_csv, split="all")

    limit = args.limit if args.limit > 0 else None
    indices = resolve_indices(cache_dir, args.split, limit=None)
    if limit is not None:
        indices = _stratified_subsample(merged, indices, limit, args.seed)

    rows = []
    for idx in indices:
        row = merged.loc[merged["idx"].astype(int) == idx]
        if row.empty:
            continue
        meta = row.iloc[0]
        stats = _compute_image_stats(memmap[idx])
        rows.append(
            {
                "idx": idx,
                "ID": meta["ID"],
                "slide_id": meta["slide_id"],
                "split_name": meta.get("split_name"),
                "CD": meta.get("CD"),
                "CV": meta.get("CV"),
                "HEX": meta.get("HEX"),
                **stats,
            }
        )

    stats_df = pd.DataFrame(rows)
    stats_df.to_csv(out_dir / "image_stats.csv", index=False)

    _plot_intensity_histogram(stats_df, out_dir / "intensity_histogram.png")
    _plot_blur_vs_cd(stats_df, out_dir / "blur_vs_cd_scatter.png")
    if not stats_df.empty:
        _plot_montage(memmap, stats_df, out_dir / "metric_montage.png")

    print(f"Phase 2 complete -> {out_dir} ({len(stats_df)} images)")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 2: image EDA.")
    add_common_args(parser)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
