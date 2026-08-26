"""Phase 1 — label distributions and split audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eda._cache import (
    add_common_args,
    load_eda_frame,
    resolve_indices,
    resolve_path,
)

METRICS = ["CD", "CV", "HEX"]
OUTLIER_RULES = {
    "CD_low": lambda df: df["CD"] < 500,
    "CD_high": lambda df: df["CD"] > 4500,
    "CV_high": lambda df: df["CV"] > 1.5,
    "HEX_low": lambda df: df["HEX"] < 0.05,
}


def _summary_stats(df: pd.DataFrame, group: str) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        if metric not in df.columns:
            continue
        series = df[metric].dropna()
        if series.empty:
            continue
        rows.append(
            {
                "group": group,
                "metric": metric,
                "count": len(series),
                "mean": series.mean(),
                "std": series.std(),
                "min": series.min(),
                "q25": series.quantile(0.25),
                "median": series.median(),
                "q75": series.quantile(0.75),
                "max": series.max(),
            }
        )
    return pd.DataFrame(rows)


def _split_summary(merged: pd.DataFrame, splits: dict[str, list[int]]) -> pd.DataFrame:
    rows = []
    for split_name, idx_list in splits.items():
        sub = merged[merged["idx"].astype(int).isin(idx_list)]
        rows.append(
            {
                "split": split_name,
                "n_images": len(sub),
                "n_slides": sub["slide_id"].nunique(),
            }
        )
    rows.append(
        {
            "split": "all",
            "n_images": len(merged),
            "n_slides": merged["slide_id"].nunique(),
        }
    )
    return pd.DataFrame(rows)


def _flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    flags = []
    for _, row in df.iterrows():
        reasons = []
        for name, rule in OUTLIER_RULES.items():
            if rule(pd.DataFrame([row])).iloc[0]:
                reasons.append(name)
        if reasons:
            flags.append({"idx": row["idx"], "ID": row["ID"], "flags": ";".join(reasons)})
    return pd.DataFrame(flags)


def _plot_histograms(df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, metric in zip(axes, METRICS):
        data = df[metric].dropna()
        use_log = metric == "CV"
        ax.hist(data, bins=40, color="steelblue", edgecolor="white")
        if use_log:
            ax.set_xscale("log")
        ax.set_title(f"{metric} distribution")
        ax.set_xlabel(metric)
        ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_correlation(df: pd.DataFrame, out_path: Path) -> None:
    sub = df[METRICS].dropna()
    if sub.empty:
        return
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    pairs = [(METRICS[i], METRICS[j]) for i in range(3) for j in range(3)]
    for ax, (x, y) in zip(axes.ravel(), pairs):
        if x == y:
            ax.hist(sub[x], bins=30, color="gray", alpha=0.7)
            ax.set_title(x)
        else:
            ax.scatter(sub[x], sub[y], s=8, alpha=0.5)
            ax.set_xlabel(x)
            ax.set_ylabel(y)
        ax.tick_params(labelsize=8)
    fig.suptitle("Metric correlation matrix", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_cd_by_slide(df: pd.DataFrame, out_path: Path, top_n: int = 15) -> None:
    counts = df.groupby("slide_id").size().sort_values(ascending=False)
    top_slides = counts.head(top_n).index.tolist()
    sub = df[df["slide_id"].isin(top_slides)]
    if sub.empty:
        return

    slide_order = (
        sub.groupby("slide_id")["CD"].median().sort_values(ascending=False).index.tolist()
    )
    data = [sub.loc[sub["slide_id"] == s, "CD"].dropna().values for s in slide_order]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(data, tick_labels=slide_order, vert=True)
    ax.set_xticklabels(slide_order, rotation=45, ha="right")
    ax.set_ylabel("CD")
    ax.set_title(f"CD by slide (top {len(slide_order)} by count)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    cache_dir = resolve_path(args.cache_dir)
    labels_csv = resolve_path(args.labels_csv)
    out_dir = resolve_path(args.output_dir) / "phase1_labels"
    out_dir.mkdir(parents=True, exist_ok=True)

    _, merged, splits = load_eda_frame(cache_dir, labels_csv, split="all")

    limit = args.limit if args.limit > 0 else None
    indices = resolve_indices(cache_dir, args.split, limit=limit)
    if args.split != "all":
        merged = merged[merged["idx"].astype(int).isin(indices)].reset_index(drop=True)

    summary_parts = [_summary_stats(merged, "all")]
    for split_name in ("train", "val", "test"):
        sub = merged[merged["split_name"] == split_name]
        if not sub.empty:
            summary_parts.append(_summary_stats(sub, split_name))

    summary_df = pd.concat(summary_parts, ignore_index=True)
    summary_df = summary_df.round(4)
    summary_df.to_csv(out_dir / "summary_stats.csv", index=False)

    split_df = _split_summary(merged, splits)
    split_df.to_csv(out_dir / "split_summary.csv", index=False)

    outliers_df = _flag_outliers(merged)
    if not outliers_df.empty:
        outliers_df.to_csv(out_dir / "outlier_flags.csv", index=False)

    missing = merged[METRICS].isna().sum()
    if missing.any():
        print("Warning: NaN label counts after merge:")
        print(missing[missing > 0])

    _plot_histograms(merged, out_dir / "cd_cv_hex_histograms.png")
    _plot_correlation(merged, out_dir / "metric_correlation.png")
    _plot_cd_by_slide(merged, out_dir / "cd_by_slide_boxplot.png")

    print(f"Phase 1 complete -> {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 1: label EDA.")
    add_common_args(parser)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
