"""Phase 4 — baseline error analysis via cached masks."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cellpose import models
from tqdm import tqdm

from eda._cache import (
    add_common_args,
    load_eda_frame,
    resolve_indices,
    resolve_path,
)
from src.data.config import MetricConfig, SegConfig, config_hash
from src.data.crop import crop_rng_for_image, random_crop_bbox
from src.data.mask_cache import get_or_compute_mask, load_mask, mask_path
from src.data.metrics import _image_for_crop, metrics_from_mask
from src.infer_cellpose_sam import visualize_segmentation
from src.utils.evaluate import evaluate_results

METRICS = ["CD", "CV", "HEX"]
ERROR_COLS = [f"{m} Error (%)" for m in METRICS]


def _crop_bbox_for_image(image_gray: np.ndarray, metric_config: MetricConfig):
    if metric_config.random_crop_frac is None:
        return None
    image_for_crop = _image_for_crop(image_gray)
    rng = crop_rng_for_image(image_for_crop, metric_config.random_crop_seed)
    return random_crop_bbox(
        image_for_crop.shape,
        frac=metric_config.random_crop_frac,
        rng=rng,
    )


def _plot_error_by_metric(error_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, METRICS):
        col = f"{metric} Error (%)"
        data = error_df[col].dropna()
        if data.empty:
            ax.set_visible(False)
            continue
        ax.hist(data, bins=30, color="coral", edgecolor="white")
        ax.set_title(f"{metric} percent error")
        ax.set_xlabel("error (%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_pred_vs_actual(pred_df: pd.DataFrame, gt_df: pd.DataFrame, out_path: Path) -> None:
    merged = pred_df.merge(gt_df, on="ID", suffixes=("_pred", "_gt"))
    if merged.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, METRICS):
        x = merged[f"{metric}_gt"]
        y = merged[f"{metric}_pred"]
        ax.scatter(x, y, s=15, alpha=0.6)
        lims = [min(x.min(), y.min()), max(x.max(), y.max())]
        ax.plot(lims, lims, "k--", linewidth=1)
        ax.set_xlabel(f"GT {metric}")
        ax.set_ylabel(f"Pred {metric}")
        ax.set_title(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_error_vs_blur(
    error_df: pd.DataFrame,
    image_stats_path: Path,
    out_path: Path,
) -> None:
    if not image_stats_path.exists():
        return
    stats = pd.read_csv(image_stats_path)
    if "idx" not in stats.columns or "laplacian_var" not in stats.columns:
        return

    merged = error_df.merge(stats[["idx", "laplacian_var"]], on="idx", how="inner")
    if merged.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, METRICS):
        col = f"{metric} Error (%)"
        sub = merged.dropna(subset=[col, "laplacian_var"])
        ax.scatter(sub["laplacian_var"], sub[col], s=15, alpha=0.6)
        ax.set_xlabel("Laplacian variance")
        ax.set_ylabel(col)
    fig.suptitle("Error vs blur (phase 2 join)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_error_vs_roi_iou(
    error_df: pd.DataFrame,
    roi_csv_path: Path,
    out_path: Path,
) -> None:
    if not roi_csv_path.exists() or roi_csv_path.stat().st_size == 0:
        return
    try:
        roi = pd.read_csv(roi_csv_path)
    except pd.errors.EmptyDataError:
        return
    if roi.empty or "idx" not in roi.columns or "bbox_iou" not in roi.columns:
        return

    merged = error_df.merge(roi[["idx", "bbox_iou"]], on="idx", how="inner")
    if merged.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metric in zip(axes, METRICS):
        col = f"{metric} Error (%)"
        sub = merged.dropna(subset=[col, "bbox_iou"])
        ax.scatter(sub["bbox_iou"], sub[col], s=30, alpha=0.7)
        ax.set_xlabel("GT ROI vs random crop IoU")
        ax.set_ylabel(col)
    fig.suptitle("Error vs ROI-crop IoU (phase 3 join)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_worst_cases(
    error_df: pd.DataFrame,
    memmap: np.memmap,
    merged: pd.DataFrame,
    cache_dir: Path,
    seg_hash: str,
    metric_config: MetricConfig,
    out_dir: Path,
    top_n: int = 5,
) -> None:
    worst_dir = out_dir / "worst_cases"
    worst_dir.mkdir(parents=True, exist_ok=True)

    id_to_idx = dict(zip(merged["ID"].astype(str), merged["idx"].astype(int)))
    for metric in METRICS:
        col = f"{metric} Error (%)"
        sub = error_df.dropna(subset=[col]).nlargest(top_n, col)
        for _, row in sub.iterrows():
            image_id = str(row["ID"]).replace(".bmp", "")
            idx = int(row["idx"]) if "idx" in row and pd.notna(row["idx"]) else id_to_idx.get(image_id)
            if idx is None:
                continue
            masks = load_mask(cache_dir, seg_hash, idx)
            gray = memmap[idx]
            rgb = np.stack([gray] * 3, axis=-1)
            crop_bbox = _crop_bbox_for_image(gray, metric_config)
            save_path = worst_dir / f"{metric}_err_{image_id.replace('/', '_')}.png"
            visualize_segmentation(
                rgb,
                masks,
                image_id=image_id,
                save_path=str(save_path),
                show=False,
                crop_bbox=crop_bbox,
            )


def run(args: argparse.Namespace) -> Path:
    cache_dir = resolve_path(args.cache_dir)
    labels_csv = resolve_path(args.labels_csv)
    out_dir = resolve_path(args.output_dir) / "phase4_baseline"
    out_dir.mkdir(parents=True, exist_ok=True)

    memmap, merged, _ = load_eda_frame(cache_dir, labels_csv, split="all")
    gt_df = merged[["ID", "CD", "CV", "HEX"]].copy()

    limit = args.limit if args.limit > 0 else None
    indices = resolve_indices(cache_dir, args.split, limit=limit)

    seg_config = SegConfig()
    seg_hash = config_hash(seg_config)
    metric_config = MetricConfig(
        random_crop_frac=0.4,
        random_crop_seed=args.seed,
    )

    need_compute = [
        idx
        for idx in indices
        if not mask_path(cache_dir, seg_hash, idx).exists()
    ]
    model = None
    if need_compute:
        print(f"Computing masks for {len(need_compute)} images (GPU required)...")
        try:
            model = models.Cellpose(gpu=True, model_type=seg_config.model_type)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize Cellpose GPU model: {exc}. "
                "Run scripts/run_segmentation_cache.py first or use a GPU machine."
            ) from exc

    pred_rows = []
    for idx in tqdm(indices, desc="Phase 4 metrics"):
        row = merged.loc[merged["idx"].astype(int) == idx].iloc[0]
        image_id = str(row["ID"])
        image_gray = memmap[idx]

        if model is not None and idx in need_compute:
            get_or_compute_mask(
                idx=idx,
                image_gray=image_gray,
                seg_config=seg_config,
                model=model,
                cache_dir=cache_dir,
                skip_existing=True,
            )

        masks_full = load_mask(cache_dir, seg_hash, idx)
        pred = metrics_from_mask(
            masks_full,
            image_gray,
            image_id,
            metric_config,
        )
        if pred is not None:
            pred_rows.append(
                {
                    "idx": idx,
                    "ID": pred["ID"],
                    "CD": pred["CD"],
                    "CV": pred["CV"],
                    "HEX": pred["HEX"],
                }
            )

    pred_df = pd.DataFrame(pred_rows)
    split_tag = args.split
    pred_path = out_dir / f"predictions_{split_tag}.csv"
    pred_df.to_csv(pred_path, index=False)

    error_df, avg_error_df = evaluate_results(pred_df, gt_df)
    if "idx" not in error_df.columns:
        id_to_idx = dict(zip(merged["ID"].astype(str), merged["idx"].astype(int)))
        error_df["idx"] = error_df["ID"].str.replace(".bmp", "", regex=False).map(id_to_idx)

    err_path = out_dir / f"errors_{split_tag}.csv"
    error_df.to_csv(err_path, index=False)

    if not avg_error_df.empty:
        print("\nMean percent error:")
        print(avg_error_df.to_string(index=False))

    _plot_error_by_metric(error_df, out_dir / "error_by_metric.png")
    _plot_pred_vs_actual(pred_df, gt_df, out_dir / "pred_vs_actual.png")

    phase2_stats = resolve_path(args.output_dir) / "phase2_images" / "image_stats.csv"
    _plot_error_vs_blur(error_df, phase2_stats, out_dir / "error_vs_blur.png")

    phase3_roi = resolve_path(args.output_dir) / "phase3_roi" / "roi_vs_random_crop.csv"
    _plot_error_vs_roi_iou(error_df, phase3_roi, out_dir / "error_vs_roi_iou.png")

    if not error_df.empty:
        _save_worst_cases(
            error_df,
            memmap,
            merged,
            cache_dir,
            seg_hash,
            metric_config,
            out_dir,
            top_n=getattr(args, "worst_n", 5),
        )

    print(f"Phase 4 complete -> {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 4: baseline error analysis.")
    add_common_args(parser)
    parser.set_defaults(split="val", limit=200)
    parser.add_argument(
        "--worst_n",
        type=int,
        default=5,
        help="Number of worst-case overlay PNGs per metric.",
    )
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
