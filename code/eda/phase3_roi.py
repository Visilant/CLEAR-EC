"""Phase 3 — GT ROI from green overlays vs random crop."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.patches import Rectangle

from eda._cache import (
    add_common_args,
    load_eda_frame,
    resolve_path,
)
from src.data.cache import _build_mha_lookup, _decode_gray_mha, _normalize_id
from src.data.config import MetricConfig
from src.data.crop import crop_rng_for_image, random_crop_bbox
from src.data.metrics import _image_for_crop


def bbox_iou(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> float:
    x0 = max(b1[0], b2[0])
    y0 = max(b1[1], b2[1])
    x1 = min(b1[2], b2[2])
    y1 = min(b1[3], b2[3])
    inter = max(0, x1 - x0) * max(0, y1 - y0)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def bbox_area_frac(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> float:
    h, w = shape
    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    return float(area / (h * w))


def scale_bbox_to_image(
    bbox: tuple[int, int, int, int],
    overlay_shape: tuple[int, ...],
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Map ROI bbox from rescaled overlay (e.g. 720x540) to full MHA pixels."""
    oh, ow = overlay_shape[:2]
    ih, iw = image_shape
    sx = iw / ow
    sy = ih / oh
    x0, y0, x1, y1 = bbox
    return (
        int(round(x0 * sx)),
        int(round(y0 * sy)),
        int(round(x1 * sx)),
        int(round(y1 * sy)),
    )


def extract_gt_roi_bbox(green_overlay: np.ndarray, threshold: int = 30) -> tuple[int, int, int, int] | None:
    """Bounding box of green-channel ROI pixels in an RGB overlay TIFF."""
    if green_overlay.ndim == 2:
        mask = green_overlay > threshold
    else:
        if green_overlay.shape[2] >= 3:
            g = green_overlay[..., 1].astype(np.float32)
            r = green_overlay[..., 0].astype(np.float32)
            b = green_overlay[..., 2].astype(np.float32)
            mask = (g > threshold) & (g > r) & (g > b)
        else:
            mask = green_overlay[..., 0] > threshold

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return (x0, y0, x1, y1)


def _random_crop_for_image(
    image_gray: np.ndarray,
    metric_config: MetricConfig,
) -> tuple[int, int, int, int]:
    image_for_crop = _image_for_crop(image_gray)
    rng = crop_rng_for_image(image_for_crop, metric_config.random_crop_seed)
    return random_crop_bbox(
        image_for_crop.shape,
        frac=metric_config.random_crop_frac,
        rng=rng,
    )


def _plot_overlay_samples(
    samples: list[dict],
    out_path: Path,
    max_panels: int = 6,
) -> None:
    n = min(len(samples), max_panels)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    if n == 1:
        axes = [axes]

    for ax, sample in zip(axes, samples[:n]):
        gray = sample["gray"]
        ax.imshow(gray, cmap="gray")
        gt = sample["gt_bbox"]
        rc = sample["random_bbox"]
        ax.add_patch(
            Rectangle(
                (gt[0], gt[1]),
                gt[2] - gt[0],
                gt[3] - gt[1],
                edgecolor="lime",
                facecolor="none",
                linewidth=2,
                label="GT ROI",
            )
        )
        ax.add_patch(
            Rectangle(
                (rc[0], rc[1]),
                rc[2] - rc[0],
                rc[3] - rc[1],
                edgecolor="yellow",
                facecolor="none",
                linewidth=2,
                linestyle="--",
                label="random crop",
            )
        )
        ax.set_title(
            f"{sample['ID']}  IoU={sample['bbox_iou']:.3f}  "
            f"GT area={sample['gt_area_frac']:.2f}  crop area={sample['crop_area_frac']:.2f}"
        )
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    cache_dir = resolve_path(args.cache_dir)
    labels_csv = resolve_path(args.labels_csv)
    green_dir = resolve_path(args.green_dir)
    out_dir = resolve_path(args.output_dir) / "phase3_roi"
    out_dir.mkdir(parents=True, exist_ok=True)

    memmap, merged, _ = load_eda_frame(cache_dir, labels_csv, split="all")
    id_to_idx = dict(zip(merged["ID"].astype(str), merged["idx"].astype(int)))
    data_dir = cache_dir.parent / "train_mha"
    mha_by_id = _build_mha_lookup(data_dir) if data_dir.is_dir() else {}

    metric_config = MetricConfig(
        random_crop_frac=0.4,
        random_crop_seed=args.seed,
    )

    overlay_paths = sorted(green_dir.glob("*.tiff")) + sorted(green_dir.glob("*.tif"))
    rows = []
    plot_samples = []

    for path in overlay_paths:
        image_id = path.stem
        idx = id_to_idx.get(image_id)

        green = np.array(Image.open(path))
        gt_bbox_overlay = extract_gt_roi_bbox(green)
        if gt_bbox_overlay is None:
            print(f"Skipping {path.name}: no green ROI detected")
            continue

        if idx is not None:
            gray = memmap[idx]
        else:
            mha_path = mha_by_id.get(_normalize_id(image_id))
            if mha_path is None:
                print(f"Skipping {path.name}: ID {image_id!r} not in cache or train_mha")
                continue
            gray = _decode_gray_mha(mha_path)
            idx = -1

        gt_bbox = scale_bbox_to_image(gt_bbox_overlay, green.shape, gray.shape)
        random_bbox = _random_crop_for_image(gray, metric_config)
        iou = bbox_iou(gt_bbox, random_bbox)

        row = {
            "idx": idx,
            "ID": image_id,
            "overlay_path": str(path),
            "gt_x0": gt_bbox[0],
            "gt_y0": gt_bbox[1],
            "gt_x1": gt_bbox[2],
            "gt_y1": gt_bbox[3],
            "crop_x0": random_bbox[0],
            "crop_y0": random_bbox[1],
            "crop_x1": random_bbox[2],
            "crop_y1": random_bbox[3],
            "bbox_iou": iou,
            "gt_area_frac": bbox_area_frac(gt_bbox, gray.shape),
            "crop_area_frac": bbox_area_frac(random_bbox, gray.shape),
        }
        rows.append(row)
        plot_samples.append(
            {
                "ID": image_id,
                "gray": gray,
                "gt_bbox": gt_bbox,
                "random_bbox": random_bbox,
                "bbox_iou": iou,
                "gt_area_frac": row["gt_area_frac"],
                "crop_area_frac": row["crop_area_frac"],
            }
        )

    roi_df = pd.DataFrame(rows)
    roi_path = out_dir / "roi_vs_random_crop.csv"
    if roi_df.empty:
        roi_path.write_text(
            "idx,ID,overlay_path,gt_x0,gt_y0,gt_x1,gt_y1,"
            "crop_x0,crop_y0,crop_x1,crop_y1,bbox_iou,gt_area_frac,crop_area_frac\n"
        )
    else:
        roi_df.to_csv(roi_path, index=False)
    _plot_overlay_samples(plot_samples, out_dir / "roi_overlay_samples.png")

    if not roi_df.empty:
        print(
            f"Phase 3: {len(roi_df)} overlays, "
            f"mean IoU={roi_df['bbox_iou'].mean():.3f} -> {out_dir}"
        )
    else:
        print(f"Phase 3: no overlays matched cache index -> {out_dir}")
    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase 3: ROI vs random crop.")
    add_common_args(parser)
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
