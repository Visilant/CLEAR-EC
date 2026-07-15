"""
CLEAR-EC segmentation pipeline.

Reads images from --data_dir (e.g. /path/to/your_holdout_images), segments each FULL image with
Cellpose v1.0, then restricts metrics (CD, CV, HEX) to a random crop region
(default 40% of H x 40% of W) by keeping only cells whose centroid lies in
the crop. Cells near the crop boundary still benefit from full surrounding
context during segmentation. Per-sample predictions are written to a CSV.
Use evaluate.py to compare the resulting predictions against ground truth.
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cellpose import io

from src.infer_cellpose_sam import get_segmentation


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility.

    Note: full bit-for-bit determinism on CUDA also requires the cuDNN flags
    below; some ops still have nondeterministic CUDA kernels, but this is the
    standard best-effort recipe."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    io.logger_setup()

    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(args.vis_output_dir, exist_ok=True)

    data_dir = Path(args.data_dir).expanduser().resolve()
    files = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff", ".bmp", ".png", ".mha"))
    if args.limit:
        files = files[: args.limit]
    print(f"Found {len(files)} images in {data_dir}")

    print(f"Random crop fraction: {args.random_crop_frac} (seed={args.seed})")

    predictions, _ = get_segmentation(
        image_paths=files,
        plotting=args.plot,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size,
        model_type=args.model_type,
        diameter=args.diameter,
        annotations=None,
        match_dots=False,
        vis_output_dir=args.vis_output_dir,
        random_crop_frac=args.random_crop_frac,
        random_crop_seed=args.seed,
        tile=not args.no_tile,
    )

    pred_df = pd.DataFrame(predictions)

    # Match CLEAR-EC ground-truth ID format: bare stem, no extension.
    pred_df["ID"] = pred_df["ID"].astype(str).str.replace(r"\.(bmp|png|tif|tiff|mha)$", "", regex=True)

    # Keep only the metrics CLEAR-EC reports.
    pred_df = pred_df[["ID", "CD", "CV", "HEX"]]

    print(f"\n{'ID':<30} {'CD':>10} {'CV':>10} {'HEX':>10}")
    print("-" * 65)
    for _, row in pred_df.iterrows():
        print(f"{row['ID']:<30} {row['CD']:>10.2f} {row['CV']:>10.2f} {row['HEX']:>10.2f}")

    pred_csv = Path(args.results_dir) / f"predictions_{args.split}.csv"
    pred_df.to_csv(pred_csv, index=False)
    print(f"\nWrote predictions -> {pred_csv}")

    return pred_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLEAR-EC: segment cropped TIFFs and write per-sample predictions.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"],
                        help="Label for this run; used to name the predictions CSV (predictions_<split>.csv).")
    parser.add_argument("--data_dir", type=str, default="/path/to/your_holdout_images",
                        help="Directory of input images to segment (e.g. a held-out split of the released training data).")
    parser.add_argument("--results_dir", type=str, default="./results_mha",
                        help="Where to write the predictions CSV (default: ./results_mha).")
    parser.add_argument("--vis_output_dir", type=str, default="./results_mha/visualizations",
                        help="Where to write segmentation visualizations (default: ./results_mha/visualizations).")
    parser.add_argument("--limit", type=int, default=0,
                        help="If > 0, process only the first N images (for quick smoke tests).")

    parser.add_argument("--seed", type=int, default=42,
                        help="Global RNG seed for Python/NumPy/PyTorch and the random crop. Set for reproducible runs.")

    # Random crop (replaces bbox/annotation-based cropping).
    parser.add_argument("--random_crop_frac", type=float, default=0.4,
                        help="Fraction of H and W kept by random crop.")

    # Cellpose v1.0 parameters.
    parser.add_argument("--model_type", type=str, default="cyto",
                        choices=["cyto", "cyto2", "nuclei"],
                        help="Cellpose v1.0 pretrained model.")
    parser.add_argument("--diameter", type=float, default=None,
                        help="Average cell diameter in pixels. None auto-estimates per image (recommended to set explicitly for endothelial frames).")
    parser.add_argument("--flow_threshold", type=float, default=0.4,
                        help="Cellpose v1.0 default 0.4.")
    parser.add_argument("--cellprob_threshold", type=float, default=0.0,
                        help="Cellpose v1.0 default 0.0.")
    parser.add_argument("--min_size", type=int, default=15,
                        help="Cellpose v1.0 default 15.")
    parser.add_argument("--plot", action="store_true", default=False,
                        help="Save segmentation overlay PNGs to --vis_output_dir.")
    parser.add_argument("--no_tile", action="store_true", default=False,
                        help="Disable Cellpose internal tiling and run whole-image inference (may OOM on large images).")

    return parser


def resolve_paths(args) -> argparse.Namespace:
    if args.data_dir is None:
        args.data_dir = "/path/to/your_holdout_images"
    return args


if __name__ == "__main__":
    args = resolve_paths(build_parser().parse_args())
    main(args)
