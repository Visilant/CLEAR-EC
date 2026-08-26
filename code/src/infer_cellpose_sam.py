"""
Cellpose v1.0 inference module for corneal endothelial cell segmentation.
"""

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import label as nd_label
from scipy.optimize import linear_sum_assignment
from skimage.measure import regionprops
from tqdm import tqdm

from cellpose import models, utils
from cellpose.plot import mask_overlay
from src.data.config import MetricConfig, SegConfig
from src.data.crop import (
    crop_and_relabel_masks,
    crop_rng_for_image,
    random_crop,
    random_crop_bbox,
)
from src.data.metrics import metrics_from_mask
from src.data.segment import segment_image
from src.io_utils import load_image


# Backward-compatible re-exports for external callers.
__all__ = [
    "random_crop",
    "random_crop_bbox",
    "crop_rng_for_image",
    "crop_and_relabel_masks",
    "get_segmentation",
    "visualize_segmentation",
]

def visualize_segmentation(
    image: np.ndarray,
    masks: np.ndarray,
    flows: np.ndarray = None,
    image_id: str = None,
    save_path: str = None,
    show: bool = False,
    dots: list = None,  # [(cx, cy), ...] in full-image coordinates
    crop_bbox: tuple = None,  # (x0, y0, x1, y1) random crop region to highlight
):
    """
    Visualize segmentation results alongside the original image.

    Args:
        image: Original image array
        masks: Segmentation masks
        flows: Flow field (optional)
        image_id: Image identifier for title
        save_path: Path to save the figure (optional)
        show: Whether to display the figure
    """
    from matplotlib.patches import Rectangle

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    def _draw_crop_rect(ax):
        if crop_bbox is None:
            return
        x0, y0, x1, y1 = crop_bbox
        ax.add_patch(Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            edgecolor="yellow", facecolor="none", linewidth=2, linestyle="--",
        ))

    # Left: Original image
    axes[0].imshow(image)
    axes[0].set_title(f"Original Image: {image_id}" if image_id else "Original Image")
    axes[0].axis("off")
    _draw_crop_rect(axes[0])
    
    # Right: Image with segmentation overlay
    if masks is not None and masks.max() > 0:
        # Generate mask outlines
        outlines = utils.masks_to_outlines(masks)
        outY, outX = np.nonzero(outlines)
        
        # Create overlay image with outlines
        imgout = image.copy()
        if imgout.ndim == 2:
            imgout = np.stack([imgout] * 3, axis=-1)
        elif imgout.ndim == 3 and imgout.shape[2] == 1:
            imgout = np.stack([imgout[:, :, 0]] * 3, axis=-1)
        
        # Draw outlines in cyan
        if len(outY) > 0:
            imgout[outY, outX] = np.array([0, 255, 255])  # Cyan color
        
        # Add semi-transparent mask overlay
        mask_overlay_img = mask_overlay(image, masks)
        
        # Blend original image with mask overlay
        overlay = (0.6 * imgout.astype(np.float32) + 0.4 * mask_overlay_img.astype(np.float32)).astype(np.uint8)
        axes[1].imshow(overlay)

        # Overlay annotation dots as red crosses
        if dots:
            dot_xs = [d[0] for d in dots]
            dot_ys = [d[1] for d in dots]
            axes[1].scatter(dot_xs, dot_ys, c="red", s=20, marker="+", linewidths=1.5, zorder=5)

        num_cells = len(np.unique(masks)) - 1  # Exclude background
        n_dots = len(dots) if dots else 0
        crop_note = "  |  yellow box = metric region" if crop_bbox is not None else ""
        axes[1].set_title(f"Segmentation: {num_cells} cells (full image){crop_note}  |  {n_dots} annotation dots")
    else:
        axes[1].imshow(image)
        axes[1].set_title("Segmentation Result: No cells detected")

    _draw_crop_rect(axes[1])
    axes[1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def get_segmentation(
    image_paths: list,
    num_samples: int = 4,  # Kept for backward compatibility, but not used
    preprocess=None,
    plotting: bool = False,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    batch_size: int = 8,
    device: str = "cuda",
    model_type: str = "cyto",           # Cellpose v1.0: 'cyto' | 'cyto2' | 'nuclei'
    diameter: float = None,             # avg cell diameter in px; None -> auto-estimate
    annotations: dict = None,  # {stem: {"bbox": [x0,y0,x1,y1], "dots": [...]}}
    match_dots: bool = False,           # if True, filter cells by annotation dots
    match_threshold: int = 20,          # px in BMP space for matching
    matching_method: str = "emd",       # "emd" | "nearest"
    vis_output_dir: str = "results/visualizations",
    random_crop_frac: float = None,     # if set, take random crop of this fraction
    random_crop_seed: int = None,       # seed for reproducible random crops
    tile: bool = True,                  # Cellpose internal tiling; False = whole-image inference
):
    """
    Perform segmentation using Cellpose v1.0 on entire images.

    Args:
        image_paths: List of image file paths to process (loaded lazily per iteration)
        num_samples: Deprecated - kept for backward compatibility
        preprocess: Preprocessing function to apply (e.g., subtraction_method)
        plotting: Whether to save segmentation overlay PNGs
        flow_threshold: Flow threshold for Cellpose (v1.0 default 0.4)
        cellprob_threshold: Cell probability threshold (v1.0 default 0.0)
        min_size: Minimum size for detected cells (v1.0 default 15)
        batch_size: Batch size for model evaluation
        device: Device to use ('cuda' or 'cpu')
        model_type: Cellpose v1.0 pretrained model — 'cyto', 'cyto2', or 'nuclei'
        diameter: Average cell diameter in pixels; None auto-estimates per image

    Returns:
        tuple: (predictions, additional_data)
            - predictions: List of prediction dictionaries
            - additional_data: Additional data (currently None)
    """
    predictions = []

    # Initialize Cellpose v1.0 model (includes size model for diameter estimation)
    model = models.Cellpose(gpu=(device == "cuda"), model_type=model_type)
    seg_config = SegConfig(
        model_type=model_type,
        diameter=diameter,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        tile=tile,
        net_avg=False,
        batch_size=batch_size,
    )
    metric_config = MetricConfig(
        random_crop_frac=random_crop_frac,
        random_crop_seed=random_crop_seed,
    )

    for i in tqdm(range(len(image_paths)), desc="Cellpose v1.0 Segmentation"):
        pathname = image_paths[i]
        pathname = Path(pathname) if isinstance(pathname, str) else pathname
        image = load_image(pathname)  # lazy: decode just-in-time
        print(f"Processing image: {pathname.name}")

        stem = pathname.stem
        crop_bbox = None
        if random_crop_frac is not None:
            rng = crop_rng_for_image(image, random_crop_seed)
            crop_bbox = random_crop_bbox(image.shape, frac=random_crop_frac, rng=rng)
            print(f"  random crop ({random_crop_frac:.2f}) -> bbox={crop_bbox}, full image shape={image.shape}")
        elif annotations and stem in annotations and annotations[stem].get("bbox"):
            x0, y0, x1, y1 = annotations[stem]["bbox"]
            crop_bbox = (int(x0), int(y0), int(x1), int(y1))
            print(f"  annotation bbox crop {crop_bbox}, full image shape={image.shape}")

        # Apply preprocessing if specified
        if preprocess:
            if callable(preprocess):
                processed_image = preprocess(image)
            else:
                processed_image = image
        else:
            processed_image = image

        try:
            masks = segment_image(model, processed_image, seg_config)
            flows = None
        except Exception as e:
            print(f"Error during model evaluation: {e}")
            continue

        # Filter cells by annotation dots. Masks are in full-image coordinates
        # (segmentation ran on the whole image), so cell centroids are already
        # in the same space as `dots` — no offset adjustment needed.
        if match_dots and annotations and stem in annotations and annotations[stem].get("dots"):
            dots_bmp = np.array(annotations[stem]["dots"])   # (D, 2) [cx, cy] full-image space
            props    = regionprops(masks)

            if len(props) > 0 and len(dots_bmp) > 0:
                cell_centroids = np.array([
                    [prop.centroid[1], prop.centroid[0]]
                    for prop in props
                ])
                # pairwise distance matrix (C, D)
                diff = cell_centroids[:, None, :] - dots_bmp[None, :, :]
                cost = np.sqrt((diff ** 2).sum(axis=2))

                keep_labels = set()
                if matching_method == "emd":
                    # Hungarian 1-to-1 optimal assignment
                    cell_idx, dot_idx = linear_sum_assignment(cost)
                    for ci, di in zip(cell_idx, dot_idx):
                        if cost[ci, di] <= match_threshold:
                            keep_labels.add(props[ci].label)
                elif matching_method == "nearest":
                    # Keep cell if its centroid is within match_threshold of any dot (many-to-one allowed)
                    for ci, prop in enumerate(props):
                        if cost[ci].min() <= match_threshold:
                            keep_labels.add(prop.label)
                else:
                    raise ValueError(f"Unknown matching_method: {matching_method!r}. Choose 'emd' or 'nearest'.")

                filtered = np.zeros_like(masks)
                for new_label, lbl in enumerate(sorted(keep_labels), start=1):
                    filtered[masks == lbl] = new_label
                masks = filtered
                print(f"  [{matching_method}] matched: {len(keep_labels)}/{len(props)} cells kept  "
                      f"(threshold={match_threshold}px, dots={len(dots_bmp)})")
            else:
                masks = np.zeros_like(masks)
                print("  matching: no cells or no dots, mask cleared")

        masks_full = masks
        image_id = pathname.stem

        if crop_bbox is not None:
            masks_metric = crop_and_relabel_masks(masks_full, crop_bbox)
            print(f"  cells (incl. partial) in cropped region: {int(masks_metric.max())} "
                  f"(full image: {int(masks_full.max())})")
        else:
            masks_metric = masks_full

        prediction = metrics_from_mask(
            masks_full,
            image,
            image_id,
            metric_config,
            crop_bbox=crop_bbox,
        )

        print(prediction)

        if prediction is not None:
            predictions.append(prediction)
            print(f"Successfully processed {image_id}")
            
            # Visualize segmentation results. Image and masks are both in
            # full-image space; the random crop region is drawn as an overlay.
            if plotting:
                os.makedirs(vis_output_dir, exist_ok=True)
                vis_path = os.path.join(vis_output_dir, f"{pathname.stem}_segmentation.png")

                # Annotation dots are already in full-image (BMP) space.
                vis_dots = None
                if annotations and stem in annotations and annotations[stem].get("dots"):
                    vis_dots = [(d[0], d[1]) for d in annotations[stem]["dots"]]

                visualize_segmentation(
                    image=image,
                    masks=masks_full,  # full-image segmentation, not crop-filtered
                    flows=flows[0] if flows is not None else None,
                    image_id=image_id,
                    save_path=vis_path,
                    show=False,
                    dots=vis_dots,
                    crop_bbox=crop_bbox,
                )
        else:
            print(f"Warning: No valid predictions for {image_id}")

    return predictions, None
