"""Metric-only path from cached segmentation masks."""

from __future__ import annotations

import numpy as np

from src.data.config import MetricConfig
from src.data.crop import (
    crop_and_relabel_masks,
    crop_rng_for_image,
    random_crop_bbox,
)
from src.utils.evaluate import calculate_metrics_from_masks


def _image_for_crop(image: np.ndarray) -> np.ndarray:
    """Match load_image() RGB layout so crop RNG is identical to main.py."""
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    return image


def metrics_from_mask(
    masks_full: np.ndarray,
    image: np.ndarray,
    image_id: str,
    metric_config: MetricConfig,
    crop_bbox: tuple[int, int, int, int] | None = None,
) -> dict | None:
    """Compute CD/CV/HEX from a full-image mask with optional random crop."""
    if crop_bbox is None and metric_config.random_crop_frac is not None:
        image_for_crop = _image_for_crop(image)
        rng = crop_rng_for_image(image_for_crop, metric_config.random_crop_seed)
        crop_bbox = random_crop_bbox(
            image_for_crop.shape,
            frac=metric_config.random_crop_frac,
            rng=rng,
        )

    if crop_bbox is not None:
        masks_metric = crop_and_relabel_masks(masks_full, crop_bbox)
    else:
        masks_metric = masks_full

    return calculate_metrics_from_masks(masks_metric, ID=image_id)
