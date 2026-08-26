"""Cached segmentation mask I/O."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.data.config import SegConfig, config_hash
from src.data.segment import segment_image


def mask_path(cache_dir: Path, seg_hash: str, idx: int) -> Path:
    return Path(cache_dir) / "masks" / seg_hash / f"{idx:06d}.npz"


def load_mask(cache_dir: Path, seg_hash: str, idx: int) -> np.ndarray:
    path = mask_path(cache_dir, seg_hash, idx)
    with np.load(path) as data:
        return data["masks"]


def save_mask(cache_dir: Path, seg_hash: str, idx: int, masks: np.ndarray) -> Path:
    path = mask_path(cache_dir, seg_hash, idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, masks=masks)
    return path


def get_or_compute_mask(
    idx: int,
    image_gray: np.ndarray,
    seg_config: SegConfig,
    model,
    cache_dir: Path,
    *,
    skip_existing: bool = True,
) -> np.ndarray:
    """Return cached mask or run Cellpose, save, and return."""
    seg_hash = config_hash(seg_config)
    path = mask_path(cache_dir, seg_hash, idx)
    if skip_existing and path.exists():
        return load_mask(cache_dir, seg_hash, idx)

    masks = segment_image(model, image_gray, seg_config)
    save_mask(cache_dir, seg_hash, idx, masks)
    return masks
