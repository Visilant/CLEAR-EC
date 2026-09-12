"""Cached segmentation mask I/O."""

from __future__ import annotations

from pathlib import Path
import os
import tempfile

import numpy as np

from src.data.config import SegConfig, config_hash


def mask_path(cache_dir: Path, seg_hash: str, idx: int) -> Path:
    return Path(cache_dir) / "masks" / seg_hash / f"{idx:06d}.npz"


def load_mask(cache_dir: Path, seg_hash: str, idx: int) -> np.ndarray:
    path = mask_path(cache_dir, seg_hash, idx)
    with np.load(path) as data:
        return data["masks"]


def save_mask(cache_dir: Path, seg_hash: str, idx: int, masks: np.ndarray) -> Path:
    path = mask_path(cache_dir, seg_hash, idx)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Readers must never see a half-written zip during parallel experiments.
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as f:
        temporary = Path(f.name)
    try:
        np.savez_compressed(temporary, masks=masks)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
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

    from src.data.segment import segment_image

    masks = segment_image(model, image_gray, seg_config)
    save_mask(cache_dir, seg_hash, idx, masks)
    return masks
