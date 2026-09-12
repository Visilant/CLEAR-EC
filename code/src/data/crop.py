"""Random crop helpers shared by inference and metric sweeps."""

from __future__ import annotations

import hashlib
import random

import numpy as np


def random_crop(image: np.ndarray, frac: float = 0.4, rng: random.Random = None):
    """
    Crop a random rectangular region of size (frac*H, frac*W) from `image`.
    Returns (cropped_image, (x0, y0, x1, y1)) in image space.
    """
    if rng is None:
        rng = random
    h, w = image.shape[:2]
    ch = max(1, int(h * frac))
    cw = max(1, int(w * frac))
    y0 = rng.randint(0, max(0, h - ch))
    x0 = rng.randint(0, max(0, w - cw))
    return image[y0 : y0 + ch, x0 : x0 + cw], (x0, y0, x0 + cw, y0 + ch)


def random_crop_bbox(shape, frac: float, rng: random.Random = None):
    """
    Return a random crop bbox (x0, y0, x1, y1) of size (frac*H, frac*W) for an
    image of the given `shape`. Same RNG semantics as `random_crop` so existing
    seed-based reproducibility is preserved.
    """
    if rng is None:
        rng = random
    h, w = shape[:2]
    ch = max(1, int(h * frac))
    cw = max(1, int(w * frac))
    y0 = rng.randint(0, max(0, h - ch))
    x0 = rng.randint(0, max(0, w - cw))
    return (x0, y0, x0 + cw, y0 + ch)


def crop_rng_for_image(image: np.ndarray, base_seed) -> random.Random:
    """Deterministic per-image RNG for the random crop."""
    if base_seed is None:
        return random
    digest = hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()
    return random.Random(f"{base_seed}:{digest}")


def crop_and_relabel_masks(masks: np.ndarray, bbox) -> np.ndarray:
    """Slice masks to bbox and relabel surviving cells contiguously from 1."""
    x0, y0, x1, y1 = bbox
    cropped = masks[y0:y1, x0:x1]

    # One pass through the crop instead of one scan per cell. Inserting zero
    # preserves the background convention even if every pixel is foreground.
    labels, inverse = np.unique(cropped, return_inverse=True)
    offset = int(labels.size > 0 and labels[0] != 0)
    return (inverse.reshape(cropped.shape) + offset).astype(masks.dtype)
