"""Single-image Cellpose segmentation helpers."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cellpose import models

from src.data.cache import open_image_cache
from src.data.config import SegConfig, config_hash
from src.data.splits import load_split


def segment_image(model, image_gray: np.ndarray, seg_config: SegConfig) -> np.ndarray:
    """Run Cellpose on a 2D grayscale image and return the label mask."""
    if image_gray.ndim == 3 and image_gray.shape[2] == 3:
        eval_image = image_gray[..., 0]
    else:
        eval_image = image_gray

    channels = [0, 0]
    masks, _, _, _ = model.eval(
        eval_image,
        diameter=seg_config.diameter,
        channels=channels,
        batch_size=seg_config.batch_size,
        flow_threshold=seg_config.flow_threshold,
        cellprob_threshold=seg_config.cellprob_threshold,
        min_size=seg_config.min_size,
        net_avg=seg_config.net_avg,
        tile=seg_config.tile,
    )
    return masks


def run_segmentation_shard(
    cache_dir: Path,
    seg_config: SegConfig,
    gpu_id: int,
    shard_id: int,
    num_shards: int,
    indices: list[int],
    *,
    skip_existing: bool = True,
) -> None:
    """Process a disjoint shard of indices on one GPU."""
    from src.data.mask_cache import get_or_compute_mask

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    memmap, index_df = open_image_cache(cache_dir)
    seg_hash = config_hash(seg_config)
    shard_indices = [i for i in indices if i % num_shards == shard_id]

    model = models.Cellpose(gpu=True, model_type=seg_config.model_type)
    desc = f"GPU {gpu_id} shard {shard_id}/{num_shards}"
    for idx in tqdm(shard_indices, desc=desc):
        image_gray = memmap[idx]
        get_or_compute_mask(
            idx=idx,
            image_gray=image_gray,
            seg_config=seg_config,
            model=model,
            cache_dir=cache_dir,
            skip_existing=skip_existing,
        )


def resolve_indices(
    cache_dir: Path,
    split: str,
    limit: int | None = None,
) -> list[int]:
    """Resolve which cache indices to process."""
    if split == "all":
        _, index_df = open_image_cache(cache_dir)
        indices = index_df["idx"].astype(int).tolist()
    else:
        indices = load_split(cache_dir, split)

    if limit is not None and limit > 0:
        indices = indices[:limit]
    return indices
