"""CLEAR-EC data pipeline: image cache, splits, mask cache, and metric sweeps."""

from src.data.config import MetricConfig, SegConfig, config_hash
from src.data.cache import build_image_cache, open_image_cache
from src.data.splits import build_splits, load_split, load_splits

__all__ = [
    "MetricConfig",
    "SegConfig",
    "config_hash",
    "build_image_cache",
    "open_image_cache",
    "build_splits",
    "load_split",
    "load_splits",
]
