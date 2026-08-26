"""Frozen configuration dataclasses with deterministic hashing."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


def config_hash(cfg) -> str:
    """Return a stable 12-char hash for a frozen dataclass config."""
    return hashlib.sha256(repr(cfg).encode()).hexdigest()[:12]


@dataclass(frozen=True)
class SegConfig:
    """Parameters that affect Cellpose segmentation output."""

    model_type: str = "cyto"
    diameter: float | None = None
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0
    min_size: int = 15
    tile: bool = True
    net_avg: bool = False
    batch_size: int = 8


@dataclass(frozen=True)
class MetricConfig:
    """Post-processing parameters (crop/threshold sweeps)."""

    random_crop_frac: float | None = 0.4
    random_crop_seed: int | None = 42
