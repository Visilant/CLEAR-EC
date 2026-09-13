"""Target-space transforms (linear or log/logit) and their z-scoring statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.training.common import METRICS


def _forward_target_space(raw: np.ndarray, target_space: str) -> np.ndarray:
    """CD, CV -> log; HEX -> logit (clipped). No-op for target_space='linear'."""
    if target_space == "linear":
        return raw.astype(np.float64)
    if target_space != "log":
        raise ValueError(f"Unknown target_space: {target_space}")
    out = raw.astype(np.float64).copy()
    out[:, 0] = np.log(np.clip(out[:, 0], 1e-6, None))
    out[:, 1] = np.log(np.clip(out[:, 1], 1e-6, None))
    hex_clipped = np.clip(out[:, 2], 1e-3, 1 - 1e-3)
    out[:, 2] = np.log(hex_clipped / (1 - hex_clipped))
    return out


def _inverse_target_space(vals: np.ndarray, target_space: str) -> np.ndarray:
    if target_space == "linear":
        return vals
    out = vals.copy()
    out[:, 0] = np.exp(out[:, 0])
    out[:, 1] = np.exp(out[:, 1])
    out[:, 2] = 1.0 / (1.0 + np.exp(-out[:, 2]))
    return out


def _inverse_target_space_torch(x: torch.Tensor, target_space: str) -> torch.Tensor:
    if target_space == "linear":
        return x
    out = torch.empty_like(x)
    out[..., 0] = torch.exp(x[..., 0])
    out[..., 1] = torch.exp(x[..., 1])
    out[..., 2] = torch.sigmoid(x[..., 2])
    return out


def target_space_stats(df: pd.DataFrame, target_space: str) -> dict[str, dict[str, float]]:
    """Mean/std of the (possibly log/logit transformed) targets, for z-scoring."""
    raw = df[list(METRICS)].to_numpy(dtype=np.float64)
    transformed = _forward_target_space(raw, target_space)
    stats: dict[str, dict[str, float]] = {}
    for j, col in enumerate(METRICS):
        vals = transformed[:, j]
        stats[col] = {"mean": float(np.mean(vals)), "std": float(np.std(vals) + 1e-8)}
    return stats
