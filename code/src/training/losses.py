"""Training losses; the relative loss is the challenge MAPE in raw metric units."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.training.common import METRICS
from src.training.targets import _inverse_target_space_torch


class RelativeAbsoluteErrorLoss(nn.Module):
    """Mean absolute relative error in original (raw) metric units."""

    def __init__(self, stats: dict[str, dict[str, float]], target_space: str = "linear"):
        super().__init__()
        self.register_buffer("means", torch.tensor([stats[m]["mean"] for m in METRICS]))
        self.register_buffer("stds", torch.tensor([stats[m]["std"] for m in METRICS]))
        self.target_space = target_space

    def forward(self, pred: torch.Tensor, target: torch.Tensor, target_raw: torch.Tensor | None = None) -> torch.Tensor:
        pred_transformed = pred * self.stds + self.means
        pred_raw = _inverse_target_space_torch(pred_transformed, self.target_space)
        if target_raw is None:
            target_transformed = target * self.stds + self.means
            target_raw = _inverse_target_space_torch(target_transformed, self.target_space)
        # Exact zero labels are excluded by the challenge MAPE definition. A
        # small tolerance accounts for normalized float32 round trips.
        valid = target_raw.abs() > 1e-5
        relative = (pred_raw - target_raw).abs() / target_raw.abs().clamp_min(1e-5)
        return relative[valid].mean()


def _build_loss(
    name: str,
    stats: dict[str, dict[str, float]] | None = None,
    target_space: str = "linear",
) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "relative":
        if stats is None:
            raise ValueError("Relative loss requires target statistics")
        return RelativeAbsoluteErrorLoss(stats, target_space=target_space)
    if name != "huber":
        raise ValueError(f"Unknown loss: {name}")
    return nn.HuberLoss(delta=1.0)


def _apply_criterion(
    criterion: nn.Module, pred: torch.Tensor, y: torch.Tensor, y_raw: torch.Tensor
) -> torch.Tensor:
    if isinstance(criterion, RelativeAbsoluteErrorLoss):
        return criterion(pred, y, y_raw)
    return criterion(pred, y)
