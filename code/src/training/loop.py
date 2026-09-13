"""One training/evaluation epoch, weight EMA, and the LR schedule."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.common import METRICS, denormalize_targets, mape_per_metric
from src.training.data import _photometric_augment, _random_crop_batch
from src.training.losses import _apply_criterion
from src.training.targets import _inverse_target_space


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stats: dict[str, dict[str, float]],
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    amp: bool = False,
    target_space: str = "linear",
    photometric: bool = False,
    ema_update=None,
    lr_scheduler=None,
    crop_scale: float = 1.0,
    clip_grad: float = 0.0,
) -> tuple[float, dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = torch.zeros((), device=device)
    n_images = 0
    preds_list: list[torch.Tensor] = []
    index_list: list[np.ndarray] = []

    for x, y, y_raw, indices in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        y_raw = y_raw.to(device, non_blocking=True)
        if train_mode and photometric:
            x = _photometric_augment(x)
        restore_context = None
        if train_mode and crop_scale < 1.0:
            full_h, full_w = x.shape[-2:]
            x = _random_crop_batch(x, crop_scale)
            base = getattr(model, "context_size", None)
            if base is not None:
                restore_context = base
                model.context_size = (max(32, int(round(base[0] * x.shape[-2] / full_h))),
                                      max(32, int(round(base[1] * x.shape[-1] / full_w))))
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                pred = model(x)
                loss = _apply_criterion(criterion, pred, y, y_raw)
            if train_mode:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if clip_grad > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if clip_grad > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
                    optimizer.step()
                if ema_update is not None:
                    ema_update()
                if lr_scheduler is not None:
                    lr_scheduler.step()
        if restore_context is not None:
            model.context_size = restore_context
        total_loss += loss.detach() * len(x)
        n_images += len(x)
        preds_list.append(pred.detach().float())
        index_list.append(indices.numpy())

    if not n_images:
        raise ValueError("Cannot run an epoch on an empty dataset")
    avg_loss = float(total_loss.item()) / n_images
    preds_transformed = denormalize_targets(torch.cat(preds_list).cpu().numpy(), stats)
    preds = _inverse_target_space(preds_transformed, target_space)
    # Never reconstruct ground truth from normalized float32 tensors: an exact
    # zero HEX target can round-trip to ~6e-8 and turn MAPE into millions of %.
    # Join original CSV targets in loader order so shuffle remains correct.
    gt = loader.dataset.frame.set_index("idx").loc[
        np.concatenate(index_list), list(METRICS)
    ].to_numpy(dtype=float)
    if not np.isfinite(avg_loss) or not np.isfinite(preds).all():
        raise FloatingPointError("Non-finite training loss or predictions")
    return avg_loss, mape_per_metric(preds, gt)


class _EMA:
    """Manual EMA of model weights, evaluated/saved in place of the raw model."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.model = copy.deepcopy(model)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.model.eval()

    @torch.no_grad()
    def update(self, source: nn.Module) -> None:
        src_state = source.state_dict()
        for k, v in self.model.state_dict().items():
            new_val = src_state[k]
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(new_val.detach(), alpha=1 - self.decay)
            else:
                v.copy_(new_val)


def _build_scheduler(optimizer, sched: str, warmup_epochs: int, epochs: int, iters_per_epoch: int):
    if sched == "none":
        return None
    if sched != "cosine":
        raise ValueError(f"Unknown sched: {sched}")
    warmup_iters = max(0, warmup_epochs) * max(1, iters_per_epoch)
    total_iters = max(1, epochs * max(1, iters_per_epoch))
    min_factor = 0.01

    def lr_lambda(it: int) -> float:
        if warmup_iters > 0 and it < warmup_iters:
            return (it + 1) / warmup_iters
        t = (it - warmup_iters) / max(1, total_iters - warmup_iters)
        t = min(max(t, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * t))
        return min_factor + (1 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
