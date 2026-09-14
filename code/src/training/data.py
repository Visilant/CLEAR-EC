"""Dataset over the uint8 memmap cache plus the training-time augmentations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.training.common import METRICS
from src.training.targets import _forward_target_space


def _random_crop_batch(x: torch.Tensor, min_scale: float) -> torch.Tensor:
    """Per-batch random crop that keeps pixel scale: one crop size per batch, random offset per image."""
    h, w = x.shape[-2:]
    s = float(np.random.uniform(min_scale, 1.0))
    ch, cw = max(64, int(round(h * s))), max(64, int(round(w * s)))
    if ch >= h and cw >= w:
        return x
    out = torch.empty((x.shape[0], x.shape[1], ch, cw), dtype=x.dtype, device=x.device)
    for i in range(x.shape[0]):
        y0 = int(np.random.randint(0, h - ch + 1))
        x0 = int(np.random.randint(0, w - cw + 1))
        out[i] = x[i, :, y0:y0 + ch, x0:x0 + cw]
    return out


def _zoom_batch(x: torch.Tensor, scales: torch.Tensor, canvas: tuple[int, int],
                antialias: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Zoom each full-resolution frame by its scale s about the centre, onto one shared canvas.

    Zoom is measured against the model's own resample of the whole frame to `canvas` (s = 1 reproduces
    it). Every sample is a centre window of the native frame resampled once (bilinear, `antialias` as in
    the ConvNeXt input path). The canvas is min(1, min s) times the model's input size: a zoom-in (s > 1)
    is a smaller native window filling the canvas, a zoom-out (s < 1) is the whole native frame on a
    smaller canvas. The cache frame is the entire 972x1296 capture, so a zoom-out cannot draw on a larger
    source region; shrinking the canvas keeps every pixel real (zero padding would put empty area into the
    global average that the CD heads rely on; reflection would fabricate mirrored cells). The global-pool
    and density heads are size-agnostic, as the crop_scale lever already relies on.

    Returns (batch float32 in [0, 1] of shape (B, C, ch, cw), cd_mult) where cd_mult[i] is the exact
    factor 1 / (s_y * s_x) the true cell density must be multiplied by, from the integer window actually
    resampled (cells appear s times larger, so there are 1/s^2 as many per unit area)."""
    batch, height, width = x.shape[0], int(x.shape[-2]), int(x.shape[-1])
    scales = scales.to(torch.float64).cpu()
    if batch != scales.numel():
        raise ValueError(f"one scale per sample: batch {batch}, scales {scales.numel()}")
    if not bool((scales > 0).all()):
        raise ValueError("scales must be positive")
    c = min(1.0, float(scales.min()))
    ch, cw = max(32, int(round(canvas[0] * c))), max(32, int(round(canvas[1] * c)))
    out = torch.empty((batch, x.shape[1], ch, cw), dtype=torch.float32, device=x.device)
    cd_mult = torch.empty(batch, dtype=torch.float32)
    for i in range(batch):
        s = float(scales[i])
        wh = min(height, max(2, int(round(height * c / s))))
        ww = min(width, max(2, int(round(width * c / s))))
        y0, x0 = (height - wh) // 2, (width - ww) // 2
        window = x[i:i + 1, :, y0:y0 + wh, x0:x0 + ww]
        window = window.float().div(255.0) if window.dtype == torch.uint8 else window.float()
        if (wh, ww) == (ch, cw):
            out[i] = window[0]
        else:
            out[i] = torch.nn.functional.interpolate(window, size=(ch, cw), mode="bilinear",
                                                     align_corners=False, antialias=antialias)[0]
        s_y = (ch / wh) * (height / canvas[0])
        s_x = (cw / ww) * (width / canvas[1])
        cd_mult[i] = 1.0 / (s_y * s_x)
    return out, cd_mult.to(x.device)


def _scale_jitter_batch(x: torch.Tensor, scale_jitter: float, canvas: tuple[int, int],
                        antialias: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample log-uniform zoom s = exp(U(-scale_jitter, +scale_jitter)); see _zoom_batch."""
    j = float(scale_jitter)
    scales = torch.exp(torch.empty(x.shape[0]).uniform_(-j, j))
    return _zoom_batch(x, scales, canvas, antialias=antialias)


def _rescale_cd_targets(y: torch.Tensor, y_raw: torch.Tensor, cd_mult: torch.Tensor,
                        stats: dict[str, dict[str, float]], target_space: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Multiply the raw CD label by cd_mult and re-normalise the CD column of y the way the dataset does
    (target space transform, then z-score); CV and HEX are unchanged. Returns new tensors."""
    y_raw = y_raw.clone()
    y = y.clone()
    cd_mult = cd_mult.to(y_raw.device, y_raw.dtype)
    y_raw[:, 0] = y_raw[:, 0] * cd_mult
    cd = y_raw[:, 0]
    if target_space == "log":
        cd = torch.log(cd.clamp_min(1e-6))
    elif target_space != "linear":
        raise ValueError(f"Unknown target_space: {target_space}")
    cd_name = METRICS[0]
    y[:, 0] = ((cd - stats[cd_name]["mean"]) / stats[cd_name]["std"]).to(y.dtype)
    return y, y_raw


def _photometric_augment(x: torch.Tensor) -> torch.Tensor:
    """GPU-side photometric jitter on a float [0,1] image batch (B,1,H,W)."""
    import torchvision.transforms.functional as TF
    x = x.float()
    if x.max() > 1.5:
        x = x / 255.0
    batch = x.shape[0]
    device = x.device
    gamma = torch.empty(batch, 1, 1, 1, device=device).uniform_(0.7, 1.4)
    contrast = torch.empty(batch, 1, 1, 1, device=device).uniform_(0.7, 1.3)
    brightness = torch.empty(batch, 1, 1, 1, device=device).uniform_(-20.0 / 255.0, 20.0 / 255.0)
    x = x.clamp(0.0, 1.0).pow(gamma)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * contrast + mean + brightness
    x = x.clamp(0.0, 1.0)

    blur_mask = torch.rand(batch, device=device) < 0.3
    if blur_mask.any():
        sigmas = torch.empty(batch, device=device).uniform_(0.0, 1.2)
        for i in range(batch):
            if blur_mask[i] and sigmas[i] > 0.05:
                k = int(2 * round(3 * float(sigmas[i])) + 1)
                k = max(3, k + (1 - k % 2))
                x[i : i + 1] = TF.gaussian_blur(x[i : i + 1], kernel_size=k, sigma=float(sigmas[i]))

    noise_mask = torch.rand(batch, device=device) < 0.3
    if noise_mask.any():
        sigma_n = torch.empty(batch, 1, 1, 1, device=device).uniform_(0.0, 6.0 / 255.0)
        noise = torch.randn_like(x) * sigma_n
        x = torch.where(noise_mask.view(batch, 1, 1, 1), x + noise, x)
    return x.clamp(0.0, 1.0)


class MetricRegressionDataset(Dataset):
    def __init__(
        self,
        memmap: np.memmap,
        frame: pd.DataFrame,
        target_stats_dict: dict[str, dict[str, float]],
        uint8_inputs: bool = True,
        augment_flips: bool = False,
        target_space: str = "linear",
    ):
        self.memmap = memmap
        self.frame = frame.reset_index(drop=True)
        self.target_stats_dict = target_stats_dict
        self.target_space = target_space
        raw = self.frame[list(METRICS)].to_numpy(dtype=np.float64)
        transformed = _forward_target_space(raw, target_space)
        targets = np.zeros_like(transformed, dtype=np.float32)
        for j, col in enumerate(METRICS):
            mean = target_stats_dict[col]["mean"]
            std = target_stats_dict[col]["std"]
            targets[:, j] = (transformed[:, j] - mean) / std
        self.targets = targets
        self.raw_targets = raw.astype(np.float32)
        self.indices = self.frame["idx"].to_numpy(dtype=np.int64)
        self.uint8_inputs = uint8_inputs
        self.augment_flips = augment_flips

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        idx = int(self.indices[i])
        img = self.memmap[idx].copy() if self.uint8_inputs else self.memmap[idx].astype(np.float32) / 255.0
        if self.augment_flips:
            if torch.rand(()) < 0.5:
                img = np.flip(img, axis=1).copy()
            if torch.rand(()) < 0.5:
                img = np.flip(img, axis=0).copy()
        x = torch.from_numpy(img).unsqueeze(0)
        y = torch.from_numpy(self.targets[i])
        y_raw = torch.from_numpy(self.raw_targets[i])
        return x, y, y_raw, idx


def apply_exclusions(train_idx: list[int], exclude_idx_file: str) -> list[int]:
    """Drop cache indices listed in exclude_idx_file from a training index list (validation is untouched)."""
    if not exclude_idx_file:
        return list(train_idx)
    text = Path(exclude_idx_file).read_text().split()
    excluded = {int(t) for t in text}
    kept = [i for i in train_idx if i not in excluded]
    print(f"[regression_cnn] exclude_idx_file={exclude_idx_file}: dropped {len(train_idx) - len(kept)} of {len(train_idx)} train images", flush=True)
    return kept
