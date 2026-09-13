"""Direct CD/CV/HEX regression CNN on grayscale microscopy images."""

from __future__ import annotations

import copy
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.cache import open_image_cache
from src.data.splits import load_split
from src.artifacts import atomic_path
from src.training.common import (
    METRICS,
    denormalize_targets,
    labels_for_indices,
    mape_per_metric,
    score_by_id,
    write_manifest,
)


@dataclass
class RegressionConfig:
    epochs: int = 30
    batch_size: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-5
    patience: int = 5
    seed: int = 42
    loss: str = "huber"  # huber | mse
    downsample: int = 4
    num_workers: int = 4
    amp: bool = False
    uint8_inputs: bool = True
    cpu_threads: int = 4
    channels_last: bool = False
    normalization: str = "batch"
    architecture: dict | None = None
    model: str = "small"  # small | convnext_tiny
    input_mode: str = "whole"  # whole | fixed | quality
    pretrained: bool = True
    context_height: int = 486
    context_width: int = 648
    patch_size: int = 384
    num_patches: int = 4
    augment_flips: bool = False
    ema: float = 0.0  # 0 disables EMA; else decay, e.g. 0.999
    sched: str = "none"  # none | cosine
    warmup_epochs: int = 0
    target_space: str = "linear"  # linear | log
    photometric: bool = False
    fold: int = -1
    n_folds: int = 0
    all_data: bool = False
    clip_grad: float = 0.0  # gradient-norm clipping (0 disables); guards the relative-loss/AMP spikes seen with ConvNeXt-V2 at seed 42
    drop_path: float = 0.1  # ConvNeXt stochastic depth (torchvision default 0.1)
    antialias: bool = False  # antialiased bilinear downsampling in the ConvNeXt input path
    train_fraction: float = 1.0  # <1 keeps this fraction of training slides (learning-curve runs)
    crop_scale: float = 1.0  # <1 enables scale-preserving random crops (min side fraction) in training


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


class SmallRegressionCNN(nn.Module):
    """Lightweight CNN for 972x1296 -> CD/CV/HEX (downsampled internally)."""

    def __init__(self, downsample: int = 4, normalization: str = "batch"):
        super().__init__()
        if normalization not in ("batch", "group"):
            raise ValueError(f"Unknown normalization: {normalization}")
        def norm(channels: int) -> nn.Module:
            return (nn.BatchNorm2d(channels) if normalization == "batch"
                    else nn.GroupNorm(8, channels))
        self.downsample = downsample
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            norm(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            norm(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            norm(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(64, len(METRICS)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        if self.downsample > 1:
            x = nn.functional.interpolate(
                x,
                scale_factor=1.0 / self.downsample,
                mode="bilinear",
                align_corners=False,
            )
        return self.head(self.features(x))


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


@torch.no_grad()
def predict_indices(
    model: nn.Module,
    memmap: np.memmap,
    frame: pd.DataFrame,
    target_stats_dict: dict[str, dict[str, float]],
    device: torch.device,
    batch_size: int = 16,
    uint8_inputs: bool = True,
    target_space: str = "linear",
) -> pd.DataFrame:
    ds = MetricRegressionDataset(memmap, frame, target_stats_dict, uint8_inputs=uint8_inputs, target_space=target_space)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows = []
    offset = 0
    for x, _, _, _ in loader:
        pred_norm = model(x.to(device)).cpu().numpy()
        batch_len = len(pred_norm)
        batch_frame = frame.iloc[offset : offset + batch_len]
        pred_transformed = denormalize_targets(pred_norm, target_stats_dict)
        pred = _inverse_target_space(pred_transformed, target_space)
        for i in range(batch_len):
            row = batch_frame.iloc[i]
            rows.append(
                {
                    "idx": int(row["idx"]),
                    "ID": str(row["ID"]),
                    "CD": float(pred[i, 0]),
                    "CV": float(pred[i, 1]),
                    "HEX": float(pred[i, 2]),
                }
            )
        offset += batch_len
    return pd.DataFrame(rows)


def build_regression_model(cfg: RegressionConfig, *, load_pretrained: bool | None = None) -> nn.Module:
    if cfg.architecture is not None:
        from src.training.nas_model import SearchRegressionCNN
        return SearchRegressionCNN(cfg.architecture)
    if cfg.model in {"convnext_tiny", "convnext_small", "convnext_base"}:
        from src.training.convnext_regression import ConvNeXtTinyRegression
        return ConvNeXtTinyRegression(
            input_mode=cfg.input_mode,
            pretrained=cfg.pretrained if load_pretrained is None else load_pretrained,
            context_size=(cfg.context_height, cfg.context_width),
            patch_size=cfg.patch_size,
            num_patches=cfg.num_patches,
            arch=cfg.model.split("_", 1)[1],
            antialias=cfg.antialias,
            drop_path=cfg.drop_path,
        )
    if cfg.model.startswith("timm:"):
        from src.training.convnext_regression import TimmWholeImageRegression
        return TimmWholeImageRegression(
            name=cfg.model[len("timm:"):],
            pretrained=cfg.pretrained if load_pretrained is None else load_pretrained,
            context_size=(cfg.context_height, cfg.context_width),
        )
    if cfg.model != "small":
        raise ValueError(f"Unknown regression model: {cfg.model}")
    return SmallRegressionCNN(downsample=cfg.downsample, normalization=cfg.normalization)


def _slide_group_folds(cache_dir: Path, n_folds: int, seed: int = 42) -> dict[int, list[int]]:
    """Deterministic slide-grouped K-fold split over all labelled indices (train+val+test)."""
    _, index_df = open_image_cache(cache_dir)
    combined = sorted(
        set(load_split(cache_dir, "train"))
        | set(load_split(cache_dir, "val"))
        | set(load_split(cache_dir, "test"))
    )
    idx_to_slide = {int(idx): str(slide) for idx, slide in zip(index_df["idx"], index_df["slide_id"])}
    slides = sorted({idx_to_slide[i] for i in combined})
    rng = np.random.default_rng(seed)
    shuffled = list(rng.permutation(slides))
    fold_of_slide = {slide: i % n_folds for i, slide in enumerate(shuffled)}
    folds: dict[int, list[int]] = {k: [] for k in range(n_folds)}
    for i in combined:
        folds[fold_of_slide[idx_to_slide[i]]].append(i)
    for k in folds:
        folds[k].sort()
    return folds


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


def _load_checkpoint(results_dir: Path, device: torch.device) -> tuple[nn.Module, dict, RegressionConfig]:
    results_dir = Path(results_dir)
    ckpt_path = results_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = results_dir / "last.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = dict(ckpt["config"])
    config.setdefault("uint8_inputs", False)  # Historical checkpoints used CPU float conversion.
    cfg = RegressionConfig(**config)
    model = build_regression_model(cfg, load_pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt["target_stats"], cfg


def train_regression_cnn(
    cache_dir: Path,
    labels_csv: Path,
    results_dir: Path,
    *,
    gpu: int = 0,
    config: RegressionConfig | None = None,
) -> dict:
    """Fit on train, select checkpoint on val. Does not load the test split.

    --fold/--n_folds replace train/val with a slide-grouped fold; --all_data trains
    on train+val+test with no validation at all (fixed epochs, last-epoch checkpoint).
    """
    cfg = config or RegressionConfig()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if (results_dir / "best_model.pt").exists() or (results_dir / "last.pt").exists():
        raise FileExistsError(f"Checkpoint already exists in {results_dir}; use a new results_dir")
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.num_workers < 0 or cfg.cpu_threads < 1:
        raise ValueError("Invalid training epochs/batch_size/num_workers/cpu_threads")
    if cfg.fold >= 0 and cfg.all_data:
        raise ValueError("--fold and --all_data are mutually exclusive")
    if cfg.fold >= 0 and (cfg.n_folds < 2 or cfg.fold >= cfg.n_folds):
        raise ValueError("--fold requires 0 <= fold < n_folds and n_folds >= 2")
    torch.set_num_threads(cfg.cpu_threads)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    memmap, _index_df = open_image_cache(cache_dir)

    has_val = True
    if cfg.all_data:
        train_idx = sorted(
            set(load_split(cache_dir, "train"))
            | set(load_split(cache_dir, "val"))
            | set(load_split(cache_dir, "test"))
        )
        val_idx: list[int] = []
        has_val = False
    elif cfg.fold >= 0:
        folds = _slide_group_folds(cache_dir, cfg.n_folds, seed=42)
        val_idx = folds[cfg.fold]
        train_idx = sorted(i for k, idxs in folds.items() if k != cfg.fold for i in idxs)
    else:
        train_idx = load_split(cache_dir, "train")
        val_idx = load_split(cache_dir, "val")
    if cfg.train_fraction < 1.0:
        slide_of = dict(zip(_index_df["idx"].astype(int), _index_df["slide_id"].astype(str)))
        slides = sorted({slide_of[i] for i in train_idx})
        keep = set(np.random.default_rng(cfg.seed).permutation(slides)[: int(round(len(slides) * cfg.train_fraction))])
        train_idx = sorted(i for i in train_idx if slide_of[i] in keep)
        print(f"[regression_cnn seed={cfg.seed}] train_fraction={cfg.train_fraction}: {len(train_idx)} train images", flush=True)

    train_df = labels_for_indices(cache_dir, labels_csv, train_idx)
    val_df = labels_for_indices(cache_dir, labels_csv, val_idx) if has_val else None

    stats = target_space_stats(train_df, cfg.target_space)
    train_ds = MetricRegressionDataset(memmap, train_df, stats, cfg.uint8_inputs,
                                       augment_flips=cfg.augment_flips, target_space=cfg.target_space)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
    val_loader = None
    if has_val:
        val_ds = MetricRegressionDataset(memmap, val_df, stats, cfg.uint8_inputs, target_space=cfg.target_space)
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            pin_memory=device.type == "cuda",
            persistent_workers=cfg.num_workers > 0,
        )

    model = build_regression_model(cfg).to(device)
    if cfg.channels_last:
        model = model.to(memory_format=torch.channels_last)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    criterion = _build_loss(cfg.loss, stats, target_space=cfg.target_space).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    lr_scheduler = _build_scheduler(optimizer, cfg.sched, cfg.warmup_epochs, cfg.epochs, len(train_loader))

    ema = _EMA(model, cfg.ema) if cfg.ema > 0 else None
    ema_update = (lambda: ema.update(model)) if ema is not None else None
    eval_model = ema.model if ema is not None else model

    no_early_stop = cfg.patience <= 0
    best_val = float("inf")
    best_state: dict | None = None
    stale = 0
    history: list[dict] = []
    last_epoch_completed = 0

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_mape = _run_epoch(
            model, train_loader, criterion, device, stats, optimizer, scaler=scaler, amp=cfg.amp,
            target_space=cfg.target_space, photometric=cfg.photometric,
            ema_update=ema_update, lr_scheduler=lr_scheduler, crop_scale=cfg.crop_scale, clip_grad=cfg.clip_grad,
        )
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_start,
            "train_loss": train_loss,
            "lr": optimizer.param_groups[0]["lr"],
            **{f"train_mape_{k}": v for k, v in train_mape.items()},
        }
        val_mape = None
        if has_val:
            val_loss, val_mape = _run_epoch(eval_model, val_loader, criterion, device, stats,
                                            target_space=cfg.target_space)
            row["val_loss"] = val_loss
            row.update({f"val_mape_{k}": v for k, v in val_mape.items()})
        history.append(row)
        msg = (f"[regression_cnn seed={cfg.seed}] epoch {epoch}/{cfg.epochs} "
               f"train_loss={train_loss:.4f} lr={row['lr']:.2e}")
        if has_val:
            msg += f" val_loss={val_loss:.4f} val_mape_mean={val_mape.get('mean', float('nan')):.2f}%"
        print(msg, flush=True)

        last_epoch_completed = epoch
        if has_val:
            val_key = float(val_mape["mean"])
            if val_key < best_val:
                best_val = val_key
                best_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
                with atomic_path(results_dir / "best_model.pt") as temporary:
                    torch.save({"model_state": best_state, "target_stats": stats,
                                "config": asdict(cfg), "epoch": epoch,
                                "val_mape": val_mape}, temporary)
                stale = 0
            else:
                stale += 1
        with atomic_path(results_dir / "history.csv") as temporary:
            pd.DataFrame(history).to_csv(temporary, index=False)
        if has_val and not no_early_stop and stale >= cfg.patience:
            print(f"[regression_cnn seed={cfg.seed}] early stop at epoch {epoch}")
            break

    # 'last' checkpoint: --patience 0 keeps it for resuming/inspection; a no-val
    # run (--all_data / --fold has_val still true, so only all_data lands here
    # without a best checkpoint) always needs a checkpoint of its own.
    last_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
    if no_early_stop or not has_val:
        with atomic_path(results_dir / "last.pt") as temporary:
            torch.save({"model_state": last_state, "target_stats": stats,
                        "config": asdict(cfg), "epoch": last_epoch_completed}, temporary)

    if has_val:
        if best_state is None:
            raise RuntimeError("No finite validation checkpoint was produced")
        eval_model.load_state_dict(best_state)
        ckpt_path = results_dir / "best_model.pt"
    else:
        eval_model.load_state_dict(last_state)
        ckpt_path = results_dir / "last.pt"

    pd.DataFrame(history).to_csv(results_dir / "history.csv", index=False)

    metrics: dict = {
        "n_train": len(train_df),
        "seed": cfg.seed,
        "epochs_run": last_epoch_completed,
    }
    if has_val:
        val_pred = predict_indices(eval_model, memmap, val_df, stats, device, batch_size=cfg.batch_size,
                                   uint8_inputs=cfg.uint8_inputs, target_space=cfg.target_space)
        val_pred.to_csv(results_dir / "predictions_val.csv", index=False)
        val_mape = score_by_id(val_pred, val_df, expected_ids=val_df.ID)
        metrics.update({"val_mape": val_mape, "val_best_mape_mean": best_val, "n_val": len(val_df)})
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    write_manifest(
        results_dir.parent.parent if results_dir.name.startswith("seed_") else results_dir.parent,
        method=f"regression_cnn_seed_{cfg.seed}",
        metrics=metrics,
        extra={"checkpoint": str(ckpt_path), "config": asdict(cfg)},
    )
    if has_val:
        print(f"[regression_cnn seed={cfg.seed}] val MAPE: {metrics['val_mape']}")
    else:
        print(f"[regression_cnn seed={cfg.seed}] all_data run complete, no validation")
    return metrics


def evaluate_regression_split(
    cache_dir: Path,
    labels_csv: Path,
    results_dir: Path,
    *,
    split: str,
    gpu: int = 0,
) -> dict:
    """Score a frozen regression checkpoint on a split (test only after freeze)."""
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    model, stats, cfg = _load_checkpoint(Path(results_dir), device)
    memmap, _ = open_image_cache(cache_dir)
    indices = load_split(cache_dir, split)
    frame = labels_for_indices(cache_dir, labels_csv, indices)
    pred = predict_indices(model, memmap, frame, stats, device, batch_size=cfg.batch_size,
                           uint8_inputs=cfg.uint8_inputs, target_space=cfg.target_space)
    pred.to_csv(Path(results_dir) / f"predictions_{split}.csv", index=False)
    scores = score_by_id(pred, frame, expected_ids=frame.ID)
    metrics_path = Path(results_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics[f"{split}_mape"] = scores
    metrics[f"n_{split}"] = len(frame)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[regression_cnn seed={cfg.seed}] {split} MAPE: {scores}")
    return scores
