"""Direct CD/CV/HEX regression CNN on grayscale microscopy images."""

from __future__ import annotations

import json
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
    normalize_targets,
    score_by_id,
    target_stats,
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


class MetricRegressionDataset(Dataset):
    def __init__(
        self,
        memmap: np.memmap,
        frame: pd.DataFrame,
        target_stats_dict: dict[str, dict[str, float]],
        uint8_inputs: bool = True,
        augment_flips: bool = False,
    ):
        self.memmap = memmap
        self.frame = frame.reset_index(drop=True)
        self.target_stats_dict = target_stats_dict
        self.targets = normalize_targets(self.frame, target_stats_dict)
        self.indices = self.frame["idx"].to_numpy(dtype=np.int64)
        self.uint8_inputs = uint8_inputs
        self.augment_flips = augment_flips

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx = int(self.indices[i])
        img = self.memmap[idx].copy() if self.uint8_inputs else self.memmap[idx].astype(np.float32) / 255.0
        if self.augment_flips:
            if torch.rand(()) < 0.5:
                img = np.flip(img, axis=1).copy()
            if torch.rand(()) < 0.5:
                img = np.flip(img, axis=0).copy()
        x = torch.from_numpy(img).unsqueeze(0)
        y = torch.from_numpy(self.targets[i])
        return x, y, idx


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
    """Mean absolute relative error in original metric units."""

    def __init__(self, stats: dict[str, dict[str, float]]):
        super().__init__()
        self.register_buffer("means", torch.tensor([stats[m]["mean"] for m in METRICS]))
        self.register_buffer("stds", torch.tensor([stats[m]["std"] for m in METRICS]))

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_original = pred * self.stds + self.means
        target_original = target * self.stds + self.means
        # Exact zero labels are excluded by the challenge MAPE definition. A
        # small tolerance accounts for normalized float32 round trips.
        valid = target_original.abs() > 1e-5
        relative = (pred_original - target_original).abs() / target_original.abs().clamp_min(1e-5)
        return relative[valid].mean()


def _build_loss(name: str, stats: dict[str, dict[str, float]] | None = None) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    if name == "relative":
        if stats is None:
            raise ValueError("Relative loss requires target statistics")
        return RelativeAbsoluteErrorLoss(stats)
    if name != "huber":
        raise ValueError(f"Unknown loss: {name}")
    return nn.HuberLoss(delta=1.0)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stats: dict[str, dict[str, float]],
    optimizer: torch.optim.Optimizer | None = None,
    scaler=None,
    amp: bool = False,
) -> tuple[float, dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = torch.zeros((), device=device)
    n_images = 0
    preds_list: list[torch.Tensor] = []
    index_list: list[np.ndarray] = []

    for x, y, indices in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                pred = model(x)
                loss = criterion(pred, y)
            if train_mode:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        total_loss += loss.detach() * len(x)
        n_images += len(x)
        preds_list.append(pred.detach().float())
        index_list.append(indices.numpy())

    if not n_images:
        raise ValueError("Cannot run an epoch on an empty dataset")
    avg_loss = float(total_loss.item()) / n_images
    preds = denormalize_targets(torch.cat(preds_list).cpu().numpy(), stats)
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
) -> pd.DataFrame:
    ds = MetricRegressionDataset(memmap, frame, target_stats_dict, uint8_inputs=uint8_inputs)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    rows = []
    offset = 0
    for x, _, _ in loader:
        pred_norm = model(x.to(device)).cpu().numpy()
        batch_len = len(pred_norm)
        batch_frame = frame.iloc[offset : offset + batch_len]
        pred = denormalize_targets(pred_norm, target_stats_dict)
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
    if cfg.model == "convnext_tiny":
        from src.training.convnext_regression import ConvNeXtTinyRegression
        return ConvNeXtTinyRegression(
            input_mode=cfg.input_mode,
            pretrained=cfg.pretrained if load_pretrained is None else load_pretrained,
            context_size=(cfg.context_height, cfg.context_width),
            patch_size=cfg.patch_size,
            num_patches=cfg.num_patches,
        )
    if cfg.model != "small":
        raise ValueError(f"Unknown regression model: {cfg.model}")
    return SmallRegressionCNN(downsample=cfg.downsample, normalization=cfg.normalization)


def _load_checkpoint(results_dir: Path, device: torch.device) -> tuple[nn.Module, dict, RegressionConfig]:
    ckpt = torch.load(results_dir / "best_model.pt", map_location=device, weights_only=False)
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
    """Fit on train, select checkpoint on val. Does not load the test split."""
    cfg = config or RegressionConfig()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    if (results_dir / "best_model.pt").exists():
        raise FileExistsError(f"Checkpoint already exists in {results_dir}; use a new results_dir")
    if cfg.epochs < 1 or cfg.batch_size < 1 or cfg.num_workers < 0 or cfg.cpu_threads < 1:
        raise ValueError("Invalid training epochs/batch_size/num_workers/cpu_threads")
    torch.set_num_threads(cfg.cpu_threads)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    memmap, _index_df = open_image_cache(cache_dir)

    train_idx = load_split(cache_dir, "train")
    val_idx = load_split(cache_dir, "val")
    train_df = labels_for_indices(cache_dir, labels_csv, train_idx)
    val_df = labels_for_indices(cache_dir, labels_csv, val_idx)

    stats = target_stats(train_df)
    train_ds = MetricRegressionDataset(memmap, train_df, stats, cfg.uint8_inputs,
                                       augment_flips=cfg.augment_flips)
    val_ds = MetricRegressionDataset(memmap, val_df, stats, cfg.uint8_inputs)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
    )
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
    criterion = _build_loss(cfg.loss, stats).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val = float("inf")
    best_state: dict | None = None
    stale = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        train_loss, train_mape = _run_epoch(
            model, train_loader, criterion, device, stats, optimizer, scaler=scaler, amp=cfg.amp
        )
        val_loss, val_mape = _run_epoch(model, val_loader, criterion, device, stats)
        row = {
            "epoch": epoch,
            "seconds": time.perf_counter() - epoch_start,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_mape_{k}": v for k, v in train_mape.items()},
            **{f"val_mape_{k}": v for k, v in val_mape.items()},
        }
        history.append(row)
        print(
            f"[regression_cnn seed={cfg.seed}] epoch {epoch}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mape_mean={val_mape.get('mean', float('nan')):.2f}%",
            flush=True,
        )

        val_key = float(val_mape["mean"])
        if val_key < best_val:
            best_val = val_key
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            with atomic_path(results_dir / "best_model.pt") as temporary:
                torch.save({"model_state": best_state, "target_stats": stats,
                            "config": asdict(cfg), "epoch": epoch,
                            "val_mape": val_mape}, temporary)
            stale = 0
        else:
            stale += 1
        with atomic_path(results_dir / "history.csv") as temporary:
            pd.DataFrame(history).to_csv(temporary, index=False)
        if stale >= cfg.patience:
            print(f"[regression_cnn seed={cfg.seed}] early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = results_dir / "best_model.pt"
    if best_state is None:
        raise RuntimeError("No finite validation checkpoint was produced")

    pd.DataFrame(history).to_csv(results_dir / "history.csv", index=False)

    val_pred = predict_indices(model, memmap, val_df, stats, device, batch_size=cfg.batch_size,
                               uint8_inputs=cfg.uint8_inputs)
    val_pred.to_csv(results_dir / "predictions_val.csv", index=False)
    val_mape = score_by_id(val_pred, val_df, expected_ids=val_df.ID)

    metrics = {
        "val_mape": val_mape,
        "val_best_mape_mean": best_val,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "seed": cfg.seed,
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    write_manifest(
        results_dir.parent.parent if results_dir.name.startswith("seed_") else results_dir.parent,
        method=f"regression_cnn_seed_{cfg.seed}",
        metrics=metrics,
        extra={"checkpoint": str(ckpt_path), "config": asdict(cfg)},
    )
    print(f"[regression_cnn seed={cfg.seed}] val MAPE: {val_mape}")
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
                           uint8_inputs=cfg.uint8_inputs)
    pred.to_csv(Path(results_dir) / f"predictions_{split}.csv", index=False)
    scores = score_by_id(pred, frame, expected_ids=frame.ID)
    metrics_path = Path(results_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics[f"{split}_mape"] = scores
    metrics[f"n_{split}"] = len(frame)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[regression_cnn seed={cfg.seed}] {split} MAPE: {scores}")
    return scores
