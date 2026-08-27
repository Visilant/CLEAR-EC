"""Direct CD/CV/HEX regression CNN on grayscale microscopy images."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.data.cache import open_image_cache
from src.data.splits import load_split
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


class MetricRegressionDataset(Dataset):
    def __init__(
        self,
        memmap: np.memmap,
        frame: pd.DataFrame,
        target_stats_dict: dict[str, dict[str, float]],
    ):
        self.memmap = memmap
        self.frame = frame.reset_index(drop=True)
        self.target_stats_dict = target_stats_dict
        self.targets = normalize_targets(self.frame, target_stats_dict)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        idx = int(self.frame.iloc[i]["idx"])
        img = self.memmap[idx].astype(np.float32) / 255.0
        x = torch.from_numpy(img).unsqueeze(0)
        y = torch.from_numpy(self.targets[i])
        return x, y, idx


class SmallRegressionCNN(nn.Module):
    """Lightweight CNN for 972x1296 -> CD/CV/HEX (downsampled internally)."""

    def __init__(self, downsample: int = 4):
        super().__init__()
        self.downsample = downsample
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
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
        if self.downsample > 1:
            x = nn.functional.interpolate(
                x,
                scale_factor=1.0 / self.downsample,
                mode="bilinear",
                align_corners=False,
            )
        return self.head(self.features(x))


def _build_loss(name: str) -> nn.Module:
    if name == "mse":
        return nn.MSELoss()
    return nn.HuberLoss(delta=1.0)


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stats: dict[str, dict[str, float]],
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, dict[str, float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    total_loss = 0.0
    n_batches = 0
    preds_list: list[np.ndarray] = []
    gt_list: list[np.ndarray] = []

    for x, y, _ in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train_mode):
            pred = model(x)
            loss = criterion(pred, y)
            if train_mode:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item())
        n_batches += 1
        preds_list.append(pred.detach().cpu().numpy())
        gt_list.append(y.detach().cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)
    preds = denormalize_targets(np.concatenate(preds_list, axis=0), stats)
    gt = denormalize_targets(np.concatenate(gt_list, axis=0), stats)
    return avg_loss, mape_per_metric(preds, gt)


@torch.no_grad()
def predict_indices(
    model: nn.Module,
    memmap: np.memmap,
    frame: pd.DataFrame,
    target_stats_dict: dict[str, dict[str, float]],
    device: torch.device,
    batch_size: int = 16,
) -> pd.DataFrame:
    ds = MetricRegressionDataset(memmap, frame, target_stats_dict)
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


def _load_checkpoint(results_dir: Path, device: torch.device) -> tuple[nn.Module, dict, RegressionConfig]:
    ckpt = torch.load(results_dir / "best_model.pt", map_location=device, weights_only=False)
    cfg = RegressionConfig(**ckpt["config"])
    model = SmallRegressionCNN(downsample=cfg.downsample).to(device)
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
    train_ds = MetricRegressionDataset(memmap, train_df, stats)
    val_ds = MetricRegressionDataset(memmap, val_df, stats)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = SmallRegressionCNN(downsample=cfg.downsample).to(device)
    criterion = _build_loss(cfg.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    best_val = float("inf")
    best_state: dict | None = None
    stale = 0
    history: list[dict] = []

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_mape = _run_epoch(
            model, train_loader, criterion, device, stats, optimizer
        )
        val_loss, val_mape = _run_epoch(model, val_loader, criterion, device, stats)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_mape_{k}": v for k, v in train_mape.items()},
            **{f"val_mape_{k}": v for k, v in val_mape.items()},
        }
        history.append(row)
        print(
            f"[regression_cnn seed={cfg.seed}] epoch {epoch}/{cfg.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_mape_mean={val_mape.get('mean', float('nan')):.2f}%"
        )

        val_key = float(val_mape["mean"])
        if val_key < best_val:
            best_val = val_key
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= cfg.patience:
                print(f"[regression_cnn seed={cfg.seed}] early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt_path = results_dir / "best_model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "target_stats": stats,
            "config": asdict(cfg),
        },
        ckpt_path,
    )

    pd.DataFrame(history).to_csv(results_dir / "history.csv", index=False)

    val_pred = predict_indices(model, memmap, val_df, stats, device)
    val_pred.to_csv(results_dir / "predictions_val.csv", index=False)
    val_mape = score_by_id(val_pred, val_df)

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
    pred = predict_indices(model, memmap, frame, stats, device, batch_size=cfg.batch_size)
    pred.to_csv(Path(results_dir) / f"predictions_{split}.csv", index=False)
    scores = score_by_id(pred, frame)
    metrics_path = Path(results_dir) / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics[f"{split}_mape"] = scores
    metrics[f"n_{split}"] = len(frame)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[regression_cnn seed={cfg.seed}] {split} MAPE: {scores}")
    return scores
