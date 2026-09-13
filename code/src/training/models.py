"""Model factory and checkpoint loading. The state-dict layout of every model here is frozen."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.training.common import METRICS
from src.training.config import RegressionConfig, config_from_checkpoint


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


def build_regression_model(cfg: RegressionConfig, *, load_pretrained: bool | None = None) -> nn.Module:
    if cfg.architecture is not None:
        raise ValueError("NAS architectures were removed on 2026-09-13; the field is kept for checkpoint compatibility")
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


def load_checkpoint_file(path: Path, device: torch.device) -> tuple[nn.Module, dict, RegressionConfig, int]:
    """Load one checkpoint file (best_model.pt, last.pt, epoch_NNN.pt) -> (model, target_stats, cfg, epoch)."""
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = config_from_checkpoint(ckpt["config"])
    model = build_regression_model(cfg, load_pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["target_stats"], cfg, int(ckpt.get("epoch", -1))


def _load_checkpoint(results_dir: Path, device: torch.device) -> tuple[nn.Module, dict, RegressionConfig]:
    """Load best_model.pt if present, else last.pt, from a run directory."""
    results_dir = Path(results_dir)
    ckpt_path = results_dir / "best_model.pt"
    if not ckpt_path.exists():
        ckpt_path = results_dir / "last.pt"
    model, stats, cfg, _epoch = load_checkpoint_file(ckpt_path, device)
    return model, stats, cfg
