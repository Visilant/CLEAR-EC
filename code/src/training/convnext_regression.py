"""ConvNeXt-Tiny regressors with deterministic whole-image and region inputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _import_torchvision_models():
    # Some training hosts provide a CPU torchvision wheel beside a newer CUDA
    # PyTorch build. ConvNeXt itself does not use NMS, but torchvision registers
    # an NMS fake kernel during import and expects the operator to exist.
    try:
        return __import__("torchvision.models", fromlist=["models"])
    except RuntimeError as exc:
        if "torchvision::nms does not exist" not in str(exc):
            raise
        lib = torch.library.Library("torchvision", "DEF")
        lib.define("nms(Tensor boxes, Tensor scores, float iou_threshold) -> Tensor")
        import sys
        for name in list(sys.modules):
            if name == "torchvision" or name.startswith("torchvision."):
                del sys.modules[name]
        models = __import__("torchvision.models", fromlist=["models"])
        models._clear_ec_library = lib
        return models


class ConvNeXtTinyRegression(nn.Module):
    """ImageNet ConvNeXt-Tiny using either the full frame or four shared patches."""

    def __init__(
        self,
        input_mode: str = "whole",
        pretrained: bool = True,
        context_size: tuple[int, int] = (486, 648),
        patch_size: int = 384,
        num_patches: int = 4,
    ):
        super().__init__()
        if input_mode not in {"whole", "fixed", "quality"}:
            raise ValueError(f"Unknown ConvNeXt input mode: {input_mode}")
        if num_patches != 4:
            raise ValueError("The validated regional design uses exactly four patches")
        models = _import_torchvision_models()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        self.feature_dim = backbone.classifier[-1].in_features
        self.input_mode = input_mode
        self.context_size = tuple(context_size)
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)
        self.register_buffer("image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])
        head_in = self.feature_dim if input_mode == "whole" else 2 * self.feature_dim
        self.head = nn.Sequential(
            nn.LayerNorm(head_in), nn.Dropout(0.2), nn.Linear(head_in, 3)
        )

    def _prepare(self, x: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        else:
            x = x.float()
        if size is not None and tuple(x.shape[-2:]) != tuple(size):
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.image_mean) / self.image_std

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def _grid_patches(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        height, width = x.shape[-2:]
        if height < self.patch_size or width < self.patch_size:
            raise ValueError(f"Input {height}x{width} is smaller than patch size {self.patch_size}")
        ys = torch.linspace(0, height - self.patch_size, 4, device=x.device).round().long()
        xs = torch.linspace(0, width - self.patch_size, 5, device=x.device).round().long()
        patches, coords = [], []
        for y in ys.tolist():
            for x0 in xs.tolist():
                patches.append(x[..., y:y + self.patch_size, x0:x0 + self.patch_size])
                coords.append((y, x0))
        return torch.stack(patches, dim=1), torch.tensor(coords, device=x.device)

    def _select_patches(self, x: torch.Tensor) -> torch.Tensor:
        patches, coords = self._grid_patches(x)
        if self.input_mode == "fixed":
            # Four interior quadrants in the 4x5 grid.
            return patches[:, [6, 8, 11, 13]]

        gray = patches.float()
        if gray.dtype == torch.uint8:
            gray = gray.div(255.0)
        elif gray.detach().amax() > 2:
            gray = gray.div(255.0)
        flat = gray.flatten(2)
        contrast = flat.std(dim=2)
        dynamic_range = flat.quantile(0.95, dim=2) - flat.quantile(0.05, dim=2)
        edge = (gray[..., 1:, :] - gray[..., :-1, :]).abs().mean(dim=(-3, -2, -1))
        edge += (gray[..., :, 1:] - gray[..., :, :-1]).abs().mean(dim=(-3, -2, -1))
        saturation = ((gray < 0.02) | (gray > 0.98)).float().mean(dim=(-3, -2, -1))
        scores = contrast + 0.5 * dynamic_range + 0.5 * edge - 0.5 * saturation

        chosen = []
        centers = coords.float() + self.patch_size / 2
        for b in range(len(x)):
            order = torch.argsort(scores[b], descending=True).tolist()
            keep: list[int] = []
            for candidate in order:
                if all(torch.linalg.vector_norm(centers[candidate] - centers[j]).item() >= 240 for j in keep):
                    keep.append(candidate)
                if len(keep) == self.num_patches:
                    break
            for candidate in order:
                if len(keep) == self.num_patches:
                    break
                if candidate not in keep:
                    keep.append(candidate)
            chosen.append(patches[b, keep])
        return torch.stack(chosen)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_mode == "whole":
            return self.head(self._encode(self._prepare(x, self.context_size)))
        context = self._encode(self._prepare(x, (243, 324)))
        patches = self._select_patches(x)
        batch, count = patches.shape[:2]
        patches = patches.reshape(batch * count, *patches.shape[2:])
        patch_features = self._encode(self._prepare(patches)).reshape(batch, count, -1).mean(1)
        return self.head(torch.cat([context, patch_features], dim=1))
