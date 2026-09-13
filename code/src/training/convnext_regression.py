"""ConvNeXt regressors with deterministic whole-image and region inputs, and the density-map CD head."""

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


CD_HEADS = ("gap", "density")
FEATURE_STRIDE = 32  # ConvNeXt: 4x4/4 stem then three 2x2/2 downsamples, each a floor division


def _feature_grid(context_size: tuple[int, int]) -> tuple[int, int]:
    """Final feature-map size of a ConvNeXt for an input of context_size."""
    return (int(context_size[0]) // FEATURE_STRIDE, int(context_size[1]) // FEATURE_STRIDE)


class _ChannelLayerNorm(nn.LayerNorm):
    """LayerNorm over the channel axis of an NCHW map (per position), as in the GAP head's LayerNorm."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class DensityCDHead(nn.Module):
    """CD as the mean of a non-negative 1x1-conv map: exp(log_scale) * mean(softplus(conv(norm(F)))) + bias.

    The frame is 1000 x 750 um, so 0.75 x CD is a count and a summed map carries counting semantics; the
    mean is that sum divided by the fixed grid area, so the conv sees the same mean-pooled gradient scale as
    the GAP head's linear layer. log_scale starts at 0: softplus of a default-initialised conv over
    LayerNormed features averages about log(2), the same order as the GAP head's initial output (tested
    within 2x). bias starts at 0 and absorbs the target normalisation offset (normalised CD is negative
    below the dataset mean, which a non-negative map cannot reach on its own).

    The first screen (results/cnn_20260913/c1/density0) parameterised this as scale * sum with scale
    starting at 1 / (H' * W'); such checkpoints load through the pre-hook below, forward-equivalent."""

    def __init__(self, in_channels: int, grid: tuple[int, int]):
        super().__init__()
        self.grid = (int(grid[0]), int(grid[1]))
        self.norm = _ChannelLayerNorm(in_channels, eps=1e-6)
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))
        self._register_load_state_dict_pre_hook(self._upgrade_legacy_scale)

    def _upgrade_legacy_scale(self, state_dict, prefix, *args) -> None:
        legacy, current = prefix + "scale", prefix + "log_scale"
        if legacy in state_dict and current not in state_dict:
            scale = state_dict.pop(legacy).float() * float(self.grid[0] * self.grid[1])
            if not scale > 0:
                raise ValueError(f"legacy density scale {float(scale)} cannot be expressed as exp(log_scale)")
            state_dict[current] = scale.log().to(state_dict.get(prefix + "bias", scale).dtype)

    def density_map(self, features: torch.Tensor) -> torch.Tensor:
        return F.softplus(self.conv(self.norm(features)))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.log_scale.exp() * self.density_map(features).mean(dim=(1, 2, 3)) + self.bias


def _check_cd_head(cd_head: str, input_mode: str = "whole") -> str:
    if cd_head not in CD_HEADS:
        raise ValueError(f"Unknown cd_head: {cd_head!r} (expected one of {CD_HEADS})")
    if cd_head == "density" and input_mode != "whole":
        raise ValueError("cd_head='density' needs the whole-image input mode")
    return cd_head


class ConvNeXtTinyRegression(nn.Module):
    """ImageNet ConvNeXt-Tiny using either the full frame or four shared patches."""

    def __init__(
        self,
        input_mode: str = "whole",
        pretrained: bool = True,
        context_size: tuple[int, int] = (486, 648),
        patch_size: int = 384,
        num_patches: int = 4,
        arch: str = "tiny",
        antialias: bool = False,
        drop_path: float = 0.1,
        cd_head: str = "gap",
    ):
        super().__init__()
        self.antialias = bool(antialias)
        self.cd_head = _check_cd_head(cd_head, input_mode)
        if arch not in {"tiny", "small", "base"}:
            raise ValueError(f"Unknown ConvNeXt arch: {arch}")
        if input_mode not in {"whole", "fixed", "quality"}:
            raise ValueError(f"Unknown ConvNeXt input mode: {input_mode}")
        if num_patches != 4:
            raise ValueError("The validated regional design uses exactly four patches")
        models = _import_torchvision_models()
        weights_enum = getattr(models, f"ConvNeXt_{arch.capitalize()}_Weights")
        weights = weights_enum.DEFAULT if pretrained else None
        backbone = getattr(models, f"convnext_{arch}")(weights=weights, stochastic_depth_prob=float(drop_path))
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
        if self.cd_head == "density":
            self.density = DensityCDHead(self.feature_dim, _feature_grid(self.context_size))

    def _prepare(self, x: torch.Tensor, size: tuple[int, int] | None = None) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        else:
            x = x.float()
        if size is not None and tuple(x.shape[-2:]) != tuple(size):
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=self.antialias)
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
            if self.cd_head == "density":
                features = self.features(self._prepare(x, self.context_size))
                out = self.head(torch.flatten(self.avgpool(features), 1))
                return torch.cat([self.density(features)[:, None], out[:, 1:]], dim=1)
            return self.head(self._encode(self._prepare(x, self.context_size)))
        context = self._encode(self._prepare(x, (243, 324)))
        patches = self._select_patches(x)
        batch, count = patches.shape[:2]
        patches = patches.reshape(batch * count, *patches.shape[2:])
        patch_features = self._encode(self._prepare(patches)).reshape(batch, count, -1).mean(1)
        return self.head(torch.cat([context, patch_features], dim=1))


class TimmWholeImageRegression(nn.Module):
    """Whole-image regressor on a timm backbone with its own pretrained normalisation."""

    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        context_size: tuple[int, int] = (486, 648),
        cd_head: str = "gap",
    ):
        super().__init__()
        import timm

        self.cd_head = _check_cd_head(cd_head)

        kwargs = {}
        if name.startswith("vit"):
            # ViTs need a fixed token grid: context_size must be a multiple of the patch size.
            kwargs = {"img_size": tuple(context_size), "dynamic_img_size": True}
        self.backbone = timm.create_model(name, pretrained=pretrained, num_classes=0, in_chans=3, **kwargs)
        cfg = self.backbone.pretrained_cfg
        mean = cfg.get("mean", (0.485, 0.456, 0.406))
        std = cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("image_mean", torch.tensor(mean)[None, :, None, None])
        self.register_buffer("image_std", torch.tensor(std)[None, :, None, None])
        self.context_size = tuple(context_size)
        self.feature_dim = self.backbone.num_features
        self.head = nn.Sequential(
            nn.LayerNorm(self.feature_dim), nn.Dropout(0.2), nn.Linear(self.feature_dim, 3)
        )
        if self.cd_head == "density":
            self.density = DensityCDHead(self.feature_dim, _feature_grid(self.context_size))

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if x.dtype == torch.uint8:
            x = x.float().div(255.0)
        else:
            x = x.float()
        if tuple(x.shape[-2:]) != self.context_size:
            x = F.interpolate(x, size=self.context_size, mode="bilinear", align_corners=False)
        if x.shape[1] == 1:
            x = x.expand(-1, 3, -1, -1)
        return (x - self.image_mean) / self.image_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cd_head == "density":
            # backbone(x) == forward_head(forward_features(x)); CV and HEX keep that pooled path exactly.
            feature_map = self.backbone.forward_features(self._prepare(x))
            out = self.head(self.backbone.forward_head(feature_map))
            return torch.cat([self.density(feature_map)[:, None], out[:, 1:]], dim=1)
        features = self.backbone(self._prepare(x))
        return self.head(features)
