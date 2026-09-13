"""Small, offline-loadable architecture space for endothelial regression NAS.

All candidates are trained from scratch. These are configurable block families,
not pretrained ResNet/EfficientNet/ConvNeXt checkpoints.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Architecture:
    block: str = 'residual'
    width: int = 24
    depths: tuple = (1, 1, 2, 1)
    kernel: int = 3
    downsample: int = 4
    pooling: str = 'mean_std'
    normalization: str = 'batch'

    def __post_init__(self):
        if self.block not in ('residual', 'inverted', 'convnext'):
            raise ValueError('Unknown NAS block')
        if self.pooling not in ('mean', 'mean_std', 'attention', 'multiscale'):
            raise ValueError('Unknown NAS pooling')
        if self.normalization not in ('batch', 'group'):
            raise ValueError('Unknown NAS normalization')
        if self.width not in (16, 24, 32) or self.kernel not in (3, 5, 7):
            raise ValueError('Invalid NAS width/kernel')
        if len(self.depths) != 4 or any(d not in (1, 2, 3) for d in self.depths):
            raise ValueError('Invalid NAS stage depths')
        if self.downsample not in (2, 4):
            raise ValueError('Invalid NAS resolution')


def norm(channels, name):
    return nn.BatchNorm2d(channels) if name == 'batch' else nn.GroupNorm(8, channels)


class Block(nn.Module):
    def __init__(self, channels, spec):
        super().__init__()
        c, k = channels, spec.kernel
        if spec.block == 'residual':
            self.body = nn.Sequential(
                nn.Conv2d(c, c, k, padding=k//2, bias=False), norm(c, spec.normalization), nn.SiLU(),
                nn.Conv2d(c, c, k, padding=k//2, bias=False), norm(c, spec.normalization))
        elif spec.block == 'inverted':
            self.body = nn.Sequential(
                nn.Conv2d(c, 3*c, 1, bias=False), norm(3*c, spec.normalization), nn.SiLU(),
                nn.Conv2d(3*c, 3*c, k, padding=k//2, groups=3*c, bias=False),
                norm(3*c, spec.normalization), nn.SiLU(),
                nn.Conv2d(3*c, c, 1, bias=False), norm(c, spec.normalization))
        else:
            self.body = nn.Sequential(
                nn.Conv2d(c, c, k, padding=k//2, groups=c), nn.GroupNorm(1, c),
                nn.Conv2d(c, 4*c, 1), nn.GELU(), nn.Conv2d(4*c, c, 1))
        # Stabilizes randomly initialized residual branches without new tuning.
        self.scale = nn.Parameter(torch.full((1, c, 1, 1), 0.1))

    def forward(self, x):
        return x + self.scale * self.body(x)


class SearchRegressionCNN(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        spec = Architecture(**architecture)
        self.spec = spec
        widths = [spec.width * m for m in (1, 2, 4, 8)]
        self.stem = nn.Sequential(nn.Conv2d(1, widths[0], 5, stride=2, padding=2, bias=False),
                                  norm(widths[0], spec.normalization), nn.SiLU())
        self.stages = nn.ModuleList()
        for i, (width, depth) in enumerate(zip(widths, spec.depths)):
            layers = []
            if i:
                layers += [nn.Conv2d(widths[i-1], width, 3, stride=2, padding=1, bias=False),
                           norm(width, spec.normalization), nn.SiLU()]
            layers += [Block(width, spec) for _ in range(depth)]
            self.stages.append(nn.Sequential(*layers))
        last = widths[-1]
        self.attention = nn.Conv2d(last, 1, 1) if spec.pooling == 'attention' else None
        features = last * 2 if spec.pooling == 'mean_std' else last
        if spec.pooling == 'multiscale':
            features += widths[-2]
        self.head = nn.Sequential(nn.Linear(features, 128), nn.SiLU(), nn.Dropout(0.2), nn.Linear(128, 3))

    def forward(self, x):
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = F.interpolate(x, scale_factor=1.0/self.spec.downsample, mode='bilinear', align_corners=False)
        x = self.stem(x)
        maps = []
        for stage in self.stages:
            x = stage(x)
            maps.append(x)
        avg = x.mean((2, 3))
        if self.spec.pooling == 'mean_std':
            pooled = torch.cat((avg, (x.var((2, 3), unbiased=False) + 1e-6).sqrt()), dim=1)
        elif self.spec.pooling == 'attention':
            weights = self.attention(x).flatten(2).softmax(dim=-1)
            pooled = (x.flatten(2) * weights).sum(-1)
        elif self.spec.pooling == 'multiscale':
            pooled = torch.cat((maps[-2].mean((2, 3)), avg), dim=1)
        else:
            pooled = avg
        return self.head(pooled)
