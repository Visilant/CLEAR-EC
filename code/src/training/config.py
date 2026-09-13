"""Regression run configuration. Field names and defaults are frozen: every saved checkpoint stores asdict(RegressionConfig)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


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
    sched_epochs: int = 0  # 0 uses epochs; otherwise hold the cosine floor after this budget
    save_epochs: tuple[int, ...] = ()  # retain fixed-budget comparison checkpoints
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
    exclude_idx_file: str = ""  # whitespace-separated cache indices dropped from TRAINING only (label-cleaning runs)
    crop_scale: float = 1.0  # <1 enables scale-preserving random crops (min side fraction) in training
    cd_head: str = "gap"  # gap | density: CD as the spatial sum of a softplus 1x1-conv map (ConvNeXt whole-image only)
    loss_trim: float = 0.0  # relative loss: drop the ceil(loss_trim * batch) largest per-sample errors when batch >= 4

    def __post_init__(self) -> None:
        # JSON transport (config.json, checkpoints) turns the tuple into a list; normalise it back.
        self.save_epochs = tuple(int(e) for e in (self.save_epochs or ()))


# Fields that identify a particular run rather than the recipe it follows.
RUN_IDENTITY_FIELDS = ("seed", "fold", "n_folds", "all_data", "num_workers", "cpu_threads")


def recipe_hash(cfg: RegressionConfig) -> str:
    """12-hex digest of the recipe (config minus run identity); five folds of one recipe share it."""
    payload = {k: v for k, v in asdict(cfg).items() if k not in RUN_IDENTITY_FIELDS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:12]


def config_from_checkpoint(config: dict) -> RegressionConfig:
    """Rebuild a config saved in a checkpoint; fields added later take their defaults."""
    config = dict(config)
    config.setdefault("uint8_inputs", False)  # Historical checkpoints used CPU float conversion.
    return RegressionConfig(**config)
