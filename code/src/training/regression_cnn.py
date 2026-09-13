"""Compatibility facade for the regression trainer.

The implementation was split into config/targets/data/models/losses/loop/folds/predict/train on
2026-09-13. Every name that existed here before is re-exported so checkpoints, inference.py, the
submission worktree and older scripts keep importing from this module.
"""

from __future__ import annotations

from src.training.config import RegressionConfig, config_from_checkpoint, recipe_hash
from src.training.data import MetricRegressionDataset, _photometric_augment, _random_crop_batch, apply_exclusions
from src.training.folds import _slide_group_folds, all_labelled_indices, resolve_indices
from src.training.loop import _EMA, _build_scheduler, _run_epoch
from src.training.losses import RelativeAbsoluteErrorLoss, _apply_criterion, _build_loss
from src.training.models import SmallRegressionCNN, _load_checkpoint, build_regression_model, load_checkpoint_file
from src.training.predict import VIEWS, predict_frame_tta, predict_indices, predict_tta
from src.training.targets import (
    _forward_target_space,
    _inverse_target_space,
    _inverse_target_space_torch,
    target_space_stats,
)
from src.training.train import evaluate_regression_split, train_regression_cnn

__all__ = [
    "RegressionConfig", "config_from_checkpoint", "recipe_hash",
    "MetricRegressionDataset", "_photometric_augment", "_random_crop_batch", "apply_exclusions",
    "_slide_group_folds", "all_labelled_indices", "resolve_indices",
    "_EMA", "_build_scheduler", "_run_epoch",
    "RelativeAbsoluteErrorLoss", "_apply_criterion", "_build_loss",
    "SmallRegressionCNN", "_load_checkpoint", "build_regression_model", "load_checkpoint_file",
    "VIEWS", "predict_frame_tta", "predict_indices", "predict_tta",
    "_forward_target_space", "_inverse_target_space", "_inverse_target_space_torch", "target_space_stats",
    "evaluate_regression_split", "train_regression_cnn",
]
