"""Inference over cache indices, with optional flip test-time augmentation (geometric mean)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.common import METRICS, denormalize_targets
from src.training.config import RegressionConfig
from src.training.data import MetricRegressionDataset
from src.training.targets import _inverse_target_space

# TTA views as (hflip, vflip); hvflip equals a 180-degree rotation. No 90-degree views: frames are not square.
VIEWS = {"none": [(False, False)], "flips": [(False, False), (True, False), (False, True), (True, True)]}


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


@torch.no_grad()
def predict_tta(model: nn.Module, stats: dict, cfg: RegressionConfig, memmap: np.memmap, indices: list[int],
                device: torch.device, views: list[tuple[bool, bool]], batch_size: int = 16) -> np.ndarray:
    """Raw-unit predictions (n, 3) as the geometric mean over the TTA views. Reference implementation
    for the submitted container (night_predict.py, 2026-09-12)."""
    preds = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        x = torch.from_numpy(np.stack([memmap[i] for i in chunk])).unsqueeze(1).to(device)
        if not cfg.uint8_inputs:
            x = x.float() / 255.0
        logs = []
        for hflip, vflip in views:
            xv = x
            if hflip:
                xv = torch.flip(xv, dims=[-1])
            if vflip:
                xv = torch.flip(xv, dims=[-2])
            out = model(xv).float().cpu().numpy()
            out = _inverse_target_space(denormalize_targets(out, stats), cfg.target_space)
            logs.append(np.log(np.clip(out, 1e-6, None)))
        preds.append(np.exp(np.mean(logs, axis=0)))
    return np.concatenate(preds)


def predict_frame_tta(model: nn.Module, stats: dict, cfg: RegressionConfig, memmap: np.memmap,
                      frame: pd.DataFrame, device: torch.device, views=None, batch_size: int = 16) -> pd.DataFrame:
    """predict_tta over a labels frame -> DataFrame[idx, ID, CD, CV, HEX] in frame order."""
    views = VIEWS["flips"] if views is None else views
    model.eval()
    values = predict_tta(model, stats, cfg, memmap, [int(i) for i in frame["idx"]], device, views, batch_size)
    out = pd.DataFrame({"idx": frame["idx"].to_numpy(dtype=int), "ID": frame["ID"].astype(str).to_numpy()})
    for j, metric in enumerate(METRICS):
        out[metric] = values[:, j]
    return out
