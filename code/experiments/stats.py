"""Paired, slide-clustered bootstrap for comparing two prediction sets on the same images."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.training.common import METRICS, mape_per_metric


def paired_cluster_ci(candidate, baseline, targets, groups, n_boot=2000):
    """Candidate minus baseline, preserving image weighting and per-target masks."""
    valid = np.abs(targets) > 1e-8
    denominator = np.where(valid, np.abs(targets), 1.0)
    delta = np.where(valid, (np.abs(candidate-targets) - np.abs(baseline-targets))
                     / denominator * 100, 0.0)
    _, inverse = np.unique(groups, return_inverse=True)
    count = inverse.max() + 1
    sums = np.zeros((count, 3))
    sizes = np.zeros((count, 3))
    np.add.at(sums, inverse, delta)
    np.add.at(sizes, inverse, valid)
    rng = np.random.default_rng(20260911)
    samples = []
    for _ in range(n_boot):
        selected = rng.integers(0, count, size=count)
        denominators = sizes[selected].sum(axis=0)
        if np.all(denominators > 0):
            samples.append(np.mean(sums[selected].sum(axis=0) / denominators))
    if not samples:
        raise ValueError('No bootstrap resample contains all three scored metrics')
    return np.quantile(samples, [0.025, 0.975]).tolist()


def slide_column(frame: pd.DataFrame) -> str:
    """labels_for_indices merges index.csv with the labels CSV, and both carry slide_id."""
    for name in ("slide_id", "slide_id_x", "slide_id_y"):
        if name in frame.columns:
            return name
    raise ValueError("frame has no slide_id column")


def compare_frames(candidate: pd.DataFrame, reference: pd.DataFrame, labels: pd.DataFrame,
                   n_boot: int = 2000) -> dict:
    """Score two prediction frames (ID, CD, CV, HEX) on their common IDs and bootstrap the paired delta.

    `labels` must carry ID, a slide column and the metrics (as returned by labels_for_indices)."""
    for name, frame in (("candidate", candidate), ("reference", reference), ("labels", labels)):
        missing = {"ID", *METRICS} - set(frame.columns)
        if missing:
            raise ValueError(f"{name} missing columns {sorted(missing)}")
    cand = candidate.assign(ID=candidate["ID"].astype(str).str.strip()).set_index("ID")
    ref = reference.assign(ID=reference["ID"].astype(str).str.strip()).set_index("ID")
    lab = labels.assign(ID=labels["ID"].astype(str).str.strip()).set_index("ID")
    common = cand.index.intersection(ref.index).intersection(lab.index)
    if len(common) == 0:
        raise ValueError("candidate and reference share no IDs")
    targets = lab.loc[common, list(METRICS)].to_numpy(float)
    groups = lab.loc[common, slide_column(lab)].astype(str).to_numpy()
    a = cand.loc[common, list(METRICS)].to_numpy(float)
    b = ref.loc[common, list(METRICS)].to_numpy(float)
    low, high = paired_cluster_ci(a, b, targets, groups, n_boot=n_boot)
    score_a, score_b = mape_per_metric(a, targets), mape_per_metric(b, targets)
    return {"n": int(len(common)), "n_slides": int(len(np.unique(groups))), "n_boot": int(n_boot),
            "candidate_mape": score_a, "reference_mape": score_b,
            "delta_mean": float(score_a["mean"] - score_b["mean"]),
            "delta_ci95_slide_boot": [float(low), float(high)]}
