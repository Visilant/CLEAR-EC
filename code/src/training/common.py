"""Shared helpers for CLEAR-EC training and scoring."""

from __future__ import annotations

import json
import fcntl
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cache import open_image_cache
from src.data.config import MetricConfig, SegConfig, config_hash
from src.data.splits import ROLES, assert_protocol_splits, load_split, load_splits
from src.artifacts import atomic_path

METRICS = ("CD", "CV", "HEX")


def resolve_path(code_root: Path, path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (code_root / path).resolve()


def load_labels(labels_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(labels_csv)
    df["ID"] = df["ID"].astype(str).str.strip()
    missing = {"ID", *METRICS} - set(df.columns)
    if missing:
        raise ValueError(f"Labels CSV missing columns: {missing}")
    return df


def labels_for_indices(
    cache_dir: Path,
    labels_csv: Path,
    indices: list[int],
) -> pd.DataFrame:
    """Return rows with idx, ID, CD, CV, HEX for cache indices."""
    _, index_df = open_image_cache(cache_dir)
    labels_df = load_labels(labels_csv)
    merged = index_df.merge(labels_df, on="ID", how="inner", validate="one_to_one")
    merged = merged[merged["idx"].astype(int).isin(indices)].copy()
    merged["idx"] = merged["idx"].astype(int)
    if set(merged.idx) != set(indices):
        raise ValueError("Labels do not cover every requested cache index")
    if not np.isfinite(merged[list(METRICS)].to_numpy(dtype=float)).all():
        raise ValueError("Labels contain non-finite targets")
    return merged.reset_index(drop=True)


def target_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in METRICS:
        vals = df[col].astype(float).values
        stats[col] = {"mean": float(np.mean(vals)), "std": float(np.std(vals) + 1e-8)}
    return stats


def normalize_targets(
    df: pd.DataFrame,
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    out = np.zeros((len(df), len(METRICS)), dtype=np.float32)
    for j, col in enumerate(METRICS):
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        out[:, j] = (df[col].astype(float).values - mean) / std
    return out


def denormalize_targets(
    arr: np.ndarray,
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    out = arr.copy()
    for j, col in enumerate(METRICS):
        mean = stats[col]["mean"]
        std = stats[col]["std"]
        out[:, j] = out[:, j] * std + mean
    return out


def mape_per_metric(pred: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    """Mean absolute percent error per metric (skip zero GT)."""
    result: dict[str, float] = {}
    for j, col in enumerate(METRICS):
        gt_col = gt[:, j]
        pred_col = pred[:, j]
        mask = np.abs(gt_col) > 1e-8
        if not mask.any():
            result[col] = float("nan")
            continue
        ape = np.abs(pred_col[mask] - gt_col[mask]) / np.abs(gt_col[mask]) * 100.0
        result[col] = float(np.mean(ape))
    result["mean"] = float(np.nanmean([result[col] for col in METRICS]))
    return result


def _normalize_id_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ID"] = out["ID"].astype(str).str.strip()
    return out


def score_by_id(pred_df: pd.DataFrame, gt_df: pd.DataFrame, *, expected_ids=None) -> dict[str, float]:
    """Join predictions to labels by ID, then compute full-precision MAPE."""
    missing_pred = {"ID", *METRICS} - set(pred_df.columns)
    missing_gt = {"ID", *METRICS} - set(gt_df.columns)
    if missing_pred:
        raise ValueError(f"Predictions missing columns: {missing_pred}")
    if missing_gt:
        raise ValueError(f"Labels missing columns: {missing_gt}")
    pred = _normalize_id_frame(pred_df)
    gt = _normalize_id_frame(gt_df)
    for name, frame in (("Predictions", pred), ("Labels", gt)):
        if frame.ID.duplicated().any():
            raise ValueError(f"{name} contain duplicate IDs")
        if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
            raise ValueError(f"{name} contain non-finite metric values")
    if expected_ids is not None:
        expected = {str(i).strip() for i in expected_ids}
        if set(pred.ID) != expected or not expected.issubset(set(gt.ID)):
            raise ValueError("Prediction/label coverage does not match expected IDs")

    merged = pred[["ID", *METRICS]].merge(
        gt[["ID", *METRICS]],
        on="ID",
        how="inner",
        suffixes=("_pred", "_gt"),
    )
    if merged.empty:
        raise ValueError("No overlapping IDs between predictions and ground truth.")

    pred_arr = merged[[f"{m}_pred" for m in METRICS]].astype(float).to_numpy()
    gt_arr = merged[[f"{m}_gt" for m in METRICS]].astype(float).to_numpy()
    return mape_per_metric(pred_arr, gt_arr)


def paired_bootstrap_mape(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    *,
    n_boot: int = 1000,
    seed: int = 42,
    ci: float = 0.95,
) -> dict[str, float]:
    """Paired bootstrap of mean MAPE over images joined by ID."""
    pred = _normalize_id_frame(pred_df)
    gt = _normalize_id_frame(gt_df)
    merged = pred[["ID", *METRICS]].merge(
        gt[["ID", *METRICS]],
        on="ID",
        how="inner",
        suffixes=("_pred", "_gt"),
    )
    if merged.empty:
        raise ValueError("No overlapping IDs for bootstrap.")

    point = score_by_id(pred, gt)
    rng = np.random.default_rng(seed)
    n = len(merged)
    samples = np.empty(n_boot, dtype=np.float64)
    pred_cols = [f"{m}_pred" for m in METRICS]
    gt_cols = [f"{m}_gt" for m in METRICS]
    pred_arr = merged[pred_cols].astype(float).to_numpy()
    gt_arr = merged[gt_cols].astype(float).to_numpy()
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = mape_per_metric(pred_arr[idx], gt_arr[idx])["mean"]
    alpha = (1.0 - ci) / 2.0
    return {
        **point,
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
        "n": int(n),
        "n_boot": int(n_boot),
    }


def experiment_hash(seg_config: SegConfig, metric_config: MetricConfig) -> str:
    return config_hash((config_hash(seg_config), metric_config))


def pred_artifact_path(cache_dir: Path, exp_hash: str, split: str, *, limit: int = 0) -> Path:
    if split not in (*ROLES, "all"):
        raise ValueError(f"Unknown split for prediction artifact: {split}")
    suffix = f".limit_{limit}" if limit > 0 else ""
    return Path(cache_dir) / "preds" / exp_hash / f"{split}{suffix}.csv"


def write_manifest(
    results_dir: Path,
    *,
    method: str,
    metrics: dict,
    extra: dict | None = None,
) -> None:
    entry = {"metrics": metrics, "updated_at": datetime.now(timezone.utc).isoformat()}
    if extra:
        entry.update(extra)
    update_manifest(results_dir, {}, method_entry=(method, entry))


def update_manifest(results_dir: Path, fields: dict, *, method_entry=None) -> None:
    """Serialize read/modify/write so independent GPU jobs cannot lose entries."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = results_dir / "manifest.json"
    with open(results_dir / ".manifest.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        manifest = (json.loads(manifest_path.read_text()) if manifest_path.exists() else
                    {"created_at": datetime.now(timezone.utc).isoformat(), "methods": {}})
        manifest.update(fields)
        if method_entry:
            method, entry = method_entry
            manifest.setdefault("methods", {})[method] = entry
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        with atomic_path(manifest_path) as temporary:
            temporary.write_text(json.dumps(manifest, indent=2))


def split_summary(cache_dir: Path) -> dict[str, int]:
    splits = load_splits(cache_dir)
    index_df = pd.read_csv(Path(cache_dir) / "index.csv")
    report = assert_protocol_splits(index_df, splits)
    return {
        "train": len(splits["train"]),
        "val": len(splits["val"]),
        "test": len(splits["test"]),
        "n_slides_train": int(report["n_slides"].get("train", 0)),
        "n_slides_val": int(report["n_slides"].get("val", 0)),
        "n_slides_test": int(report["n_slides"].get("test", 0)),
    }


def load_role_frame(
    cache_dir: Path,
    labels_csv: Path,
    split: str,
) -> pd.DataFrame:
    if split not in ROLES:
        raise ValueError(f"Unknown role {split}; expected one of {ROLES}")
    return labels_for_indices(cache_dir, labels_csv, load_split(cache_dir, split))
