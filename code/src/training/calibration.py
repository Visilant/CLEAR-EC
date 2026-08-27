"""Linear/ridge calibration of baseline Cellpose metric predictions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.cache import open_image_cache
from src.data.splits import load_split
from src.training.common import (
    METRICS,
    labels_for_indices,
    score_by_id,
    write_manifest,
)

ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)


@dataclass
class CalibrationConfig:
    alpha: float = 1.0
    fit_intercept: bool = True
    per_metric: bool = True


def _fit_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Return coefficients including intercept as first element."""
    n, p = X.shape
    del n
    reg = np.eye(p)
    if p > 0:
        reg[0, 0] = 0.0  # do not penalize intercept
    xtx = X.T @ X + alpha * reg
    xty = X.T @ y
    return np.linalg.solve(xtx, xty)


def _design_matrix(preds: np.ndarray, fit_intercept: bool) -> np.ndarray:
    if fit_intercept:
        return np.column_stack([np.ones(len(preds)), preds])
    return preds


def _require_pred_csv(path: Path, split: str) -> pd.DataFrame:
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            f"Missing {split} prediction CSV for calibration: {path}. "
            "Pass the frozen Cellpose artifact explicitly."
        )
    pred_df = pd.read_csv(path)
    pred_df["ID"] = pred_df["ID"].astype(str).str.strip()
    missing = {"ID", *METRICS} - set(pred_df.columns)
    if missing:
        raise ValueError(f"{path} missing columns {missing}")
    return pred_df


def fit_calibration(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    indices: list[int],
    index_df: pd.DataFrame,
    config: CalibrationConfig,
) -> dict:
    """Fit ridge calibration on the provided indices only (train)."""
    sub = index_df[index_df["idx"].astype(int).isin(indices)][["idx", "ID"]].copy()
    sub["ID"] = sub["ID"].astype(str).str.strip()
    gt = gt_df.copy()
    gt["ID"] = gt["ID"].astype(str).str.strip()
    pred = pred_df.copy()
    pred["ID"] = pred["ID"].astype(str).str.strip()

    merged = sub.merge(pred, on="ID", how="inner").merge(
        gt[["ID", *METRICS]].rename(columns={m: f"{m}_gt" for m in METRICS}),
        on="ID",
        how="inner",
    )
    if merged.empty:
        raise ValueError("No overlapping rows for calibration fit.")

    pred_cols = [f"{m}_pred" for m in METRICS]
    merged = merged.rename(columns={m: f"{m}_pred" for m in METRICS if m in merged.columns})
    preds = merged[pred_cols].astype(float).values
    gt_arr = merged[[f"{m}_gt" for m in METRICS]].astype(float).values

    coefs: dict[str, list[float]] = {}
    if config.per_metric:
        for j, metric in enumerate(METRICS):
            X = _design_matrix(preds[:, [j]], config.fit_intercept)
            y = gt_arr[:, j]
            beta = _fit_ridge(X, y, alpha=config.alpha)
            coefs[metric] = beta.tolist()
    else:
        X = _design_matrix(preds, config.fit_intercept)
        beta = _fit_ridge(X, gt_arr, alpha=config.alpha)
        coefs["joint"] = beta.tolist()

    return {
        "coefs": coefs,
        "n_fit": len(merged),
        "fit_ids": merged["ID"].astype(str).tolist(),
        "alpha": config.alpha,
    }


def apply_calibration(
    pred_df: pd.DataFrame,
    coefs: dict[str, list[float]],
    *,
    per_metric: bool = True,
    fit_intercept: bool = True,
) -> pd.DataFrame:
    out = pred_df.copy()
    preds = out[[*METRICS]].astype(float).values
    calibrated = preds.copy()

    if per_metric:
        for j, metric in enumerate(METRICS):
            beta = np.array(coefs[metric], dtype=float)
            x = _design_matrix(preds[:, [j]], fit_intercept)
            calibrated[:, j] = x @ beta
    else:
        beta = np.array(coefs["joint"], dtype=float)
        x = _design_matrix(preds, fit_intercept)
        calibrated = x @ beta

    for j, metric in enumerate(METRICS):
        out[metric] = calibrated[:, j]
    return out


def _score_split(
    calibrated: pd.DataFrame,
    gt_df: pd.DataFrame,
    indices: list[int],
    index_df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, float]]:
    role = index_df[index_df["idx"].astype(int).isin(indices)][["idx", "ID"]].copy()
    role["ID"] = role["ID"].astype(str).str.strip()
    pred = calibrated.merge(role[["ID"]], on="ID", how="inner")
    gt = gt_df.copy()
    gt["ID"] = gt["ID"].astype(str).str.strip()
    gt_role = role.merge(gt, on="ID", how="inner")
    if pred.empty or gt_role.empty:
        raise ValueError("Calibration scoring produced an empty split join.")
    scores = score_by_id(pred, gt_role)
    return pred, scores


def train_calibration(
    cache_dir: Path,
    labels_csv: Path,
    results_dir: Path,
    *,
    pred_csv_train: Path,
    pred_csv_val: Path,
    config: CalibrationConfig | None = None,
    alpha_grid: tuple[float, ...] = ALPHA_GRID,
) -> dict:
    """Fit on train, select alpha on val. Does not read test predictions."""
    cfg = config or CalibrationConfig()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    _, index_df = open_image_cache(cache_dir)
    train_idx = load_split(cache_dir, "train")
    val_idx = load_split(cache_dir, "val")
    gt_df = labels_for_indices(cache_dir, labels_csv, train_idx + val_idx)

    train_pred = _require_pred_csv(Path(pred_csv_train), "train")
    val_pred = _require_pred_csv(Path(pred_csv_val), "val")

    best: dict | None = None
    best_val = float("inf")
    for alpha in alpha_grid:
        trial = CalibrationConfig(
            alpha=alpha,
            fit_intercept=cfg.fit_intercept,
            per_metric=cfg.per_metric,
        )
        fit_info = fit_calibration(train_pred, gt_df, train_idx, index_df, trial)
        val_cal = apply_calibration(
            val_pred,
            fit_info["coefs"],
            per_metric=trial.per_metric,
            fit_intercept=trial.fit_intercept,
        )
        _, val_scores = _score_split(val_cal, gt_df, val_idx, index_df)
        if val_scores["mean"] < best_val:
            best_val = val_scores["mean"]
            best = {
                "fit_info": fit_info,
                "trial": trial,
                "val_scores": val_scores,
                "val_pred": val_cal,
            }

    if best is None:
        raise RuntimeError("Calibration alpha grid produced no fits.")

    trial: CalibrationConfig = best["trial"]
    fit_info = best["fit_info"]
    val_pred_cal: pd.DataFrame = best["val_pred"]
    val_pred_cal.to_csv(results_dir / "predictions_val.csv", index=False)

    train_cal = apply_calibration(
        train_pred,
        fit_info["coefs"],
        per_metric=trial.per_metric,
        fit_intercept=trial.fit_intercept,
    )
    _, train_scores = _score_split(train_cal, gt_df, train_idx, index_df)

    payload = {
        "config": asdict(trial),
        "pred_csv_train": str(pred_csv_train),
        "pred_csv_val": str(pred_csv_val),
        "alpha_grid": list(alpha_grid),
        **{k: v for k, v in fit_info.items() if k != "fit_ids"},
        "fit_ids": fit_info["fit_ids"],
        "n_fit_ids": len(fit_info["fit_ids"]),
    }
    with open(results_dir / "calibration.json", "w") as f:
        json.dump(payload, f, indent=2)

    metrics = {
        "train_mape": train_scores,
        "val_mape": best["val_scores"],
        "n_fit": fit_info["n_fit"],
        "n_val": len(val_idx),
        "selected_alpha": trial.alpha,
        "pred_csv_train": str(pred_csv_train),
        "pred_csv_val": str(pred_csv_val),
    }
    with open(results_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    write_manifest(
        results_dir.parent,
        method="calibration",
        metrics=metrics,
        extra={"calibration_json": str(results_dir / "calibration.json")},
    )
    print(f"[calibration] val MAPE: {best['val_scores']} alpha={trial.alpha}")
    return metrics


def evaluate_calibration_split(
    cache_dir: Path,
    labels_csv: Path,
    results_dir: Path,
    *,
    pred_csv: Path,
    split: str,
) -> dict:
    """Apply frozen calibration to a split. Call on test only after freeze."""
    results_dir = Path(results_dir)
    payload = json.loads((results_dir / "calibration.json").read_text())
    pred_df = _require_pred_csv(Path(pred_csv), split)
    calibrated = apply_calibration(
        pred_df,
        payload["coefs"],
        per_metric=payload["config"]["per_metric"],
        fit_intercept=payload["config"]["fit_intercept"],
    )
    out_csv = results_dir / f"predictions_{split}.csv"
    calibrated.to_csv(out_csv, index=False)

    indices = load_split(cache_dir, split)
    gt_df = labels_for_indices(cache_dir, labels_csv, indices)
    scores = score_by_id(calibrated, gt_df)
    metrics_path = results_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    metrics[f"{split}_mape"] = scores
    metrics[f"n_{split}"] = len(indices)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"[calibration] {split} MAPE: {scores}")
    return scores
