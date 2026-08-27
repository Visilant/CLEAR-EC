"""
CLEAR-EC evaluation: compare a predictions CSV (from main.py) against a
ground-truth CSV (columns: ID, CD, CV, HEX) that you build from the released
training data — e.g. /path/to/your_holdout_ground_truth.csv.

Reports per-case CD/CV/HEX side-by-side, per-slide percent error, a
best->worst ranking, and the average percent error across all slides.
"""

import argparse
from pathlib import Path

import pandas as pd

from src.utils.evaluate import evaluate_results


METRIC_COLS = ["CD", "CV", "HEX"]


def load_predictions(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = {"ID", *METRIC_COLS} - set(df.columns)
    if missing:
        raise ValueError(f"Predictions CSV {csv_path} is missing columns: {missing}")
    df["ID"] = df["ID"].astype(str).str.replace(r"\.(bmp|png|tif|tiff)$", "", regex=True)
    return df[["ID", *METRIC_COLS]].copy()


def load_gt_from_csv(csv_path: Path) -> pd.DataFrame:
    """
    Ground-truth CSV with columns ID, CD, CV, HEX (built from the released
    training data). The image filename stem matches the `ID` column directly,
    so we keep `ID` as-is.
    """
    df = pd.read_csv(csv_path)
    missing = {"ID", *METRIC_COLS} - set(df.columns)
    if missing:
        raise ValueError(f"Ground-truth CSV {csv_path} is missing columns: {missing}")
    out = df[["ID", *METRIC_COLS]].copy()
    out["ID"] = out["ID"].astype(str)
    return out


def main(args):
    pred_df = load_predictions(Path(args.predictions_csv))
    gt_df = load_gt_from_csv(Path(args.gt_csv))

    merged = pd.merge(pred_df, gt_df, on="ID", suffixes=("_pred", "_gt"), how="inner")
    if merged.empty:
        print("WARNING: no overlapping IDs between predictions and ground truth.")
        print(f"  pred IDs (first 5): {pred_df['ID'].head().tolist()}")
        print(f"  gt   IDs (first 5): {gt_df['ID'].head().tolist()}")
        return None

    print(f"\n{'ID':<30} {'Metric':<10} {'Pred':>10} {'GT':>10}")
    print("-" * 65)
    for _, row in merged.iterrows():
        for col in METRIC_COLS:
            print(f"{row['ID']:<30} {col:<10} {row[f'{col}_pred']:>10.2f} {row[f'{col}_gt']:>10.2f}")
        print()

    errors_df, avg_error_df = evaluate_results(pred_df, gt_df)

    error_cols = [c for c in errors_df.columns if c != "ID"]
    if error_cols:
        errors_df["Mean Error (%)"] = errors_df[error_cols].mean(axis=1, skipna=True)
        ranked = errors_df.sort_values("Mean Error (%)").reset_index(drop=True)
        ranked.index += 1
        print("\n=== Per-slide ranking (best -> worst) ===")
        print(ranked[["ID", "Mean Error (%)"] + error_cols].to_string())

    print("\n=== Average percent error across all slides ===")
    print(avg_error_df.to_string(index=False))

    if args.results_dir:
        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        err_csv = results_dir / f"errors_{args.split}.csv"
        avg_csv = results_dir / f"avg_error_{args.split}.csv"
        errors_df.to_csv(err_csv, index=False)
        avg_error_df.to_csv(avg_csv, index=False)
        print(f"\nWrote per-slide errors -> {err_csv}")
        print(f"Wrote average errors  -> {avg_csv}")

    return avg_error_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLEAR-EC: compare prediction CSV against ground truth.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which CLEAR-EC split to evaluate. Sets default --predictions_csv / --gt_csv.")
    parser.add_argument("--predictions_csv", type=str, default='./results_mha/predictions_test.csv',
                        help="Path to predictions CSV (default: ./results_mha/predictions_<split>.csv).")
    parser.add_argument("--gt_csv", type=str, default='/path/to/your_holdout_ground_truth.csv',
                        help="Path to ground-truth CSV with columns ID, CD, CV, HEX (build this from the released training data).")
    parser.add_argument("--results_dir", type=str, default="./results_mha",
                        help="Where to write error CSVs (default: ./results_mha). Pass empty string to skip writing.")
    return parser


def resolve_paths(args) -> argparse.Namespace:
    if args.predictions_csv is None:
        args.predictions_csv = str(Path(args.results_dir) / f"predictions_{args.split}.csv")
    if args.gt_csv is None:
        args.gt_csv = "/path/to/your_holdout_ground_truth.csv"
    return args


if __name__ == "__main__":
    args = resolve_paths(build_parser().parse_args())
    main(args)
