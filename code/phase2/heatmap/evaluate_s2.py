"""S2 step 3-4: score the geometry CD estimators on val 892 (and train), the CNN OOF reference,
a fold-honest ridge stack, and the 25-overlay hull-vs-window diagnostic.

Writes results/<dir>/eval_estimators.csv, eval_stack.json, overlay_diagnostic.csv and prints a summary.
Test-split (906) labels are never read: overlay rows in the test split report only the
window / hull ratio of detector readouts.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", "detector"))

from data import LABELS_CSV, ROOT, UM_PER_PX, hull_mask, load_overlays, median_nn, points_in_mask  # noqa: E402
from features import WIN_H, WIN_W, largest_cluster  # noqa: E402
from readout import voronoi_metrics  # noqa: E402

SPLITS_JSON = os.path.join(ROOT, "data/cache/splits.json")
OOF_GLOB = os.path.join(ROOT, "results/night_20260912/oof/v2fold[0-4]_last_flips.csv")


def mape(pred, gt):
    m = np.abs(gt) > 1e-8
    return float(np.mean(np.abs(pred[m] - gt[m]) / np.abs(gt[m]) * 100))


def log_corr(pred, gt):
    m = (pred > 0) & (gt > 0)
    return float(np.corrcoef(np.log(pred[m]), np.log(gt[m]))[0, 1]) if m.sum() > 2 else np.nan


def score_estimator(df, col, train_mask, val_mask, cnn_col="CD_cnn"):
    """MAPE, log-corr, train-fitted global log-scale MAPE; NaN estimates fall back to the CNN."""
    out = {}
    est = df[col].to_numpy(float)
    gt = df["CD_label"].to_numpy(float)
    ok = np.isfinite(est) & (est > 0)
    scale = float(np.exp(np.median(np.log(gt[train_mask & ok]) - np.log(est[train_mask & ok]))))
    for name, mask in (("val", val_mask), ("train", train_mask)):
        v = mask & ok
        filled = np.where(ok, est, df[cnn_col].to_numpy(float))
        filled_scaled = np.where(ok, est * scale, df[cnn_col].to_numpy(float))
        out[f"{name}_n_valid"] = int(v.sum())
        out[f"{name}_mape"] = mape(est[v], gt[v])
        out[f"{name}_logcorr"] = log_corr(est[v], gt[v])
        out[f"{name}_scaled_mape"] = mape(est[v] * scale, gt[v])
        out[f"{name}_scaled_mape_cnnfill"] = mape(filled_scaled[mask], gt[mask])
        out[f"{name}_mape_cnnfill"] = mape(filled[mask], gt[mask])
    out["scale"] = scale
    return out


def ridge_fit(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    A = np.c_[Xs, np.ones(len(Xs))]
    reg = lam * np.eye(A.shape[1]); reg[-1, -1] = 0
    w = np.linalg.solve(A.T @ A + reg, A.T @ y)
    return dict(w=w, mu=mu, sd=sd)


def ridge_predict(fit, X):
    Xs = (X - fit["mu"]) / fit["sd"]
    return np.c_[Xs, np.ones(len(Xs))] @ fit["w"]


def stack(df, est_col, train_mask, val_mask, lam=1.0):
    """Ridge on train of log CD_label on [log CD_cnn, log est (one column or a list), log n_total, mean_score];
    val rows whose features are not finite fall back to the CNN."""
    est_cols = [est_col] if isinstance(est_col, str) else list(est_col)
    F = np.c_[np.log(df["CD_cnn"]), np.log(df[est_cols].clip(lower=1e-6)), np.log(df["n_total"].clip(lower=1)), df["mean_score"]]
    est_col = "+".join(est_cols)
    y = np.log(df["CD_label"].to_numpy(float))
    ok = np.isfinite(F).all(1) & np.isfinite(y)
    fit = ridge_fit(F[train_mask & ok], y[train_mask & ok], lam)
    pred = df["CD_cnn"].to_numpy(float).copy()
    pred[ok] = np.exp(ridge_predict(fit, F[ok]))
    gt = df["CD_label"].to_numpy(float)
    return dict(est=est_col, lam=lam, coef=fit["w"].round(4).tolist(), n_train=int((train_mask & ok).sum()),
                val_n_fallback=int((val_mask & ~ok).sum()),
                val_mape=mape(pred[val_mask], gt[val_mask]), train_mape=mape(pred[train_mask], gt[train_mask]),
                cnn_val_mape=mape(df["CD_cnn"].to_numpy(float)[val_mask], gt[val_mask]))


def tertile_table(df, cols, val_mask):
    """MAPE and median estimate/label ratio by label tertile (edges from the val labels)."""
    gt = df["CD_label"].to_numpy(float)
    edges = np.quantile(gt[val_mask], [1 / 3, 2 / 3])
    tert = np.digitize(gt, edges)
    rows = []
    for c in cols:
        est = df[c].to_numpy(float)
        for t, name in enumerate(("low", "mid", "high")):
            m = val_mask & (tert == t) & np.isfinite(est) & (est > 0)
            rows.append(dict(estimator=c, tertile=name, n=int(m.sum()), cd_range=f"{gt[m].min():.0f}-{gt[m].max():.0f}",
                             mape=mape(est[m], gt[m]), median_ratio=float(np.median(est[m] / gt[m]))))
    return pd.DataFrame(rows)


def overlay_diagnostic(det, feats, labels, index, splits_of):
    """Detector readout inside the true click hull vs inside the emulated (conf) window vs oracle."""
    overlays = load_overlays()
    o = det["offsets"]
    rows = []
    for image_id, v in overlays.items():
        i = int(index.loc[image_id, "idx"])
        pts = np.stack([det["x"][o[i]:o[i + 1]], det["y"][o[i]:o[i + 1]]], 1).astype(float)
        dots = v["dots"]
        region = hull_mask(dots, 0.5 * median_nn(dots))
        in_hull = points_in_mask(pts, region)
        m_hull = voronoi_metrics(in_hull, UM_PER_PX)
        f = feats.loc[image_id]
        x0, y0 = int(f["x0_winconf"]), int(f["y0_winconf"])
        inside = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + WIN_W) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + WIN_H)
        m_win = voronoi_metrics(largest_cluster(pts[inside]), UM_PER_PX)
        bx0, by0, bx1, by1 = v["box"]
        ov = (max(0, min(x0 + WIN_W, bx1) - max(x0, bx0)) * max(0, min(y0 + WIN_H, by1) - max(y0, by0))
              / (WIN_W * WIN_H))
        row = dict(ID=image_id, overlay_split=v["split"], cache_split=splits_of[i], n_clicks=len(dots),
                   n_det_hull=len(in_hull), n_det_win=int(inside.sum()), CD_det_hull=m_hull["CD"],
                   CD_det_win=m_win["CD"], win_box_overlap=ov, win_hull_ratio=m_win["CD"] / m_hull["CD"])
        if splits_of[i] != "test":
            lab = float(labels.loc[image_id, "CD"])
            m_or = voronoi_metrics(dots, UM_PER_PX)
            row.update(CD_label=lab, CD_oracle=m_or["CD"],
                       ape_oracle=abs(m_or["CD"] - lab) / lab * 100, ape_hull=abs(m_hull["CD"] - lab) / lab * 100,
                       ape_win=abs(m_win["CD"] - lab) / lab * 100)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "../../../results/detector_20260913/s2"))
    args = ap.parse_args()
    feats = pd.read_csv(os.path.join(args.dir, "features.csv"))
    labels = pd.read_csv(LABELS_CSV)
    labels["ID"] = labels["ID"].astype(str).str.strip()  # 33 label IDs carry trailing spaces (as score_by_id strips)
    labels = labels.set_index("ID")
    splits = json.load(open(SPLITS_JSON))
    splits_of = {i: k for k, v in splits.items() for i in v}
    oof = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(OOF_GLOB))]).drop_duplicates("ID")
    assert len(oof) == 9000, len(oof)
    df = feats.merge(oof[["ID", "CD"]].rename(columns={"CD": "CD_cnn"}), on="ID", how="left")
    df["split"] = df["idx"].map(splits_of)
    df = df[df["split"] != "test"].copy()  # never touch test-split labels
    df["CD_label"] = labels.loc[df.ID, "CD"].values
    train_mask = (df["split"] == "train").to_numpy()
    val_mask = (df["split"] == "val").to_numpy()
    assert val_mask.sum() == 892 and train_mask.sum() == 7202

    est_cols = [c for c in df.columns if c.startswith("CD_") and c not in ("CD_label", "CD_cnn")]
    rows = []
    for c in est_cols:
        rows.append(dict(estimator=c, **score_estimator(df, c, train_mask, val_mask)))
    cnn = dict(estimator="CD_cnn (OOF ref)", val_n_valid=int(val_mask.sum()),
               val_mape=mape(df.CD_cnn.to_numpy(float)[val_mask], df.CD_label.to_numpy(float)[val_mask]),
               val_logcorr=log_corr(df.CD_cnn.to_numpy(float)[val_mask], df.CD_label.to_numpy(float)[val_mask]),
               train_n_valid=int(train_mask.sum()),
               train_mape=mape(df.CD_cnn.to_numpy(float)[train_mask], df.CD_label.to_numpy(float)[train_mask]),
               train_logcorr=log_corr(df.CD_cnn.to_numpy(float)[train_mask], df.CD_label.to_numpy(float)[train_mask]))
    res = pd.DataFrame(rows + [cnn])
    res.to_csv(os.path.join(args.dir, "eval_estimators.csv"), index=False)
    cols = ["estimator", "val_n_valid", "val_mape", "val_logcorr", "val_scaled_mape", "val_scaled_mape_cnnfill",
            "train_mape", "train_logcorr", "train_scaled_mape", "scale"]
    print(res[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # best estimator chosen on TRAIN scaled MAPE (val untouched for selection)
    geo = res[res.estimator.isin(est_cols)]
    best = geo.sort_values("train_scaled_mape").iloc[0]["estimator"]
    stacks = [stack(df, best, train_mask, val_mask, lam) for lam in (1.0, 10.0)]
    stacks += [stack(df, c, train_mask, val_mask, 1.0) for c in est_cols if c != best]
    if "CD_direct" in est_cols:
        stacks.append(stack(df, ["CD_winfocus", "CD_direct"], train_mask, val_mask, 1.0))
    # CNN-only ridge control (same recipe without the geometry column)
    F = np.c_[np.log(df["CD_cnn"]), np.log(df["n_total"].clip(lower=1)), df["mean_score"]]
    y = np.log(df["CD_label"].to_numpy(float)); ok = np.isfinite(F).all(1)
    fit = ridge_fit(F[train_mask & ok], y[train_mask & ok], 1.0)
    pred = df["CD_cnn"].to_numpy(float).copy(); pred[ok] = np.exp(ridge_predict(fit, F[ok]))
    control = dict(est="none (cnn + n_total + mean_score)", lam=1.0, coef=fit["w"].round(4).tolist(),
                   val_mape=mape(pred[val_mask], df.CD_label.to_numpy(float)[val_mask]))
    json.dump(dict(best_by_train=best, stacks=stacks, control=control), open(os.path.join(args.dir, "eval_stack.json"), "w"), indent=1)
    print(f"\nbest geometry estimator by TRAIN scaled MAPE: {best}")
    for s in stacks[:2]:
        print(f"stack [{s['est']}] lam={s['lam']}: val MAPE {s['val_mape']:.3f} (train {s['train_mape']:.3f}), "
              f"CNN val {s['cnn_val_mape']:.3f}, coef {s['coef']}, fallback rows {s['val_n_fallback']}")
    print(f"control ridge without geometry: val MAPE {control['val_mape']:.3f}")
    print("other stacks (lam 1):", {s["est"]: round(s["val_mape"], 3) for s in stacks[2:]})
    tcols = [c for c in ("CD_winfocus", "CD_direct") if c in est_cols] + ["CD_cnn"]
    tert = tertile_table(df, tcols, val_mask)
    tert.to_csv(os.path.join(args.dir, "tertiles.csv"), index=False)
    print("\nval MAPE / median ratio by label tertile:")
    print(tert.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    det = np.load(os.path.join(args.dir, "detections.npz"))
    index = pd.read_csv(os.path.join(ROOT, "data/cache/index.csv")).set_index("ID")
    diag = overlay_diagnostic(det, feats.set_index("ID"), labels, index, splits_of)
    diag.to_csv(os.path.join(args.dir, "overlay_diagnostic.csv"), index=False)
    print("\noverlay diagnostic:")
    print(diag.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    lab = diag[diag.cache_split != "test"]
    print(f"\n21 train+val overlays: CD APE oracle {lab.ape_oracle.mean():.2f}, det-in-hull {lab.ape_hull.mean():.2f}, "
          f"det-in-window {lab.ape_win.mean():.2f}; median win/hull ratio {diag.win_hull_ratio.median():.3f} "
          f"(all 25), mean window-box overlap {diag.win_box_overlap.mean():.2f}")


if __name__ == "__main__":
    main()
