"""Voronoi center-method metrics and greedy nearest-neighbour matching."""
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, Voronoi, cKDTree
from tqdm import tqdm

from overlays import load_overlays

UM = 0.7716049


def poly_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def voronoi_metrics(pts, um_per_px=UM):
    pts = np.asarray(pts, float)
    if len(pts) < 8:
        return dict(CD=np.nan, CV=np.nan, HEX=np.nan, n_interior=0)
    vor = Voronoi(pts)
    hull = Delaunay(pts[ConvexHull(pts).vertices])
    areas, nsides = [], []
    for ri in vor.point_region:
        reg = vor.regions[ri]
        if -1 in reg or len(reg) == 0:
            continue
        poly = vor.vertices[reg]
        if hull.find_simplex(poly).min() < 0:
            continue
        areas.append(poly_area(poly))
        nsides.append(len(reg))
    if not areas:
        return dict(CD=np.nan, CV=np.nan, HEX=np.nan, n_interior=0)
    a = np.array(areas) * um_per_px**2
    return dict(
        CD=len(a) / a.sum() * 1e6,
        CV=a.std() / a.mean(),
        HEX=float(np.mean(np.array(nsides) == 6)),
        n_interior=len(a),
    )


def match(det_xy, gt_xy, tol_px):
    det_xy = np.asarray(det_xy, float)
    gt_xy = np.asarray(gt_xy, float)
    if len(det_xy) == 0 or len(gt_xy) == 0:
        return dict(f1=0.0, precision=0.0, recall=0.0, count_ratio=0.0, med_loc_err=np.nan)

    tree = cKDTree(det_xy)
    used_det = np.zeros(len(det_xy), dtype=bool)
    dists, idxs = tree.query(gt_xy, k=min(5, len(det_xy)))
    dists = np.atleast_2d(dists)
    idxs = np.atleast_2d(idxs)
    order = np.argsort(dists.min(axis=1))
    errs, tp = [], 0
    for gi in order:
        for d, di in zip(dists[gi], idxs[gi]):
            if d > tol_px:
                break
            if not used_det[di]:
                used_det[di] = True
                tp += 1
                errs.append(d)
                break

    fp = len(det_xy) - tp
    fn = len(gt_xy) - tp
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return dict(
        f1=f1,
        precision=precision,
        recall=recall,
        count_ratio=len(det_xy) / len(gt_xy),
        med_loc_err=float(np.median(errs)) if errs else np.nan,
    )


if __name__ == "__main__":
    labels = pd.read_csv("/home/visilant/CLEAR-EC/data/final_train_ids.csv").set_index("ID")
    overlays = load_overlays()

    rows = []
    for image_id, v in tqdm(overlays.items(), desc="voronoi on GT dots"):
        m = voronoi_metrics(v["dots"])
        lab = labels.loc[image_id]
        rows.append(dict(ID=image_id, split=v["split"], n_dots=len(v["dots"]), **m,
                          CD_gt=lab["CD"], CV_gt=lab["CV"], HEX_gt=lab["HEX"]))

    df = pd.DataFrame(rows)
    df["CD_ape"] = (df["CD"] - df["CD_gt"]).abs() / df["CD_gt"] * 100
    df["HEX_ape"] = (df["HEX"] - df["HEX_gt"]).abs() / df["HEX_gt"].clip(lower=1e-6) * 100
    print(df[["ID", "split", "n_dots", "CD", "CD_gt", "CD_ape", "CV", "CV_gt", "HEX", "HEX_gt", "HEX_ape"]]
          .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    cv_corr = np.corrcoef(df["CV"], df["CV_gt"])[0, 1]
    cv_scale = (df["CV"] / df["CV_gt"]).median()
    print("\nsummary (25 GT dot sets vs final_train_ids.csv labels):")
    print(f"  CD MAPE:  {df['CD_ape'].mean():.2f}%")
    print(f"  HEX MAPE: {df['HEX_ape'].mean():.2f}%")
    print(f"  CV correlation: {cv_corr:.3f}  (CV scale ~= {cv_scale:.2f}x)")
