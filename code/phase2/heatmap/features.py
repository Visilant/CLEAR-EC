"""S2 step 2: per-image geometry-based CD/CV/HEX estimators from the saved detections.

Reads results/<dir>/detections.npz (from infer_all.py) and writes results/<dir>/features.csv with,
per frame: three annotator-box emulations (538x408 window chosen by summed peak score, by peak
count, or by band-pass focus energy; largest Delaunay cluster inside; Voronoi readout), whole-frame
Voronoi cell-area statistics (median / top-confidence-half / trimmed-mean area -> CD), the largest
whole-frame cluster readout, the frame-count CD, and focus / saturation / confidence summaries.
"""
import argparse
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, Voronoi, cKDTree

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", "detector"))

from data import CACHE_INDEX, FRAME_SHAPE, UM_PER_PX  # noqa: E402
from readout import poly_area, voronoi_metrics  # noqa: E402

WIN_W, WIN_H = 538, 408
STRIDE = 64
BLOCK = 27
FRAME_AREA_MM2 = FRAME_SHAPE[0] * FRAME_SHAPE[1] * UM_PER_PX ** 2 / 1e6
WIN_AREA_MM2 = WIN_W * WIN_H * UM_PER_PX ** 2 / 1e6
_DET = {}


def largest_cluster(pts, edge_factor=1.6):
    """Largest connected component of the Delaunay graph with edges < edge_factor * median NN distance
    (same rule as phase2/detector/eval_val.py, vectorised)."""
    pts = np.asarray(pts, float)
    if len(pts) < 4:
        return pts
    nn_d, _ = cKDTree(pts).query(pts, k=2)
    thresh = edge_factor * np.median(nn_d[:, 1])
    tri = Delaunay(pts)
    s = tri.simplices
    a = np.concatenate([s[:, 0], s[:, 1], s[:, 2]])
    b = np.concatenate([s[:, 1], s[:, 2], s[:, 0]])
    keep = np.linalg.norm(pts[a] - pts[b], axis=1) < thresh
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(pts)
    g = coo_matrix((np.ones(keep.sum()), (a[keep], b[keep])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    big = np.bincount(lab).argmax()
    return pts[lab == big]


def voronoi_cells(pts):
    """Interior Voronoi cells (finite, inside the convex hull): point indices, areas (px^2), side counts."""
    pts = np.asarray(pts, float)
    if len(pts) < 8:
        return np.zeros(0, int), np.zeros(0), np.zeros(0, int)
    vor = Voronoi(pts)
    hull = Delaunay(pts[ConvexHull(pts).vertices])
    idx, areas, nsides = [], [], []
    for pi, ri in enumerate(vor.point_region):
        reg = vor.regions[ri]
        if -1 in reg or len(reg) == 0:
            continue
        poly = vor.vertices[reg]
        if hull.find_simplex(poly).min() < 0:
            continue
        idx.append(pi)
        areas.append(poly_area(poly))
        nsides.append(len(reg))
    return np.array(idx, int), np.array(areas), np.array(nsides, int)


def window_scores(values_at_pts, pts):
    """Sum of values_at_pts over every 538x408 window on a STRIDE grid via an integral image.
    Returns (scores[ny, nx], x0s, y0s)."""
    H, W = FRAME_SHAPE
    img = np.zeros((H + 1, W + 1))
    np.add.at(img, (pts[:, 1].astype(int) + 1, pts[:, 0].astype(int) + 1), values_at_pts)
    ii = img.cumsum(0).cumsum(1)
    y0s = np.arange(0, H - WIN_H + 1, STRIDE)
    x0s = np.arange(0, W - WIN_W + 1, STRIDE)
    Y0, X0 = np.meshgrid(y0s, x0s, indexing="ij")
    Y1, X1 = Y0 + WIN_H, X0 + WIN_W
    return ii[Y1, X1] - ii[Y0, X1] - ii[Y1, X0] + ii[Y0, X0], x0s, y0s


def block_window_scores(block_map):
    """Mean of a 36x48 block map over 20x15-block (540x405 px) windows on a 2-block (54 px) grid."""
    bw, bh = WIN_W // BLOCK, WIN_H // BLOCK  # 19, 15 -> use 20 x 15 blocks
    bw = 20
    ii = np.zeros((block_map.shape[0] + 1, block_map.shape[1] + 1))
    ii[1:, 1:] = block_map.astype(float).cumsum(0).cumsum(1)
    ys = np.arange(0, block_map.shape[0] - bh + 1, 2)
    xs = np.arange(0, block_map.shape[1] - bw + 1, 2)
    Y0, X0 = np.meshgrid(ys, xs, indexing="ij")
    Y1, X1 = Y0 + bh, X0 + bw
    return (ii[Y1, X1] - ii[Y0, X1] - ii[Y1, X0] + ii[Y0, X0]) / (bw * bh), xs * BLOCK, ys * BLOCK


def readout_in_window(pts, scores, x0, y0):
    inside = (pts[:, 0] >= x0) & (pts[:, 0] < x0 + WIN_W) & (pts[:, 1] >= y0) & (pts[:, 1] < y0 + WIN_H)
    clu = largest_cluster(pts[inside])
    m = voronoi_metrics(clu, UM_PER_PX)
    return dict(CD=m["CD"], CV=m["CV"], HEX=m["HEX"], n=len(clu), n_in=int(inside.sum()),
                score=float(scores[inside].mean()) if inside.any() else np.nan, x0=int(x0), y0=int(y0))


def block_mean_in_window(block_map, x0, y0):
    bx0, by0 = x0 // BLOCK, y0 // BLOCK
    return float(block_map[by0:by0 + WIN_H // BLOCK + 1, bx0:bx0 + WIN_W // BLOCK + 1].astype(float).mean())


def features_for(i):
    z = _DET
    o = z["offsets"]
    pts = np.stack([z["x"][o[i]:o[i + 1]], z["y"][o[i]:o[i + 1]]], 1).astype(float)
    sc = z["score"][o[i]:o[i + 1]].astype(float)
    focus_blk, sat_blk, hm_blk = z["focus_block"][i], z["sat_block"][i], z["hm_block"][i]
    row = dict(idx=int(i), n_total=len(pts), mean_score=float(sc.mean()) if len(sc) else np.nan,
               CD_frame=len(pts) / FRAME_AREA_MM2, sat_frac=float(sat_blk.astype(float).mean()),
               focus_mean=float(focus_blk.astype(float).mean()), hm_mean=float(hm_blk.astype(float).mean()))
    if "win_sum" in z:  # direct differentiable count over the focus window (S3 detector)
        row["CD_direct"] = float(z["win_sum"][i]) / (2 * np.pi * float(z["sigma"]) ** 2) / WIN_AREA_MM2
    fs, fx, fy = block_window_scores(focus_blk)
    k = np.unravel_index(fs.argmax(), fs.shape)
    row["focus_max"] = float(fs[k])
    if len(pts) < 8:
        return row
    # (a) annotator-box emulation: three window choices
    for name, vals in (("conf", sc), ("cnt", np.ones(len(pts)))):
        ws, x0s, y0s = window_scores(vals, pts)
        j = np.unravel_index(ws.argmax(), ws.shape)
        r = readout_in_window(pts, sc, x0s[j[1]], y0s[j[0]])
        row.update({f"{kk}_win{name}": v for kk, v in r.items()})
        row[f"focus_win{name}"] = block_mean_in_window(focus_blk, x0s[j[1]], y0s[j[0]])
    r = readout_in_window(pts, sc, fx[k[1]], fy[k[0]])
    row.update({f"{kk}_winfocus": v for kk, v in r.items()})
    row["focus_winfocus"] = row["focus_max"]
    # (b) whole-frame Voronoi cell areas
    nn_d, _ = cKDTree(pts).query(pts, k=2)
    row["med_nn_px"] = float(np.median(nn_d[:, 1]))
    idx, areas, nsides = voronoi_cells(pts)
    if len(areas) >= 8:
        a_um2 = areas * UM_PER_PX ** 2
        row["n_interior"] = len(a_um2)
        row["CD_med"] = 1e6 / np.median(a_um2)
        row["CD_all"] = 1e6 / a_um2.mean()
        top = sc[idx] >= np.median(sc[idx])
        row["CD_top"] = 1e6 / a_um2[top].mean()
        lo, hi = np.percentile(a_um2, [10, 90])
        trim = (a_um2 >= lo) & (a_um2 <= hi)
        row["CD_trim"] = 1e6 / a_um2[trim].mean()
        row["CV_all"] = float(a_um2.std() / a_um2.mean())
        row["HEX_all"] = float(np.mean(nsides == 6))
        row["hull_frac"] = float(ConvexHull(pts).volume / (FRAME_SHAPE[0] * FRAME_SHAPE[1]))
    clu = largest_cluster(pts)
    m = voronoi_metrics(clu, UM_PER_PX)
    row.update(CD_clu=m["CD"], CV_clu=m["CV"], HEX_clu=m["HEX"], n_clu=len(clu))
    return row


def _init(path):
    global _DET
    z = np.load(path)
    _DET = {k: z[k] for k in z}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(HERE, "../../../results/detector_20260913/s2"))
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    det_path = os.path.join(args.dir, "detections.npz")
    index = pd.read_csv(CACHE_INDEX)
    n = len(index) if args.limit is None else args.limit
    with Pool(args.workers, initializer=_init, initargs=(det_path,)) as pool:
        rows = pool.map(features_for, range(n), chunksize=20)
    df = pd.DataFrame(rows)
    df.insert(1, "ID", index.ID.values[:n])
    df.insert(2, "slide_id", index.slide_id.values[:n])
    out = os.path.join(args.dir, "features.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}: {df.shape}; NaN CD_winconf {df.CD_winconf.isna().sum()}, NaN CD_med {df.CD_med.isna().sum()}")


if __name__ == "__main__":
    main()
