"""A1/A2: counting-statistics and process-noise floor from the 25 annotated overlays.

Reuses the Voronoi readout from spike/sam/reference_readout.py (copied verbatim, plus an
`all_finite` option) and the parsed click sets in spike/sam/overlays.json.
Outputs A1_counting_floor.json, A2_oracle_floor.json, A2_cv_formulas.csv, per-image tables.
"""
from __future__ import annotations
import json, sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
from scipy.spatial import Voronoi, ConvexHull, Delaunay, cKDTree

UM = 0.7716049
ROUND_ULP = {"CD": 0.5, "CV": 0.005, "HEX": 0.005}   # half-unit of label quantisation


def poly_area(pts):
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def voronoi_cells(pts, um_per_px=UM, interior_only=True):
    """Per-cell areas (um^2), side counts, and boundary flag for Voronoi cells of pts."""
    pts = np.asarray(pts, float)
    if len(pts) < 8:
        return None
    vor = Voronoi(pts)
    hull = Delaunay(pts[ConvexHull(pts).vertices])
    areas, nsides, keep_idx = [], [], []
    for pi, ri in enumerate(vor.point_region):
        reg = vor.regions[ri]
        if -1 in reg or len(reg) == 0:
            continue
        poly = vor.vertices[reg]
        if interior_only and hull.find_simplex(poly).min() < 0:
            continue
        areas.append(poly_area(poly)); nsides.append(len(reg)); keep_idx.append(pi)
    if not areas:
        return None
    return dict(area=np.array(areas) * um_per_px ** 2, nsides=np.array(nsides), idx=np.array(keep_idx), pts=pts)


def metrics_from_cells(area, nsides):
    return dict(CD=len(area) / area.sum() * 1e6, CV=area.std() / area.mean(), HEX=float(np.mean(nsides == 6)), n=len(area))


def voronoi_metrics(pts):
    c = voronoi_cells(pts)
    if c is None:
        return dict(CD=np.nan, CV=np.nan, HEX=np.nan, n=0)
    return metrics_from_cells(c["area"], c["nsides"])


def cell_bootstrap(cells, n_boot=2000, rng=None):
    """Nonparametric bootstrap over interior cells: relative SD of each metric at the actual n."""
    rng = rng or np.random.default_rng(0)
    a, s = cells["area"], cells["nsides"]
    n = len(a)
    out = {m: [] for m in ("CD", "CV", "HEX")}
    for _ in range(n_boot):
        k = rng.integers(0, n, n)
        m = metrics_from_cells(a[k], s[k])
        for key in out: out[key].append(m[key])
    base = metrics_from_cells(a, s)
    return {m: float(np.std(out[m]) / base[m]) for m in out}, base


def subbox_resample(pts, box, fracs=(0.5, 0.6, 0.7, 0.8, 0.9), grid=(5, 4)):
    """Slide a sub-box of area f*box over the box; relative SD of metrics across positions."""
    x0, y0, x1, y1 = box
    W, H = x1 - x0, y1 - y0
    rows = []
    for f in fracs:
        s = np.sqrt(f); w, h = W * s, H * s
        vals = {m: [] for m in ("CD", "CV", "HEX")}; ns = []
        for gx in np.linspace(0, W - w, grid[0]):
            for gy in np.linspace(0, H - h, grid[1]):
                bx0, by0 = x0 + gx, y0 + gy
                m = (pts[:, 0] >= bx0) & (pts[:, 0] <= bx0 + w) & (pts[:, 1] >= by0) & (pts[:, 1] <= by0 + h)
                r = voronoi_metrics(pts[m])
                if r["n"] < 20: continue
                for k in vals: vals[k].append(r[k])
                ns.append(r["n"])
        if len(ns) < 4: continue
        rows.append(dict(f=f, n_mean=float(np.mean(ns)), n_pos=len(ns),
                         **{f"relsd_{k}": float(np.std(vals[k]) / np.mean(vals[k])) for k in vals}))
    return rows


def cv_candidates(cells_int, cells_all, pts):
    """Alternative CV definitions on the click set."""
    a = cells_int["area"]; s = cells_int["nsides"]
    out = {}
    out["area_interior"] = a.std() / a.mean()
    out["area_interior_ddof1"] = a.std(ddof=1) / a.mean()
    aa = cells_all["area"]
    out["area_all_finite"] = aa.std() / aa.mean()
    out["sqrt_area_interior"] = np.sqrt(a).std() / np.sqrt(a).mean()
    lo, hi = np.percentile(a, [5, 95]); t = a[(a >= lo) & (a <= hi)]
    out["area_trimmed5"] = t.std() / t.mean()
    # nearest-neighbour distance CV over interior points
    tree = cKDTree(pts); d, _ = tree.query(pts[cells_int["idx"]], k=2); nn = d[:, 1]
    out["nn_dist"] = nn.std() / nn.mean()
    # second ring removed: interior cells whose Delaunay neighbours are all interior
    tri = Delaunay(pts); indptr, nbrs = tri.vertex_neighbor_vertices
    interior = set(cells_int["idx"].tolist())
    core = [i for i, pi in enumerate(cells_int["idx"]) if all(int(q) in interior for q in nbrs[indptr[pi]:indptr[pi + 1]])]
    if len(core) >= 8:
        ac = a[core]; out["area_core2"] = ac.std() / ac.mean()
    else:
        out["area_core2"] = np.nan
    # side-length CV: perimeter/nsides proxy
    out["mean_side_proxy"] = (np.sqrt(a) / s).std() / (np.sqrt(a) / s).mean()
    # range-based dispersion (max-min)/mean
    out["area_range_over_mean"] = (a.max() - a.min()) / a.mean()
    out["area_iqr_over_median"] = (np.percentile(a, 75) - np.percentile(a, 25)) / np.median(a)
    out["area_mad_over_median"] = np.median(np.abs(a - np.median(a))) / np.median(a)
    out["area_std_over_min"] = a.std() / a.min()
    out["log_area_std"] = np.log(a).std()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlays", default="/home/visilant/CLEAR-EC/.worktrees/spike-sam/spike/sam/overlays.json")
    ap.add_argument("--labels", default="/home/visilant/CLEAR-EC/data/final_train_ids.csv")
    ap.add_argument("--out", default="/home/visilant/CLEAR-EC/results/noise_floor_20260912")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ov = json.load(open(args.overlays))
    lab = pd.read_csv(args.labels); lab["ID"] = lab["ID"].str.strip(); lab = lab.set_index("ID")
    rng = np.random.default_rng(0)

    per_img, sub_rows, cv_rows = [], [], []
    for iid, v in ov.items():
        pts = np.asarray(v["dots"], float); box = v["box"]
        ci = voronoi_cells(pts, interior_only=True); ca = voronoi_cells(pts, interior_only=False)
        relsd, base = cell_bootstrap(ci, rng=rng)
        g = lab.loc[iid]
        row = dict(ID=iid, split=v["split"], n_dots=len(pts), n_interior=int(base["n"]),
                   CD=base["CD"], CV=base["CV"], HEX=base["HEX"], CD_gt=g.CD, CV_gt=g.CV, HEX_gt=g.HEX,
                   **{f"boot_relsd_{m}": relsd[m] for m in relsd})
        row["CD_ape"] = abs(base["CD"] - g.CD) / g.CD * 100
        row["HEX_ape"] = abs(base["HEX"] - g.HEX) / max(g.HEX, 1e-6) * 100
        row["CV_ratio"] = g.CV / base["CV"]
        p = base["HEX"]; n = base["n"]
        row["binom_relsd_HEX"] = np.sqrt(p * (1 - p) / n) / p
        cv = base["CV"]; row["closed_relsd_CV"] = np.sqrt((1 + 2 * cv ** 2) / (2 * n))
        per_img.append(row)
        for r in subbox_resample(pts, box): sub_rows.append(dict(ID=iid, **r))
        cv_rows.append(dict(ID=iid, CV_gt=g.CV, **cv_candidates(ci, ca, pts)))

    df = pd.DataFrame(per_img); df.to_csv(out / "A1_per_image.csv", index=False)
    sub = pd.DataFrame(sub_rows); sub.to_csv(out / "A1_subbox.csv", index=False)
    cvdf = pd.DataFrame(cv_rows); cvdf.to_csv(out / "A2_cv_per_image.csv", index=False)

    # ---- A1: counting floor. MAPE of a perfect predictor vs a label with relative SD s: E|N(0,s)| = s*sqrt(2/pi)
    def mape_from_relsd(s): return s * np.sqrt(2 / np.pi) * 100
    a1 = {}
    for m in ("CD", "CV", "HEX"):
        s_boot = df[f"boot_relsd_{m}"].values
        # rounding adds an independent uniform term: relSD = ulp/sqrt(3)/label
        r_ulp = ROUND_ULP[m] / np.sqrt(3) / df[f"{m}_gt"].values
        tot = np.sqrt(s_boot ** 2 + r_ulp ** 2)
        bs = [mape_from_relsd(np.mean(rng.choice(tot, len(tot)))) for _ in range(2000)]
        a1[m] = dict(relsd_cell_bootstrap=float(s_boot.mean()), relsd_rounding=float(r_ulp.mean()),
                     floor_mape=float(mape_from_relsd(tot.mean())), ci95=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                     n_interior_mean=float(df.n_interior.mean()))
    a1["HEX"]["binomial_closed_form_relsd"] = float(df.binom_relsd_HEX.mean())
    a1["CV"]["closed_form_relsd"] = float(df.closed_relsd_CV.mean())
    # sub-box spatial check: fit relsd = c / sqrt(n) per metric and extrapolate to the full-box n
    spatial = {}
    for m in ("CD", "CV", "HEX"):
        x = 1 / np.sqrt(sub.n_mean.values); y = sub[f"relsd_{m}"].values
        c = float(np.sum(x * y) / np.sum(x * x))
        spatial[m] = dict(c=c, relsd_at_full_n=float(c / np.sqrt(df.n_interior.mean())),
                          relsd_by_f=sub.groupby("f")[f"relsd_{m}"].mean().round(4).to_dict())
    a1["spatial_subbox"] = spatial
    a1["mean_floor_mape"] = float(np.mean([a1[m]["floor_mape"] for m in ("CD", "CV", "HEX")]))
    json.dump(a1, open(out / "A1_counting_floor.json", "w"), indent=2)

    # ---- A2: oracle process noise
    a2 = dict(CD_oracle_mape=float(df.CD_ape.mean()), HEX_oracle_mape=float(df.HEX_ape.mean()),
              CD_oracle_mape_ci95=[float(x) for x in np.percentile([rng.choice(df.CD_ape, 25).mean() for _ in range(2000)], [2.5, 97.5])],
              HEX_oracle_mape_ci95=[float(x) for x in np.percentile([rng.choice(df.HEX_ape, 25).mean() for _ in range(2000)], [2.5, 97.5])],
              CD_oracle_bias_pct=float(((df.CD - df.CD_gt) / df.CD_gt * 100).mean()),
              HEX_oracle_bias_pct=float(((df.HEX - df.HEX_gt) / df.HEX_gt * 100).mean()),
              by_split={s: dict(CD=float(g.CD_ape.mean()), HEX=float(g.HEX_ape.mean()), n=int(len(g))) for s, g in df.groupby("split")},
              CV_corr_interior=float(np.corrcoef(df.CV, df.CV_gt)[0, 1]),
              CV_ratio_median=float(df.CV_ratio.median()), CV_ratio_logsd=float(np.log(df.CV_ratio).std()))
    # per-metric process noise: oracle APE in excess of the counting floor (in quadrature, relSD units)
    for m in ("CD", "HEX"):
        obs = df[f"{m}_ape"].mean() / 100 / np.sqrt(2 / np.pi)
        cnt = a1[m]["floor_mape"] / 100 / np.sqrt(2 / np.pi)
        a2[f"{m}_process_relsd_excess"] = float(np.sqrt(max(obs ** 2 - cnt ** 2, 0)))
    # CV formula table: fit a single scale k (median ratio) then MAPE and log-ratio SD
    rows = []
    for col in [c for c in cvdf.columns if c not in ("ID", "CV_gt")]:
        v = cvdf[col].values; g = cvdf.CV_gt.values; ok = np.isfinite(v) & (v > 0)
        k = float(np.median(g[ok] / v[ok])); pred = k * v[ok]
        rows.append(dict(formula=col, scale_k=k, corr=float(np.corrcoef(v[ok], g[ok])[0, 1]),
                         mape_scaled=float(np.mean(np.abs(pred - g[ok]) / g[ok] * 100)),
                         logratio_sd=float(np.std(np.log(g[ok] / v[ok]))), n=int(ok.sum())))
    cvt = pd.DataFrame(rows).sort_values("mape_scaled"); cvt.to_csv(out / "A2_cv_formulas.csv", index=False)
    a2["cv_best_formula"] = cvt.iloc[0].to_dict()
    json.dump(a2, open(out / "A2_oracle_floor.json", "w"), indent=2)
    print(json.dumps(a1, indent=1)); print(json.dumps(a2, indent=1)); print(cvt.to_string(index=False))
    print(df[["ID", "split", "n_dots", "n_interior", "CD_ape", "HEX_ape", "CV_ratio", "boot_relsd_CD", "boot_relsd_CV", "boot_relsd_HEX"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
