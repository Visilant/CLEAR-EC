"""A3/A4: residual anatomy of the night_20260912 OOF predictions and test-set visibility.

Inputs: results/night_20260912/oof/{fold,v2fold}{0..4}_last_flips.csv, labels, cache index.
Outputs under results/noise_floor_20260912/: A3_residuals.csv, A3_summary.json, A3_flagged_top{2,5}.txt,
A3_contact_sheet.png, A4_visibility.json.
"""
from __future__ import annotations
import json, re, argparse
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[3]  # repository root
from scipy import stats

METS = ["CD", "CV", "HEX"]
CLAMP = {"CD": (372.0, 4500.0), "CV": (0.03, 1.5), "HEX": (0.0, 1.0)}
N_INT = 113.8  # mean interior cells per annotated box (A1)


def mape(p, g):
    k = g != 0
    return float(np.mean(np.abs(p[k] - g[k]) / np.abs(g[k]) * 100))


def load_oof(oof_dir, prefix):
    parts = [pd.read_csv(oof_dir / f"{prefix}{k}_last_flips.csv") for k in range(5)]
    df = pd.concat(parts, ignore_index=True)
    df["ID"] = df.ID.astype(str).str.strip()
    assert df.ID.is_unique and len(df) == 9000, (prefix, len(df), df.ID.is_unique)
    return df.set_index("ID")


def geo(frames, weights):
    w = np.array(weights, float)
    return np.exp(sum(wi * np.log(f[METS]) for wi, f in zip(w, frames)) / w.sum())


def parse_id(s):
    m = re.match(r"^(\d{4})-(\d{4})[ -]?(ODCN|OSCN)$", s)
    if m: return ("B", m.group(1), m.group(2), m.group(3))
    m = re.match(r"^(\d{4})-(\d{2})$", s)
    if m: return ("A", m.group(2), m.group(1), "")
    return ("C", "", "", "")


def neighbours(num):
    n = int(num); out = set()
    for d in (1, 10, 100, 1000):
        for s in (n - d, n + d):
            if 0 <= s <= 9999: out.add(f"{s:04d}")
    for i in range(3):  # adjacent digit transposition
        l = list(num); l[i], l[i + 1] = l[i + 1], l[i]; out.add("".join(l))
    out.discard(num); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=str(REPO / "results/night_20260912/oof"))
    ap.add_argument("--labels", default=str(REPO / "data/final_train_ids.csv"))
    ap.add_argument("--cache", default=str(REPO / "data/cache"))
    ap.add_argument("--out", default=str(REPO / "results/noise_floor_20260912"))
    args = ap.parse_args()
    out = Path(args.out); oof = Path(args.oof); rng = np.random.default_rng(0)

    lab = pd.read_csv(args.labels); lab["ID"] = lab.ID.str.strip(); lab = lab.set_index("ID")
    idx = pd.read_csv(Path(args.cache) / "index.csv"); idx["ID"] = idx.ID.str.strip(); idx = idx.set_index("ID")
    v2 = load_oof(oof, "v2fold"); ti = load_oof(oof, "fold")
    ids = lab.index; assert set(ids) == set(v2.index) == set(ti.index)
    v2, ti = v2.loc[ids], ti.loc[ids]
    ens = geo([v2, ti], [2, 1])
    for m in METS: ens[m] = ens[m].clip(*CLAMP[m])
    G = lab[METS].astype(float)
    summ = {"scores": {}}
    for name, P in (("v2", v2), ("tiny", ti), ("ens", ens)):
        s = {m: mape(P[m].values, G[m].values) for m in METS}; s["mean"] = float(np.mean(list(s.values())))
        summ["scores"][name] = s
    print("scores", json.dumps(summ["scores"], indent=1))
    assert abs(summ["scores"]["ens"]["mean"] - 8.84) < 0.03, summ["scores"]["ens"]

    # residual table
    R = pd.DataFrame(index=ids)
    R["idx"] = idx.loc[ids, "idx"].values; R["slide_id"] = lab.slide_id.values
    fam = [parse_id(s) for s in ids]; R["family"] = [f[0] for f in fam]
    for m in METS:
        R[f"{m}_gt"] = G[m].values; R[f"{m}_ens"] = ens[m].values; R[f"{m}_v2"] = v2[m].values; R[f"{m}_tiny"] = ti[m].values
        g = G[m].values; ok = g > 0
        R[f"{m}_ape"] = np.where(ok, np.abs(ens[m].values - g) / np.where(ok, g, 1) * 100, np.nan)
        R[f"{m}_logres"] = np.where(ok, np.log(ens[m].values / np.where(ok, g, 1)), np.nan)
        R[f"{m}_disagree"] = np.abs(np.log(v2[m].values) - np.log(ti[m].values))
    R["mean_ape"] = R[[f"{m}_ape" for m in METS]].mean(axis=1)

    # 2. distribution
    dist = {}
    for m in METS:
        r = R[f"{m}_logres"].dropna().values; a = R[f"{m}_ape"].dropna().values; med = np.median(a)
        dist[m] = dict(logres_mean=float(r.mean()), logres_sd=float(r.std()), skew=float(stats.skew(r)), excess_kurtosis=float(stats.kurtosis(r)),
                       ape_median=float(med), frac_ape_gt_3x_median=float(np.mean(a > 3 * med)), frac_ape_gt_5x_median=float(np.mean(a > 5 * med)),
                       share_of_mape_from_top5pct=float(np.sort(a)[-int(0.05 * len(a)):].sum() / a.sum()),
                       mape_if_top2pct_removed=float(np.sort(a)[:-int(0.02 * len(a))].mean()))
    summ["distribution"] = dist

    # 3a. fellow-eye residual correlation
    fe = {}
    two = R.groupby("slide_id").filter(lambda g: len(g) == 2).sort_values("slide_id")
    a, b = two.iloc[0::2], two.iloc[1::2]
    for m in METS:
        x, y = a[f"{m}_logres"].values, b[f"{m}_logres"].values; k = np.isfinite(x) & np.isfinite(y)
        fe[m] = dict(pearson=float(np.corrcoef(x[k], y[k])[0, 1]), spearman=float(stats.spearmanr(x[k], y[k])[0]), n_pairs=int(k.sum()),
                     label_corr=float(np.corrcoef(a[f"{m}_gt"], b[f"{m}_gt"])[0, 1]), pred_corr=float(np.corrcoef(a[f"{m}_ens"], b[f"{m}_ens"])[0, 1]))
    summ["fellow_eye_residual"] = fe

    # 3b. member disagreement vs residual
    dis = {}
    for m in METS:
        d = R[f"{m}_disagree"].values; a_ = R[f"{m}_ape"].values; k = np.isfinite(a_)
        q = pd.qcut(d[k], 5, labels=False)
        dis[m] = dict(spearman=float(stats.spearmanr(d[k], a_[k])[0]), mape_by_disagree_quintile=[float(a_[k][q == i].mean()) for i in range(5)],
                      disagree_median=float(np.median(d)))
    summ["disagreement"] = dis

    # 3c. numeric-neighbour ID check (families A and B)
    lab_by = {s: G.loc[s].values for s in ids}
    id_index = {}
    for s, f in zip(ids, fam):
        if f[0] in ("A", "B"): id_index[(f[0], f[1], f[2], f[3])] = s
    swaps = []
    P = ens[METS].values; gt = G.values; own = R.mean_ape.values
    for i, (s, f) in enumerate(zip(ids, fam)):
        if f[0] not in ("A", "B") or own[i] < 25: continue
        best = None
        for nb in neighbours(f[2]):
            t = id_index.get((f[0], f[1], nb, f[3]))
            if t is None: continue
            g2 = lab_by[t]
            if (g2 == 0).any(): continue
            ape2 = float(np.mean(np.abs(P[i] - g2) / g2 * 100))
            if best is None or ape2 < best[1]: best = (t, ape2)
        if best and best[1] < 0.3 * own[i]:
            swaps.append(dict(ID=s, own_ape=float(own[i]), neighbour=best[0], neighbour_ape=best[1]))
    # null rate: same test with a random other ID of the same family/suffix instead of a numeric neighbour
    null_hits = 0; null_tries = 0
    pool = {}
    for s, f in zip(ids, fam): pool.setdefault((f[0], f[3]), []).append(s)
    for i, (s, f) in enumerate(zip(ids, fam)):
        if f[0] not in ("A", "B") or own[i] < 25: continue
        cands = rng.choice(pool[(f[0], f[3])], 7, replace=False)
        best = min(float(np.mean(np.abs(P[i] - lab_by[t]) / np.maximum(lab_by[t], 1e-9) * 100)) for t in cands if t != s)
        null_tries += 1; null_hits += best < 0.3 * own[i]
    summ["numeric_neighbour"] = dict(n_candidates_own_ape_ge_25=int((own >= 25).sum()), n_collapse=len(swaps), null_rate=float(null_hits / max(null_tries, 1)),
                                     null_expected=float(null_hits / max(null_tries, 1) * (own >= 25).sum()), examples=swaps[:25])

    # 3d. label quintiles vs counting floor
    quint = {}
    for m in METS:
        g = R[f"{m}_gt"].values; a_ = R[f"{m}_ape"].values; k = np.isfinite(a_)
        q = pd.qcut(g[k], 5, labels=False); rows = []
        for i in range(5):
            gm = float(np.mean(g[k][q == i]))
            if m == "HEX": relsd = np.sqrt(gm * (1 - gm) / N_INT) / gm
            elif m == "CV": relsd = np.sqrt((1 + 2 * gm ** 2) / (2 * N_INT))
            else: relsd = 0.019
            rows.append(dict(q=i, label_mean=gm, mape=float(a_[k][q == i].mean()), counting_floor=float(relsd * np.sqrt(2 / np.pi) * 100)))
        quint[m] = rows
    summ["quintiles"] = quint

    # 4. flagging by robust standardized log residual
    Z = np.zeros((len(R), 3))
    for j, m in enumerate(METS):
        r = R[f"{m}_logres"].values; med = np.nanmedian(r); mad = np.nanmedian(np.abs(r - med)) * 1.4826
        Z[:, j] = np.abs((r - med) / mad)
    R["zmax"] = np.nanmax(Z, axis=1)
    order = R.sort_values("zmax", ascending=False)
    for pct in (2, 5):
        n = int(round(len(R) * pct / 100))
        np.savetxt(out / f"A3_flagged_top{pct}.txt", order.idx.values[:n], fmt="%d")
    summ["flagging"] = dict(zmax_top2_threshold=float(order.zmax.values[int(0.02 * len(R))]), zmax_top5_threshold=float(order.zmax.values[int(0.05 * len(R))]),
                            top2_family_counts=order.head(180).family.value_counts().to_dict(), all_family_counts=R.family.value_counts().to_dict(),
                            mean_ape_top2=float(order.head(180).mean_ape.mean()))
    R.to_csv(out / "A3_residuals.csv")

    # contact sheet of top 20
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        mm = np.memmap(Path(args.cache) / "images_u8.npy", dtype=np.uint8, mode="r", shape=(9000, 972, 1296))
        top = order.head(20)
        fig, axes = plt.subplots(4, 5, figsize=(25, 16))
        for ax, (iid, row) in zip(axes.ravel(), top.iterrows()):
            ax.imshow(np.asarray(mm[int(row.idx)])[::3, ::3], cmap="gray", vmin=0, vmax=255); ax.axis("off")
            ax.set_title(f"{iid} z={row.zmax:.1f}\nGT {row.CD_gt:.0f}/{row.CV_gt:.2f}/{row.HEX_gt:.2f}  P {row.CD_ens:.0f}/{row.CV_ens:.2f}/{row.HEX_ens:.2f}", fontsize=9)
        plt.tight_layout(); plt.savefig(out / "A3_contact_sheet.png", dpi=60); plt.close()
    except Exception as e:
        summ["contact_sheet_error"] = repr(e)

    # A4 visibility: subsample SD and paired-difference SD for n = 100 and 1000, image-level and slide-level
    def per_image(P): return np.nanmean(np.stack([np.abs(P[m].values - G[m].values) / np.where(G[m].values > 0, G[m].values, np.nan) * 100 for m in METS]), axis=0)
    apes = {"ens": per_image(ens), "v2": per_image(v2), "tiny": per_image(ti)}
    slides = R.slide_id.values; uniq = np.unique(slides); slide_rows = {s: np.where(slides == s)[0] for s in uniq}
    vis = {}
    for n in (100, 1000):
        draws_img = [rng.choice(len(R), n, replace=False) for _ in range(4000)]
        draws_sl = []
        for _ in range(4000):
            sel = []; 
            for s in rng.permutation(uniq):
                sel.extend(slide_rows[s]); 
                if len(sel) >= n: break
            draws_sl.append(np.array(sel[:n]))
        for tag, draws in (("image", draws_img), ("slide", draws_sl)):
            res = {}
            for k, a in apes.items():
                v = np.array([np.nanmean(a[d]) for d in draws]); res[f"{k}_sd"] = float(v.std()); res[f"{k}_p_lt_8.91"] = float(np.mean(v < 8.91)); res[f"{k}_p_lt_8.70"] = float(np.mean(v < 8.70))
            for x, y in (("ens", "tiny"), ("v2", "tiny"), ("ens", "v2")):
                dv = np.array([np.nanmean(apes[x][d]) - np.nanmean(apes[y][d]) for d in draws])
                res[f"diff_{x}_minus_{y}"] = dict(oof_delta=float(np.nanmean(apes[x]) - np.nanmean(apes[y])), sd=float(dv.std()), p_sign_preserved=float(np.mean(dv < 0)),
                                                  min_delta_80pct_power=float(0.84 * dv.std()))
            vis[f"n{n}_{tag}"] = res
    json.dump(vis, open(out / "A4_visibility.json", "w"), indent=2)
    json.dump(summ, open(out / "A3_summary.json", "w"), indent=2, default=float)
    print(json.dumps(summ, indent=1, default=float)); print(json.dumps(vis, indent=1))


if __name__ == "__main__":
    main()
