"""Paired, slide-clustered bootstrap comparison of two prediction CSVs on the same images (B1/B3 gates)."""
import argparse, json
import numpy as np, pandas as pd
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]  # repository root
METS = ["CD", "CV", "HEX"]

def per_image(P, G):
    return np.nanmean(np.stack([np.abs(P[m].values - G[m].values) / np.where(G[m].values > 0, G[m].values, np.nan) * 100 for m in METS]), axis=0)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b", required=True)
    ap.add_argument("--labels", default=str(REPO / "data/final_train_ids.csv")); ap.add_argument("--out"); ap.add_argument("--exclude_idx_file", default="")
    args = ap.parse_args()
    lab = pd.read_csv(args.labels); lab["ID"] = lab.ID.str.strip(); lab = lab.set_index("ID")
    A = pd.read_csv(args.a); B = pd.read_csv(args.b)
    for d in (A, B): d["ID"] = d.ID.astype(str).str.strip()
    A = A.set_index("ID"); B = B.set_index("ID").loc[A.index]
    G = lab.loc[A.index]; slides = G.slide_id.values
    ea, eb = per_image(A, G), per_image(B, G)
    res = dict(n=len(A), a=args.a, b=args.b, a_mean=float(ea.mean()), b_mean=float(eb.mean()), delta_a_minus_b=float(ea.mean() - eb.mean()),
               per_metric={m: dict(a=float(np.mean(np.abs(A[m].values[G[m].values > 0] - G[m].values[G[m].values > 0]) / G[m].values[G[m].values > 0] * 100)), b=float(np.mean(np.abs(B[m].values[G[m].values > 0] - G[m].values[G[m].values > 0]) / G[m].values[G[m].values > 0] * 100))) for m in METS})
    rng = np.random.default_rng(0); uniq = np.unique(slides); rows = {s: np.where(slides == s)[0] for s in uniq}
    d = ea - eb; boots = []
    for _ in range(2000):
        pick = rng.choice(len(uniq), len(uniq)); sel = np.concatenate([rows[uniq[i]] for i in pick]); boots.append(d[sel].mean())
    res["delta_ci95_slide_boot"] = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]; res["delta_se"] = float(np.std(boots))
    if args.exclude_idx_file:
        ex = set(int(t) for t in open(args.exclude_idx_file).read().split()); k = ~A.idx.isin(ex).values
        res["delta_on_unflagged_only"] = float(ea[k].mean() - eb[k].mean()); res["n_unflagged"] = int(k.sum())
    print(json.dumps(res, indent=1))
    if args.out: json.dump(res, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
