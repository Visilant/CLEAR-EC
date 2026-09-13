#!/usr/bin/env python3
"""Summarize the 2026-09-12 night: val table for every arm, OOF scores for the folds, subsample noise."""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.splits import load_split  # noqa: E402

R = Path("/home/visilant/CLEAR-EC/results/night_20260912")
LAB = pd.read_csv("/home/visilant/CLEAR-EC/data/final_train_ids.csv"); LAB["ID"] = LAB["ID"].astype(str).str.strip(); LAB = LAB.set_index("ID")
CACHE = Path("/home/visilant/CLEAR-EC/data/cache"); METS = ["CD", "CV", "HEX"]
INCUMBENT = pd.read_csv("/home/visilant/CLEAR-EC/results/convnext_20260911/whole_relative/regression_cnn/seed_123/predictions_val.csv")
INCUMBENT["ID"] = INCUMBENT["ID"].astype(str).str.strip(); INCUMBENT = INCUMBENT.set_index("ID")

def mape(df: pd.DataFrame) -> dict:
    g = LAB.loc[df.index]; out = {}
    for m in METS:
        t = g[m].values.astype(float); p = df[m].values.astype(float); k = t != 0
        out[m] = 100 * np.mean(np.abs(p[k] - t[k]) / np.abs(t[k]))
    out["mean"] = float(np.mean([out[m] for m in METS])); return out

def geo(dfs): 
    idx = dfs[0].index
    return pd.DataFrame({m: np.exp(np.mean([np.log(d.loc[idx, m].clip(lower=1e-6)) for d in dfs], axis=0)) for m in METS}, index=idx)

def load_pred(p):
    d = pd.read_csv(p); d["ID"] = d["ID"].astype(str).str.strip(); return d.set_index("ID").sort_index()

def fmt(s): return "  ".join(f"{m} {s[m]:.2f}" for m in METS) + f"  | mean {s['mean']:.2f}"

print("== Single-split arms (892 val) ==")
rows = []
for mj in sorted(glob.glob(str(R / "*/regression_cnn/seed_*/metrics.json"))):
    name = Path(mj).parents[2].name; d = json.load(open(mj))
    if "val_mape" not in d: rows.append((name, "no-val (all_data)", None, None)); continue
    seed = d["seed"]; best = d["val_mape"]["mean"]
    lastj = R / "tta" / f"{name}_val_last.json"
    last = json.load(open(lastj))["mape"]["mean"] if lastj.exists() else None
    lastt = R / "tta" / f"{name}_val_last_flips.json"
    lastf = json.load(open(lastt))["mape"]["mean"] if lastt.exists() else None
    rows.append((name, f"seed {seed} ep{d.get('epochs_run','?')}", best, last, lastf))
print(f"{'arm':<16}{'note':<20}{'best-epoch':>11}{'last-epoch':>11}{'last+TTA':>10}")
for r in rows:
    print(f"{r[0]:<16}{r[1]:<20}" + "".join(f"{(v if v is not None else float('nan')):>11.2f}" if i < 2 else f"{(v if v is not None else float('nan')):>10.2f}" for i, v in enumerate(r[2:])))
print("incumbent whole_relative best-epoch 9.54 (epoch 7, patience 5)")

# seed ensemble on val from r3 seeds (last epoch)
r3 = [R / "tta" / f"r3_seed{s}_val_last.csv" for s in (123, 42)]
if all(p.exists() for p in r3):
    dfs = [load_pred(p) for p in r3]; e = geo(dfs)
    print("\nr3 two-seed geometric ensemble (last epoch):", fmt(mape(e)))
    print("with incumbent added:", fmt(mape(geo(dfs + [INCUMBENT]))))

print("\n== Fold OOF over all 9000 (each image predicted by the model that did not train on it) ==")
val_ids = set(LAB.index[[i in set(load_split(CACHE, 'val')) for i in range(len(LAB))]]) if False else None
for which in ("last", "best"):
    for tta in ("none", "flips"):
        files = sorted(glob.glob(str(R / "oof" / f"fold*_{which}_{tta}.csv")))
        if not files: continue
        parts = [load_pred(f) for f in files]; oof = pd.concat(parts)
        per_fold = [mape(p)["mean"] for p in parts]
        s = mape(oof)
        print(f"{which:<5}{tta:<6} n={len(oof):<5} {fmt(s)}   per-fold: {' '.join(f'{v:.2f}' for v in per_fold)} (spread {max(per_fold)-min(per_fold):.2f})")
        common = oof.index.intersection(INCUMBENT.index)
        if len(common) > 800:
            print(f"      on the 892-val subset: OOF {mape(oof.loc[common])['mean']:.2f} vs incumbent {mape(INCUMBENT.loc[common])['mean']:.2f}")
        ape = np.mean([np.abs(oof[m] - LAB.loc[oof.index, m]) / LAB.loc[oof.index, m].replace(0, np.nan) * 100 for m in METS], axis=0)
        rng = np.random.default_rng(0); sub = [np.nanmean(rng.choice(ape, 100, replace=False)) for _ in range(5000)]
        print(f"      100-image subsample: sd {np.std(sub):.2f}, P(<8.91)={np.mean(np.array(sub) < 8.91):.2f}")
