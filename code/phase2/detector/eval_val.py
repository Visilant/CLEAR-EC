"""CLI: run a detector on val-split memmap images, apply the annotator-like region
selection + largest-cluster filter, compute Voronoi metrics, compare to labels."""
import argparse
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import Delaunay, cKDTree
from tqdm import tqdm

from readout import voronoi_metrics

IMAGES_PATH = "/home/visilant/CLEAR-EC/data/cache/images_u8.npy"
INDEX_CSV = "/home/visilant/CLEAR-EC/data/cache/index.csv"
SPLITS_JSON = "/home/visilant/CLEAR-EC/data/cache/splits.json"
LABELS_CSV = "/home/visilant/CLEAR-EC/data/final_train_ids.csv"
IMG_SHAPE = (972, 1296)
WIN_W, WIN_H = 540, 408
STRIDE = 68


def load_images():
    n = len(pd.read_csv(INDEX_CSV))
    return np.memmap(IMAGES_PATH, dtype=np.uint8, mode="r", shape=(n,) + IMG_SHAPE)


def band_energy(image_u8):
    f = image_u8.astype(np.float32)
    return np.abs(gaussian_filter(f, 2) - gaussian_filter(f, 6))


def choose_region(image_u8):
    """Slide a WIN_W x WIN_H window, score by mean band-pass energy, pick the argmax."""
    energy = band_energy(image_u8)
    h, w = image_u8.shape
    best_score, best_xy = -np.inf, (0, 0)
    for y0 in range(0, h - WIN_H + 1, STRIDE):
        for x0 in range(0, w - WIN_W + 1, STRIDE):
            score = energy[y0 : y0 + WIN_H, x0 : x0 + WIN_W].mean()
            if score > best_score:
                best_score, best_xy = score, (x0, y0)
    return best_xy[0], best_xy[1], best_xy[0] + WIN_W, best_xy[1] + WIN_H


def largest_cluster(pts, edge_factor=1.6):
    """Largest connected component of the Delaunay graph with edges < edge_factor * median NN dist."""
    if len(pts) < 4:
        return pts
    tree = cKDTree(pts)
    nn_d, _ = tree.query(pts, k=2)
    med = np.median(nn_d[:, 1])
    thresh = edge_factor * med

    tri = Delaunay(pts)
    n = len(pts)
    adj = [[] for _ in range(n)]
    for simplex in tri.simplices:
        for i in range(3):
            a, b = simplex[i], simplex[(i + 1) % 3]
            if np.linalg.norm(pts[a] - pts[b]) < thresh:
                adj[a].append(b)
                adj[b].append(a)

    seen = np.zeros(n, dtype=bool)
    best_comp = []
    for start in range(n):
        if seen[start]:
            continue
        stack, comp = [start], []
        seen[start] = True
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
        if len(comp) > len(best_comp):
            best_comp = comp
    return pts[best_comp]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cpsam", choices=["cpsam", "sam_vit_b"])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--diameter", type=float, default=None)
    ap.add_argument("--flow", type=float, default=0.4)
    ap.add_argument("--cellprob", type=float, default=0.0)
    ap.add_argument("--clahe", action="store_true")
    args = ap.parse_args()

    detect_kwargs = {}
    if args.model == "cpsam":
        detect_kwargs = dict(diameter=args.diameter, flow_threshold=args.flow, cellprob_threshold=args.cellprob, clahe=args.clahe)

    index = pd.read_csv(INDEX_CSV)
    splits = json.load(open(SPLITS_JSON))
    val_idx = splits["val"][: args.n]
    val_ids = index.set_index("idx").loc[val_idx, "ID"].tolist()
    labels = pd.read_csv(LABELS_CSV).set_index("ID")

    out_path = f"results/val_{args.model}_e{int(args.clahe)}.csv"
    os.makedirs("results", exist_ok=True)
    done_ids = set()
    if os.path.exists(out_path):
        done_ids = set(pd.read_csv(out_path)["ID"])

    images = load_images()
    from detect import detect

    rows = []
    for i, image_id in zip(tqdm(val_idx, desc=f"eval_val {args.model}"), val_ids):
        if image_id in done_ids or image_id not in labels.index:
            continue
        image_u8 = np.asarray(images[i])
        det = detect(args.model, image_u8, **detect_kwargs)

        x0, y0, x1, y1 = choose_region(image_u8)
        in_win = (det[:, 0] >= x0) & (det[:, 0] < x1) & (det[:, 1] >= y0) & (det[:, 1] < y1)
        cluster = largest_cluster(det[in_win])

        m = voronoi_metrics(cluster)
        lab = labels.loc[image_id]
        row = dict(ID=image_id, n_det=len(det), n_win=int(in_win.sum()), n_cluster=len(cluster), **m)
        for metric in ("CD", "CV", "HEX"):
            row[f"{metric}_gt"] = lab[metric]
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
        rows = []

    df = pd.read_csv(out_path)
    for metric in ("CD", "CV", "HEX"):
        df[f"{metric}_ape"] = (df[metric] - df[f"{metric}_gt"]).abs() / df[f"{metric}_gt"] * 100
    scale = (df["CD_gt"] / df["CD"]).median()
    df["CD_scaled_ape"] = (df["CD"] * scale - df["CD_gt"]).abs() / df["CD_gt"] * 100

    print(f"\n{len(df)} val images evaluated ({args.model})")
    for metric in ("CD", "CV", "HEX"):
        valid = df[metric].notna() & (df[metric] > 0) & (df[f"{metric}_gt"] > 0)
        corr = np.corrcoef(np.log(df.loc[valid, metric]), np.log(df.loc[valid, f"{metric}_gt"]))[0, 1]
        print(f"  {metric}: MAPE={df[f'{metric}_ape'].mean():.2f}%  log-corr={corr:.3f}")
    print(f"  CD MAPE after global scale ({scale:.3f}x): {df['CD_scaled_ape'].mean():.2f}%")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
