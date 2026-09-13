"""CLI: evaluate a detector against the 25 train_green overlays (19 train / 6 held-out)."""
import argparse
import os

import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from tqdm import tqdm

from overlays import load_overlays
from readout import match, voronoi_metrics

MHA_DIR = "/home/visilant/CLEAR-EC/data/train_mha"
LABELS_CSV = "/home/visilant/CLEAR-EC/data/final_train_ids.csv"


def load_image(image_id):
    img = sitk.ReadImage(os.path.join(MHA_DIR, f"{image_id}.mha"))
    return sitk.GetArrayFromImage(img).astype(np.uint8)


def median_nn_dist(pts):
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=2)
    return float(np.median(d[:, 1]))


def dist_to_hull_edges(query, hull_pts):
    n = len(hull_pts)
    best = np.full(len(query), np.inf)
    for i in range(n):
        a, b = hull_pts[i], hull_pts[(i + 1) % n]
        ab = b - a
        t = np.clip(((query - a) @ ab) / (ab @ ab), 0, 1)
        proj = a + t[:, None] * ab
        d = np.linalg.norm(query - proj, axis=1)
        best = np.minimum(best, d)
    return best


def keep_in_dilated_hull(det, gt, dilation):
    if len(det) == 0:
        return det
    hull_pts = gt[ConvexHull(gt).vertices]
    inside = Delaunay(hull_pts).find_simplex(det) >= 0
    dist = dist_to_hull_edges(det, hull_pts)
    return det[inside | (dist <= dilation)]


def fake_detect(gt, rng):
    return gt + rng.normal(0, 2, size=gt.shape)


def run(model_name, ids, overlays, labels, detect_kwargs, fake, rng):
    rows = []
    for image_id in tqdm(ids, desc=f"eval {model_name} ({'fake' if fake else 'real'})"):
        gt = overlays[image_id]["dots"]
        if fake:
            det = fake_detect(gt, rng)
        else:
            from detect import detect
            image_u8 = load_image(image_id)
            det = detect(model_name, image_u8, **detect_kwargs)

        cell_diam = median_nn_dist(gt)
        tol = 0.5 * cell_diam
        det_kept = keep_in_dilated_hull(det, gt, 0.5 * cell_diam)

        m = match(det_kept, gt, tol)
        oracle = voronoi_metrics(gt)
        vdet = voronoi_metrics(det_kept)
        lab = labels.loc[image_id]

        row = dict(ID=image_id, split=overlays[image_id]["split"], n_gt=len(gt), n_det=len(det_kept), **m)
        for metric in ("CD", "CV", "HEX"):
            row[f"{metric}_gt"] = lab[metric]
            row[f"{metric}_oracle"] = oracle[metric]
            row[f"{metric}_det"] = vdet[metric]
            row[f"{metric}_ape_oracle"] = abs(oracle[metric] - lab[metric]) / max(lab[metric], 1e-6) * 100
            row[f"{metric}_ape_det"] = abs(vdet[metric] - lab[metric]) / max(lab[metric], 1e-6) * 100
        rows.append(row)
    return pd.DataFrame(rows)


def print_block(name, df):
    print(f"\n--- {name} ({len(df)} images) ---")
    cols = ["ID", "n_gt", "n_det", "f1", "count_ratio", "med_loc_err",
            "CD_ape_oracle", "CD_ape_det", "HEX_ape_oracle", "HEX_ape_det"]
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"  mean F1={df['f1'].mean():.3f}  count_ratio={df['count_ratio'].mean():.3f}  "
          f"CD_MAPE(oracle/det)={df['CD_ape_oracle'].mean():.2f}/{df['CD_ape_det'].mean():.2f}%  "
          f"CV_MAPE(oracle/det)={df['CV_ape_oracle'].mean():.2f}/{df['CV_ape_det'].mean():.2f}%  "
          f"HEX_MAPE(oracle/det)={df['HEX_ape_oracle'].mean():.2f}/{df['HEX_ape_det'].mean():.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="cpsam", choices=["cpsam", "sam_vit_b"])
    ap.add_argument("--diameter", type=float, default=None)
    ap.add_argument("--flow", type=float, default=0.4)
    ap.add_argument("--cellprob", type=float, default=0.0)
    ap.add_argument("--clahe", action="store_true")
    ap.add_argument("--fake", action="store_true", help="use GT dots jittered by 2px instead of detect.py")
    args = ap.parse_args()

    detect_kwargs = {}
    if args.model == "cpsam":
        detect_kwargs = dict(diameter=args.diameter, flow_threshold=args.flow, cellprob_threshold=args.cellprob, clahe=args.clahe)

    overlays = load_overlays()
    labels = pd.read_csv(LABELS_CSV).set_index("ID")
    rng = np.random.default_rng(0)

    train_ids = [i for i, v in overlays.items() if v["split"] == "train"]
    heldout_ids = [i for i, v in overlays.items() if v["split"] in ("val", "test")]

    df_train = run(args.model, train_ids, overlays, labels, detect_kwargs, args.fake, rng)
    df_held = run(args.model, heldout_ids, overlays, labels, detect_kwargs, args.fake, rng)

    print_block("train (19)", df_train)
    print_block("held-out (val+test, 6)", df_held)

    df_all = pd.concat([df_train, df_held], ignore_index=True)
    os.makedirs("results", exist_ok=True)
    if args.fake:
        out_path = "results/fake.csv"
    else:
        params = f"d{args.diameter}_f{args.flow}_c{args.cellprob}_e{int(args.clahe)}" if args.model == "cpsam" else "default"
        out_path = f"results/{args.model}_{params}.csv"
    df_all.to_csv(out_path, index=False)
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
