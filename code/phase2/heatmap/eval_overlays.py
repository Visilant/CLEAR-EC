"""Score a heatmap checkpoint on the val / test overlays: detection F1 and Voronoi readout APE.

Detections are kept inside the supervised region (click convex hull dilated by half the
image's median click spacing, as in phase2/detector/eval_overlays.py); matching tolerance
is half the image's median click spacing; NMS distance is half the train-pooled median.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.append(os.path.join(HERE, "..", "detector"))  # readout.match / voronoi_metrics; appended so heatmap modules win

from data import LABELS_CSV, UM_PER_PX, build_samples, median_nn, points_in_mask  # noqa: E402
from model import HeatmapNet, predict_heatmap  # noqa: E402
from peaks import extract_peaks  # noqa: E402
from readout import match, voronoi_metrics  # noqa: E402

METRICS = ("CD", "CV", "HEX")


def score_detections(sample, det_xy, labels=None):
    """Match detections (already restricted to the supervised region) to clicks; Voronoi readout."""
    gt = sample.dots
    tol = 0.5 * median_nn(gt)
    m = match(det_xy, gt, tol)
    row = dict(ID=sample.image_id, split=sample.split, n_gt=len(gt), n_det=len(det_xy), tol_px=tol, **m)
    if labels is not None:
        oracle = voronoi_metrics(gt, UM_PER_PX)
        vdet = voronoi_metrics(det_xy, UM_PER_PX)
        lab = labels.loc[sample.image_id]
        for k in METRICS:
            row[f"{k}_gt"] = lab[k]
            row[f"{k}_oracle"] = oracle[k]
            row[f"{k}_det"] = vdet[k]
            row[f"{k}_ape_oracle"] = abs(oracle[k] - lab[k]) / max(lab[k], 1e-6) * 100
            row[f"{k}_ape_det"] = abs(vdet[k] - lab[k]) / max(lab[k], 1e-6) * 100
    return row


def detections_for(sample, heatmap, threshold, nms_px):
    return points_in_mask(extract_peaks(heatmap, threshold, nms_px, region=sample.region), sample.region)


def evaluate(samples, heatmaps, threshold, nms_px, labels=None):
    rows = [score_detections(s, detections_for(s, heatmaps[s.image_id], threshold, nms_px), labels)
            for s in samples]
    return pd.DataFrame(rows)


def summary(df):
    out = dict(n=len(df), recall=df.recall.mean(), precision=df.precision.mean(), f1=df.f1.mean(),
               count_ratio=df.count_ratio.mean(), max_abs_count_dev=(df.count_ratio - 1).abs().max(),
               med_loc_err=df.med_loc_err.mean())
    for k in METRICS:
        if f"{k}_ape_det" in df:
            out[f"{k}_ape_det"] = df[f"{k}_ape_det"].mean()
            out[f"{k}_ape_oracle"] = df[f"{k}_ape_oracle"].mean()
    return out


def gate_table(df):
    s = summary(df)
    gates = [("recall >= 0.90", s["recall"], s["recall"] >= 0.90),
             ("F1 >= 0.85", s["f1"], s["f1"] >= 0.85),
             ("|count ratio - 1| <= 0.05 (mean)", abs(s["count_ratio"] - 1), abs(s["count_ratio"] - 1) <= 0.05)]
    return gates, s


def load_model(ckpt_path, device):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = HeatmapNet(pretrained=False)
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--threshold", type=float, default=None, help="default: value stored in the checkpoint")
    ap.add_argument("--out", default=None, help="CSV path for per-image rows")
    args = ap.parse_args()
    device = torch.device("cuda")
    model, ck = load_model(args.ckpt, device)
    threshold = ck["threshold"] if args.threshold is None else args.threshold
    nms_px = ck["nms_px"]
    labels = pd.read_csv(LABELS_CSV).set_index("ID")
    samples = list(build_samples(splits=(args.split,)).values())
    heatmaps = {s.image_id: predict_heatmap(model, s.frame, device) for s in samples}
    df = evaluate(samples, heatmaps, threshold, nms_px, labels)
    cols = ["ID", "n_gt", "n_det", "recall", "precision", "f1", "count_ratio", "med_loc_err",
            "CD_gt", "CD_oracle", "CD_det", "CD_ape_oracle", "CD_ape_det",
            "CV_ape_oracle", "CV_ape_det", "HEX_ape_oracle", "HEX_ape_det"]
    print(f"split={args.split} threshold={threshold:.3f} nms_px={nms_px:.2f}")
    print(df[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    gates, s = gate_table(df)
    print("\nsummary:", {k: round(float(v), 4) for k, v in s.items()})
    if args.split == "test":
        for name, val, ok in gates:
            print(f"  gate {name}: {val:.4f} -> {'PASS' if ok else 'FAIL'}")
    if args.out:
        df.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
