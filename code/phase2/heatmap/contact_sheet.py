"""PNG contact sheet: clicks (green) vs detections (red) for chosen overlays, cropped to the supervised region."""
import argparse
import os
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import build_samples  # noqa: E402
from eval_overlays import detections_for, load_model  # noqa: E402
from model import predict_heatmap  # noqa: E402


def panel(sample, det, margin=30):
    x0, y0, x1, y1 = sample.bbox
    x0, y0 = max(x0 - margin, 0), max(y0 - margin, 0)
    x1, y1 = min(x1 + margin, sample.frame.shape[1]), min(y1 + margin, sample.frame.shape[0])
    im = Image.fromarray(sample.frame[y0:y1, x0:x1]).convert("RGB")
    dr = ImageDraw.Draw(im)
    for x, y in sample.dots:
        dr.ellipse([x - x0 - 3, y - y0 - 3, x - x0 + 3, y - y0 + 3], outline=(0, 255, 0), width=2)
    for x, y in det:
        dr.ellipse([x - x0 - 2, y - y0 - 2, x - x0 + 2, y - y0 + 2], fill=(255, 0, 0))
    dr.text((4, 4), f"{sample.image_id} clicks={len(sample.dots)} det={len(det)}", fill=(255, 255, 0))
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ids", nargs="+", default=["0324-23", "0471-23"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda")
    model, ck = load_model(args.ckpt, device)
    samples = build_samples(splits=("val", "test"))
    panels = []
    for image_id in args.ids:
        s = samples[image_id]
        det = detections_for(s, predict_heatmap(model, s.frame, device), ck["threshold"], ck["nms_px"])
        panels.append(panel(s, det))
    W = sum(p.width for p in panels) + 10 * (len(panels) - 1)
    H = max(p.height for p in panels)
    sheet = Image.new("RGB", (W, H), (0, 0, 0))
    x = 0
    for p in panels:
        sheet.paste(p, (x, 0))
        x += p.width + 10
    sheet = sheet.resize((sheet.width * 2, sheet.height * 2), Image.NEAREST)
    sheet.save(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
