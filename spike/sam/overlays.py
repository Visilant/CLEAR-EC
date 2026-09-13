"""Parse train_green overlays -> box + GT dot centers in full-res coords."""
import json
import os

import numpy as np
from PIL import Image
from scipy.ndimage import binary_opening, distance_transform_edt
from skimage.feature import peak_local_max
from tqdm import tqdm

OVERLAY_DIR = "/home/visilant/CLEAR-EC/data/train_green"
CACHE_PATH = "overlays.json"
SCALE_X = 1296 / 720
SCALE_Y = 972 / 540

VAL_IDS = {"0334-23", "0587-23"}
TEST_IDS = {"0324-23", "0329-23", "0471-23", "0586-23"}


def split_of(image_id):
    if image_id in VAL_IDS:
        return "val"
    if image_id in TEST_IDS:
        return "test"
    return "train"


def parse_overlay(path):
    rgba = np.array(Image.open(path))
    r, g, b = rgba[:, :, 0].astype(int), rgba[:, :, 1].astype(int), rgba[:, :, 2].astype(int)
    green = (g > 100) & (g - r > 50) & (g - b > 50)

    row_frac = green.mean(axis=1)
    col_frac = green.mean(axis=0)
    rows = np.where(row_frac > 0.25)[0]
    cols = np.where(col_frac > 0.25)[0]
    y0, y1 = rows.min(), rows.max()
    x0, x1 = cols.min(), cols.max()

    box_mask = np.zeros_like(green)
    box_mask[y0 : y1 + 1, :] |= green[y0 : y1 + 1, :] & (row_frac[y0 : y1 + 1] > 0.25)[:, None]
    box_mask[:, x0 : x1 + 1] |= green[:, x0 : x1 + 1] & (col_frac[x0 : x1 + 1] > 0.25)[None, :]

    dots_mask = green & ~box_mask
    dots_mask = binary_opening(dots_mask, structure=np.ones((3, 3)))
    dist = distance_transform_edt(dots_mask)
    coords = peak_local_max(dist, min_distance=3, threshold_abs=2)  # (row, col)

    dots_xy = np.stack([coords[:, 1] * SCALE_X, coords[:, 0] * SCALE_Y], axis=1)
    box_full = (x0 * SCALE_X, y0 * SCALE_Y, x1 * SCALE_X, y1 * SCALE_Y)
    return box_full, dots_xy


def load_overlays():
    if os.path.exists(CACHE_PATH):
        raw = json.load(open(CACHE_PATH))
        return {
            k: dict(box=tuple(v["box"]), dots=np.array(v["dots"]), split=v["split"])
            for k, v in raw.items()
        }

    ids = sorted(f[:-5] for f in os.listdir(OVERLAY_DIR) if f.endswith(".tiff"))
    out = {}
    for image_id in tqdm(ids, desc="parsing overlays"):
        box, dots = parse_overlay(os.path.join(OVERLAY_DIR, f"{image_id}.tiff"))
        out[image_id] = dict(box=box, dots=dots, split=split_of(image_id))

    json.dump(
        {k: dict(box=list(v["box"]), dots=v["dots"].tolist(), split=v["split"]) for k, v in out.items()},
        open(CACHE_PATH, "w"),
    )
    return out


if __name__ == "__main__":
    overlays = load_overlays()
    counts = {"train": 0, "val": 0, "test": 0}
    print(f"{'ID':10s} {'split':6s} {'n_dots':7s} {'box_w':7s} {'box_h':7s}")
    for image_id, v in overlays.items():
        counts[v["split"]] += 1
        x0, y0, x1, y1 = v["box"]
        print(f"{image_id:10s} {v['split']:6s} {len(v['dots']):7d} {x1-x0:7.1f} {y1-y0:7.1f}")
    print(f"\ntotal overlays: {len(overlays)}")
    print(f"split counts: train={counts['train']} val={counts['val']} test={counts['test']}")
