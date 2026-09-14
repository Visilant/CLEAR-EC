"""Overlays -> frames, Gaussian heatmap targets, loss masks, random training crops.

The clicks in overlays.json cover only a compact blob inside the green box (about 150 of
the ~400 cells the box holds), so the supervised region is the convex hull of the clicks
dilated by half the median nearest-neighbour spacing, not the box.
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from scipy.spatial import ConvexHull, cKDTree
from skimage.draw import polygon as draw_polygon

ROOT = "/home/visilant/CLEAR-EC"
OVERLAYS_JSON = os.path.join(ROOT, "results/sam_spike_20260912/overlays.json")
CACHE_NPY = os.path.join(ROOT, "data/cache/images_u8.npy")
CACHE_INDEX = os.path.join(ROOT, "data/cache/index.csv")
LABELS_CSV = os.path.join(ROOT, "data/final_train_ids.csv")
FRAME_SHAPE = (972, 1296)
UM_PER_PX = 0.7716049


def load_overlays():
    raw = json.load(open(OVERLAYS_JSON))
    return {k: dict(box=tuple(v["box"]), dots=np.asarray(v["dots"], float), split=v["split"])
            for k, v in raw.items()}


def open_frames():
    index = pd.read_csv(CACHE_INDEX).set_index("ID")
    n = len(index)
    mm = np.memmap(CACHE_NPY, dtype=np.uint8, mode="r", shape=(n, *FRAME_SHAPE))
    return mm, index


def load_frame(mm, index, image_id):
    return np.array(mm[int(index.loc[image_id, "idx"])])


def median_nn(pts):
    pts = np.asarray(pts, float)
    d, _ = cKDTree(pts).query(pts, k=2)
    return float(np.median(d[:, 1]))


def train_median_nn(overlays):
    """Median nearest-neighbour click spacing pooled over the train overlays."""
    d = []
    for v in overlays.values():
        if v["split"] == "train":
            p = v["dots"]
            dd, _ = cKDTree(p).query(p, k=2)
            d.extend(dd[:, 1].tolist())
    return float(np.median(d))


def hull_mask(pts, dilation_px, shape=FRAME_SHAPE):
    """Boolean mask of the convex hull of pts dilated by dilation_px (full-frame coordinates)."""
    pts = np.asarray(pts, float)
    hull = pts[ConvexHull(pts).vertices]
    mask = np.zeros(shape, bool)
    rr, cc = draw_polygon(hull[:, 1], hull[:, 0], shape)
    mask[rr, cc] = True
    if dilation_px > 0:
        mask = distance_transform_edt(~mask) <= dilation_px  # fast exact disk dilation
    return mask


def points_in_mask(pts, mask):
    pts = np.asarray(pts, float)
    if len(pts) == 0:
        return pts
    x = np.clip(np.round(pts[:, 0]).astype(int), 0, mask.shape[1] - 1)
    y = np.clip(np.round(pts[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return pts[mask[y, x]]


def gaussian_heatmap(pts, sigma, shape=FRAME_SHAPE):
    """Max over per-click Gaussians (peak value 1) at sub-pixel positions."""
    hm = np.zeros(shape, np.float32)
    r = int(np.ceil(4 * sigma))
    for x, y in np.asarray(pts, float):
        cx, cy = int(round(x)), int(round(y))
        x0, x1 = max(cx - r, 0), min(cx + r + 1, shape[1])
        y0, y1 = max(cy - r, 0), min(cy + r + 1, shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        gy, gx = np.mgrid[y0:y1, x0:x1]
        g = np.exp(-((gx - x) ** 2 + (gy - y) ** 2) / (2 * sigma ** 2)).astype(np.float32)
        hm[y0:y1, x0:x1] = np.maximum(hm[y0:y1, x0:x1], g)
    return hm


class Sample:
    """One overlay: full frame (uint8), heatmap target, loss mask, eval region, click coordinates."""

    def __init__(self, image_id, frame, dots, split, sigma, mask_dilation):
        self.image_id = image_id
        self.split = split
        self.frame = frame
        self.dots = dots
        self.target = gaussian_heatmap(dots, sigma)
        self.mask = hull_mask(dots, mask_dilation)  # loss mask (train-pooled spacing)
        self.region = hull_mask(dots, 0.5 * median_nn(dots))  # eval region (image spacing)
        ys, xs = np.where(self.mask)
        self.bbox = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)  # x0,y0,x1,y1


def build_samples(sigma=3.0, mask_dilation=None, splits=("train", "val", "test")):
    overlays = load_overlays()
    if mask_dilation is None:
        mask_dilation = 0.5 * train_median_nn(overlays)
    mm, index = open_frames()
    out = {}
    for image_id, v in overlays.items():
        if v["split"] not in splits:
            continue
        frame = load_frame(mm, index, image_id)
        out[image_id] = Sample(image_id, frame, v["dots"], v["split"], sigma, mask_dilation)
    return out


def random_crop(sample, size, rng):
    """Random size x size crop whose centre lies in the supervised region's bounding box."""
    x0, y0, x1, y1 = sample.bbox
    H, W = sample.frame.shape
    cx = rng.integers(x0, x1)
    cy = rng.integers(y0, y1)
    left = int(np.clip(cx - size // 2, 0, W - size))
    top = int(np.clip(cy - size // 2, 0, H - size))
    sl = (slice(top, top + size), slice(left, left + size))
    return sample.frame[sl], sample.target[sl], sample.mask[sl]


def augment(img, target, mask, rng):
    """Flips, rot90, brightness/contrast/gamma jitter on a float [0,1] image."""
    k = int(rng.integers(0, 4))
    if k:
        img, target, mask = (np.rot90(a, k) for a in (img, target, mask))
    if rng.random() < 0.5:
        img, target, mask = (a[:, ::-1] for a in (img, target, mask))
    if rng.random() < 0.5:
        img, target, mask = (a[::-1, :] for a in (img, target, mask))
    img = intensity_jitter(img, rng)
    return (np.ascontiguousarray(img, np.float32), np.ascontiguousarray(target, np.float32),
            np.ascontiguousarray(mask, np.float32))


def intensity_jitter(img, rng):
    """Gamma / contrast / brightness jitter and occasional noise on a float [0,1] image."""
    gamma = float(np.exp(rng.uniform(-0.3, 0.3)))
    img = np.power(img, gamma)
    a = float(rng.uniform(0.75, 1.25))
    b = float(rng.uniform(-0.12, 0.12))
    img = np.clip(a * (img - 0.5) + 0.5 + b, 0.0, 1.0)
    if rng.random() < 0.3:
        img = np.clip(img + rng.normal(0, 0.02, img.shape), 0.0, 1.0)
    return img


def make_batch(train_samples, batch_size, crop_size, rng):
    imgs, tgts, msks = [], [], []
    for _ in range(batch_size):
        s = train_samples[int(rng.integers(len(train_samples)))]
        img, tgt, msk = random_crop(s, crop_size, rng)
        img = img.astype(np.float32) / 255.0
        img, tgt, msk = augment(img, tgt, msk, rng)
        imgs.append(img)
        tgts.append(tgt)
        msks.append(msk)
    return np.stack(imgs)[:, None], np.stack(tgts)[:, None], np.stack(msks)[:, None]
