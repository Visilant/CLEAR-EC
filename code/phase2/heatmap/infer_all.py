"""S2 step 1: run the S1 heatmap detector on every cached frame; save peaks + coarse confidence maps.

Per frame we save (results/<out>/det/<idx>.npz): peak x, y (int16), peak score (float16, smoothed
heatmap value), a 36x48 block-mean of the heatmap (27 px blocks, saturated plateaus zeroed), the plateau
fraction per block, and a 36x48 block-mean of the band-pass focus energy |G2 - G6| of the frame.
The loss was masked to the click hull, so on the unsupervised black background the model emits
flat saturated 1.0 plateaus; a peak is kept only where the smoothed heatmap has a range of at least
PLATEAU_RANGE over the NMS window. Peaks are extracted on the GPU (Gaussian
smooth 1 px, 21x21 max-pool NMS = the 10.26 px checkpoint NMS, threshold from the checkpoint);
the first batch is cross-checked against peaks.extract_peaks. Resumable: existing npz are skipped.
"""
import argparse
import glob
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import CACHE_INDEX, CACHE_NPY, FRAME_SHAPE  # noqa: E402
from model import HeatmapNet, pad_to_multiple  # noqa: E402
from peaks import extract_peaks  # noqa: E402

BLOCK = 27  # 972 / 27 = 36 rows, 1296 / 27 = 48 cols
WIN_W, WIN_H = 538, 408  # annotator-box emulation window (features.py)
PLATEAU_RANGE = 0.2  # min (max - min) of the smoothed heatmap over the NMS window for a real peak


def gaussian_kernel1d(sigma, device):
    r = int(4 * sigma + 0.5)
    x = torch.arange(-r, r + 1, dtype=torch.float32, device=device)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return (k / k.sum()), r


def blur(x, sigma):
    """Separable Gaussian blur of (B,1,H,W), reflect padding (scipy truncate=4 kernel)."""
    k, r = gaussian_kernel1d(sigma, x.device)
    x = F.pad(x, (r, r, 0, 0), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.pad(x, (0, 0, r, r), mode="reflect")
    return F.conv2d(x, k.view(1, 1, -1, 1))


def block_mean(x, block=BLOCK):
    return F.avg_pool2d(x, block, stride=block)


@torch.no_grad()
def run_batch(model, frames_u8, device, threshold, nms_px, amp=True, windows=None):
    """frames_u8: (B,H,W) uint8 numpy -> list of per-frame dicts. windows: optional per-frame (x0, y0)
    of the 538x408 focus window; the raw heatmap sum over it is saved as win_sum (direct count readout)."""
    x = torch.from_numpy(frames_u8).to(device).float().div_(255.0)[:, None]
    xp, (H, W) = pad_to_multiple(x)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=amp):
        logits = model(xp)
    hm = torch.sigmoid(logits.float())[:, :, :H, :W]
    hm_s = blur(hm, 1.0)
    r = int(round(nms_px))
    mx = F.max_pool2d(hm_s, 2 * r + 1, stride=1, padding=r)
    mn = -F.max_pool2d(-hm_s, 2 * r + 1, stride=1, padding=r)
    plateau = (mx - mn) < PLATEAU_RANGE  # saturated flat output on unsupervised background
    is_peak = (hm_s >= mx) & (hm_s > threshold) & ~plateau
    focus = (blur(x, 2.0) - blur(x, 6.0)).abs() * 255.0
    hm_blk = block_mean(hm * (~plateau).float()).cpu().numpy()[:, 0]
    sat_blk = block_mean(plateau.float()).cpu().numpy()[:, 0]
    focus_blk = block_mean(focus).cpu().numpy()[:, 0]
    out = []
    for b in range(len(frames_u8)):
        ys, xs = torch.nonzero(is_peak[b, 0], as_tuple=True)
        sc = hm_s[b, 0, ys, xs]
        out.append(dict(x=xs.cpu().numpy().astype(np.int16), y=ys.cpu().numpy().astype(np.int16),
                        score=sc.cpu().numpy().astype(np.float16),
                        hm_block=hm_blk[b].astype(np.float16), sat_block=sat_blk[b].astype(np.float16),
                        focus_block=focus_blk[b].astype(np.float16)))
        if windows is not None:
            x0, y0 = windows[b]
            out[-1]["win_sum"] = np.float32(hm[b, 0, y0:y0 + WIN_H, x0:x0 + WIN_W].sum().item())
    return out, hm.cpu().numpy()[:, 0]


def crosscheck(hm_np, gpu_out, threshold, nms_px, log):
    """Compare GPU peaks with peaks.extract_peaks on the same heatmaps (first batch only)."""
    from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
    for h, o in zip(hm_np, gpu_out):
        ref = extract_peaks(h, threshold, nms_px)
        hs = gaussian_filter(h, 1.0)
        size = 2 * int(round(nms_px)) + 1
        rng = maximum_filter(hs, size=size, mode="nearest") - minimum_filter(hs, size=size, mode="nearest")
        ri = np.clip(np.round(ref).astype(int), 0, None)
        ref = ref[rng[ri[:, 1], ri[:, 0]] >= PLATEAU_RANGE] if len(ref) else ref
        got = np.stack([o["x"], o["y"]], 1).astype(float)
        if len(ref) and len(got):
            from scipy.spatial import cKDTree
            d, _ = cKDTree(ref).query(got, k=1)
            agree = float((d <= 1.0).mean())
        else:
            agree = float(len(ref) == len(got))
        log(f"crosscheck: cpu peaks {len(ref)}, gpu peaks {len(got)}, gpu within 1 px of a cpu peak {agree:.4f}")


def consolidate(det_dir, n, out_path, log, sigma=None):
    xs, ys, ss, offs = [], [], [], [0]
    win_sum = []
    hm_blk = np.zeros((n, FRAME_SHAPE[0] // BLOCK, FRAME_SHAPE[1] // BLOCK), np.float16)
    fo_blk = np.zeros_like(hm_blk)
    sa_blk = np.zeros_like(hm_blk)
    for i in range(n):
        z = np.load(os.path.join(det_dir, f"{i:05d}.npz"))
        xs.append(z["x"]); ys.append(z["y"]); ss.append(z["score"])
        offs.append(offs[-1] + len(z["x"]))
        hm_blk[i] = z["hm_block"]; fo_blk[i] = z["focus_block"]; sa_blk[i] = z["sat_block"]
        if "win_sum" in z:
            win_sum.append(float(z["win_sum"]))
    extra = dict(win_sum=np.array(win_sum, np.float32), sigma=np.float32(sigma)) if len(win_sum) == n else {}
    np.savez_compressed(out_path, x=np.concatenate(xs), y=np.concatenate(ys), score=np.concatenate(ss),
                        offsets=np.array(offs, np.int64), hm_block=hm_blk, focus_block=fo_blk, sat_block=sa_blk,
                        **extra)
    log(f"consolidated {n} frames, {offs[-1]} detections -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(HERE, "../../../results/detector_20260913/s1/best_by_val.pt"))
    ap.add_argument("--out", default=os.path.join(HERE, "../../../results/detector_20260913/s2"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--windows", default=None,
                    help="features.csv with x0_winfocus/y0_winfocus: save the heatmap sum over that window per frame")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    det_dir = os.path.join(args.out, "det")
    os.makedirs(det_dir, exist_ok=True)
    log_f = open(os.path.join(args.out, "infer.log"), "a")

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        log_f.write(line + "\n"); log_f.flush(); print(line, flush=True)

    device = torch.device("cuda")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = HeatmapNet(pretrained=False)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    threshold, nms_px = float(ck["threshold"]), float(ck["nms_px"])
    log(f"ckpt iter {ck['iter']} threshold {threshold} nms_px {nms_px:.3f} batch {args.batch}")

    index = pd.read_csv(CACHE_INDEX)
    n = len(index) if args.limit is None else min(args.limit, len(index))
    mm = np.memmap(CACHE_NPY, dtype=np.uint8, mode="r", shape=(len(index), *FRAME_SHAPE))
    windows = None
    if args.windows:
        wf = pd.read_csv(args.windows, usecols=["idx", "x0_winfocus", "y0_winfocus"]).set_index("idx")
        windows = np.stack([wf.loc[np.arange(len(index)), "x0_winfocus"], wf.loc[np.arange(len(index)), "y0_winfocus"]], 1).astype(int)
        log(f"direct-count windows from {args.windows}")
    todo = [i for i in range(n) if not os.path.exists(os.path.join(det_dir, f"{i:05d}.npz"))]
    log(f"{n} frames, {len(todo)} to do")

    pool = ThreadPoolExecutor(4)
    t0, done, checked = time.time(), 0, False
    for s in range(0, len(todo), args.batch):
        ids = todo[s:s + args.batch]
        frames = np.stack([np.asarray(mm[i]) for i in ids])
        outs, hm_np = run_batch(model, frames, device, threshold, nms_px,
                                windows=None if windows is None else windows[ids])
        if not checked:
            crosscheck(hm_np, outs, threshold, nms_px, log)
            checked = True
        for i, o in zip(ids, outs):
            pool.submit(np.savez_compressed, os.path.join(det_dir, f"{i:05d}.npz"), **o)
        done += len(ids)
        if done % 200 < args.batch or done == len(todo):
            el = time.time() - t0
            log(f"{done}/{len(todo)} frames, {el / done:.3f} s/frame, eta {(len(todo) - done) * el / done / 60:.1f} min")
    pool.shutdown(wait=True)
    el = time.time() - t0
    json.dump(dict(n=n, n_new=len(todo), seconds=el, s_per_frame=el / max(len(todo), 1), threshold=threshold,
                   nms_px=nms_px, batch=args.batch), open(os.path.join(args.out, "infer_timing.json"), "w"), indent=1)
    if len(glob.glob(os.path.join(det_dir, "*.npz"))) >= n:
        consolidate(det_dir, n, os.path.join(args.out, "detections.npz"), log, sigma=float(ck.get("sigma", 3.0)))


if __name__ == "__main__":
    main()
