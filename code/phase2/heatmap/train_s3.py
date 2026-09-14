"""S3: fine-tune the S1 heatmap detector with weak count supervision from the training labels.

Each batch has two streams: (a) overlay crops with the dense Gaussian heatmap loss masked to the
click hull (exactly as S1); (b) train-split population frames, the S2 focus-chosen 538x408 window,
with a count loss: count = sum(sigmoid heatmap over the window) / (2 pi sigma^2), CD = count / window
area (mm2), loss = |CD - CD_label| / CD_label. lambda (weight of the count loss) is set so the two
losses are equal on the first batches. Checkpoint selection uses the 2 val overlays (detection F1,
threshold sweep) and a fixed 500-image subset of the val split (CD MAPE of the peak-based winfocus
readout and of the direct count readout). Val/test overlay IDs are excluded from the population
stream; val labels are read only for the selection subset; test labels are never read.
"""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import (CACHE_INDEX, CACHE_NPY, FRAME_SHAPE, LABELS_CSV, ROOT, UM_PER_PX,  # noqa: E402
                  build_samples, gaussian_heatmap, intensity_jitter, load_overlays, make_batch, train_median_nn)
from eval_overlays import load_model  # noqa: E402
from features import WIN_H, WIN_W, readout_in_window  # noqa: E402
from infer_all import PLATEAU_RANGE, blur, run_batch  # noqa: E402
from model import HeatmapNet, masked_mse  # noqa: E402
from train import THRESHOLDS, select_key, val_scores  # noqa: E402

SPLITS_JSON = os.path.join(ROOT, "data/cache/splits.json")
S2_FEATURES = os.path.join(ROOT, "results/detector_20260913/s2/features.csv")
WIN_AREA_MM2 = WIN_W * WIN_H * UM_PER_PX ** 2 / 1e6
PAD_W, PAD_H = 544, 416  # window padded to multiples of 32 with real pixels; count mask = 538x408


def gaussian_integral(sigma):
    return 2 * np.pi * sigma ** 2


def verify_gaussian_integral(sigma, rng, n=300, spacing=20.0):
    """Synthetic check: sum of the max-of-Gaussians target over n jittered points on a ~spacing grid
    divided by 2 pi sigma^2 must recover n (overlap between neighbours is negligible at 20 px / sigma 3)."""
    g = np.stack(np.meshgrid(np.arange(60, 1200, spacing), np.arange(60, 900, spacing)), -1).reshape(-1, 2)
    pts = g[rng.choice(len(g), n, replace=False)] + rng.uniform(-3, 3, (n, 2))
    hm = gaussian_heatmap(pts, sigma)
    est = hm.sum() / gaussian_integral(sigma)
    return float(est), n


class Population:
    """Train-split frames with labels and their S2 focus-chosen window."""

    def __init__(self, exclude_ids, rng):
        index = pd.read_csv(CACHE_INDEX)
        feats = pd.read_csv(S2_FEATURES, usecols=["idx", "ID", "x0_winfocus", "y0_winfocus"]).set_index("idx")
        labels = pd.read_csv(LABELS_CSV)
        labels["ID"] = labels["ID"].astype(str).str.strip()
        labels = labels.set_index("ID")
        train_idx = json.load(open(SPLITS_JSON))["train"]
        rows = []
        for i in train_idx:
            image_id = index.ID.iloc[i]
            if image_id in exclude_ids:
                continue
            rows.append((i, int(feats.loc[i, "x0_winfocus"]), int(feats.loc[i, "y0_winfocus"]),
                         float(labels.loc[image_id, "CD"])))
        self.rows = rows
        self.mm = np.memmap(CACHE_NPY, dtype=np.uint8, mode="r", shape=(len(index), *FRAME_SHAPE))
        self.rng = rng

    def batch(self, batch_size):
        imgs, msks, cds = [], [], []
        for _ in range(batch_size):
            i, x0, y0, cd = self.rows[int(self.rng.integers(len(self.rows)))]
            H, W = FRAME_SHAPE
            left = int(np.clip(x0 - (PAD_W - WIN_W) // 2, 0, W - PAD_W))
            top = int(np.clip(y0 - (PAD_H - WIN_H) // 2, 0, H - PAD_H))
            img = np.asarray(self.mm[i][top:top + PAD_H, left:left + PAD_W]).astype(np.float32) / 255.0
            mask = np.zeros((PAD_H, PAD_W), np.float32)
            mask[y0 - top:y0 - top + WIN_H, x0 - left:x0 - left + WIN_W] = 1.0
            if self.rng.random() < 0.5:
                img, mask = img[:, ::-1], mask[:, ::-1]
            if self.rng.random() < 0.5:
                img, mask = img[::-1, :], mask[::-1, :]
            imgs.append(intensity_jitter(img, self.rng)); msks.append(mask); cds.append(cd)
        imgs, msks = np.stack(imgs)[:, None], np.stack(msks)[:, None]
        k = int(self.rng.integers(0, 4))  # one rot90 per batch keeps the batch shape uniform
        if k:
            imgs, msks = np.rot90(imgs, k, axes=(2, 3)), np.rot90(msks, k, axes=(2, 3))
        return (np.ascontiguousarray(imgs, np.float32), np.ascontiguousarray(msks, np.float32),
                np.array(cds, np.float32))


def set_train_frozen_bn(model):
    """Train mode with BatchNorm frozen (running statistics, as at inference). With two streams of
    different content (overlay crops, population windows) forwarded separately, batch statistics differ
    per stream and from the running average, and the count integral is very sensitive to the sigmoid
    baseline: at it 500 of the first attempt the same window read 0.81x the label with batch statistics
    and 1.23x with running statistics. Freezing BN makes the trained function the evaluated function."""
    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def count_loss(logits, mask, cd_label, sigma, drop_plateaus_px=None):
    """Relative CD error of the differentiable count over the window; returns (loss, CD_pred).
    drop_plateaus_px (diagnostic only): zero saturated plateaus (S2's range test over the NMS window)."""
    pred = torch.sigmoid(logits.float())
    if drop_plateaus_px is not None:
        r = int(round(drop_plateaus_px))
        hm_s = blur(pred, 1.0)
        rng = F.max_pool2d(hm_s, 2 * r + 1, stride=1, padding=r) + F.max_pool2d(-hm_s, 2 * r + 1, stride=1, padding=r)
        pred = pred * (rng >= PLATEAU_RANGE).float()
    count = (pred * mask).sum(dim=(1, 2, 3)) / gaussian_integral(sigma)
    cd = count / WIN_AREA_MM2
    return (torch.abs(cd - cd_label) / cd_label).mean(), cd


class ValSubset:
    """Fixed 500-image subset of the val split: CD MAPE of the peak-based winfocus readout and of the
    direct count readout (S2 window). Val labels are read here only."""

    def __init__(self, n, seed, out):
        index = pd.read_csv(CACHE_INDEX)
        val_idx = np.array(json.load(open(SPLITS_JSON))["val"])
        sub = np.sort(np.random.default_rng(seed).choice(val_idx, n, replace=False))
        json.dump(dict(seed=seed, n=n, idx=sub.tolist()), open(os.path.join(out, "val_subset_idx.json"), "w"))
        feats = pd.read_csv(S2_FEATURES, usecols=["idx", "x0_winfocus", "y0_winfocus"]).set_index("idx")
        labels = pd.read_csv(LABELS_CSV)
        labels["ID"] = labels["ID"].astype(str).str.strip()
        labels = labels.set_index("ID")
        self.idx = sub
        self.x0 = feats.loc[sub, "x0_winfocus"].to_numpy(int)
        self.y0 = feats.loc[sub, "y0_winfocus"].to_numpy(int)
        self.cd = labels.loc[index.ID.iloc[sub], "CD"].to_numpy(float)
        self.mm = np.memmap(CACHE_NPY, dtype=np.uint8, mode="r", shape=(len(index), *FRAME_SHAPE))

    @torch.no_grad()
    def score(self, model, threshold, nms_px, sigma, device, batch=4):
        cd_win, cd_dir, sat = [], [], []
        for s in range(0, len(self.idx), batch):
            ids = self.idx[s:s + batch]
            frames = np.stack([np.asarray(self.mm[i]) for i in ids])
            outs, hm = run_batch(model, frames, device, threshold, nms_px)
            for k, o in enumerate(outs):
                x0, y0 = self.x0[s + k], self.y0[s + k]
                pts = np.stack([o["x"], o["y"]], 1).astype(float)
                sc = o["score"].astype(float)
                r = readout_in_window(pts, sc, x0, y0) if len(pts) >= 8 else dict(CD=np.nan)
                cd_win.append(r["CD"])
                cd_dir.append(hm[k, y0:y0 + WIN_H, x0:x0 + WIN_W].sum() / gaussian_integral(sigma) / WIN_AREA_MM2)
                sat.append(float(o["sat_block"].astype(float).mean()))
        cd_win, cd_dir = np.array(cd_win), np.array(cd_dir)
        ok = np.isfinite(cd_win)
        ape_win = np.abs(cd_win[ok] - self.cd[ok]) / self.cd[ok] * 100
        ape_dir = np.abs(cd_dir - self.cd) / self.cd * 100
        return dict(sub_mape_winfocus=float(ape_win.mean()), sub_mape_direct=float(ape_dir.mean()),
                    sub_n_valid=int(ok.sum()), sub_sat_frac=float(np.mean(sat)),
                    sub_ratio_winfocus=float(np.median(cd_win[ok] / self.cd[ok])),
                    sub_ratio_direct=float(np.median(cd_dir / self.cd)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(ROOT, "results/detector_20260913/s1/best_by_val.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results/detector_20260913/s3"))
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--batch-overlay", type=int, default=4)
    ap.add_argument("--batch-pop", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    ap.add_argument("--lam", type=float, default=None, help="count-loss weight; default: equalise at start")
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-subset", type=int, default=500)
    ap.add_argument("--val-seed", type=int, default=20260913)
    ap.add_argument("--min-val-f1", type=float, default=0.75, help="val-overlay F1 floor for selection")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")

    model, ck = load_model(args.init, device)
    sigma = float(ck["sigma"])
    nms_px = float(ck["nms_px"])
    est, n = verify_gaussian_integral(sigma, rng)
    print(f"gaussian integral check: sum/(2 pi sigma^2) = {est:.2f} for {n} synthetic points "
          f"(ratio {est / n:.4f})", flush=True)
    assert abs(est / n - 1) < 0.01

    overlays = load_overlays()
    exclude = {k for k, v in overlays.items() if v["split"] != "train"}
    assert nms_px == 0.5 * train_median_nn(overlays)
    samples = build_samples(sigma=sigma, splits=("train", "val"))
    train_s = [s for s in samples.values() if s.split == "train"]
    val_s = [s for s in samples.values() if s.split == "val"]
    pop = Population(exclude, rng)
    valsub = ValSubset(args.val_subset, args.val_seed, args.out)
    print(f"overlays train {len(train_s)} val {len(val_s)}; population frames {len(pop.rows)} "
          f"(train split, overlays of other splits excluded); val subset {len(valsub.idx)}", flush=True)

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.99))

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        p = (it - args.warmup) / max(1, args.iters - args.warmup)
        return args.lr * 0.5 * (1 + np.cos(np.pi * p))

    def to_dev(*arrs):
        return [torch.from_numpy(a).to(device, non_blocking=True) for a in arrs]

    def forward(net, imgs, tgts, msks, wimgs, wmsks, wcd, drop_plateaus=False):
        x, y, m = to_dev(imgs, tgts, msks)
        wx, wm, wc = to_dev(wimgs, wmsks, wcd)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            lo = net(x)
            lw = net(wx)
        l_hm = masked_mse(lo, y, m, args.pos_weight)
        l_cnt, cd_pred = count_loss(lw, wm, wc, sigma, nms_px if drop_plateaus else None)
        return l_hm, l_cnt, cd_pred

    # lambda: equalise the two losses on the first batches (initial weights); the plateau-masked count
    # loss is also measured (diagnostic: at init the excess window mass comes from broad low blobs,
    # not from the saturated background plateaus, so the two values nearly coincide).
    lam_info = {}
    if args.lam is None:
        model.eval()
        hs, cs, cm = [], [], []
        with torch.no_grad():
            for _ in range(16):
                imgs, tgts, msks = make_batch(train_s, args.batch_overlay, args.crop, rng)
                wimgs, wmsks, wcd = pop.batch(args.batch_pop)
                l_hm, l_cnt, _ = forward(model, imgs, tgts, msks, wimgs, wmsks, wcd)
                _, l_cnt_m, _ = forward(model, imgs, tgts, msks, wimgs, wmsks, wcd, drop_plateaus=True)
                hs.append(l_hm.item()); cs.append(l_cnt.item()); cm.append(l_cnt_m.item())
        lam_info = dict(heatmap_init=float(np.mean(hs)), count_init_raw=float(np.mean(cs)),
                        count_init_plateau_masked=float(np.mean(cm)),
                        lam_raw_equalised=float(np.mean(hs) / np.mean(cs)),
                        lam_masked_equalised=float(np.mean(hs) / np.mean(cm)))
        args.lam = lam_info["lam_raw_equalised"]
        print(f"initial losses: heatmap {np.mean(hs):.5f}, count raw {np.mean(cs):.4f} (plateau-masked "
              f"{np.mean(cm):.4f}) -> lambda {args.lam:.4f} (plateau-masked-equalised would be "
              f"{lam_info['lam_masked_equalised']:.4f})", flush=True)
    lam = args.lam

    # baseline scores of the initial checkpoint on the selection sets
    model.eval()
    rows0 = val_scores(model, val_s, nms_px, device)
    top0 = max(rows0, key=select_key)
    sub0 = valsub.score(model, float(top0["threshold"]), nms_px, sigma, device)
    log = [dict(iter=0, weights="init", **top0, **sub0)]
    print(f"init: val thr {top0['threshold']:.2f} F1 {top0['f1']:.4f} R {top0['recall']:.4f} ratio "
          f"{top0['count_ratio']:.3f} | subset MAPE winfocus {sub0['sub_mape_winfocus']:.2f} direct "
          f"{sub0['sub_mape_direct']:.2f} sat {sub0['sub_sat_frac']:.3f}", flush=True)

    def sel_key(r):
        feasible = r["f1"] >= args.min_val_f1
        return (feasible, -min(r["sub_mape_winfocus"], r["sub_mape_direct"]))

    best = None
    t0 = time.time()
    run = dict(hm=0.0, cnt=0.0, ratio=0.0)
    for it in range(args.iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        imgs, tgts, msks = make_batch(train_s, args.batch_overlay, args.crop, rng)
        wimgs, wmsks, wcd = pop.batch(args.batch_pop)
        set_train_frozen_bn(model)
        l_hm, l_cnt, cd_pred = forward(model, imgs, tgts, msks, wimgs, wmsks, wcd)
        loss = l_hm + lam * l_cnt
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        with torch.no_grad():
            d = min(args.ema, (1 + it) / (10 + it))
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            for be, bm in zip(ema.buffers(), model.buffers()):
                be.copy_(bm)
        run["hm"] += l_hm.item(); run["cnt"] += l_cnt.item()
        run["ratio"] += float((cd_pred.detach().cpu().numpy() / wcd).mean())
        if (it + 1) % 50 == 0:
            print(f"it {it+1:5d} hm {run['hm']/50:.5f} cnt {run['cnt']/50:.4f} (x{lam:.4f}) "
                  f"pred/label {run['ratio']/50:.3f} lr {lr_at(it):.2e} {time.time()-t0:.0f}s", flush=True)
            run = dict(hm=0.0, cnt=0.0, ratio=0.0)
        if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
            for name, net in (("ema", ema), ("raw", model)):
                net.eval()
                rows = val_scores(net, val_s, nms_px, device)
                top = max(rows, key=select_key)
                sub = valsub.score(net, float(top["threshold"]), nms_px, sigma, device)
                row = dict(iter=it + 1, weights=name, **top, **sub)
                log.append(row)
                print(f"  [{name}] it {it+1}: val thr {top['threshold']:.2f} F1 {top['f1']:.4f} R {top['recall']:.4f} "
                      f"ratio {top['count_ratio']:.3f} | subset MAPE winfocus {sub['sub_mape_winfocus']:.2f} "
                      f"(ratio {sub['sub_ratio_winfocus']:.3f}) direct {sub['sub_mape_direct']:.2f} "
                      f"(ratio {sub['sub_ratio_direct']:.3f}) sat {sub['sub_sat_frac']:.3f}", flush=True)
                state = dict(model=net.state_dict(), iter=it + 1, weights=name, threshold=float(top["threshold"]),
                             nms_px=nms_px, sigma=sigma, lam=lam, args=vars(args), val_summary=top, val_subset=sub,
                             init=args.init)
                torch.save(state, os.path.join(args.out, f"ckpt_{it+1:05d}_{name}.pt"))
                if best is None or sel_key(row) > sel_key(best):
                    best = row
                    torch.save(state, os.path.join(args.out, "best_by_val.pt"))
            json.dump(dict(best=best, log=log, lam=lam, lam_info=lam_info,elapsed_s=time.time() - t0, args=vars(args)),
                      open(os.path.join(args.out, "train_log.json"), "w"), indent=1, default=float)
    elapsed = time.time() - t0
    print(f"done in {elapsed/60:.1f} min; selected iter {best['iter']} ({best['weights']}) thr {best['threshold']:.2f}: "
          f"val F1 {best['f1']:.4f} R {best['recall']:.4f}; subset MAPE winfocus {best['sub_mape_winfocus']:.2f} "
          f"direct {best['sub_mape_direct']:.2f}", flush=True)
    json.dump(dict(best=best, log=log, lam=lam, lam_info=lam_info,elapsed_s=elapsed, args=vars(args)),
              open(os.path.join(args.out, "train_log.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
