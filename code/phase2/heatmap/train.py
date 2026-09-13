"""Train the heatmap detector on the 19 train overlays; select checkpoint and threshold on the 2 val overlays."""
import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from data import build_samples, load_overlays, make_batch, train_median_nn  # noqa: E402
from eval_overlays import evaluate, summary  # noqa: E402
from model import HeatmapNet, masked_mse, predict_heatmap  # noqa: E402

COUNT_TOL = 0.05


def select_key(r):
    """Gate-aligned val selection: feasible count ratio first, then recall, then F1."""
    return (abs(r["count_ratio"] - 1) <= COUNT_TOL, r["recall"], r["f1"])


THRESHOLDS = np.round(np.concatenate([np.arange(0.02, 0.10, 0.02), np.arange(0.10, 0.61, 0.05)]), 2)


def val_scores(model, val_samples, nms_px, device):
    heatmaps = {s.image_id: predict_heatmap(model, s.frame, device) for s in val_samples}
    rows = []
    for t in THRESHOLDS:
        s = summary(evaluate(val_samples, heatmaps, float(t), nms_px))
        rows.append(dict(threshold=float(t), **s))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/visilant/CLEAR-EC/results/detector_20260913/s1")
    ap.add_argument("--iters", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--pos-weight", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.998)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")

    nms_px = 0.5 * train_median_nn(load_overlays())
    samples = build_samples(sigma=args.sigma, splits=("train", "val"))
    train_s = [s for s in samples.values() if s.split == "train"]
    val_s = [s for s in samples.values() if s.split == "val"]
    print(f"train overlays {len(train_s)} ({sum(len(s.dots) for s in train_s)} clicks), "
          f"val {len(val_s)}, nms_px {nms_px:.2f}")

    model = HeatmapNet(pretrained=True).to(device)
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, betas=(0.9, 0.99))

    def lr_at(it):
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        p = (it - args.warmup) / max(1, args.iters - args.warmup)
        return args.lr * 0.5 * (1 + np.cos(np.pi * p))

    best = dict(f1=-1.0, recall=-1.0, count_ratio=0.0)
    log = []
    t0 = time.time()
    run_loss = 0.0
    for it in range(args.iters):
        for g in opt.param_groups:
            g["lr"] = lr_at(it)
        imgs, tgts, msks = make_batch(train_s, args.batch, args.crop, rng)
        x = torch.from_numpy(imgs).to(device, non_blocking=True)
        y = torch.from_numpy(tgts).to(device, non_blocking=True)
        m = torch.from_numpy(msks).to(device, non_blocking=True)
        model.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(x)
        loss = masked_mse(logits, y, m, args.pos_weight)
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
        run_loss += loss.item()
        if (it + 1) % 50 == 0:
            print(f"it {it+1:5d} loss {run_loss/50:.5f} lr {lr_at(it):.2e} {time.time()-t0:.0f}s", flush=True)
            run_loss = 0.0
        if (it + 1) % args.eval_every == 0 or it + 1 == args.iters:
            for name, net in (("ema", ema), ("raw", model)):
                net.eval()
                rows = val_scores(net, val_s, nms_px, device)
                top = max(rows, key=select_key)
                log.append(dict(iter=it + 1, weights=name, **top))
                print(f"  val[{name}] it {it+1}: best thr {top['threshold']:.2f} F1 {top['f1']:.4f} "
                      f"R {top['recall']:.4f} P {top['precision']:.4f} ratio {top['count_ratio']:.3f}", flush=True)
                if select_key(top) > select_key(best):
                    best = dict(f1=top["f1"], recall=top["recall"], count_ratio=top["count_ratio"],
                                iter=it + 1, weights=name, threshold=top["threshold"], val_rows=rows)
                    torch.save(dict(model=net.state_dict(), iter=it + 1, weights=name,
                                    threshold=float(top["threshold"]), nms_px=float(nms_px),
                                    sigma=args.sigma, args=vars(args), val_summary=top),
                               os.path.join(args.out, "best_by_val.pt"))
    elapsed = time.time() - t0
    print(f"done in {elapsed/60:.1f} min; selected iter {best['iter']} ({best['weights']}) threshold "
          f"{best['threshold']:.2f}: val recall {best['recall']:.4f} F1 {best['f1']:.4f} ratio {best['count_ratio']:.3f}")
    json.dump(dict(best={k: v for k, v in best.items() if k != "val_rows"}, val_threshold_sweep=best["val_rows"],
                   log=log, elapsed_s=elapsed, args=vars(args)),
              open(os.path.join(args.out, "train_log.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
