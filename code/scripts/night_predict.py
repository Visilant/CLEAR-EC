#!/usr/bin/env python3
"""Score a saved regression checkpoint on any index set, with optional flip TTA.

Examples
  night_predict.py --ckpt_dir R/whole_relative/regression_cnn/seed_123 --which best --indices val --tta none
  night_predict.py --ckpt_dir R/fold0/regression_cnn/seed_123 --which last --indices fold:0/5 --tta flips
Writes <out> CSV (idx, ID, CD, CV, HEX) and prints per-metric MAPE when labels cover the indices.
TTA views: identity, hflip, vflip, hvflip (== 180 deg rotation); combined by geometric mean.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.cache import open_image_cache  # noqa: E402
from src.data.splits import load_split  # noqa: E402
from src.training.common import METRICS, labels_for_indices, score_by_id, denormalize_targets  # noqa: E402
from src.training.regression_cnn import (  # noqa: E402
    RegressionConfig, _inverse_target_space, _slide_group_folds, build_regression_model,
)

VIEWS = {"none": [(False, False)], "flips": [(False, False), (True, False), (False, True), (True, True)]}


def resolve_indices(cache_dir: Path, spec: str) -> list[int]:
    if spec in {"train", "val", "test"}:
        return load_split(cache_dir, spec)
    if spec == "all":
        return sorted(set(load_split(cache_dir, "train")) | set(load_split(cache_dir, "val")) | set(load_split(cache_dir, "test")))
    if spec.startswith("fold:"):
        k, n = spec[5:].split("/")
        return _slide_group_folds(cache_dir, int(n), seed=42)[int(k)]
    return [int(x) for x in Path(spec).read_text().split()]


def load(ckpt_dir: Path, which: str, device: torch.device):
    path = ckpt_dir / ("best_model.pt" if which == "best" else "last.pt")
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = dict(ckpt["config"])
    config.setdefault("uint8_inputs", False)
    cfg = RegressionConfig(**config)
    model = build_regression_model(cfg, load_pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["target_stats"], cfg, int(ckpt.get("epoch", -1))


@torch.no_grad()
def predict(model, stats, cfg, memmap, indices, device, views, batch_size=16):
    preds = []
    for start in range(0, len(indices), batch_size):
        chunk = indices[start:start + batch_size]
        x = torch.from_numpy(np.stack([memmap[i] for i in chunk])).unsqueeze(1).to(device)
        if not cfg.uint8_inputs:
            x = x.float() / 255.0
        logs = []
        for hflip, vflip in views:
            xv = x
            if hflip:
                xv = torch.flip(xv, dims=[-1])
            if vflip:
                xv = torch.flip(xv, dims=[-2])
            out = model(xv).float().cpu().numpy()
            out = _inverse_target_space(denormalize_targets(out, stats), cfg.target_space)
            logs.append(np.log(np.clip(out, 1e-6, None)))
        preds.append(np.exp(np.mean(logs, axis=0)))
    return np.concatenate(preds)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_dir", required=True)
    ap.add_argument("--which", choices=["best", "last"], default="best")
    ap.add_argument("--indices", required=True, help="train|val|test|all|fold:k/n|<file of idx>")
    ap.add_argument("--tta", choices=list(VIEWS), default="none")
    ap.add_argument("--cache_dir", default="/home/visilant/CLEAR-EC/data/cache")
    ap.add_argument("--labels_csv", default="/home/visilant/CLEAR-EC/data/final_train_ids.csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    cache_dir = Path(args.cache_dir)
    memmap, _ = open_image_cache(cache_dir)
    indices = resolve_indices(cache_dir, args.indices)
    frame = labels_for_indices(cache_dir, Path(args.labels_csv), indices)
    model, stats, cfg, epoch = load(Path(args.ckpt_dir), args.which, device)
    arr = predict(model, stats, cfg, memmap, list(frame["idx"]), device, VIEWS[args.tta])
    out = pd.DataFrame({"idx": frame["idx"], "ID": frame["ID"]})
    for j, m in enumerate(METRICS):
        out[m] = arr[:, j]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    scores = score_by_id(out, frame, expected_ids=frame.ID)
    summary = {"ckpt_dir": args.ckpt_dir, "which": args.which, "epoch": epoch, "indices": args.indices,
               "n": len(out), "tta": args.tta, "mape": scores}
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
