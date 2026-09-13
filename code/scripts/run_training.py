#!/usr/bin/env python3
"""Train one regression run (or one per seed) from CLI flags or a JSON config.

The runner (experiments/run.py) passes --config_json <job>/config.json so the RegressionConfig
travels as data; the flag form below is generated from the dataclass fields for humans.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.cache import open_image_cache
from src.training.common import resolve_path, split_summary, update_manifest
from src.training.config import RegressionConfig
from src.training.train import train_regression_cnn

VALID_METHODS = ("regression",)
# Fields whose CLI flag is the negation of the field (defaults are True).
INVERTED_FLAGS = {"pretrained": "no_pretrained", "uint8_inputs": "float_inputs"}
# Fields not exposed as flags: seed comes from --seeds, architecture (NAS) was removed.
SKIPPED_FIELDS = {"seed", "architecture"}
CHOICES = {
    "loss": ["huber", "mse", "relative"],
    "input_mode": ["whole", "fixed", "quality"],
    "sched": ["none", "cosine"],
    "target_space": ["linear", "log"],
    "normalization": ["batch", "group"],
    "cd_head": ["gap", "density"],
}
HELP = {
    "model": "small | convnext_tiny | convnext_small | convnext_base | timm:<name>",
    "ema": "EMA decay (0 disables).",
    "sched_epochs": "Cosine budget; hold its final LR for remaining training epochs.",
    "save_epochs": "Comma-separated epochs to retain; also saves recovery state each epoch.",
    "crop_scale": "<1 enables scale-preserving random crops with this min side fraction.",
    "clip_grad": "Gradient-norm clipping (0 disables). Use 1.0 for ConvNeXt-V2.",
    "exclude_idx_file": "Whitespace-separated cache indices dropped from training only.",
    "cd_head": "gap (frozen default) | density: CD as the summed softplus 1x1-conv map (ConvNeXt whole-image).",
    "loss_trim": "Relative loss: drop the ceil(loss_trim * batch) largest per-sample errors when batch >= 4.",
    "float_inputs": "Legacy FP32 CPU preprocessing (default: uint8 inputs).",
}


def parse_methods(value: str) -> list[str]:
    methods = [item.strip() for item in value.split(",") if item.strip()]
    if not methods:
        raise SystemExit("No training methods specified.")
    unknown = [item for item in methods if item not in VALID_METHODS]
    if unknown:
        raise SystemExit(f"Unknown or removed method(s): {unknown}. Valid: {', '.join(VALID_METHODS)} "
                         "(ridge calibration lives in code/legacy/).")
    return methods


def _method_arg(value: str) -> str:
    parse_methods(value)
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLEAR-EC regression training.")
    parser.add_argument("--method", type=_method_arg, default="regression", help="regression (only)")
    parser.add_argument("--config_json", type=str, default="",
                        help="JSON file with a 'config' object (asdict(RegressionConfig)); overrides every flag below.")
    parser.add_argument("--gpu", type=int, default=0, help="CUDA device index (after CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--cache_dir", type=str, default="../data/cache")
    parser.add_argument("--labels_csv", type=str, default="../data/final_train_ids.csv")
    parser.add_argument("--results_dir", type=str, default="../results/training",
                        help="Root for regression_cnn/seed_<s>/ outputs and manifest.json.")
    parser.add_argument("--seeds", type=str, default="42", help="Comma-separated seeds (default: 42).")
    parser.add_argument("--min_cache_count", type=int, default=9000,
                        help="Warn if fewer cached images than this.")
    group = parser.add_argument_group("RegressionConfig fields")
    for f in fields(RegressionConfig):
        if f.name in SKIPPED_FIELDS:
            continue
        if f.name in INVERTED_FLAGS:
            flag = INVERTED_FLAGS[f.name]
            group.add_argument(f"--{flag}", action="store_true", help=HELP.get(flag, f"Disable {f.name}."))
            continue
        if f.name == "save_epochs":
            group.add_argument("--save_epochs", type=str, default="", help=HELP["save_epochs"])
            continue
        if isinstance(f.default, bool):
            group.add_argument(f"--{f.name}", action="store_true", help=HELP.get(f.name, ""))
            continue
        group.add_argument(f"--{f.name}", type=type(f.default), default=f.default,
                           choices=CHOICES.get(f.name), help=HELP.get(f.name, f"default {f.default}"))
    return parser


def config_from_args(args: argparse.Namespace, seed: int) -> RegressionConfig:
    values: dict = {}
    for f in fields(RegressionConfig):
        if f.name in SKIPPED_FIELDS:
            continue
        if f.name in INVERTED_FLAGS:
            values[f.name] = not getattr(args, INVERTED_FLAGS[f.name])
        elif f.name == "save_epochs":
            values[f.name] = tuple(int(e) for e in args.save_epochs.split(",") if e.strip())
        else:
            values[f.name] = getattr(args, f.name)
    return RegressionConfig(seed=seed, **values)


def _parse_seeds(value: str) -> list[int]:
    seeds = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not seeds:
        raise SystemExit("No regression seeds specified.")
    return seeds


def main() -> None:
    args = build_parser().parse_args()
    parse_methods(args.method)
    code_root = Path(__file__).resolve().parents[1]
    cache_dir = resolve_path(code_root, args.cache_dir)
    labels_csv = resolve_path(code_root, args.labels_csv)
    results_root = resolve_path(code_root, args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    _, index_df = open_image_cache(cache_dir)
    n_cached = len(index_df)
    summary = split_summary(cache_dir)
    print(f"Cache: {n_cached} images | splits: {summary}")
    if n_cached < args.min_cache_count:
        print(f"WARNING: cache has {n_cached} images (< {args.min_cache_count}). Training will still run.")

    if args.config_json:
        payload = json.loads(Path(args.config_json).read_text())
        configs = [RegressionConfig(**payload["config"])]
    else:
        configs = [config_from_args(args, seed) for seed in _parse_seeds(args.seeds)]

    errors: dict[str, str] = {}
    for cfg in configs:
        run_dir = results_root / "regression_cnn" / f"seed_{cfg.seed}"
        try:
            train_regression_cnn(cache_dir, labels_csv, run_dir, gpu=args.gpu, config=cfg)
        except Exception as exc:  # keep going for the other seeds, report at the end
            errors[f"seed_{cfg.seed}"] = str(exc)
            print(f"[seed {cfg.seed}] ERROR: {exc}")

    update_manifest(results_root, {"errors": errors, "n_cached": n_cached, "split_summary": summary,
                                   "cache_dir": str(cache_dir), "labels_csv": str(labels_csv)})
    if errors:
        raise SystemExit(f"Training finished with errors: {errors}")
    print(f"Training complete -> {results_root}")


if __name__ == "__main__":
    main()
