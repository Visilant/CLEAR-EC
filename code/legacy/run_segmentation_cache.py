#!/usr/bin/env python3
"""Run Cellpose segmentation once per image and cache label masks."""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.config import SegConfig, config_hash
from src.data.segment import resolve_indices, run_segmentation_shard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cache Cellpose masks for CLEAR-EC.")
    parser.add_argument("--cache_dir", type=str, default="../data/cache")
    parser.add_argument(
        "--split",
        type=str,
        default="all",
        choices=["train", "val", "test", "all"],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0, help="GPU id for single-process mode.")
    parser.add_argument(
        "--gpus",
        type=str,
        default="",
        help="Comma-separated GPU ids for multi-process sharding (e.g. 0,1).",
    )
    parser.add_argument("--model_type", type=str, default="cyto", choices=["cyto", "cyto2", "nuclei"])
    parser.add_argument("--diameter", type=float, default=None)
    parser.add_argument("--flow_threshold", type=float, default=0.4)
    parser.add_argument("--cellprob_threshold", type=float, default=0.0)
    parser.add_argument("--min_size", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--no_tile", action="store_true")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip indices whose mask file already exists (default: True).",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Recompute masks even when cache files exist.",
    )
    return parser


def _seg_config_from_args(args) -> SegConfig:
    return SegConfig(
        model_type=args.model_type,
        diameter=args.diameter,
        flow_threshold=args.flow_threshold,
        cellprob_threshold=args.cellprob_threshold,
        min_size=args.min_size,
        tile=not args.no_tile,
        net_avg=False,
        batch_size=args.batch_size,
    )


def _worker(kwargs: dict) -> None:
    run_segmentation_shard(**kwargs)


def main() -> None:
    args = build_parser().parse_args()
    code_root = Path(__file__).resolve().parents[1]
    cache_dir = (code_root / args.cache_dir).resolve()
    limit = args.limit if args.limit > 0 else None
    skip_existing = args.skip_existing and not args.no_skip_existing

    seg_config = _seg_config_from_args(args)
    seg_hash = config_hash(seg_config)
    print(f"SegConfig hash: {seg_hash}")

    indices = resolve_indices(cache_dir, args.split, limit=limit)
    print(f"Processing {len(indices)} images (split={args.split})")

    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",") if g.strip() != ""]
    else:
        gpu_ids = [args.gpu]

    num_shards = len(gpu_ids)
    if num_shards == 1:
        run_segmentation_shard(
            cache_dir=cache_dir,
            seg_config=seg_config,
            gpu_id=gpu_ids[0],
            shard_id=0,
            num_shards=1,
            indices=indices,
            skip_existing=skip_existing,
        )
        return

    ctx = mp.get_context("spawn")
    processes = []
    for shard_id, gpu_id in enumerate(gpu_ids):
        kwargs = dict(
            cache_dir=cache_dir,
            seg_config=seg_config,
            gpu_id=gpu_id,
            shard_id=shard_id,
            num_shards=num_shards,
            indices=indices,
            skip_existing=skip_existing,
        )
        p = ctx.Process(target=_worker, args=(kwargs,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
        if p.exitcode != 0:
            raise RuntimeError(f"Segmentation worker failed with exit code {p.exitcode}")


if __name__ == "__main__":
    main()
