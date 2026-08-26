#!/usr/bin/env python3
"""Orchestrator for CLEAR-EC exploratory data analysis phases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

from eda._cache import add_common_args, ensure_cache_exists, resolve_path
from eda import phase1_labels, phase2_images, phase3_roi, phase4_baseline

PHASE_MODULES = {
    "1": phase1_labels,
    "2": phase2_images,
    "3": phase3_roi,
    "4": phase4_baseline,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CLEAR-EC EDA phases (labels, images, ROI, baseline)."
    )
    add_common_args(parser)
    parser.add_argument(
        "--phases",
        type=str,
        default="1,2,3,4",
        help="Comma-separated phase numbers to run (1-4).",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="../data/train_mha",
        help="MHA directory used when building a missing cache.",
    )
    parser.add_argument(
        "--build_limit",
        type=int,
        default=0,
        help="If cache is missing, build with this many images (0 = full).",
    )
    return parser


def _parse_phases(value: str) -> list[str]:
    phases = [p.strip() for p in value.split(",") if p.strip()]
    unknown = [p for p in phases if p not in PHASE_MODULES]
    if unknown:
        raise ValueError(f"Unknown phases: {unknown}; choose from {list(PHASE_MODULES)}")
    return phases


def main() -> None:
    args = build_parser().parse_args()
    cache_dir = resolve_path(args.cache_dir)
    labels_csv = resolve_path(args.labels_csv)
    build_limit = args.build_limit if args.build_limit > 0 else None

    ensure_cache_exists(
        cache_dir=cache_dir,
        labels_csv=labels_csv,
        data_dir=resolve_path(args.data_dir),
        seed=args.seed,
        limit=build_limit,
    )

    phases = _parse_phases(args.phases)
    for phase_id in phases:
        print(f"\n=== Running phase {phase_id} ===")
        PHASE_MODULES[phase_id].run(args)


if __name__ == "__main__":
    main()
