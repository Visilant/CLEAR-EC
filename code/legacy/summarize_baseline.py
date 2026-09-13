#!/usr/bin/env python3
"""Summarize overnight experiment manifest into REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize CLEAR-EC overnight results.")
    parser.add_argument(
        "--manifest",
        type=str,
        default="../results/overnight/manifest.json",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="../results/overnight",
    )
    parser.add_argument("--cache_dir", type=str, default="../data/cache")
    parser.add_argument("--gt_csv", type=str, default="../data/final_train_ids.csv")
    return parser


def _fmt_mape(block: dict | None) -> str:
    if not block:
        return "—"
    mean = block.get("mean")
    if mean is None:
        return "—"
    extra = ""
    if "ci_low" in block and "ci_high" in block:
        extra = f" [{block['ci_low']:.2f}, {block['ci_high']:.2f}]"
    return (
        f"mean {mean:.2f}%{extra} "
        f"(CD {block.get('CD', float('nan')):.2f}, "
        f"CV {block.get('CV', float('nan')):.2f}, "
        f"HEX {block.get('HEX', float('nan')):.2f})"
    )


def main() -> None:
    args = build_parser().parse_args()
    code_root = Path(__file__).resolve().parents[1]
    manifest_path = (code_root / args.manifest).resolve()
    results_dir = (code_root / args.results_dir).resolve()

    with open(manifest_path) as f:
        manifest = json.load(f)

    results_dir.mkdir(parents=True, exist_ok=True)
    git = manifest.get("git", {})
    splits = manifest.get("split_summary", {})
    configs = pd.DataFrame(manifest.get("configs", []))
    if not configs.empty:
        comparison_path = results_dir / "config_comparison.csv"
        configs.to_csv(comparison_path, index=False)
    else:
        comparison_path = results_dir / "config_comparison.csv"

    lines = [
        "# CLEAR-EC Fair Overnight Report",
        "",
        f"Started: {manifest.get('started_at', '—')}",
        f"Finished: {manifest.get('finished_at', '—')}",
        f"Dry run: {manifest.get('dry_run', False)}",
        f"Branch: `{git.get('branch', '—')}` @ `{git.get('commit', '—')[:12] if git.get('commit') else '—'}`",
        f"Dirty tree: {git.get('dirty', '—')}",
        f"Test first accessed: {manifest.get('test_access_at', 'not yet')}",
        "",
        "## Protocol",
        "",
        "- `train`: fit only",
        "- `val`: configuration / checkpoint / alpha selection",
        "- `test`: scored once after every arm is frozen",
        "",
        f"Splits: train={splits.get('train', '—')} "
        f"val={splits.get('val', '—')} test={splits.get('test', '—')}",
        f"Slides: train={splits.get('n_slides_train', '—')} "
        f"val={splits.get('n_slides_val', '—')} test={splits.get('n_slides_test', '—')}",
        "",
        "## Frozen Cellpose config",
        "",
        f"- Name: **{manifest.get('best_seg_name', '—')}**",
        f"- Hash: `{manifest.get('best_seg_hash', '—')}`",
        f"- Crop: **{manifest.get('best_crop_frac', '—')}**",
        f"- Val mean error: {manifest.get('best_val_mean_error_pct', '—')}",
        "",
    ]

    test_scores = manifest.get("test_scores") or {}
    if test_scores:
        lines.extend(["## Test scores (one-shot)", ""])
        if "cellpose_baseline" in test_scores:
            lines.append(f"- **Cellpose baseline**: {_fmt_mape(test_scores['cellpose_baseline'])}")
        if "ridge_calibration" in test_scores:
            lines.append(
                f"- **Ridge calibration**: {_fmt_mape(test_scores['ridge_calibration'])}"
            )
        if "regression_cnn" in test_scores:
            lines.append("- **Regression CNN**")
            for row in test_scores["regression_cnn"]:
                lines.append(f"  - seed {row.get('seed')}: {_fmt_mape(row)}")
        lines.append("")
    elif manifest.get("dry_run"):
        lines.extend(
            [
                "## Test scores",
                "",
                "Dry run: GPU arms were not launched. Manifest records resolved splits, git state, and protocol only.",
                "",
            ]
        )

    if not configs.empty:
        lines.extend(["## Val tuning table", "", "```", configs.to_string(index=False), "```", ""])

    lines.extend(
        [
            "## Artifacts",
            "",
            f"- Manifest: `{manifest_path}`",
            f"- Config comparison: `{comparison_path}`",
            f"- Log: `{results_dir / 'overnight.log'}`",
        ]
    )

    report_path = results_dir / "REPORT.md"
    report_path.write_text("\n".join(lines) + "\n")
    print(f"Wrote {report_path}")
    if not configs.empty:
        print(f"Wrote {comparison_path} ({len(configs)} rows)")


if __name__ == "__main__":
    main()
