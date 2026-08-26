# CLEAR-EC Exploratory Data Analysis

Modular EDA pipeline under `code/eda/` that reads from the memmap cache in `src/data/` and writes figures and CSVs to `results/eda/`.

## Prerequisite — memmap cache

All phases except pure label stats assume the cache exists:

```bash
cd code
python scripts/build_cache.py \
  --data_dir ../data/train_mha \
  --labels_csv ../data/final_train_ids.csv \
  --cache_dir ../data/cache
```

This produces:

```
data/cache/
├── images_u8.npy      # (N, 972, 1296) uint8 memmap
├── index.csv          # idx, ID, slide_id, mha_path
└── splits.json        # train/val/test index lists (slide-grouped, seed=42)
```

For smoke tests: `python scripts/build_cache.py --limit 500`

If the cache is missing, `run_eda.py` can build it automatically via `--build_limit`.

## Quick start

```bash
cd code

# All phases on val split (phase 4 defaults: val, limit 200)
python -m eda.run_eda --split val --limit 100

# Labels + images only
python -m eda.run_eda --phases 1,2

# Baseline error analysis (GPU for uncached masks)
python -m eda.run_eda --phases 4 --split val
```

Individual phases can also be run directly:

```bash
python -m eda.phase1_labels --split all
python -m eda.phase2_images --split val --limit 500
python -m eda.phase3_roi
python -m eda.phase4_baseline --split val --limit 200
```

## CLI defaults (relative to `code/`)

| Argument | Default | Notes |
|----------|---------|-------|
| `--cache_dir` | `../data/cache` | Memmap + index + splits |
| `--labels_csv` | `../data/final_train_ids.csv` | CD/CV/HEX ground truth |
| `--green_dir` | `../data/train_green` | Phase 3 green ROI overlays |
| `--output_dir` | `../results/eda` | All figure/CSV output |
| `--split` | `all` | `train` / `val` / `test` / `all` |
| `--limit` | `0` | Cap indices within split (phase 4 default: 200) |
| `--seed` | `42` | Subsampling and crop RNG seed |

## Splits

Splits are slide-level (no leakage): `splits.json` in the cache directory maps each image `idx` to `train`, `val`, or `test`. Use `--split` to restrict analysis to one partition.

Phase 2 uses stratified subsampling when `--limit` is set: samples are drawn evenly across CD quartiles within the chosen split.

## Phase 3 — ROI vs random crop

Green overlays in `train_green/` are rescaled renderings (720×540) of the full MHA (1296×972); phase 3 scales the extracted ROI bbox to full image coordinates before comparing IoU with the random crop. Overlay IDs not present in the memmap cache are loaded directly from `data/train_mha/`.

## Outputs

```
results/eda/
├── phase1_labels/
│   ├── summary_stats.csv
│   ├── split_summary.csv
│   ├── outlier_flags.csv          # if any outliers detected
│   ├── cd_cv_hex_histograms.png
│   ├── metric_correlation.png
│   └── cd_by_slide_boxplot.png
├── phase2_images/
│   ├── image_stats.csv            # idx for downstream joins
│   ├── intensity_histogram.png
│   ├── blur_vs_cd_scatter.png
│   └── metric_montage.png
├── phase3_roi/
│   ├── roi_vs_random_crop.csv     # up to 25 green-overlay rows
│   └── roi_overlay_samples.png
└── phase4_baseline/
    ├── predictions_{split}.csv
    ├── errors_{split}.csv
    ├── error_by_metric.png
    ├── pred_vs_actual.png
    ├── error_vs_blur.png          # joins phase 2 if available
    ├── error_vs_roi_iou.png       # joins phase 3 if available
    └── worst_cases/               # top-N error overlays per metric
```

## Phase 4 — mask cache

Phase 4 uses cached Cellpose masks keyed by `SegConfig` hash. Pre-compute masks:

```bash
python scripts/run_segmentation_cache.py \
  --cache_dir ../data/cache --split val --limit 200
```

Or let phase 4 call `get_or_compute_mask()` inline (requires CUDA). Once masks exist, metric-only re-runs are fast (same pattern as `scripts/sweep_metrics.py`).

## Dependencies

Uses packages from `code/requirements.txt` only. Phases 1–3 are CPU-only; phase 4 needs a GPU for initial mask caching.
