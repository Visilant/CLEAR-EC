# CLEAR-EC Challenge

**CLEAR-EC**: **C**orneal **L**earning for **E**ndothelial **A**ssessment and **R**eview using AI for **E**ndothelial **C**ount

This repository provides a **baseline pipeline** and a **ready-to-submit algorithm container** for the [CLEAR-EC challenge](https://clear-ec.grand-challenge.org/) on Grand Challenge. The goal is to estimate clinically relevant endothelial metrics from corneal endothelial microscopy images:

1. **Cell segmentation**
2. **Instance-level cell analysis**
3. **Metric calculation** — **Cell Density (CD)**, **Coefficient of Variation (CV)**, and **Hexagonality (HEX)**

The baseline is a transparent starting point: not the strongest possible solution, but a working reference showing how a raw image becomes the three challenge outputs. Everything in [`code/`](code/) is set up so you can **swap in your own method and submit it as a Docker container** without rebuilding the plumbing.

---

## The baseline pipeline

Corneal endothelial assessment is important for evaluating corneal graft quality. The baseline formulates the task as:

**Image → Cell Segmentation → Cell Morphology → Clinical Metrics**

which offers interpretability and a direct link between the model's masks and the downstream measurements.

### Step 1 — Cell segmentation
Segment the corneal endothelial image to identify individual cells and obtain per-cell masks. The baseline uses **Cellpose v1.0** (`cyto` model), vendored in [`code/src/models/cellpose/`](code/src/models/cellpose/).

### Step 2 — Cell morphology extraction
From the masks, derive per-cell geometric properties: area, perimeter, number of neighbors/sides, and centroid. These feed the summary metrics.

### Step 3 — Metric calculation
Three metrics are computed from the masks (restricted to the metric region — see the note on the random crop below). All are implemented in [`code/src/utils/evaluate.py`](code/src/utils/evaluate.py).

First, each cell's pixel area is converted to physical units with a fixed pixel-to-micron ratio:

```
pix_to_um = 1000 / 1296        # µm per pixel
area_µm²  = area_pixels × pix_to_um²
```

**1. Cell Density (CD)** — cells per unit area, in **cells/mm²**:

```
CD = (N_cells × 1e6) / Σ(area_µm²)
```

where `N_cells` is the number of segmented cells and `Σ(area_µm²)` is the total cell area (the `1e6` converts µm² → mm²). Returns `0` if no cells are found.

**2. Coefficient of Variation (CV)** — dimensionless spread of cell areas:

```
CV = std(area_µm²) / mean(area_µm²)
```

Returned as a **ratio** (e.g. `0.30`), not a percent; multiply by 100 for a percentage. Returns `0` if there are no cells or the mean area is `0`.

**3. Hexagonality (HEX)** — mean shape-regularity score in **[0, 1]**:

```
For each cell i:
  fit an oriented hexagon from the cell's major/minor axis lengths
  (scaled by 0.8) and orientation angle, centered on the cell centroid
  IoU_i = |cell ∩ hexagon| / |cell ∪ hexagon|
HEX = mean(IoU_i)
```

Higher HEX means shapes closer to a regular hexagon. Returns `0` if no cells are found. (This IoU-to-fitted-hexagon definition is the implemented baseline; it is not a discrete "6-sided cell" count.)

> **Region of interest (random crop).** The baseline segments the **full** image (so cells near the region boundary still get full surrounding context), then restricts the metrics to a deterministic random crop — 40% of H × 40% of W, seeded per image — by keeping only cells whose centroid lies inside the crop. See `code/inference.py` and `code/src/infer_cellpose_sam.py`.

Each processed image yields a prediction row with `ID, CD, CV, HEX` (plus `SD`, `Total Area (µm²)`, and `Number of Cells`, which are not scored).

---

## Quick Start

```bash
cd code

./do_test_run.sh    # build the image + run it on the sample case in test/  (do this first)
./do_save.sh        # export the image + model as .tar.gz files for upload
```

Then upload the resulting `clear_ec_algorithm_*.tar.gz` (and `model.tar.gz`) to your Algorithm page on Grand Challenge and submit it to the challenge.

---

## What runs on the platform

There are two "modes" in this repo, and it is important to keep them straight:

| File | Runs where | Purpose |
| --- | --- | --- |
| **`code/inference.py`** | **Inside the container, on Grand Challenge** | The real submission. Reads one image, writes three JSON predictions. **This is what is scored.** |
| `code/main.py` | Your local machine (dev only) | Batch-runs the pipeline over a folder of images and writes a CSV. Handy for experimenting. **Not** included in the container (see `.dockerignore`). |
| `code/evaluate.py` | Your local machine (dev only) | Compares a predictions CSV against ground truth to estimate your score locally. |

Both share the same core logic in [`code/src/`](code/src/), so improvements you make in `src/` show up in both your local experiments and your submission.

---

## The Grand Challenge runtime contract

When your algorithm runs on the platform, the container is started **once per case** with:

- **No network** (`--network none`) — everything, including model weights, must already be inside the image.
- **Read-only input** mounted at `/input`.
- **Writable output** at `/output`.

Your job, implemented in `code/inference.py`, is to read the input image and write exactly three JSON files:

```
/input/inputs.json                                          # metadata describing the case
/input/images/corneal-specular-microscopy-image/<name>.mha  # the input image (MHA/TIFF)

        │  your algorithm  ▼

/output/cell-density.json               # a single float: cells / mm²
/output/coefficient-of-variation.json   # a single float: ratio (e.g. 0.30)
/output/hexagonality.json               # a single float in [0, 1]
```

Each output file contains just a JSON-encoded number, e.g. `2750.0`. See `interf0_handler()` in [`code/inference.py`](code/inference.py) for the reference implementation — it locates the image, runs segmentation, and calls `write_json_file(...)` three times.

> **Socket / interface.** The challenge input socket is `corneal-specular-microscopy-image`. `inputs.json` lists which sockets are present; `inference.py` reads it via `get_interface_key()` and dispatches to the matching handler. Unless the challenge adds new inputs, you only need the single `interf0_handler`.

---

## Data and local validation

**The test images and test ground truth are hidden** — they live only on Grand Challenge and are never released to participants. When you submit, the platform runs your container on those hidden cases and scores the outputs for you.

What you **do** have is the **released training data** (images + their CD/CV/HEX ground truth), published on Zenodo:

**➡️ https://zenodo.org/records/21270595**

Download it and use it to build your own local validation set so you can measure your algorithm before every submission instead of burning submission attempts:

1. **Split the training data yourself.** Hold out a portion of the released cases as a personal "test" set (e.g. 80/20), and keep it separate from anything you train or tune on.
2. **Point `main.py` at your held-out images** to produce predictions:
   ```bash
   python main.py --data_dir /path/to/your_holdout_images --results_dir ./my_val
   ```
3. **Score against the training ground truth** with `evaluate.py`:
   ```bash
   python evaluate.py --predictions_csv ./my_val/predictions_test.csv \
                      --gt_csv /path/to/your_holdout_ground_truth.csv
   ```

This gives you a fast, private proxy for the leaderboard: iterate locally as much as you want, and only submit when your held-out score looks good. Just remember your local numbers are an *estimate* — the hidden test set will differ, so avoid overfitting to your own split.

---

## Setup for local development

Iterating inside Docker is slow. Do your experimentation with `main.py` on a folder of images first, then containerize once it works.

**Requirements:** Python 3.9+, a CUDA 12.1-compatible GPU (optional but much faster), and the dependencies in `code/requirements.txt` / `code/environment.yml`.

```bash
cd code

# Option A — conda
conda env create -f environment.yml
conda activate CLEAR-EC

# Option B — pip
pip install -r requirements.txt
```

**Run the baseline over a split and score it:**

```bash
python main.py --split test --plot --seed 42
python evaluate.py --split test --gt_csv /path/to/ground_truth.csv
```

Key `main.py` arguments (run `python main.py -h` for the full list):

- `--split` — which split to process: `train` or `test` (default `test`)
- `--data_dir` — folder of input images (defaults to `<repo>/data/<split>_mha`)
- `--results_dir` — where the predictions CSV is written (default `./results_mha`)
- `--plot` — also save segmentation overlay PNGs for visual inspection
- `--limit N` — process only the first N images (quick smoke test)
- `--diameter`, `--flow_threshold`, `--cellprob_threshold`, `--model_type`, `--min_size` — Cellpose knobs

`evaluate.py` merges your `predictions_<split>.csv` with a ground-truth CSV on `ID` and reports per-slide percent error, a best→worst ranking, and the average error — a quick local proxy for the leaderboard.

---

## Building your own method

The container contract never changes — only what happens **between** reading the image and writing the three numbers. In practice you edit `code/src/` and leave the build/test/save machinery alone.

### 1. Swap in your algorithm

`inference.py` currently calls `get_segmentation(...)` from `src/infer_cellpose_sam.py`, which returns a prediction dict with `CD`, `CV`, and `HEX`. To use your own method, the only hard requirement is: **read the image from `/input`, write three floats to `/output`, exit 0.**

> **Metric definitions must match the challenge.** CD, CV, and HEX are defined in [`code/src/utils/evaluate.py`](code/src/utils/evaluate.py) and in *The baseline pipeline* above. 

### 2. Bundle your model weights (offline!)

The container runs with **no internet**, so every weight file must be inside the image before submission. Two supported patterns:

- **Download at build time** (what the baseline does): the `Dockerfile` runs
  ```dockerfile
  RUN python -c "from cellpose import models; models.Cellpose(gpu=False, model_type='cyto')"
  ```
  which caches the weights into the image. Add an equivalent line for your model, or `COPY` a checkpoint you ship in the build context.
- **Ship as a separate Model** (the `code/model/` directory → `model.tar.gz`, mounted at `/opt/ml/model`): use this for large checkpoints you upload once and reuse across algorithm versions. `do_save.sh` already packs `model/` into `model.tar.gz`; load from `/opt/ml/model` inside `inference.py`. 

If your image tries to reach the network at run time, it will fail on the platform. **Test with `--network none` locally** (`do_test_run.sh` already does this).

### 3. Update dependencies

Add any new Python packages to `requirements.txt` (this is what the image installs). Keep `environment.yml` in sync so your local runs match the container. Pin versions — the base image is `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime` on `linux/amd64`.

### 4. Edit the Dockerfile only if you must

You usually don't need to touch the `Dockerfile`. Edit it if you add system libraries (`apt-get`), change how weights are fetched, or need extra files `COPY`-d into the image. Note that only `src/` and `inference.py` are copied in — `main.py`, `evaluate.py`, `data/`, etc. are excluded by `.dockerignore` on purpose.

---

## Build, test, and save the container

Run these from **inside `code/`**. They require Docker with the `buildx` plugin; a GPU is optional locally (the platform always provides one).

### `./do_test_run.sh` — build + smoke test (run this first, and after every change)

Builds the image, then runs it against the sample case in `test/input/interf0/` **exactly the way Grand Challenge will**: `--network none`, read-only `/input`, writable `/output`, GPU if available. Afterwards, check the outputs:

```bash
cat test/output/interf0/cell-density.json
cat test/output/interf0/coefficient-of-variation.json
cat test/output/interf0/hexagonality.json
```

If those three files contain sensible numbers, your container works. **A green run here is the single best predictor that your submission will run on the platform.**

### `./do_build.sh` — build only

Just builds the `clear_ec_algorithm` image (tagged `linux/amd64`). Used internally by the other two scripts; call it directly when you only want to check the image builds.

### `./do_save.sh` — export for upload

Produces two artifacts in `code/`:

- `clear_ec_algorithm_<timestamp>.tar.gz` — the **algorithm image**, exported in the legacy Docker tar format Grand Challenge requires (the script uses a `docker-container` buildx builder to guarantee this; a plain `docker save` on Docker 25+ produces an OCI archive that GC rejects).
- `model.tar.gz` — the contents of `model/`, to be uploaded as a **separate Model** and attached to your Algorithm.

---

## Submitting to Grand Challenge

The full walkthrough is:

1. **Test locally** — `./do_test_run.sh` passes and writes valid JSON.
2. **Save** — `./do_save.sh` produces the image and model tarballs.
3. **Create an Algorithm** on Grand Challenge and upload `clear_ec_algorithm_*.tar.gz`; upload `model.tar.gz` as a Model and attach it.
4. **Try it** on a single case in the platform UI to confirm it runs there too.
5. **Submit** the algorithm to the CLEAR-EC challenge phase. The evaluation method (see [`code/eval/`](code/eval/)) scores your three outputs against the hidden ground truth and updates the leaderboard.

---

## Common pitfalls

- **Network access at run time.** The container has none. Bake weights into the image or ship them via `model/`. Verify with the `--network none` smoke test.
- **Wrong output shape.** Exactly three files, each a single JSON number, at the exact paths above. No arrays, no extra keys.
- **Wrong metric units.** CD in cells/mm², CV as a ratio (not %), HEX in [0, 1]. Reuse `code/src/utils/evaluate.py`.
- **OCI vs. legacy tar.** Always export with `do_save.sh`; don't hand-roll `docker save`.
- **Platform is `linux/amd64`.** If you build on Apple Silicon, keep the `--platform=linux/amd64` flags (the scripts already set them).
- **Forgetting to rebuild.** Editing `src/` does nothing until you rebuild. `do_test_run.sh` and `do_save.sh` rebuild for you.

