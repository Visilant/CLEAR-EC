# Phase I Review Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `/home/visilant/clear-ec-phase1-submission`, a fresh, signature-free repository from which the CLEAR-EC evaluators can install, run, and reproduce Visilant's Phase I entries (v2ens and v2ens_plus) end to end.

**Architecture:** A `clear_ec` package under `src/` holds the ported data cache, slide-grouped splits and folds, the ConvNeXt regression trainer, TTA prediction, and a new `ensemble` module shared by the container entrypoint, the OOF scorer and the container replay. Seventeen LFS-tracked checkpoints plus two small ensemble manifests define the entries; `scripts/` wraps every step (cache, train, OOF, bundle, container). Documentation follows the organisers' numbering.

**Tech Stack:** Python 3.10 (container) / 3.12 (training host), PyTorch 2.1.2+cu121 (container) / 2.13.0+cu130 (host), timm 1.0.29, torchvision, numpy 1.26.4, pandas 2.3.3, SimpleITK, Docker 29 with buildx, Git LFS 3.4.1, pdflatex.

**Spec:** `/home/visilant/CLEAR-EC/docs/superpowers/specs/2026-09-14-phase1-review-repo-design.md`

## Global Constraints

- New repo path: `/home/visilant/clear-ec-phase1-submission` (`$NEW` below). Research repo `/home/visilant/CLEAR-EC` (`$OLD`); submission worktree `/home/visilant/CLEAR-EC-phase1-v2ens` (`$WT`).
- No occurrence of `Claude`, `Anthropic`, `Co-Authored-By`, `claude.ai`, or tool names in any tracked file or commit message. Commits: `git -c user.name=Adi -c user.email=adi@visilant.org commit`, no trailers.
- Behaviour of ported modules is unchanged; only the import root changes from `src.` to `clear_ec.`. `RegressionConfig` fields and defaults stay byte-identical.
- Python in `$NEW` runs with `$OLD/code/.venv/bin/python` during development (it has torch 2.13, timm 1.0.29); scripts insert `$NEW/src` into `sys.path` so no install is required.
- Do not push, do not create a GitHub repository, do not add collaborators.
- Checkpoint names and weights: v2fold0..4 (V2-Tiny, w2), fold0..4 (Tiny, w1), v2refit_seed123 / v2refit_seed7 (V2-Tiny all-data, w2), v2base_fold0..4 (V2-Base, w3). v2ens = first ten; v2ens_plus = all seventeen.
- Clamps: CD [372, 4500], CV [0.03, 1.5], HEX [0, 1]. TTA: flips (4 views). Combination: weighted geometric mean.
- Fixture expected values (GPU, A5000): v2ens CD 2479.2796266090510, CV 0.43580039758981637, HEX 0.46232458644902247; v2ens_plus CD 2524.9313273744374, CV 0.43571988526284405, HEX 0.46188458487183903.

---

### Task 1: Scaffold the repository

**Files:**
- Create: `$NEW/.gitignore`, `$NEW/.gitattributes`, `$NEW/src/clear_ec/__init__.py`, `$NEW/src/clear_ec/data/__init__.py`, `$NEW/src/clear_ec/training/__init__.py`, `$NEW/tests/__init__.py`, `$NEW/tests/_path.py`

**Interfaces:**
- Produces: importable `clear_ec` package once `src/` is on `sys.path`; `tests/_path.py` performs that insertion for tests.

- [ ] **Step 1: Create directories and git init**

```bash
NEW=/home/visilant/clear-ec-phase1-submission
mkdir -p $NEW/{src/clear_ec/data,src/clear_ec/training,configs/members,configs/ensembles,scripts,checkpoints/members,examples/input/images,examples/expected_output,tests/fixtures,docs}
cd $NEW && git init -q -b main && git lfs install --local
```

- [ ] **Step 2: Write .gitignore, .gitattributes, package inits**

`.gitignore`:
```
__pycache__/
*.pyc
.venv/
build/
data/
results/
*.tar.gz
*.tar
.pytest_cache/
Technical_Report.aux
Technical_Report.log
Technical_Report.out
Technical_Report.toc
```
`.gitattributes`:
```
checkpoints/members/*.pt filter=lfs diff=lfs merge=lfs -text
```
`src/clear_ec/__init__.py`:
```python
"""CLEAR-EC Phase I submission: direct regression of CD, CV and HEX with ConvNeXt ensembles."""
__version__ = "1.0.0"
```
Empty `src/clear_ec/data/__init__.py`, `src/clear_ec/training/__init__.py`, `tests/__init__.py`.
`tests/_path.py`:
```python
"""Put src/ on sys.path so tests import clear_ec without installation."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

- [ ] **Step 3: Verify import and commit**

Run: `cd $NEW && /home/visilant/CLEAR-EC/code/.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import clear_ec; print(clear_ec.__version__)"` -> `1.0.0`
```bash
git add -A && git -c user.name=Adi -c user.email=adi@visilant.org commit -q -m "Scaffold repository layout"
```

---

### Task 2: Port the data and training package

**Files:**
- Create: `$NEW/src/clear_ec/artifacts.py`, `io.py`, `data/cache.py`, `data/splits.py`, `training/{common,config,targets,data,models,losses,loop,folds,predict,train,convnext_regression}.py`
- Test: `$NEW/tests/test_config.py`, `$NEW/tests/test_splits_and_scoring.py`

**Interfaces:**
- Produces (used by later tasks): `clear_ec.data.cache.open_image_cache(cache_dir) -> (memmap, index_df)`, `build_image_cache(data_dir, labels_csv, cache_dir, *, seed, force, limit)`; `clear_ec.data.splits.load_split(cache_dir, name)`; `clear_ec.training.common.METRICS`, `load_labels`, `labels_for_indices`, `score_by_id`, `denormalize_targets`; `clear_ec.training.config.RegressionConfig`, `config_from_checkpoint`; `clear_ec.training.models.load_checkpoint_file(path, device) -> (model, stats, cfg, epoch)`, `build_regression_model`; `clear_ec.training.predict.VIEWS`, `predict_tta`, `predict_frame_tta`; `clear_ec.training.folds._slide_group_folds(cache_dir, n, seed=42)`, `resolve_indices`; `clear_ec.training.train.train_regression_cnn(cache_dir, labels_csv, results_dir, *, gpu, config)`; `clear_ec.training.targets._inverse_target_space`; `clear_ec.io.load_image`.

- [ ] **Step 1: Copy modules with the import rewrite**

```bash
OLD=/home/visilant/CLEAR-EC/code; NEW=/home/visilant/clear-ec-phase1-submission
cp $OLD/src/artifacts.py $NEW/src/clear_ec/artifacts.py
cp $OLD/src/io_utils.py  $NEW/src/clear_ec/io.py
cp $OLD/src/data/cache.py $OLD/src/data/splits.py $NEW/src/clear_ec/data/
for m in common config targets data models losses loop folds predict train convnext_regression; do cp $OLD/src/training/$m.py $NEW/src/clear_ec/training/; done
cd $NEW/src && sed -i -e 's/from src\.io_utils import/from clear_ec.io import/' -e 's/from src\./from clear_ec./g' -e 's/import src\./import clear_ec./g' clear_ec/*.py clear_ec/*/*.py
grep -rn "src\." clear_ec | grep -v "clear_ec\." || echo "no stale imports"
```

- [ ] **Step 2: Trim common.py of the legacy Cellpose experiment helpers**

In `training/common.py` delete the import line `from clear_ec.data.config import MetricConfig, SegConfig, config_hash` and the two functions `experiment_hash` and `pred_artifact_path` (they only served the segmentation cache). Everything else stays. Confirm nothing imports them:
```bash
grep -rn "experiment_hash\|pred_artifact_path\|data.config" $NEW/src || echo clean
```

- [ ] **Step 3: Adjust io.py docstring**

In `io.py` change the comment `# Cellpose downstream expects 3-channel RGB; replicate grayscale.` to `# Callers expect a 3-channel array; replicate grayscale.` and the module docstring line about `convert_to_mha.py` to `Read an uncompressed 2D MetaImage (.mha).` No code change.

- [ ] **Step 4: Port the config freeze test and the pure scoring/splits tests**

`tests/test_config.py`: copy `$OLD/tests/test_config.py`, replace `from src.training.config import` with
```python
from tests._path import SRC  # noqa: F401
from clear_ec.training.config import RegressionConfig, config_from_checkpoint, recipe_hash
```
`tests/test_splits_and_scoring.py`:
```python
"""Slide-disjoint splits and ID-joined scoring are the protocol every result in this repo relies on."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from tests._path import SRC  # noqa: F401
from clear_ec.data.splits import assert_protocol_splits, build_splits, load_split
from clear_ec.training.common import METRICS, score_by_id


def _index(n_slides=30, per_slide=3):
    rows = []
    for s in range(n_slides):
        for j in range(per_slide):
            rows.append({"idx": len(rows), "ID": f"{s:04d}-{j}", "slide_id": f"{s:04d}"})
    return pd.DataFrame(rows)


class SplitTests(unittest.TestCase):
    def test_build_splits_are_image_and_slide_disjoint(self):
        index = _index()
        with tempfile.TemporaryDirectory() as tmp:
            splits = build_splits(index, Path(tmp), seed=42)
            report = assert_protocol_splits(index, splits)
            self.assertEqual(report["n_images"], len(index))
            self.assertEqual(load_split(Path(tmp), "val"), splits["val"])
            written = json.loads((Path(tmp) / "splits.json").read_text())
            self.assertEqual(written, splits)

    def test_assert_protocol_splits_rejects_slide_leakage(self):
        index = _index(n_slides=4)
        leaky = {"train": [0, 1, 2, 3], "val": [4, 5, 6, 7, 8], "test": [9, 10, 11]}
        with self.assertRaises(ValueError):
            assert_protocol_splits(index, leaky)


class ScoringTests(unittest.TestCase):
    def test_score_by_id_joins_by_id_not_order(self):
        gt = pd.DataFrame({"ID": ["a", "b", "c"], "CD": [1000, 2000, 3000], "CV": [0.3, 0.4, 0.5], "HEX": [0.5, 0.6, 0.7]})
        pred = pd.DataFrame({"ID": ["c", "a", "b"], "CD": [3300, 1100, 2200], "CV": [0.5, 0.3, 0.4], "HEX": [0.7, 0.5, 0.6]})
        s = score_by_id(pred, gt, expected_ids=gt.ID)
        self.assertAlmostEqual(s["CD"], 10.0)
        self.assertAlmostEqual(s["CV"], 0.0)
        self.assertAlmostEqual(s["mean"], 10.0 / 3)

    def test_score_by_id_rejects_non_finite(self):
        gt = pd.DataFrame({"ID": ["a"], "CD": [1000.0], "CV": [0.3], "HEX": [0.5]})
        pred = gt.copy(); pred.loc[0, "CD"] = np.nan
        with self.assertRaises(ValueError):
            score_by_id(pred, gt)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 5: Run tests, commit**

Run: `cd $NEW && $OLD/.venv/bin/python -m unittest discover -s tests -t . -v` -> 7 tests OK.
```bash
git add -A && git -c user.name=Adi -c user.email=adi@visilant.org commit -q -m "Port data cache, splits, folds and the regression trainer"
```

---

### Task 3: Ensemble module and container entrypoint

**Files:**
- Create: `$NEW/src/clear_ec/ensemble.py`, `$NEW/inference.py`
- Test: `$NEW/tests/test_ensemble.py`

**Interfaces:**
- Produces: `clear_ec.ensemble.load_manifest(model_dir) -> dict`; `verify_members(model_dir, manifest)`; `combine(log_preds: list[np.ndarray], weights: list[float], clamp: dict) -> np.ndarray` (shape (n,3), clamped); `member_log_prediction(model, stats, cfg, x_u8, views) -> np.ndarray` (shape (3,), log of geometric mean over views); `predict_image(image_u8: np.ndarray, model_dir: Path, device) -> dict[str,float]`; `cpu_budget() -> int`.
- `inference.py` honours env vars `CLEAR_EC_INPUT_DIR` (default `/input`), `CLEAR_EC_OUTPUT_DIR` (`/output`), `CLEAR_EC_MODEL_DIR` (default: `/opt/ml/model` if it holds `submission.json`, else `/opt/app/model`).

- [ ] **Step 1: Write the failing ensemble math test**

`tests/test_ensemble.py`:
```python
"""Weighted geometric mean over members, then physical clamps."""
import unittest

import numpy as np

from tests._path import SRC  # noqa: F401
from clear_ec.ensemble import combine

CLAMP = {"CD": [372.0, 4500.0], "CV": [0.03, 1.5], "HEX": [0.0, 1.0]}


class CombineTests(unittest.TestCase):
    def test_weighted_geometric_mean(self):
        a = np.log(np.array([[2000.0, 0.40, 0.50]]))
        b = np.log(np.array([[4000.0, 0.40, 0.50]]))
        out = combine([a, b], [2.0, 1.0], CLAMP)
        self.assertAlmostEqual(out[0, 0], np.exp((2 * np.log(2000) + np.log(4000)) / 3), places=6)
        self.assertAlmostEqual(out[0, 1], 0.40, places=9)

    def test_clamps_apply(self):
        a = np.log(np.array([[9000.0, 0.001, 1.7]]))
        out = combine([a], [1.0], CLAMP)
        np.testing.assert_allclose(out[0], [4500.0, 0.03, 1.0])

    def test_non_finite_raises(self):
        with self.assertRaises(ValueError):
            combine([np.array([[np.nan, 0.0, 0.0]])], [1.0], CLAMP)
```
Run: `python -m unittest tests.test_ensemble -v` -> ImportError.

- [ ] **Step 2: Write ensemble.py**

```python
"""Ensemble bundle inference: per-member flip TTA, weighted geometric mean, physical clamps.

A bundle directory holds submission.json and one checkpoint per member:
    {"method": "ensemble", "tta": "flips", "clamp": {...}, "members": [{"file", "weight", "sha256"}, ...]}
This module is the single implementation used by the container (inference.py), the local
OOF scorer (scripts/predict_oof.py) and the container replay (scripts/replay_container.py).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from clear_ec.training.common import METRICS, denormalize_targets
from clear_ec.training.config import config_from_checkpoint
from clear_ec.training.models import build_regression_model
from clear_ec.training.predict import VIEWS
from clear_ec.training.targets import _inverse_target_space


def load_manifest(model_dir: Path) -> dict:
    manifest = json.loads((Path(model_dir) / "submission.json").read_text())
    if manifest.get("method") != "ensemble":
        raise ValueError("Unknown submission bundle method")
    return manifest


def verify_members(model_dir: Path, manifest: dict) -> None:
    """Refuse to run on a bundle whose checkpoints differ from the manifest."""
    for member in manifest["members"]:
        path = Path(model_dir) / member["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != member["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch for {member['file']}")


def combine(log_preds: list[np.ndarray], weights: list[float], clamp: dict) -> np.ndarray:
    """Weighted geometric mean of member predictions given as logs, shape (n, 3), then clamps."""
    stacked = np.stack([np.asarray(p, dtype=np.float64) for p in log_preds])
    if not np.isfinite(stacked).all():
        raise ValueError("Non-finite member prediction")
    out = np.exp(np.average(stacked, axis=0, weights=np.asarray(weights, dtype=np.float64)))
    for k, metric in enumerate(METRICS):
        if metric in clamp:
            lo, hi = clamp[metric]
            out[:, k] = np.clip(out[:, k], lo, hi)
    if not np.isfinite(out).all():
        raise ValueError("Non-finite ensemble prediction")
    return out


@torch.inference_mode()
def member_log_prediction(model, stats: dict, cfg, x_u8: torch.Tensor, views) -> np.ndarray:
    """log of the geometric mean over TTA views for a (1, 1, H, W) uint8 tensor -> shape (1, 3)."""
    x = x_u8 if cfg.uint8_inputs else x_u8.float() / 255.0
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
    return np.mean(logs, axis=0)


def load_member(path: Path, device: torch.device):
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    cfg = config_from_checkpoint(ckpt["config"])
    model = build_regression_model(cfg, load_pretrained=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    return model, ckpt["target_stats"], cfg


def predict_image(image_u8: np.ndarray, model_dir: Path, device: torch.device, *, verbose: bool = True) -> dict:
    """Apply a bundle to one grayscale uint8 frame (H, W). Returns {"CD", "CV", "HEX"}."""
    model_dir = Path(model_dir)
    manifest = load_manifest(model_dir)
    verify_members(model_dir, manifest)
    views = VIEWS[manifest.get("tta", "none")]
    x_u8 = torch.from_numpy(np.ascontiguousarray(image_u8)).unsqueeze(0).unsqueeze(0).to(device)
    logs, weights = [], []
    for member in manifest["members"]:
        model, stats, cfg = load_member(model_dir / member["file"], device)
        logs.append(member_log_prediction(model, stats, cfg, x_u8, views))
        weights.append(float(member.get("weight", 1.0)))
        if verbose:
            print(f"member {member['file']}: {np.exp(logs[-1][0]).round(4).tolist()}", flush=True)
        del model
    out = combine(logs, weights, manifest.get("clamp", {}))[0]
    return dict(zip(METRICS, map(float, out)))


def cpu_budget() -> int:
    """Threads to use on CPU: the cgroup CPU quota if set, else the affinity count, capped at 8."""
    n = None
    for quota_file in ("/sys/fs/cgroup/cpu.max", "/sys/fs/cgroup/cpu/cpu.cfs_quota_us"):
        try:
            parts = Path(quota_file).read_text().split()
            quota = int(parts[0])
            period = int(parts[1]) if len(parts) > 1 else int(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read_text())
            if quota > 0:
                n = max(1, quota // period)
            break
        except (OSError, ValueError, IndexError):
            continue
    if n is None:
        try:
            n = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            n = os.cpu_count() or 1
    return max(1, min(int(n), 8))
```
Note: `combine` uses float64 and `np.average` exactly like the submitted `predict_model_bundle` (which stacked float64 logs from `np.mean` of float32-derived arrays; `np.log` of float32 inputs yields float32 there — keep parity by computing `np.log(np.clip(out, 1e-6, None))` on the float32 `out`, as above, so member logs are float32 and `np.mean` promotes identically). The replay in Task 8 confirms parity to 1e-6.

- [ ] **Step 3: Write inference.py**

```python
"""CLEAR-EC Phase I algorithm container entrypoint.

Grand Challenge runtime contract:
    /input/inputs.json                                         metadata
    /input/images/corneal-specular-microscopy-image/*.mha      input image
    /output/cell-density.json                                  CD prediction
    /output/coefficient-of-variation.json                      CV prediction
    /output/hexagonality.json                                  HEX prediction

The model bundle (submission.json + member checkpoints) is baked at /opt/app/model; a bundle
mounted by the platform at /opt/ml/model takes precedence. Paths can be overridden with the
environment variables CLEAR_EC_INPUT_DIR, CLEAR_EC_OUTPUT_DIR and CLEAR_EC_MODEL_DIR for local runs.
"""
from __future__ import annotations

import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from clear_ec.ensemble import cpu_budget, predict_image  # noqa: E402
from clear_ec.io import load_image  # noqa: E402

INPUT_PATH = Path(os.environ.get("CLEAR_EC_INPUT_DIR", "/input"))
OUTPUT_PATH = Path(os.environ.get("CLEAR_EC_OUTPUT_DIR", "/output"))
MODEL_PATH = Path("/opt/ml/model")
BAKED_MODEL_PATH = Path("/opt/app/model")
SEED = 42


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_model_dir() -> Path:
    override = os.environ.get("CLEAR_EC_MODEL_DIR")
    candidates = [Path(override)] if override else [MODEL_PATH, BAKED_MODEL_PATH]
    for candidate in candidates:
        if (candidate / "submission.json").exists():
            return candidate
    raise FileNotFoundError(f"No submission.json under any of {candidates}")


def find_input_image() -> Path:
    image_dir = INPUT_PATH / "images" / "corneal-specular-microscopy-image"
    files = sorted(glob.glob(str(image_dir / "*.mha")) + glob.glob(str(image_dir / "*.tif")) + glob.glob(str(image_dir / "*.tiff")))
    if not files:
        raise FileNotFoundError(f"No MHA/TIFF input found under {image_dir}")
    return Path(files[0])


def run() -> int:
    set_seed(SEED)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(cpu_budget())
    print(f"Torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}, device: {device}")
    image_path = find_input_image()
    model_dir = resolve_model_dir()
    print(f"Processing input: {image_path.name}\nUsing model bundle at {model_dir}")
    image = load_image(image_path)[..., 0].copy()
    prediction = predict_image(image, model_dir, device)
    cd, cv, hex_ = (float(prediction[k]) for k in ("CD", "CV", "HEX"))
    if not np.isfinite([cd, cv, hex_]).all():
        raise ValueError("Predictions must be finite JSON numbers")
    print(f"Predictions: CD={cd:.2f}  CV={cv:.4f}  HEX={hex_:.4f}")
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    for name, value in (("cell-density.json", cd), ("coefficient-of-variation.json", cv), ("hexagonality.json", hex_)):
        (OUTPUT_PATH / name).write_text(json.dumps(value, indent=4, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
```
Keep the `set_seed` and cudnn flags identical to the submitted entrypoint: `cudnn.deterministic=True` affects convolution algorithm choice and therefore the last digits.

- [ ] **Step 4: Run tests, commit**

Run: `python -m unittest tests.test_ensemble -v` -> 3 OK.
```bash
git add -A && git -c user.name=Adi -c user.email=adi@visilant.org commit -q -m "Add ensemble module and container entrypoint"
```

---

### Task 4: Checkpoints, ensemble manifests, member configs

**Files:**
- Create: `$NEW/checkpoints/members/*.pt` (17, LFS), `$NEW/checkpoints/SHA256SUMS`, `$NEW/checkpoints/README.md`, `$NEW/configs/ensembles/v2ens.json`, `$NEW/configs/ensembles/v2ens_plus.json`, `$NEW/configs/members/*.json` (17), `$NEW/scripts/verify_checkpoints.sh`
- Test: `$NEW/tests/test_configs_match_checkpoints.py`

**Interfaces:**
- Ensemble JSON schema: `{"name", "description", "tta": "flips", "clamp": {...}, "members": [{"name": "v2fold0", "file": "v2fold0.pt", "weight": 2}, ...]}`.
- Member JSON schema: `{"name", "family", "checkpoint": "checkpoints/members/<name>.pt", "config": {<RegressionConfig asdict>}}`.

- [ ] **Step 1: Copy checkpoints and write checksums**

```bash
WT=/home/visilant/CLEAR-EC-phase1-v2ens/results/phase1_v2ens_plus/build_ctx/model; NEW=/home/visilant/clear-ec-phase1-submission
for f in v2fold{0..4} fold{0..4} v2refit_seed123 v2refit_seed7 v2base_fold{0..4}; do cp $WT/$f.pt $NEW/checkpoints/members/$f.pt; done
cd $NEW/checkpoints && sha256sum members/*.pt > SHA256SUMS && cat SHA256SUMS
```
Cross-check against the manifests' recorded hashes:
```bash
python3 - <<'EOF'
import json,hashlib
sub=json.load(open('/home/visilant/CLEAR-EC-phase1-v2ens/results/phase1_v2ens_plus/build_ctx/model/submission.json'))
for m in sub['members']:
    h=hashlib.sha256(open(f"/home/visilant/clear-ec-phase1-submission/checkpoints/members/{m['file']}",'rb').read()).hexdigest()
    assert h==m['sha256'], m['file']
print('17 checksums match the submitted bundle')
EOF
```
`scripts/verify_checkpoints.sh`:
```bash
#!/usr/bin/env bash
# Verify that every member checkpoint is present, is a real file (not an LFS pointer), and matches SHA256SUMS.
set -euo pipefail
cd "$(dirname "$0")/../checkpoints"
for f in members/*.pt; do
  if head -c 40 "$f" | grep -q "version https://git-lfs"; then
    echo "$f is a Git LFS pointer; run: git lfs pull" >&2; exit 1
  fi
done
sha256sum -c SHA256SUMS
```

- [ ] **Step 2: Write ensemble manifests**

`configs/ensembles/v2ens.json`:
```json
{
  "name": "v2ens",
  "description": "Phase I entry 1: 5x ConvNeXt-V2-Tiny fold models (weight 2) + 5x ConvNeXt-Tiny fold models (weight 1), flip TTA, weighted geometric mean, physical clamps. Out-of-fold mean error over 9000 training images 8.8389; hidden test set 8.7583.",
  "tta": "flips",
  "clamp": {"CD": [372.0, 4500.0], "CV": [0.03, 1.5], "HEX": [0.0, 1.0]},
  "members": [
    {"name": "v2fold0", "file": "v2fold0.pt", "weight": 2}, {"name": "v2fold1", "file": "v2fold1.pt", "weight": 2},
    {"name": "v2fold2", "file": "v2fold2.pt", "weight": 2}, {"name": "v2fold3", "file": "v2fold3.pt", "weight": 2},
    {"name": "v2fold4", "file": "v2fold4.pt", "weight": 2},
    {"name": "fold0", "file": "fold0.pt", "weight": 1}, {"name": "fold1", "file": "fold1.pt", "weight": 1},
    {"name": "fold2", "file": "fold2.pt", "weight": 1}, {"name": "fold3", "file": "fold3.pt", "weight": 1},
    {"name": "fold4", "file": "fold4.pt", "weight": 1}
  ]
}
```
`configs/ensembles/v2ens_plus.json`: same head with name `v2ens_plus`, description "Prepared second candidate (not uploaded): v2ens members plus 2 all-data ConvNeXt-V2-Tiny refits (weight 2) and 5 ConvNeXt-V2-Base fold models (weight 3). Out-of-fold mean error of the fold-only part 8.814.", and members = the ten above + `v2refit_seed123` (2), `v2refit_seed7` (2), `v2base_fold0..4` (3). Member order must equal the submitted `submission.json` order (v2fold*, fold*, v2refit*, v2base*).

- [ ] **Step 3: Generate member configs from the saved run manifests**

Run in `$OLD` (one-off generator, not shipped):
```bash
cd /home/visilant/CLEAR-EC && code/.venv/bin/python - <<'EOF'
import json, pathlib
NEW = pathlib.Path('/home/visilant/clear-ec-phase1-submission/configs/members')
runs = {**{f'v2fold{k}': ('results/night_20260912', f'v2fold{k}', 123, 'convnextv2_tiny') for k in range(5)},
        **{f'fold{k}': ('results/night_20260912', f'fold{k}', 123, 'convnext_tiny') for k in range(5)},
        'v2refit_seed123': ('results/night_20260912', 'v2refit_seed123', 123, 'convnextv2_tiny_refit'),
        'v2refit_seed7': ('results/night_20260912', 'v2refit_seed7', 7, 'convnextv2_tiny_refit'),
        **{f'v2base_fold{k}': ('results/noise_floor_20260912', f'b3_v2base_fold{k}', 123, 'convnextv2_base') for k in range(5)}}
for name, (root, run, seed, family) in runs.items():
    man = json.load(open(f'{root}/{run}/manifest.json'))
    cfg = man['methods'][f'regression_cnn_seed_{seed}']['config']
    (NEW / f'{name}.json').write_text(json.dumps({'name': name, 'family': family,
        'checkpoint': f'checkpoints/members/{name}.pt', 'config': cfg}, indent=2) + '\n')
print(len(list(NEW.glob('*.json'))), 'member configs')
EOF
```

- [ ] **Step 4: Write the failing test that configs match checkpoint-embedded configs**

`tests/test_configs_match_checkpoints.py`:
```python
"""Every member config must equal the config stored inside its checkpoint, and every ensemble
manifest must reference existing members whose checksums match SHA256SUMS."""
import hashlib
import json
import unittest
from pathlib import Path

import torch

from tests._path import ROOT, SRC  # noqa: F401
from clear_ec.training.config import RegressionConfig, config_from_checkpoint

MEMBERS = sorted((ROOT / "configs" / "members").glob("*.json"))
CKPT = ROOT / "checkpoints" / "members"


def _is_pointer(path: Path) -> bool:
    return path.stat().st_size < 1024


class ConfigTests(unittest.TestCase):
    def test_seventeen_member_configs_parse(self):
        self.assertEqual(len(MEMBERS), 17)
        for path in MEMBERS:
            payload = json.loads(path.read_text())
            cfg = RegressionConfig(**payload["config"])
            self.assertEqual(payload["name"], path.stem)
            self.assertEqual(payload["checkpoint"], f"checkpoints/members/{path.stem}.pt")
            self.assertIn(cfg.model.split(":")[-1], ("convnextv2_tiny.fcmae_ft_in22k_in1k", "convnext_tiny", "convnextv2_base.fcmae_ft_in22k_in1k"))

    def test_ensembles_reference_members_and_sums(self):
        sums = dict(line.split()[::-1] for line in (ROOT / "checkpoints" / "SHA256SUMS").read_text().splitlines())
        for name, n in (("v2ens", 10), ("v2ens_plus", 17)):
            ens = json.loads((ROOT / "configs" / "ensembles" / f"{name}.json").read_text())
            self.assertEqual(len(ens["members"]), n)
            for m in ens["members"]:
                self.assertIn(f"members/{m['file']}", sums)
                self.assertTrue((ROOT / "configs" / "members" / f"{m['name']}.json").exists())

    @unittest.skipUnless(CKPT.exists() and not _is_pointer(CKPT / "fold0.pt"), "checkpoints not pulled")
    def test_config_json_equals_checkpoint_config(self):
        for path in MEMBERS:
            payload = json.loads(path.read_text())
            ckpt = torch.load(CKPT / f"{path.stem}.pt", map_location="cpu", weights_only=False)
            self.assertEqual(config_from_checkpoint(ckpt["config"]), config_from_checkpoint(payload["config"]), path.stem)
```
Run -> `test_config_json_equals_checkpoint_config` should pass immediately if the generator worked (the manifest config is the checkpoint config); if it fails, the generator picked the wrong run.

- [ ] **Step 5: checkpoints/README.md, run tests, commit (LFS)**

`checkpoints/README.md`: three short paragraphs: what the 17 files are (table: name, architecture, training set, epochs, size), how to fetch (`git lfs pull`), how to verify (`scripts/verify_checkpoints.sh`), and that a mirror URL, if one is added, goes in the README's section 3.3.
```bash
cd $NEW && git add -A && git -c user.name=Adi -c user.email=adi@visilant.org commit -q -m "Add member checkpoints (LFS), ensemble manifests and member configs"
git lfs ls-files | wc -l   # expect 17
```

---

### Task 5: Training and cache scripts

**Files:**
- Create: `$NEW/scripts/build_cache.py`, `$NEW/scripts/train_member.py`, `$NEW/scripts/train_all_members.sh`
- Test: `$NEW/tests/test_train_member_cli.py`

**Interfaces:**
- `train_member.py --config configs/members/<name>.json [--gpu 0] [--cache_dir data/cache] [--labels_csv data/final_train_ids.csv] [--results_dir results]` trains into `results/<name>/regression_cnn/seed_<seed>/` and writes `results/<name>/manifest.json`.
- `build_cache.py --data_dir data/train_mha --labels_csv data/final_train_ids.csv --cache_dir data/cache [--limit N] [--force]`.

- [ ] **Step 1: build_cache.py**

Copy `$OLD/scripts/build_cache.py`; replace the `sys.path.insert` line with `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))`, import `from clear_ec.data.cache import build_image_cache`, change the three defaults to `data/train_mha`, `data/final_train_ids.csv`, `data/cache` and resolve them against the repo root (`Path(__file__).resolve().parents[1]`) instead of `code/`. Remove the `--force_cache` alias.

- [ ] **Step 2: train_member.py**

```python
#!/usr/bin/env python3
"""Train one ensemble member from its config JSON (configs/members/<name>.json).

Writes results/<name>/regression_cnn/seed_<seed>/{last.pt, history.csv, metrics.json[, best_model.pt,
predictions_val.csv]} and results/<name>/manifest.json. Fold members validate on their held-out fold;
all-data refits train on all 9000 images with no validation and keep the last epoch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clear_ec.data.cache import open_image_cache  # noqa: E402
from clear_ec.training.common import split_summary, update_manifest  # noqa: E402
from clear_ec.training.config import RegressionConfig  # noqa: E402
from clear_ec.training.train import train_regression_cnn  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path, help="configs/members/<name>.json")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index")
    p.add_argument("--cache_dir", type=Path, default=ROOT / "data" / "cache")
    p.add_argument("--labels_csv", type=Path, default=ROOT / "data" / "final_train_ids.csv")
    p.add_argument("--results_dir", type=Path, default=ROOT / "results")
    p.add_argument("--epochs", type=int, default=None, help="Override the epoch budget (smoke tests only)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    payload = json.loads(args.config.read_text())
    cfg_dict = dict(payload["config"])
    if args.epochs is not None:
        cfg_dict["epochs"] = args.epochs
    cfg = RegressionConfig(**cfg_dict)
    _, index_df = open_image_cache(args.cache_dir)
    summary = split_summary(args.cache_dir)
    print(f"Cache: {len(index_df)} images | splits: {summary}")
    results_root = args.results_dir / payload["name"]
    run_dir = results_root / "regression_cnn" / f"seed_{cfg.seed}"
    train_regression_cnn(args.cache_dir, args.labels_csv, run_dir, gpu=args.gpu, config=cfg)
    update_manifest(results_root, {"errors": {}, "n_cached": len(index_df), "split_summary": summary,
                                   "cache_dir": str(args.cache_dir), "labels_csv": str(args.labels_csv),
                                   "member": payload["name"]})
    print(f"Done -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: train_all_members.sh**

```bash
#!/usr/bin/env bash
# Train all seventeen members. Two GPUs: V2-Tiny/V2-Base on GPU 1, Tiny and refits on GPU 0.
# Wall time on two RTX A5000: about 6.5 h (V2-Base 47 min/fold, V2-Tiny 21 min/fold, Tiny 16 min/fold, refits 28 min).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
GPU_A=${GPU_A:-0}; GPU_B=${GPU_B:-1}
(
  for m in fold0 fold1 fold2 fold3 fold4 v2refit_seed123 v2refit_seed7; do
    $PY scripts/train_member.py --config configs/members/$m.json --gpu $GPU_A
  done
) &
(
  for m in v2fold0 v2fold1 v2fold2 v2fold3 v2fold4 v2base_fold0 v2base_fold1 v2base_fold2 v2base_fold3 v2base_fold4; do
    $PY scripts/train_member.py --config configs/members/$m.json --gpu $GPU_B
  done
) &
wait
echo "All members trained under results/<name>/regression_cnn/seed_<seed>/last.pt"
```

- [ ] **Step 4: CLI test**

`tests/test_train_member_cli.py`:
```python
"""The training CLI must accept every shipped member config and refuse a missing config."""
import subprocess
import sys
import unittest

from tests._path import ROOT


class TrainCliTests(unittest.TestCase):
    def test_help_lists_config_flag(self):
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_member.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("--config", out.stdout)

    def test_missing_config_fails(self):
        out = subprocess.run([sys.executable, str(ROOT / "scripts" / "train_member.py")], capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
```

- [ ] **Step 5: Run, commit**

```bash
chmod +x $NEW/scripts/*.sh $NEW/scripts/*.py
cd $NEW && $OLD/.venv/bin/python -m unittest discover -s tests -t . -v && git add -A && git -c user.name=Adi -c user.email=adi@visilant.org commit -q -m "Add cache builder and member training scripts"
```

---

### Task 6: OOF scoring and bundle assembly

**Files:**
- Create: `$NEW/scripts/predict_oof.py`, `$NEW/scripts/assemble_bundle.py`
- Test: `$NEW/tests/test_assemble_bundle.py`

**Interfaces:**
- `predict_oof.py [--members all|name,...] [--ensembles v2ens,v2ens_plus] [--checkpoint_dir checkpoints/members] [--cache_dir] [--labels_csv] [--out results/oof] [--gpu 0]` writes `results/oof/members/<name>.csv` (idx, ID, CD, CV, HEX over the member's held-out fold) and `results/oof/summary.json` + printed table with per-family and per-ensemble MAPE.
- `assemble_bundle.py --ensemble configs/ensembles/<name>.json --out build/model_<name>` writes files + `submission.json` `{"method":"ensemble","tta","clamp","description","members":[{"file","weight","sha256"}]}`.

- [ ] **Step 1: assemble_bundle test (failing)**

`tests/test_assemble_bundle.py`:
```python
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests._path import ROOT


class AssembleTests(unittest.TestCase):
    def test_bundle_from_fake_members(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); (tmp / "members").mkdir()
            for n in ("a", "b"):
                (tmp / "members" / f"{n}.pt").write_bytes(n.encode() * 10)
            ens = {"name": "t", "description": "d", "tta": "flips", "clamp": {"CD": [1, 2]},
                   "members": [{"name": "a", "file": "a.pt", "weight": 2}, {"name": "b", "file": "b.pt", "weight": 1}]}
            (tmp / "t.json").write_text(json.dumps(ens))
            out = subprocess.run([sys.executable, str(ROOT / "scripts" / "assemble_bundle.py"), "--ensemble", str(tmp / "t.json"),
                                  "--checkpoint_dir", str(tmp / "members"), "--out", str(tmp / "bundle")], capture_output=True, text=True)
            self.assertEqual(out.returncode, 0, out.stderr)
            sub = json.loads((tmp / "bundle" / "submission.json").read_text())
            self.assertEqual(sub["method"], "ensemble")
            self.assertEqual([m["weight"] for m in sub["members"]], [2, 1])
            self.assertEqual(sub["members"][0]["sha256"], hashlib.sha256(b"a" * 10).hexdigest())
            self.assertTrue((tmp / "bundle" / "b.pt").exists())
```

- [ ] **Step 2: assemble_bundle.py**

```python
#!/usr/bin/env python3
"""Assemble a container model bundle: copy the ensemble's member checkpoints and write submission.json."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ensemble", required=True, type=Path, help="configs/ensembles/<name>.json")
    p.add_argument("--checkpoint_dir", type=Path, default=ROOT / "checkpoints" / "members")
    p.add_argument("--out", required=True, type=Path, help="bundle directory to create (must not exist)")
    args = p.parse_args()
    ens = json.loads(args.ensemble.read_text())
    args.out.mkdir(parents=True, exist_ok=False)
    members = []
    for m in ens["members"]:
        src = args.checkpoint_dir / m["file"]
        if src.stat().st_size < 1024:
            raise SystemExit(f"{src} looks like a Git LFS pointer; run git lfs pull")
        dst = args.out / m["file"]
        shutil.copyfile(src, dst)
        members.append({"file": m["file"], "weight": m["weight"], "sha256": hashlib.sha256(dst.read_bytes()).hexdigest()})
        print(f"{m['file']}  weight {m['weight']}  sha256 {members[-1]['sha256'][:16]}")
    manifest = {"method": "ensemble", "tta": ens["tta"], "clamp": ens["clamp"], "description": ens["description"], "members": members}
    (args.out / "submission.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Bundle -> {args.out} ({len(members)} members)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3: predict_oof.py**

```python
#!/usr/bin/env python3
"""Out-of-fold evaluation: each fold member predicts its held-out fold with flip TTA; ensembles are the
weighted geometric mean of families joined by image ID; scores are mean absolute percentage error.

All-data refits (v2refit_*) have no held-out images and are skipped; the v2ens_plus OOF is therefore the
score of its fold-only part with the same family weights (2:1:3), as reported in the technical report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clear_ec.data.cache import open_image_cache  # noqa: E402
from clear_ec.training.common import METRICS, labels_for_indices, load_labels, score_by_id  # noqa: E402
from clear_ec.training.folds import _slide_group_folds  # noqa: E402
from clear_ec.training.models import load_checkpoint_file  # noqa: E402
from clear_ec.training.predict import VIEWS, predict_frame_tta  # noqa: E402


def member_configs(names: str) -> list[dict]:
    paths = sorted((ROOT / "configs" / "members").glob("*.json"))
    configs = [json.loads(p.read_text()) for p in paths]
    if names != "all":
        wanted = set(names.split(","))
        configs = [c for c in configs if c["name"] in wanted]
    return configs


def predict_member(cfg_payload: dict, args, memmap, folds, device) -> pd.DataFrame | None:
    cfg = cfg_payload["config"]
    if cfg.get("all_data") or cfg.get("fold", -1) < 0:
        return None
    out_csv = args.out / "members" / f"{cfg_payload['name']}.csv"
    if out_csv.exists():
        return pd.read_csv(out_csv)
    indices = folds[cfg["fold"]]
    frame = labels_for_indices(args.cache_dir, args.labels_csv, indices)
    model, stats, rcfg, epoch = load_checkpoint_file(args.checkpoint_dir / f"{cfg_payload['name']}.pt", device)
    pred = predict_frame_tta(model, stats, rcfg, memmap, frame, device, views=VIEWS["flips"], batch_size=args.batch_size)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out_csv, index=False)
    s = score_by_id(pred, frame, expected_ids=frame.ID)
    print(f"{cfg_payload['name']:>16}  epoch {epoch}  n {len(pred):4d}  CD {s['CD']:.4f}  CV {s['CV']:.4f}  HEX {s['HEX']:.4f}  mean {s['mean']:.4f}", flush=True)
    del model
    return pred


def family_frame(preds: dict[str, pd.DataFrame], names: list[str]) -> pd.DataFrame:
    frame = pd.concat([preds[n] for n in names], ignore_index=True)
    frame["ID"] = frame["ID"].astype(str).str.strip()
    if frame.ID.duplicated().any():
        raise ValueError("fold members overlap")
    return frame.set_index("ID")


def geometric(families: dict[str, pd.DataFrame], weights: dict[str, float], ids: pd.Index) -> pd.DataFrame:
    total = sum(weights.values())
    out = pd.DataFrame(index=ids)
    for m in METRICS:
        out[m] = np.exp(sum(w * np.log(families[f].loc[ids, m].clip(lower=1e-6).to_numpy(float)) for f, w in weights.items()) / total)
    return out.reset_index()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--members", default="all")
    p.add_argument("--ensembles", default="v2ens,v2ens_plus")
    p.add_argument("--checkpoint_dir", type=Path, default=ROOT / "checkpoints" / "members")
    p.add_argument("--cache_dir", type=Path, default=ROOT / "data" / "cache")
    p.add_argument("--labels_csv", type=Path, default=ROOT / "data" / "final_train_ids.csv")
    p.add_argument("--out", type=Path, default=ROOT / "results" / "oof")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    memmap, _ = open_image_cache(args.cache_dir)
    folds = _slide_group_folds(args.cache_dir, 5, seed=42)
    preds: dict[str, pd.DataFrame] = {}
    for payload in member_configs(args.members):
        pred = predict_member(payload, args, memmap, folds, device)
        if pred is not None:
            preds[payload["name"]] = pred
    labels = load_labels(args.labels_csv)
    summary = {"members": {}, "families": {}, "ensembles": {}}
    for name, pred in preds.items():
        summary["members"][name] = score_by_id(pred, labels)
    families = {}
    for fam, names in (("convnextv2_tiny", [f"v2fold{k}" for k in range(5)]), ("convnext_tiny", [f"fold{k}" for k in range(5)]),
                       ("convnextv2_base", [f"v2base_fold{k}" for k in range(5)])):
        if all(n in preds for n in names):
            families[fam] = family_frame(preds, names)
            summary["families"][fam] = score_by_id(families[fam].reset_index(), labels, expected_ids=labels.ID)
    ids = next(iter(families.values())).index if families else None
    for ens_name in args.ensembles.split(","):
        ens = json.loads((ROOT / "configs" / "ensembles" / f"{ens_name}.json").read_text())
        weights: dict[str, float] = {}
        for m in ens["members"]:
            fam = json.loads((ROOT / "configs" / "members" / f"{m['name']}.json").read_text())["family"]
            if fam.endswith("_refit"):
                continue
            weights[fam] = float(m["weight"])
        if ids is None or any(f not in families for f in weights):
            print(f"{ens_name}: missing family predictions, skipped")
            continue
        summary["ensembles"][ens_name] = {"family_weights": weights, **score_by_id(geometric(families, weights, ids), labels, expected_ids=labels.ID)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    rows = [(k, v) for k, v in summary["families"].items()] + [(k, v) for k, v in summary["ensembles"].items()]
    print(f"\n{'set':>20}  {'CD':>8}  {'CV':>8}  {'HEX':>8}  {'mean':>8}")
    for k, v in rows:
        print(f"{k:>20}  {v['CD']:8.4f}  {v['CV']:8.4f}  {v['HEX']:8.4f}  {v['mean']:8.4f}")
    print(f"\nSummary -> {args.out / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```
Expected on the real cache (verification step): family convnextv2_tiny 8.86, convnext_tiny 9.00, ensemble v2ens mean 8.8389 (CD 6.3000, CV 10.2515, HEX 9.9652), v2ens_plus 8.814.

- [ ] **Step 4: Run tests, commit**

```bash
cd $NEW && $OLD/.venv/bin/python -m unittest discover -s tests -t . -v && git add -A && git -c user.name=Adi -c user.email=adi@visilant.org commit -q -m "Add out-of-fold scoring and bundle assembly"
```

---

### Task 7: Examples, parity fixtures, end-to-end test

**Files:**
- Create: `$NEW/examples/input/inputs.json`, `$NEW/examples/input/images/case.mha`, `$NEW/examples/expected_output/v2ens/*.json`, `$NEW/examples/expected_output/v2ens_plus/*.json`, `$NEW/examples/reference_val50_v2ens.csv`, `$NEW/examples/reference_val50_v2ens_plus.csv`, `$NEW/examples/README.md`, `$NEW/tests/fixtures/{parity_idx20.txt, parity_fold0_val20_cpu.csv, parity_v2fold0_val20_cpu.csv}`
- Create tests: `$NEW/tests/test_parity.py`, `$NEW/tests/test_folds.py`, `$NEW/tests/test_inference_example.py`

- [ ] **Step 1: Copy the fixture and expected outputs**

```bash
WT=/home/visilant/CLEAR-EC-phase1-v2ens/results
cp $WT/phase1_v2ens_plus/fixture/input/inputs.json $NEW/examples/input/
cp $WT/phase1_v2ens_plus/fixture/input/images/case.mha $NEW/examples/input/images/case.mha
mkdir -p $NEW/examples/expected_output/{v2ens,v2ens_plus}
cp $WT/phase1_v2ens/out_gpu/*.json $NEW/examples/expected_output/v2ens/
cp $WT/phase1_v2ens_plus/out_gpu/*.json $NEW/examples/expected_output/v2ens_plus/
cp $WT/phase1_v2ens_plus/local_reference_50.csv $NEW/examples/reference_val50_v2ens_plus.csv
cp $OLD/tests/fixtures/parity_idx20.txt $OLD/tests/fixtures/parity_fold0_val20_cpu.csv $OLD/tests/fixtures/parity_v2fold0_val20_cpu.csv $NEW/tests/fixtures/
```
Generate `reference_val50_v2ens.csv` from the night golden CSVs (same maths as the submitted replay):
```bash
cd /home/visilant/CLEAR-EC && code/.venv/bin/python - <<'EOF'
import json, glob, numpy as np, pandas as pd
from pathlib import Path
idx = pd.read_csv('data/cache/index.csv'); idx['ID']=idx.ID.astype(str).str.strip()
val = json.load(open('data/cache/splits.json'))['val'][:50]; rows = idx[idx.idx.isin(val)]
logs, w = [], []
for p in sorted(glob.glob('results/night_20260912/golden/*_val_flips.csv')):
    d = pd.read_csv(p); d['ID']=d.ID.astype(str).str.strip(); logs.append(np.log(d.set_index('ID').loc[rows.ID, ['CD','CV','HEX']].values.astype(float))); w.append(2 if Path(p).name.startswith('v2') else 1)
ens = np.exp(np.average(np.stack(logs), axis=0, weights=w))
clamp = {'CD':(372,4500),'CV':(0.03,1.5),'HEX':(0,1)}
ens = np.stack([np.clip(ens[:,k], *clamp[m]) for k,m in enumerate(['CD','CV','HEX'])], axis=1)
out = pd.DataFrame(ens, columns=['CD','CV','HEX']); out.insert(0,'ID',rows.ID.values); out.insert(0,'idx',rows.idx.values)
out.to_csv('/home/visilant/clear-ec-phase1-submission/examples/reference_val50_v2ens.csv', index=False); print(out.head(3))
EOF
```
Sanity: row for ID 1201-18 must be within 1e-6 relative of the fixture expected v2ens output (2479.2796, 0.43580, 0.46232).

- [ ] **Step 2: Port parity and folds tests**

`tests/test_parity.py`: copy `$OLD/tests/test_parity.py`; change constants to
```python
from tests._path import ROOT, SRC  # noqa: F401
FIXTURES = ROOT / "tests" / "fixtures"
CKPT = ROOT / "checkpoints" / "members"
CACHE = Path(os.environ.get("CLEAR_EC_CACHE_DIR", ROOT / "data" / "cache"))
LABELS = Path(os.environ.get("CLEAR_EC_LABELS_CSV", ROOT / "data" / "final_train_ids.csv"))
CASES = {"fold0": (CKPT / "fold0.pt", "parity_fold0_val20_cpu.csv", None),
         "v2fold0": (CKPT / "v2fold0.pt", "parity_v2fold0_val20_cpu.csv", "408ae5df5570ee82")}
```
drop the smallcnn case and the `golden` GPU test; keep `_check_cpu` (gated by `CLEAR_EC_FULL_PARITY=1`) and the GPU-vs-CPU-fixture test (rtol 5e-3); `_predict` loads with `load_checkpoint_file(path, device)`; imports become `clear_ec.*`; skip condition `CACHE.exists() and CKPT.exists()`.
`tests/test_folds.py`: copy, imports to `clear_ec.training.folds`, drop the `NIGHT` metrics check, keep the digest `60db43af95acb672` and the slide-disjointness check, skip when the cache is absent.

- [ ] **Step 3: End-to-end example test**

`tests/test_inference_example.py`:
```python
"""Run inference.py on the shipped example exactly as the container does and compare with the recorded outputs.
GPU results match to 1e-5 relative; CPU differs from the A5000 (TF32) at the 1e-3 level."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import torch

from tests._path import ROOT

CKPT = ROOT / "checkpoints" / "members"


@unittest.skipUnless(CKPT.exists() and (CKPT / "fold0.pt").stat().st_size > 1024, "checkpoints not pulled")
class ExampleTests(unittest.TestCase):
    def _run(self, ensemble: str) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            bundle = Path(tmp) / "model"; out = Path(tmp) / "out"
            subprocess.run([sys.executable, str(ROOT / "scripts" / "assemble_bundle.py"), "--ensemble",
                            str(ROOT / "configs" / "ensembles" / f"{ensemble}.json"), "--out", str(bundle)], check=True, capture_output=True)
            env = {**os.environ, "CLEAR_EC_INPUT_DIR": str(ROOT / "examples" / "input"), "CLEAR_EC_OUTPUT_DIR": str(out), "CLEAR_EC_MODEL_DIR": str(bundle)}
            res = subprocess.run([sys.executable, str(ROOT / "inference.py")], env=env, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr[-2000:])
            return {p.stem: json.loads(p.read_text()) for p in out.glob("*.json")}

    def _check(self, ensemble: str):
        got = self._run(ensemble)
        rtol = 1e-5 if torch.cuda.is_available() else 5e-3
        for name in ("cell-density", "coefficient-of-variation", "hexagonality"):
            want = json.loads((ROOT / "examples" / "expected_output" / ensemble / f"{name}.json").read_text())
            self.assertAlmostEqual(got[name] / want, 1.0, delta=rtol, msg=f"{ensemble} {name}: {got[name]} vs {want}")

    def test_v2ens_example(self):
        self._check("v2ens")

    def test_v2ens_plus_example(self):
        if os.environ.get("CLEAR_EC_FULL_PARITY") != "1" and not torch.cuda.is_available():
            self.skipTest("17-member CPU run is slow; set CLEAR_EC_FULL_PARITY=1")
        self._check("v2ens_plus")
```

- [ ] **Step 4: examples/README.md, run the suite on GPU, commit**

`examples/README.md`: what `input/` is (Grand Challenge interface layout, one validation image 1201-18), what `expected_output/` holds (per-ensemble outputs produced on an RTX A5000), what the two reference CSVs are (first 50 validation split images, ID order, produced by the local predictor) and how `scripts/replay_container.py` uses them.
```bash
cd $NEW && CUDA_VISIBLE_DEVICES=0 $OLD/.venv/bin/python -m unittest discover -s tests -t . -v 2>&1 | tail -20
git add -A && git -c user.name=Adi -c user.email=adi@visilant.org commit -q -m "Add example case, reference predictions, parity fixtures and end-to-end test"
```

---

### Task 8: Container build, export, smoke, replay

**Files:**
- Create: `$NEW/Dockerfile`, `$NEW/.dockerignore`, `$NEW/requirements-container.txt`, `$NEW/scripts/build_container.sh`, `$NEW/scripts/export_container.sh`, `$NEW/scripts/export_legacy_container.py`, `$NEW/scripts/run_example.sh`, `$NEW/scripts/replay_container.py`

- [ ] **Step 1: Dockerfile and requirements-container.txt**

`requirements-container.txt` (torch/torchvision come from the base image: torch 2.1.2, torchvision 0.16.2):
```
numpy==1.26.4
pandas==2.3.3
Pillow==11.3.0
SimpleITK==2.5.6
timm==1.0.29
```
`Dockerfile`:
```dockerfile
FROM --platform=linux/amd64 pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime
ENV PYTHONUNBUFFERED=1
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"
USER user
COPY --chown=user:user requirements-container.txt /opt/app/
RUN python -m pip install --user --no-cache-dir --no-color --requirement /opt/app/requirements-container.txt
COPY --chown=user:user src /opt/app/src
COPY --chown=user:user build/model /opt/app/model
COPY --chown=user:user inference.py /opt/app/
ENTRYPOINT ["python", "inference.py"]
```
`.dockerignore`:
```
*
!requirements-container.txt
!src/
!inference.py
!build/model/
**/__pycache__
```
Check `timm==1.0.29` installs on Python 3.10 / torch 2.1.2 (it did in the submitted image). pandas is imported by `clear_ec.training.common`; keep it.

- [ ] **Step 2: build_container.sh**

```bash
#!/usr/bin/env bash
# Build the algorithm image for one ensemble: scripts/build_container.sh v2ens|v2ens_plus
set -euo pipefail
cd "$(dirname "$0")/.."
ENS=${1:-v2ens}
TAG=clear_ec_phase1:$ENS
rm -rf build/model
python scripts/assemble_bundle.py --ensemble configs/ensembles/$ENS.json --out build/model
docker build --platform=linux/amd64 --tag "$TAG" .
echo "Built $TAG"
```

- [ ] **Step 3: run_example.sh**

```bash
#!/usr/bin/env bash
# Run the image offline on examples/input and print the three outputs. Usage: scripts/run_example.sh v2ens [gpu_index|cpu]
set -euo pipefail
cd "$(dirname "$0")/.."
ENS=${1:-v2ens}; DEV=${2:-0}
OUT=build/out_$ENS; mkdir -p "$OUT"; chmod o+rwX "$OUT"
GPU=(); [ "$DEV" != "cpu" ] && GPU=(--gpus "device=$DEV")
docker run --rm "${GPU[@]}" --platform=linux/amd64 --network none \
  -v "$PWD/examples/input:/input:ro" -v "$PWD/$OUT:/output" clear_ec_phase1:$ENS
for f in cell-density coefficient-of-variation hexagonality; do printf "%s: " $f; cat "$OUT/$f.json"; echo; done
echo "Expected (RTX A5000):"; for f in cell-density coefficient-of-variation hexagonality; do printf "%s: " $f; cat "examples/expected_output/$ENS/$f.json"; echo; done
```

- [ ] **Step 4: replay_container.py**

Runs inside the image (mounted), replays the 50 reference images through `clear_ec.ensemble.predict_image` with the baked bundle:
```python
"""Replay the 50 reference validation images through the container's bundle and compare to the reference CSV.
Run through docker (see README 3.8), mounting the raw MHA folder at /data and this file at /replay.py:
    docker run --rm --gpus device=0 --network none -v $PWD/scripts/replay_container.py:/replay.py:ro \
      -v $PWD/examples:/examples:ro -v $PWD/data/train_mha:/data:ro -v $PWD/build/out_replay:/output \
      --entrypoint python clear_ec_phase1:v2ens /replay.py v2ens
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/opt/app/src")
from clear_ec.ensemble import predict_image  # noqa: E402
from clear_ec.io import load_image  # noqa: E402

ens = sys.argv[1] if len(sys.argv) > 1 else "v2ens"
n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
ref = pd.read_csv(f"/examples/reference_val50_{ens}.csv").head(n)
ref["ID"] = ref.ID.astype(str).str.strip()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
bundle = Path("/opt/app/model")
out, times = [], []
for image_id in ref.ID:
    t = time.perf_counter()
    image = load_image(Path("/data") / f"{image_id}.mha")[..., 0].copy()
    p = predict_image(image, bundle, device, verbose=False)
    times.append(time.perf_counter() - t)
    out.append([p["CD"], p["CV"], p["HEX"]])
    print(f"{image_id}: CD {p['CD']:.4f} CV {p['CV']:.5f} HEX {p['HEX']:.5f} ({times[-1]:.1f} s)", flush=True)
out = np.array(out); golden = ref[["CD", "CV", "HEX"]].to_numpy(float)
rel = np.abs(out - golden) / golden
res = {"ensemble": ens, "n": int(len(out)), "torch": torch.__version__, "device": str(device),
       "sec_per_case_mean": float(np.mean(times)), "sec_per_case_max": float(np.max(times)),
       "max_rel_diff_vs_reference": {m: float(rel[:, k].max()) for k, m in enumerate(["CD", "CV", "HEX"])}}
print(json.dumps(res, indent=2))
Path("/output").mkdir(exist_ok=True)
Path(f"/output/replay_{ens}.json").write_text(json.dumps(res, indent=2))
pd.DataFrame(out, columns=["CD", "CV", "HEX"]).assign(ID=ref.ID.values, sec=times).to_csv(f"/output/replay_{ens}.csv", index=False)
```
MHA file names: `data/train_mha/<ID>.mha` (verified: `0981-18.mha`); keep the lookup by ID.

- [ ] **Step 5: export scripts**

Copy `$WT/code/scripts/export_legacy_container.py` unchanged to `scripts/`. `scripts/export_container.sh`:
```bash
#!/usr/bin/env bash
# Save the image in the legacy Docker archive format Grand Challenge accepts. Usage: scripts/export_container.sh v2ens
set -euo pipefail
cd "$(dirname "$0")/.."
ENS=${1:-v2ens}; TAG=clear_ec_phase1:$ENS; mkdir -p build
docker save "$TAG" -o build/${ENS}_docker.tar
python scripts/export_legacy_container.py build/${ENS}_docker.tar build/clear_ec_phase1_${ENS}.tar.gz
rm -f build/${ENS}_docker.tar
(cd build && sha256sum clear_ec_phase1_${ENS}.tar.gz | tee clear_ec_phase1_${ENS}.tar.gz.sha256)
```
Requires `pigz`; state it in the README.

- [ ] **Step 6: Build, smoke, replay both ensembles, commit**

```bash
cd $NEW && PYTHON=$OLD/.venv/bin/python bash -c 'sed -i "s#^python scripts#\${PYTHON:-python} scripts#" scripts/build_container.sh' # allow PYTHON override
scripts/build_container.sh v2ens 2>&1 | tail -3
scripts/run_example.sh v2ens 0
mkdir -p build/out_replay && chmod o+rwX build/out_replay
docker run --rm --gpus device=0 --platform=linux/amd64 --network none -v $PWD/scripts/replay_container.py:/replay.py:ro -v $PWD/examples:/examples:ro -v /home/visilant/CLEAR-EC/data/train_mha:/data:ro -v $PWD/build/out_replay:/output --entrypoint python clear_ec_phase1:v2ens /replay.py v2ens | tail -12
scripts/build_container.sh v2ens_plus 2>&1 | tail -3 && scripts/run_example.sh v2ens_plus 0
docker run --rm --gpus device=0 --platform=linux/amd64 --network none -v $PWD/scripts/replay_container.py:/replay.py:ro -v $PWD/examples:/examples:ro -v /home/visilant/CLEAR-EC/data/train_mha:/data:ro -v $PWD/build/out_replay:/output --entrypoint python clear_ec_phase1:v2ens_plus /replay.py v2ens_plus | tail -12
```
Pass criteria: fixture outputs equal the expected JSON to at least 1e-6 relative; replay max_rel_diff below 1e-5 for all three metrics on both ensembles. Record the numbers for README 3.7/3.8.
```bash
git add -A && git -c user.name=Adi -c user.email=adi@visilant.org commit -q -m "Add container build, export, example run and replay scripts"
```

---

### Task 9: Environment files and README

**Files:**
- Create: `$NEW/requirements.txt`, `$NEW/environment.yml`, `$NEW/README.md`

- [ ] **Step 1: requirements.txt and environment.yml**

`requirements.txt` (training/evaluation host; the versions the members were trained with):
```
--extra-index-url https://download.pytorch.org/whl/cu130
torch==2.13.0
torchvision==0.28.0
timm==1.0.29
numpy==1.26.4
pandas==2.3.3
scipy==1.13.1
SimpleITK==2.5.6
Pillow==11.3.0
PyYAML==6.0.3
```
`environment.yml`: name `clear-ec`, `python=3.12`, `pip`, and the same pins under `pip:`.
Before writing, confirm the cu130 wheel index actually serves torch 2.13.0 / torchvision 0.28.0 (`pip download --no-deps --dry-run` or `pip index versions torch --extra-index-url ...`); if not, pin what the index offers and state the tested combination in README 3.2.

- [ ] **Step 2: README.md**

Write with the organisers' numbering, every command copy-pasteable from the repo root. Required content and the measured numbers to quote:

1. Overview: task, team Visilant, method in three sentences, results table (v2ens OOF 8.8389 / hidden 8.7583, 4th of the Phase I leaderboard on 2026-09-12; v2ens_plus OOF fold-only 8.814, not uploaded).
2. Repository map (tree with one line per entry).
3.1 Environment setup: `python3.12 -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt` or `conda env create -f environment.yml`; Docker 24+ with buildx and NVIDIA container toolkit for the container path; `git lfs install`.
3.2 Versions: container Python 3.10, torch 2.1.2+cu121, torchvision 0.16.2, timm 1.0.29, CUDA 12.1 runtime, cuDNN 8; host Python 3.12.13, torch 2.13.0+cu130, timm 1.0.29, driver 580; tested GPUs RTX A5000 24 GB.
3.3 Checkpoints: `git lfs pull`, `scripts/verify_checkpoints.sh`, table of 17 members with sizes (fold*.pt 106 MiB, v2fold*/v2refit* 106 MiB, v2base* 335 MiB, total 2.9 GiB), LFS quota note, mirror subsection.
3.4 Data: download the nine `train_mha_0*.zip` and `final_train_ids.csv` from the challenge's training-data release, unzip into `data/train_mha/`, expected tree, `python scripts/build_cache.py` (11 GB memmap, about 3 minutes), the CSV columns `ID, Image Number, CD, CV, HEX, slide_id`.
3.5 Commands: (a) predict one image locally without Docker: `CLEAR_EC_INPUT_DIR=examples/input CLEAR_EC_OUTPUT_DIR=build/out CLEAR_EC_MODEL_DIR=build/model python inference.py` after `python scripts/assemble_bundle.py ...`; (b) container: build, run_example, export; (c) OOF evaluation `python scripts/predict_oof.py`; (d) training `scripts/train_all_members.sh` or one member; (e) tests.
3.6 Outputs: the three JSON files, `results/oof/`, `results/<member>/...`, `build/`.
3.7 Hardware and runtime: training epoch times (Tiny 80 s, V2-Tiny 158 s, V2-Tiny all-data 212 s, V2-Base 352 s on an A5000; 8 or 12 epochs), inference per case (v2ens 18 s GPU including start-up, 32 s CPU two cores; v2ens_plus 31 s GPU on a shared A5000), VRAM (batch 8 at 486x648 fits in 24 GB), disk (cache 11 GB, checkpoints 2.9 GB, images 5 to 7 GB).
3.8 Reproduction: exact ordered steps for (i) reproducing the submitted predictions from the shipped checkpoints (example + replay + OOF numbers), (ii) rebuilding the container archive, (iii) retraining all members and re-assembling; note that retraining yields checkpoints that differ bitwise (cuDNN nondeterminism) with OOF within about 0.05.
6. Checklist 6.1 to 6.12 with a one-line evidence pointer each.
Contact line: team Visilant, adi@visilant.org.

- [ ] **Step 3: Commit**

```bash
git add -A && git -c user.name=Adi -c user.email=adi@visilant.org commit -q -m "Add README, requirements and conda environment"
```

---

### Task 10: Technical report (Markdown + PDF)

**Files:**
- Create: `$NEW/Technical_Report.md`, `$NEW/docs/Technical_Report.tex`, `$NEW/Technical_Report.pdf`, `$NEW/scripts/build_report.sh`

- [ ] **Step 1: Write Technical_Report.md**

Sections: 1 Problem and metrics (CD, CV, HEX; mean absolute percentage error; the ground truth is a manual centre-method count in a fixed box, so segmentation-based measurement was abandoned in favour of direct regression); 2 Data (9000 frames 972x1296 uint8 from 6771 slides; label ranges; slide-grouped protocol); 3 Method (whole-frame input downsampled to 486x648; ImageNet-pretrained ConvNeXt-Tiny / ConvNeXt-V2-Tiny / ConvNeXt-V2-Base; three-output linear head on normalised targets; relative absolute error loss; AdamW lr 1e-4 wd 1e-4, batch 8, AMP, channels-last, flip augmentation, EMA 0.999, cosine schedule with one warm-up epoch, fixed budget 8 epochs (12 for Tiny), grad-clip 1.0 for V2; last-epoch checkpoint; five slide-grouped folds seed 42 over all 9000 images; flip TTA; weighted geometric mean; clamps); 4 Validation methodology (OOF over 9000, paired slide bootstrap for comparisons, noise floor estimate about 5 to 8 percent from label noise); 5 Results (table: family OOF; v2ens 8.8389; platform 8.7583; v2ens_plus 8.814; per-metric); 6 Code organisation (map of `src/clear_ec`, scripts, configs); 7 Implementation details (uint8 memmap cache; deterministic splits and folds with digests; frozen RegressionConfig embedded in checkpoints; checksum-verified bundles; container CPU budget from cgroup; TF32 note); 8 Reproducibility and limitations (bitwise retrain not guaranteed; refits unvalidatable; no test-time calibration; runtime).

- [ ] **Step 2: LaTeX and PDF**

Write `docs/Technical_Report.tex` (article class, `booktabs`, `hyperref`, no external images) mirroring the Markdown, and `scripts/build_report.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build docs/Technical_Report.tex >/dev/null
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build docs/Technical_Report.tex >/dev/null
cp build/Technical_Report.pdf Technical_Report.pdf && echo "Technical_Report.pdf written"
```
Run it; open the PDF page count (`pdfinfo` or `python -c` with PyPDF absent -> use `pdfinfo` if present, otherwise `ls -la`).

- [ ] **Step 3: Commit**

```bash
git add -A && git -c user.name=Adi -c user.email=adi@visilant.org commit -q -m "Add technical report (Markdown, LaTeX source, PDF)"
```

---

### Task 11: Clean-environment verification and signature scan

- [ ] **Step 1: Fresh virtual environment from requirements.txt**

```bash
cd $NEW && python3.12 -m venv /tmp/claude-1000/-home-visilant-CLEAR-EC/*/scratchpad/venv_check 2>/dev/null || uv venv --python 3.12 build/venv_check
# prefer uv if present: uv pip install --python build/venv_check/bin/python -r requirements.txt
build/venv_check/bin/python -m unittest discover -s tests -t . -v 2>&1 | tail -15
```
All non-skipped tests pass; record which were skipped and why (cache present -> parity + folds run; set `CLEAR_EC_FULL_PARITY=1` once for the CPU-exact checks, about 3 minutes).

- [ ] **Step 2: Training smoke**

```bash
build/venv_check/bin/python scripts/build_cache.py --data_dir /home/visilant/CLEAR-EC/data/train_mha --labels_csv /home/visilant/CLEAR-EC/data/final_train_ids.csv --cache_dir build/cache_smoke --limit 200
build/venv_check/bin/python scripts/train_member.py --config configs/members/fold0.json --cache_dir build/cache_smoke --labels_csv /home/visilant/CLEAR-EC/data/final_train_ids.csv --results_dir build/results_smoke --epochs 1 --gpu 0
ls build/results_smoke/fold0/regression_cnn/seed_123/
```
Expect `last.pt history.csv metrics.json best_model.pt predictions_val.csv` and `manifest.json` one level up. Note: the fold split over a 200-image cache is different from the full one; this checks the code path only.

- [ ] **Step 3: OOF numbers on the full cache**

```bash
build/venv_check/bin/python scripts/predict_oof.py --cache_dir /home/visilant/CLEAR-EC/data/cache --labels_csv /home/visilant/CLEAR-EC/data/final_train_ids.csv --out build/oof --gpu 1 2>&1 | tail -12
```
Expect v2ens 8.8389 and v2ens_plus 8.814 (within 1e-3). Put the resulting table in README 3.8 and the report.

- [ ] **Step 4: Signature scan and history check**

```bash
cd $NEW && grep -rIil "claude\|anthropic\|co-authored\|noreply@anthropic" --exclude-dir=.git . || echo "tree clean"
git log --format='%an <%ae>%n%B' | grep -i "claude\|anthropic\|co-authored" || echo "history clean"
git status --short | wc -l   # 0
```

- [ ] **Step 5: Final commit and hand-over note**

Update README 3.7/3.8 with the measured numbers from Steps 1-3, commit, then report: repo path, commit hash, test counts, replay deltas, LFS size, what was not done (retraining, push, collaborator, mirror).

## Self-review

- Spec coverage: layout (T1-T10), ported/dropped (T2, T3, T8), configs (T4), checkpoints/LFS (T4), container (T8), docs (T9, T10), signature policy (T11), verification 1-7 (T4, T7, T8, T11), out of scope respected.
- No placeholders: every script and test is given in full; README/report content is specified section by section with the numbers to quote.
- Names consistent: `predict_image`, `combine`, `cpu_budget`, `load_checkpoint_file`, `assemble_bundle.py --ensemble --out --checkpoint_dir`, env vars `CLEAR_EC_INPUT_DIR/OUTPUT_DIR/MODEL_DIR`, member names and weights.
