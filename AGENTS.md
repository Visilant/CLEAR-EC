# Working in CLEAR-EC

## Purpose and layout

Estimate cell density (CD), coefficient of variation (CV), and hexagonality
(HEX) from corneal endothelial microscopy images for the CLEAR-EC challenge.
Current method: direct regression with ImageNet-pretrained ConvNeXt backbones,
trained on slide-grouped folds over all 9,000 labelled images.

- `docs/EXPERIMENTS.md`: generated ledger of every experiment, leaderboard, verdicts.
- `docs/RESEARCHER_GUIDE.md`: how to run a new campaign end to end.
- `docs/HANDOFF_2026-09-12.md`: the science (noise floor, headroom, Phase II directions).
- `code/inference.py`: Grand Challenge container entrypoint (ships with `code/src/` only).
- `code/src/training/`: the trainer split into `config`, `targets`, `data`, `models`,
  `losses`, `loop`, `folds`, `predict`, `train`; `regression_cnn.py` re-exports the old names.
- `code/src/data/`: image memmap cache, slide splits, deterministic crops, config hashes.
- `code/experiments/`: YAML experiment specs, the GPU queue, scoring, ensembles, paired
  comparisons, reports, and the ledger (`python -m experiments.run`).
- `code/scripts/`: `run_training.py` (one run; used by the runner), `build_cache.py`,
  `export_regression.py`, `benchmark_local.py`, `summarize_round1.py`, `noise_floor/`.
- `code/legacy/`: the August Cellpose pipeline (segmentation cache, metric sweep, ridge
  calibration, overnight orchestrator); runnable, tested, not used by current work.
- `code/phase2/detector/`: the SAM/Cellpose-SAM cell-centre detection spike (gates failed).
- `code/eda/`: four analysis phases; `code/viewer/`: local Streamlit image/metric viewer.
- `code/tests/`: protocol, audit, parity, config/fold freezes, spec and CLI tests.
- `data/` and `results/`: ignored local datasets, caches, and experiment artifacts.

## Local commands

Run Python commands from `code/` with `.venv/bin/python` (no pip inside; add packages with
`uv pip install --python code/.venv/bin/python <pkg>`).

```bash
cd code
.venv/bin/python -m unittest discover -s tests -v      # CLEAR_EC_FULL_PARITY=1 for slow CPU parity
.venv/bin/python -m experiments.run run experiments/specs/smoke.yaml --dry-run
.venv/bin/python -m experiments.run run experiments/specs/<spec>.yaml --detach
.venv/bin/python -m experiments.run status results/<name>
.venv/bin/python -m experiments.run ledger
.venv/bin/python scripts/run_training.py --help
.venv/bin/python legacy/run_overnight.py --help
.venv/bin/python -m streamlit run viewer/app.py
```

Runtime dependencies are in `code/requirements.txt`; Streamlit is in
`code/requirements-dev.txt`. The Docker base supplies PyTorch. Do not assume
`environment.yml` exactly matches all runtime dependencies.

Container scripts are invoked from `code/` with `bash do_build.sh`,
`bash do_test_run.sh`, and `bash do_save.sh`; the submitted build lives in the
sibling worktree `/home/visilant/CLEAR-EC-phase1-v2ens` (see its docs).

## Preserve experiment integrity

- Keep train, validation, and test partitions disjoint by image and slide.
- Fit on train; select configurations/checkpoints/calibration alpha on val;
  evaluate test only after freezing the choices. Do not tune on test results.
- Preserve deterministic per-image crop behavior and config-keyed caches.
- Join predictions and ground truth by `ID`, not row order.
- Do not overwrite saved experiment artifacts or rebuild large caches unless
  the task requires it. Keep datasets, weights, and generated outputs out of Git.
- Run the protocol tests for changes to splits, scoring, or experiment logic.

## Submission contract

The container reads `/input/inputs.json` and an image under
`/input/images/corneal-specular-microscopy-image/`. It writes three scalar
JSON numbers under `/output/`: `cell-density.json`,
`coefficient-of-variation.json`, and `hexagonality.json`.

Runtime must work offline; bundle required model weights. CD is cells/mm²,
CV is a ratio, and baseline HEX is fitted-hexagon IoU, not a count of
six-sided cells. Preserve these semantics unless explicitly changing the method.

## State (2026-09-13)

- Git: everything is on `main`; tags `phase1-slot1-v2ens` (submitted container, branch
  `submission/phase1-v2ens`, worktree `/home/visilant/CLEAR-EC-phase1-v2ens`, frozen) and
  `phase1-seed123-unused`. Older branches and worktrees were merged and deleted on 2026-09-13.
- Score: Phase I slot 1 scored 8.7583 on the hidden 100 images (4th; top five 8.70 to 8.77).
  Locally the same ensemble is 8.84 out-of-fold over 9,000 images. Submissions are unlimited but
  one container upload remains (live image: v2ens_plus, `38863e52`); candidates go up as Model
  archives attached to it. The 17-member v2ens_plus scored 8.8531 on 2026-09-14 (worse than
  slot 1 on the same 100 images). Current state and open threads: docs/HANDOFF_2026-09-14.md.
- Every lever tried since (label cleaning, capacity, longer training, seeds, TTA, calibration,
  resolution, augmentation) moved the score by 0.03 or less; CV and HEX sit at the label-noise
  floor. `docs/EXPERIMENTS.md` carries the verdict table; do not rerun what it marks.
- Checkpoint compatibility: `RegressionConfig` field names and defaults are frozen by
  `tests/test_config.py`; `tests/test_parity.py` reproduces the night's golden predictions.
- The August baseline numbers (Cellpose 27.28, ridge 14.83 on the 906 test split) and the
  readiness audit (`AUDIT.md`, corrected small CNN 12.51 val) are historical; the submission
  entrypoint accepts regression bundles at `/opt/ml/model`, and `code/model/` still holds the
  audited small-CNN bundle, not the deployed ensemble.
