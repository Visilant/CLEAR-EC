# Working in CLEAR-EC

## Purpose and layout

Estimate cell density (CD), coefficient of variation (CV), and hexagonality
(HEX) from corneal endothelial microscopy images for the CLEAR-EC challenge.

- `code/inference.py`: Grand Challenge container entrypoint, one case per run.
- `code/main.py`: local batch inference; `code/evaluate.py`: CSV scoring.
- `code/src/infer_cellpose_sam.py`: segmentation pipeline; despite its name,
  the submission currently uses the installed Cellpose 1.0.2 `cyto` model.
- `code/src/utils/evaluate.py`: baseline metric definitions and evaluation.
- `code/src/data/`: image memmap cache, slide splits, deterministic crops,
  configuration hashes, segmentation masks, and metric calculation.
- `code/src/training/`: direct regression CNN, ridge calibration, scoring.
- `code/scripts/`: cache building, sweeps, training, and overnight experiments.
- `code/eda/`: four analysis phases; see its README for commands and outputs.
- `code/viewer/`: local Streamlit image, segmentation, and metric viewer.
- `code/tests/test_protocol.py`: split isolation, scoring, and CLI tests.
- `data/` and `results/`: ignored local datasets, caches, and experiment artifacts.

## Local commands

Run Python commands from `code/`. An existing local environment is at
`code/.venv`; use `.venv/bin/python` when available.

```bash
cd code
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python main.py --help
.venv/bin/python -m eda.run_eda --help
.venv/bin/python scripts/run_training.py --help
.venv/bin/python -m streamlit run viewer/app.py
```

Runtime dependencies are in `code/requirements.txt`; Streamlit is in
`code/requirements-dev.txt`. The Docker base supplies PyTorch. Do not assume
`environment.yml` exactly matches all runtime dependencies.

Container scripts are invoked from `code/` with `bash do_build.sh`,
`bash do_test_run.sh`, and `bash do_save.sh`. The smoke test needs local
`test/input/interf0/` fixtures and a `model/` directory. Fixtures are absent;
the readiness audit created a local regression bundle in `code/model/`.
Check prerequisites before running expensive builds.

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

## State observed at initialization (2026-09-11)

The local overnight report records 9,000 images split 7,202/892/906 across
train/val/test. It reports mean test errors of 27.28% for tuned Cellpose,
14.83% for ridge calibration, and 16.59–23.29% for three regression CNN seeds.
See `results/overnight/REPORT.md` and saved training artifacts for provenance;
these are local evaluation results, not verified challenge leaderboard scores.

The tuned Cellpose uses diameter 40 and crop fraction 0.5. The submission
entrypoint defaults to automatic diameter and crop fraction 0.4, with no ridge
calibration applied. It now accepts explicit regression bundles mounted at
`/opt/ml/model` with `submission.json` and `best_model.pt`; export with
`code/scripts/export_regression.py`. Do not assume a model is deployed unless
the intended bundle is attached.

See `AUDIT.md` for the subsequent readiness review and fixes. In particular,
epoch scoring must use original labels: reconstructing zero targets from
normalized float32 values corrupted prior checkpoint selection. A corrected
ten-epoch run is saved under `results/audit_20260911/selection_fix/`, with
12.51% validation error confirmed in the offline container at batch size 1.
No new test scoring or official submission was performed. Four exact image
duplicate pairs cross train/test; do not present the existing test split as
fully independent. The existing splits were preserved.

Existing user changes at initialization: modified `code/.dockerignore`,
untracked `code/requirements-dev.txt`, and untracked `code/viewer/`. Preserve
them. The top-level README describes the baseline and omits newer workflows;
verify claims against code and artifacts.
