# Running CLEAR-EC experiments as a researcher agent

This is the loop an agent (or a person) follows to test an idea on CLEAR-EC without re-deriving
anything. Read `docs/EXPERIMENTS.md` first: it lists every run so far, the verdict on every lever,
and what must not be rerun. Read `docs/HANDOFF_2026-09-12.md` for the science (noise floor, why the
remaining headroom is in CD, Phase II directions).

## 1. Environment and data

- Python: `code/.venv` (3.12, torch 2.13+cu130, timm 1.0.29, torchvision 0.28, pyyaml). It has no pip;
  add packages with `uv pip install --python code/.venv/bin/python <package>`.
- Hardware: two RTX A5000 24 GB. ConvNeXt-V2-Tiny at 486x648: about 100 s/epoch alone, 160 s with
  both GPUs busy; ConvNeXt-Tiny 85 s; V2-Base 352 s; the small CNN 5 s.
- Data: `data/final_train_ids.csv` (9,000 labels), `data/cache/` (uint8 memmap 9000x972x1296,
  `index.csv` with slide ids, `splits.json` train 7,202 / val 892 / test 906, slide-disjoint).
- Folds: `src.training.folds._slide_group_folds(cache, 5, seed=42)` is deterministic and frozen by
  `tests/test_folds.py` (sizes 1802/1783/1802/1788/1825). Every fold model in the submission used it.
- Run everything from `code/`: `cd code && .venv/bin/python -m experiments.run --help`.

## 2. The loop

1. **Decide what the run must show.** Out-of-fold over the 9,000 labelled images is the compass. On
   the 100 hidden Phase I images one candidate's score has SD 0.96, and a paired difference between
   two correlated candidates has SD 0.06 to 0.12, so an OOF gain under 0.10 is invisible there; on
   the 1,000-image Phase II set the bar is 0.03 to 0.05. State the gate before training.
2. **Write a spec** in `code/experiments/specs/<name>.yaml` (copy `night_v2_5fold.yaml`). A spec is a
   base recipe (any `RegressionConfig` field), arms (overrides plus `seeds`, `n_folds`/`folds`, or
   `all_data`), the GPUs, and post steps (score with TTA, ensembles, paired comparisons). Unknown
   fields are rejected at load time. `recipe_hash` in each job's `config.json` identifies the recipe
   so folds and seeds of one recipe group together in the ledger.
3. **Dry-run**: `python -m experiments.run run experiments/specs/<name>.yaml --dry-run` prints the
   expanded jobs, which are already done (skipped), and the exact worker command lines.
4. **Launch**: `python -m experiments.run run experiments/specs/<name>.yaml --detach`. The driver
   writes `results/<name>/{plan.json,status.json,driver.log,logs/<job>.log,source/}`; `plan.json`
   holds the spec text, git commit, and a sha256 of every source file that ran. Jobs are skipped
   when their `metrics.json` exists, so a crashed campaign is resumed by launching again.
   `--limit N` runs N pending jobs; `--gpus 0` pins one GPU.
5. **Watch**: `python -m experiments.run status results/<name>`; `tail -f results/<name>/driver.log`.
6. **Post steps** run automatically after training (or `score`, then `report`): held-out
   predictions with flip TTA per job (`scores/`), OOF concatenation per fold arm, weighted
   geometric-mean ensembles (`ensembles/`), and paired slide-clustered bootstrap comparisons
   (`compare/`). `REPORT.md` in the results dir tabulates all of it.
7. **Ledger**: `python -m experiments.run ledger` regenerates `docs/EXPERIMENTS.md` and
   `docs/experiments.csv` from every campaign, including the new one. Add prose-only facts
   (a platform score, a decision) to `code/experiments/ledger_manual.yaml`, and add a verdict row
   there for the lever you tested so nobody reruns it. Commit the docs and the spec.
8. **Decide** with the paired CI, not the point estimate; the reference is `results/night_20260912/oof/*_last_flips.csv`
   (Tiny folds) or the `v2fold` arm of a V2 rerun. Anything that does not clear its gate is a verdict, not a failure.

## 3. Recipe facts that are settled (do not rediscover)

- Recipe of the submitted models: relative (MAPE) loss, AdamW lr 1e-4, wd 1e-4, batch 8, AMP,
  channels-last, flips, EMA 0.999, cosine with 1 warm-up epoch, fixed budget (8 epochs V2-Tiny,
  12 epochs Tiny), `patience: 0`, last checkpoint. `clip_grad: 1.0` is mandatory for ConvNeXt-V2
  (seed-42 runs spiked at epoch 3 without it).
- Judge fixed-budget runs on the last epoch; best-epoch selection is worth 0.02 and biases val.
- All-data refits cannot be validated; check `history.csv` train loss is monotone before use.
- Seeds correlate 0.994; cross-family ensembles gain 0.02; recalibration is dead; resolution,
  crops, augmentation, capacity, longer training all tie. CV and HEX sit at the label-noise floor;
  the only headroom is CD, and it lives in donor-level mislabels a frame-level regressor cannot fix.
- Phase II direction: a cell-centre detector for CD (code in `code/phase2/detector`, gates in its
  `SPEC.md`), plus a donor-level label audit. The Voronoi readout on true clicks gives CD 3.1%.

## 4. Submission path

Platform rules as confirmed on 2026-09-14 (they differ from the earlier "3 submissions" reading):
Phase I submissions are unlimited, container image uploads are limited (one more image after the
v2ens_plus image, version `38863e52`, which is the live algorithm), Model archives can be uploaded
and attached without limit, and the leaderboard keeps a team's best score. Consequences: every
candidate should be a Model archive (`submission.json` + checkpoints at the archive root; the live
image reads members, weights, `tta` and `clamp` from it and can build V2-Tiny, Tiny and V2-Base),
the remaining container upload is reserved for a code change (for example a detector-based method),
and an attached Model at `/opt/ml/model` always overrides the bundle baked into the image. Repeated
submissions of near-identical candidates select on the 100 hidden images (one candidate's score has
SD about 0.06 relative to another's); record every platform score in
`code/experiments/ledger_manual.yaml` and treat the spread as noise, not as a lever.


The submitted container lives in the sibling worktree `/home/visilant/CLEAR-EC-phase1-v2ens`
(branch `submission/phase1-v2ens`, tag `phase1-slot1-v2ens`): its `docs/PHASE_I_SUBMISSION_V2ENS.md`
has the build, replay and export steps; its `inference.py` is the reference for preprocessing,
TTA and clamps (`[372, 4500]`, `[0.03, 1.5]`, `[0, 1]`). New members must be scored through
`src/training/predict.py` (the same code path, parity-tested against the night's golden
predictions) and replayed through the container before upload. Container images:
`clear_ec_phase1:v2ens` (submitted).

## 5. Rules that keep results honest

- Never overwrite a run directory; the trainer refuses to, and the runner skips finished jobs.
- Join predictions to labels by `ID`; write artifacts atomically (`src.artifacts.atomic_path`).
- Keep `data/`, `results/`, weights and images out of git; commit specs, the ledger and reports.
- Run `cd code && .venv/bin/python -m unittest discover -s tests` before committing trainer
  changes; `CLEAR_EC_FULL_PARITY=1` adds the slow CPU-exact checkpoint parity checks.
- Optional disk follow-up (not done on 2026-09-13): `best_model.pt` of fixed-budget runs and the
  checkpoints of tied `x_*` arms in `results/night_20260912` (about 6 GB) are reproducible from
  their specs and can be pruned once Phase I closes.
