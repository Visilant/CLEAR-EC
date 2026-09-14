# Phase I technical-review repository: design

Date: 2026-09-14. Deadline for the organisers' review materials: 2026-09-16.

## Goal

Produce a self-contained, signature-free repository that lets the CLEAR-EC evaluation team
set up, run, and reproduce Visilant's Phase I submission from scratch, following the
organisers' request (complete source, checkpoints, step-by-step README, technical document,
recommended structure, checklist). The repository is created fresh at
`/home/visilant/clear-ec-phase1-submission` with its own git history; nothing is filtered or
rewritten from the research repository. The research repository (`/home/visilant/CLEAR-EC`)
is unchanged apart from this spec, the plan, and a memory note.

## Targets to reproduce

| Entry | Members | Weights | Local OOF (9000) | Platform |
|---|---|---|---:|---|
| v2ens (Phase I slot 1) | 5x ConvNeXt-V2-Tiny folds, 5x ConvNeXt-Tiny folds | 2 : 1 | 8.8389 | 8.7583 hidden set, 2026-09-12 |
| v2ens_plus (prepared, not uploaded) | v2ens + 2 all-data V2-Tiny refits + 5 ConvNeXt-V2-Base folds | 2 : 1 : 2 : 3 | 8.814 (fold-only part) | not evaluated |

Both are weighted geometric means over members, each member evaluated with four-flip TTA,
followed by physical clamps CD [372, 4500], CV [0.03, 1.5], HEX [0, 1]. Seventeen unique
checkpoints in total (about 2.9 GB); the ten v2ens members are a subset of v2ens_plus.

## Repository layout

```
clear-ec-phase1-submission/
  README.md                     items 3.1-3.8 of the request, in that order
  Technical_Report.md           item 4; Technical_Report.pdf built from it with pdflatex
  requirements.txt              training / evaluation environment, pinned
  environment.yml               conda equivalent of requirements.txt
  Dockerfile, .dockerignore     Grand Challenge algorithm container (unchanged base image)
  inference.py                  container entrypoint (ensemble path only)
  src/clear_ec/                 package: data/ (cache, splits, folds), training/ (config,
                                targets, data, models, losses, loop, train, predict),
                                ensemble.py (manifest, geometric mean, clamps), io.py
  configs/members/*.json        17 member configs written from the saved run manifests
  configs/ensembles/{v2ens,v2ens_plus}.json   member list + weights + clamps
  scripts/build_cache.py        MHA folder + labels CSV -> uint8 memmap cache + index + splits
  scripts/train_member.py       one member from a config JSON (fold / all-data)
  scripts/train_all_members.sh  the 17 commands in order, two-GPU friendly
  scripts/predict_oof.py        per-member OOF predictions with TTA, ensemble score table
  scripts/assemble_bundle.py    ensemble JSON + checkpoints -> build/model/ (submission.json)
  scripts/build_container.sh    docker build with bundle baked at /opt/app/model
  scripts/export_container.sh   legacy-format tar.gz for Grand Challenge upload
  scripts/run_example.sh        offline docker run on examples/ input, prints the three JSONs
  scripts/replay_container.py   50 validation images through the container vs reference CSV
  scripts/verify_checkpoints.sh sha256sum -c over checkpoints/SHA256SUMS
  checkpoints/members/*.pt      17 LFS-tracked checkpoints
  checkpoints/SHA256SUMS
  examples/input/{inputs.json, images/case.mha}   fixture case (val image 1201-18)
  examples/expected_output/*.json                 v2ens outputs for the fixture
  examples/reference_val50_{v2ens,v2ens_plus}.csv 50-image golden predictions for replay
  tests/                        config freeze, fold assignment, CPU parity fixtures,
                                ensemble math, end-to-end fixture through inference.py
```

## What is ported and what is dropped

Ported, behaviour-identical: `code/src/data/{cache,splits,config}.py`, `code/src/artifacts.py`,
`code/src/io_utils.py`, `code/src/training/*` (minus the legacy `regression_cnn.py` re-export
shim and the SmallRegressionCNN unless the config freeze test needs it), the ensemble branch
of the worktree `inference.py`, `scripts/build_cache.py`, `scripts/run_training.py`,
`scripts/export_regression.py`, `scripts/export_legacy_container.py`, the container replay
script, and the tests `test_config`, `test_folds`, `test_parity` (adapted to the new paths and
to fixtures shipped in the repo instead of `results/`).

Dropped: the Cellpose legacy pipeline and vendored source, `infer_cellpose_sam.py`, the
`predict_baseline` fallback in `inference.py` (never reached when a bundle is present), the
Cellpose weight download in the Dockerfile, the SAM detector spike, EDA, the viewer, the
experiment runner and ledger, the noise-floor scripts. Removing Cellpose changes no
prediction; the 50-image replay proves it.

Renames: the package becomes `clear_ec` under `src/`; imports change from `src.x` to
`clear_ec.x`. No algorithmic change, no default change. The frozen `RegressionConfig`
field set and defaults stay byte-identical (config freeze test).

## Member configs

Each `configs/members/<name>.json` is `{"name", "config": {...RegressionConfig...},
"seed", "family", "weight_in": {"v2ens": w, "v2ens_plus": w}}` with the `config` object copied
from the run manifest (`results/night_20260912/<run>/manifest.json`,
`results/noise_floor_20260912/b3_v2base_fold*/manifest.json`). `train_member.py --config
configs/members/x.json` reproduces the run; `--gpu`, `--cache_dir`, `--labels_csv`,
`--results_dir` are the only CLI overrides.

## Checkpoints

`checkpoints/members/<name>.pt` are the `last.pt` files already copied into the two build
contexts, renamed to the member names used in the manifests. `.gitattributes` tracks
`checkpoints/members/*.pt` with Git LFS. `SHA256SUMS` lists all 17. The README states the
GitHub LFS quota constraint and gives `git lfs pull` plus the checksum verification as the
primary path, and reserves a "mirror" subsection for an external URL should one be added.

## Container

Same base image `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime`, same system libraries,
requirements reduced to what the ensemble path imports (numpy, SimpleITK, timm,
torchvision for `convnext_tiny`, Pillow), bundle baked at `/opt/app/model`, platform mount
`/opt/ml/model` takes precedence. Entry: `python inference.py`, reads
`/input/images/corneal-specular-microscopy-image/*.mha`, writes
`/output/cell-density.json`, `/output/coefficient-of-variation.json`,
`/output/hexagonality.json`. CPU thread budget from the cgroup quota is kept.

## Documentation

README sections, in the organisers' numbering: 1 overview and results table; 2 repository
map; 3.1 environment setup (venv and conda, Docker); 3.2 versions (Python 3.10 in the
container, torch 2.1.2 cu121, timm 1.0.29, torchvision 0.16.2; training host versions listed
separately); 3.3 checkpoints (LFS pull, checksums, placement); 3.4 data preparation (nine
zip archives -> `data/train_mha/`, `final_train_ids.csv`, cache build, expected tree);
3.5 exact commands (inference on a folder, container run, OOF evaluation, training all
members, bundle assembly, container export); 3.6 outputs and locations; 3.7 hardware and
runtime (measured numbers: 157 s/epoch V2-Tiny on A5000, 18 s and 31 s per case in the
container on GPU, 32 s CPU two cores for v2ens); 3.8 reproduction of the submitted results
step by step; then the submission checklist 6.1-6.12 with the evidence for each item.

Technical_Report: problem and metrics; data and labels; method (direct regression on
whole images at 486x648, relative-error loss, EMA, cosine, slide-grouped folds, TTA,
geometric-mean ensemble, clamps); training protocol; validation methodology (OOF over 9000,
paired comparisons); results (OOF per family, ensemble, platform score); code organisation;
implementation details (cache, determinism, config freeze, container constraints);
limitations. Built to PDF with pdflatex from a LaTeX source generated alongside the Markdown.

## Signature policy

No occurrence of "Claude", "Anthropic", "Co-Authored-By", session URLs, or tool names in
any tracked file or commit message. Commits authored `Adi <adi@visilant.org>`. A final
`grep -rIi` over the tree and `git log` is part of the verification.

## Verification before hand-over

1. `sha256sum -c checkpoints/SHA256SUMS` passes.
2. Fresh venv from `requirements.txt`; `python -m pytest tests` (or unittest) passes.
3. `docker build` from the repo; `scripts/run_example.sh` offline reproduces the fixture
   values (v2ens: CD 2479.2796, CV 0.43580, HEX 0.46232) to the recorded digits.
4. `scripts/replay_container.py 50` for both ensembles: max relative difference vs the
   reference CSVs below 1e-5 (recorded: 3.8e-7 and 2.7e-7).
5. One-epoch `train_member.py` smoke on a 200-image cache subset completes and writes the
   expected artifact tree.
6. Signature grep over tree and history returns nothing.
7. The repository is not pushed and no collaborator is added until the user says so.

## Out of scope

Retraining the 17 members; any new experiment; uploading to Grand Challenge; changes to the
research repository beyond spec, plan, memory note.

## Risks

GitHub LFS free quota (1 GiB storage, 1 GiB/month bandwidth) is below the 2.9 GB bundle;
push and reviewer download require an LFS data pack or an external mirror. The v2ens_plus
entry has no platform score and is documented as prepared but not evaluated.
