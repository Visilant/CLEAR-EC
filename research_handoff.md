# CLEAR-EC Phase I research handoff (as of 2026-09-12 17:45 EDT)

For any agent picking this up. Everything here is measured; where a number is an estimate or an
assumption it says so. Read this before touching models, slots, or the experiment plan. It supersedes
the model-selection sections of `docs/PHASE_I_EXPERIMENT_PLAN.md` and `_V2.md` and extends
`results/night_20260912/REPORT.md` and `results/noise_floor_20260912/REPORT.md`.

## 0. Status in one paragraph

Slot 1 of 3 is spent. The submitted container `clear_ec_phase1:v2ens` (ten-model ConvNeXt ensemble)
scored **8.7583** on the 100 hidden Phase I images and is **4th**; the top five span 8.7042 to 8.7720 (a
0.068 band). Out-of-fold over all 9,000 labelled images the same ensemble scores 8.84. The label-noise
floor for a frame-level predictor is bounded between 4.9 and 7.6 mean MAPE; CV and HEX are at their floor
and all remaining headroom is in CD, where the excess is a heavy tail of donor-level gross mislabels.
Every lever tested since (label cleaning, bigger backbone, longer training, ensembling, calibration,
resolution, augmentation, seeds) moves the score by 0.03 or less. Finishing order among the top five is
decided by the 100-image draw, not by anything we can engineer before the 2026-09-14 deadline. The
remaining two slots should carry the highest-expected-value candidate and nothing speculative.

## 1. Task facts that are not in the README

- Target: per image, CD (cells/mm2), CV (area coefficient of variation), HEX (fraction six-sided). Score is
  the equal-weight mean of the three MAPEs. 100 hidden test images in Phase I, 3 submissions, top 5 advance;
  Phase II has 1,000 images and 2 submissions (Oct 1 to Nov 14).
- Ground truth is the Voronoi centre method on about 110 to 180 manually clicked cells inside one
  annotator-placed box of about 538x407 px (`data/train_green` overlays, 25 images). Mean interior cells
  per box 114 (104 to 133). Labels are rounded: CD integer, CV and HEX to 2 dp.
- 9,000 images, 6,771 slides. 2,227 two-image slide groups are **fellow eyes** (ODCN/OSCN of one donor),
  not repeat measurements. One three-image slide. There are no genuine repeat frames of the same eye.
- Exact pixel duplicates: three same-donor ID variants (space vs hyphen) with identical labels, and four
  cross-donor pairs (`results/audit_20260911/evidence/duplicates.json`) with typo-like IDs
  (0894/0694, 0361/0261, 0635/0365) and conflicting labels, i.e. probable ID/label mix-ups.
- All frames 1296x972 at 0.7716049 um/px. ID families A (`nnnn-yy`, 4,514) and B (`yyyy-nnnn ODCN/OSCN`,
  4,482) share spacing and frame size; OOF bias by family is under 2 percent.

## 2. What the incumbent is

`clear_ec_phase1:v2ens` (built 2026-09-12 09:45 EDT, archive `CLEAR-EC-phase1-v2ens/results/phase1_v2ens/artifacts/`):

- 5x ConvNeXt-V2-Tiny (`timm convnextv2_tiny.fcmae_ft_in22k_in1k`, weight 2) + 5x torchvision ConvNeXt-Tiny
  (weight 1), one model per slide-grouped fold over all 9,000 images, geometric mean, flip TTA (4 views).
- Input: uint8 -> /255 -> bilinear to 486x648 -> grey to 3ch -> ImageNet normalisation. Output clamps
  CD [372, 4500], CV [0.03, 1.5], HEX [0, 1].
- Recipe: relative (MAPE-like) loss, AdamW lr 1e-4, wd 1e-4, batch 8, AMP, flips, EMA 0.999, cosine with
  1 warmup epoch, fixed 8 epochs (V2) / 12 epochs (Tiny), last checkpoint, `--clip_grad 1.0` mandatory
  for V2 (seed-42 runs spiked at epoch 3 without it).
- Container replay matched local predictions to 4e-7 relative on 50 cases; 6.9 s/image on an A5000 (max
  8.2 s), 40 forward passes per image; 37 tests passed at build time.
- Manifest and hashes: `results/night_20260912/candidate_manifest.json`.

Scores (OOF over 9,000 unless stated):

| candidate | CD | CV | HEX | mean |
|---|---:|---:|---:|---:|
| v2ens (submitted) | 6.30 | 10.25 | 9.97 | **8.84** |
| V2-Tiny folds alone | 6.35 | 10.30 | 9.93 | 8.86 |
| ConvNeXt-Tiny folds alone | 6.35 | 10.35 | 10.21 | 8.97 |
| V2-Base folds (B3, today) | 6.33 | 10.25 | 9.93 | 8.87 |
| Platform, 100 hidden images | | | | **8.7583** |
| Old 892-image val, previous incumbent | | | | 9.54 |

## 3. Noise-floor analysis (results/noise_floor_20260912/)

Method, because no repeat measurements exist: (A1) nonparametric bootstrap over the interior Voronoi cells
of each of the 25 click sets gives the finite-sample scatter of a label computed on ~114 cells; (A2) the
Voronoi readout of the annotator's own clicks vs the label gives process noise beyond that; (A3) residual
anatomy of the OOF predictions; (A4) subsampling for test-set visibility.

| metric | counting floor (A1) | + process noise (A2) | model OOF | headroom |
|---|---:|---:|---:|---:|
| CD | 1.52 (CI 1.34 to 1.72) | 1.7 | 6.30 | 4.6 |
| CV | 5.89 (5.47 to 6.36) | 10.5 | 10.25 | 0 |
| HEX | 7.09 (6.57 to 7.65) | 10.4 | 9.97 | 0 |
| mean | 4.83 | 7.5 | 8.84 | 1.3 (all CD) |

Key findings:

- HEX bootstrap relative SD 0.0887 vs binomial closed form 0.0886 (sanity passed). CD is deterministic
  given the clicks: oracle bias -3.0 percent constant, residual scatter 0.8 percent.
- HEX label differs from the standard readout by 6.4 percent even on the same clicks; the label was computed
  on a different or differently bounded cell set.
- Label CV = 2.00x interior Voronoi-area CV (corr 0.91, log-ratio SD 0.110). Thirteen alternative CV
  definitions (`A2_cv_formulas.csv`) are all worse or equal; none reaches under 9.3 percent MAPE.
- The floor excludes tissue variation across the frame outside the box, so it is a lower bound.
- **CD residuals are heavy-tailed** (skew 2.1, excess kurtosis 37): top 5 percent of images carry 24 percent
  of CD MAPE; dropping the top 1/2/5 percent gives CD 5.71/5.49/5.02. CV and HEX tails are ordinary.
- **Fellow-eye residual correlation** CD 0.50, CV 0.30, HEX 0.05: the CD error is donor-level. The
  contact sheet (`A3_contact_sheet.png`) shows donors 2020-0644, 2020-0646, 2020-0651 with both eyes
  labelled CD 700 to 900 on dense mosaics the model reads at 2,700 to 3,150. Seven donor pairs have both
  eyes in the flagged top 2 percent vs 1.8 expected by chance. Implausible labels: 6 with CV > 1, 2 with
  HEX < 0.15, 15 with CD < 800.
- Member disagreement predicts CD error (top quintile 8.95 vs 5.5) but not CV/HEX (Spearman 0.02).
- Numeric-neighbour ID test has no power (null rate 62 percent); mislabel rate cannot be estimated that way.
- Flag lists in `night_predict.py --indices` format: `A3_flagged_top2.txt` (180), `A3_flagged_top5.txt` (450).

Test-set visibility (`A4_visibility.json`):

| | 100 images | 1,000 images |
|---|---:|---:|
| SD of one candidate's score (image draws / slide draws) | 0.96 / 1.21 | 0.30 |
| P(v2ens < 8.91) / P(< 8.70) | 0.62 / 0.48 | 0.66 / 0.30 |
| paired SD, v2ens vs Tiny (OOF delta 0.13) | 0.12 | 0.036 |
| paired SD, v2ens vs V2-only (delta 0.02) | 0.06 | 0.018 |
| minimum OOF gain visible at 80 percent power | 0.10 to 0.15 | 0.03 to 0.05 |

The platform result 8.7583 vs expected 8.84 +/- 0.96 means no distribution shift; OOF is the compass.

## 4. Levers tested, with verdicts (do not rerun)

| lever | evidence | verdict |
|---|---|---|
| Label cleaning (drop top-2 percent flagged from training), V2 fold 0 | 8.721 vs 8.750 paired on 1,802 held-out images, delta -0.028 (CI -0.087 to +0.030); CD unchanged | fails 0.10 gate; test labels carry the same noise |
| Capacity: ConvNeXt-V2-Base, 5 folds, same recipe | OOF 8.871 vs V2-Tiny 8.897, delta -0.026 (CI -0.053 to -0.001), 3.5x cost; per fold 8.78/8.64/9.39/8.59/8.87 vs 8.75/8.64/9.41/8.61/8.90 | fails 0.15 gate; tie |
| ConvNeXt-Small, V2-Nano, DINOv2 ViT-S | 9.34, 9.32, 10.49 on 892 val vs 9.21 to 9.27 | lose or tie |
| Longer training (20 epochs, Tiny) | best 9.26 at epoch 11, last 9.57; 12-epoch last 9.34; with wd 0.05 + sd 0.3 at 20 epochs 9.39 | loses at the last epoch; V2 plateaus at epochs 7 to 8 |
| Seeds / seed ensembles | seeds correlate 0.994; two-seed ensemble +0.02; six arms +0.00 | nothing to average |
| Cross-family ensembles | Tiny vs V2 correlate 0.965 to 0.977; ten-model ensemble beats V2 alone by 0.02 | already in v2ens; ceiling reached |
| Flip TTA / best-epoch selection | +0.02 to 0.03 / +0.02 | in v2ens / not used (fixed budget) |
| Recalibration (scale, log-linear, quantile, per family) | under 0.02 or worse, measured fold-honestly | dead |
| Resolution 648x864, antialias, crops 0.6, log targets, wd, drop-path | all within 0.05 of baseline on 892 val | dead |
| Half the training data | costs 0.05 to 0.09 | learning curve flat; more data is worth about 0.02 |
| Detector-based Voronoi readout (SAM/Cellpose spike) | detector recall 0.5 to 0.75, CD MAPE 21 to 35 percent vs 6 percent gate; oracle readout on true clicks CD 3.1 percent | not a Phase I path; the only route to the CD headroom for Phase II |
| Pure-MAE V2 init (no ImageNet fine-tune) | 12.75 | dead |

## 5. Interpretation

Variance is gone (seed correlation 0.994, EMA removed epoch swings); what remains is bias against noisy
labels. Longer training, smoothing, and ensembling attack variance and therefore cannot move the score. CV
and HEX are at their floor for any frame-level model. The 4.6 points of CD headroom are real in principle but
sit in mislabelled donors that a MAPE-trained model already handles as well as the metric allows on a test
set carrying the same noise; the only way to reach them is a model whose CD is anchored in detected cell
geometry, which the current detectors cannot provide. Independent teams converging at 8.70 to 8.77 on the
hidden set is consistent with this floor.

## 6. Slot recommendation (2 remaining, deadline 2026-09-14)

Confirm first on the phase rules page whether the leaderboard keeps a team's best score or its latest.
Under best-of:

1. **Slot 2: all-data V2 refits folded into the ensemble.** `results/night_20260912/v2refit_seed123` and
   `v2refit_seed7` (`last.pt`, 8 epochs on all 9,000, both train-loss curves checked monotone), geometric
   mean with the ten fold models. Expected about +0.02 OOF (cannot be validated: the refits saw every
   label); paired SD vs v2ens on 100 images about 0.06, so roughly a 30 percent chance of moving above 8.70.
   Optionally add the five V2-Base fold models (`results/noise_floor_20260912/b3_v2base_fold*/…/last.pt`,
   OOF 8.87, correlation with V2-Tiny unmeasured) if the container budget allows fifteen models; check the
   per-image runtime, the T4 budget is unknown.
2. **Slot 3: nothing speculative.** Only a candidate whose OOF is at least 0.05 better than what slot 2
   sent. No such candidate exists today. Leaving the slot unused is acceptable.

Under latest-score rules, submit nothing further unless the OOF gain exceeds 0.10.

Do not: tune against the leaderboard, add seeds or TTA variants, recalibrate, run more backbone arms, or
train longer.

## 7. Phase II directions (1,000 images, visibility bar 0.03 to 0.05)

1. **Cell-centre detection for CD only.** CD is deterministic given centres (0.8 percent scatter) and
   annotator-box sampling costs only 1.5 percent. A detector with recall over 0.9 and an annotator-box
   emulation (well-focused contiguous cluster of about 114 cells) would bring CD from 6.3 toward 2 to 3,
   worth 1.0 to 1.4 on the mean. Keep the CNN for CV and HEX. Train the detector on the 25 overlays plus
   pseudo-labels from the CNN-consistent images; the earlier spike code is on branch `spike/sam-detector`
   (`.worktrees/spike-sam/spike/sam/`).
2. **Donor-level label audit** for the training set: fellow-eye pairs where both residuals exceed 25 percent
   in the same direction are candidates for exclusion or relabelling; label cleaning failed in Phase I on
   MAPE but a cleaner training set matters more for a detector supervised on labels.
3. **Resolve the CV formula** on any further overlays the organisers release; the 2.0x factor with 0.11
   log scatter is the single largest unexplained piece of the floor.
4. Do not revisit capacity, resolution, augmentation, or calibration for the whole-image regressor.

## 8. Where everything is

| item | path |
|---|---|
| Labels, cache, splits | `data/final_train_ids.csv`, `data/cache/{images_u8.npy,index.csv,splits.json}` (memmap uint8 (9000,972,1296), no header) |
| Night report and OOF predictions | `results/night_20260912/REPORT.md`, `results/night_20260912/oof/*_last_flips.csv` (`idx,ID,CD,CV,HEX`) |
| Candidate manifest, golden fixtures | `results/night_20260912/candidate_manifest.json`, `golden/` |
| Noise-floor report, JSONs, flag lists, contact sheet | `results/noise_floor_20260912/` |
| Noise-floor scripts | `code/scripts/noise_floor/{overlay_floor.py,oof_residuals.py,compare_fold.py}` |
| Tests | `code/tests/test_noise_floor.py` (3 pass; pytest is not installed anywhere, run via importlib) |
| Trainer (newest; has `--fold/--n_folds`, `--clip_grad`, `--exclude_idx_file`) | `.worktrees/regression-night/code/scripts/run_training.py`, `code/src/training/regression_cnn.py` |
| Scorer with flip TTA and index subsets | `.worktrees/regression-night/code/scripts/night_predict.py` |
| Fold definition | `regression_cnn.py::_slide_group_folds(cache_dir, 5, seed=42)` (deterministic, not on disk) |
| Submitted container and archives | `docker image clear_ec_phase1:v2ens`; `/home/visilant/CLEAR-EC-phase1-v2ens/results/phase1_v2ens/{status.json,artifacts/}` (status.json still says archive_ready_for_manual_upload and has no submission ID; update it) |
| V2-Base fold checkpoints (unused) | `results/noise_floor_20260912/b3_v2base_fold{0..4}/regression_cnn/seed_123/last.pt` |
| Overlay click sets and Voronoi readout | `.worktrees/spike-sam/spike/sam/{overlays.json,reference_readout.py,readout.py}` |
| Environment | `code/.venv` (py 3.12, torch 2.13+cu130, timm 1.0.29, torchvision 0.28; no pip, no pytest, no sklearn, no cleanlab) |
| Hardware | 2x RTX A5000 24 GB; V2-Tiny 100 s/epoch alone, 160 s when both GPUs busy; V2-Base 352 s/epoch |

## 9. Open items and caveats

- The Phase I leaderboard is JavaScript-rendered and its CDN returns 403 to headless browsers; read it by
  hand. Sixth to tenth places were not reported and matter more than first to fifth for advancement risk.
- The kparikh3 row "CLEAR EC Direct Regression Seed 123" at 8.7583 is assumed to be the v2ens submission;
  the score is consistent only with the ConvNeXt payload (the small CNN scores 12.5 locally).
- Platform per-case runtime on the T4 was not observed; locally 6.9 s/image with 40 forward passes.
- Floor estimates for CV and HEX process noise rest on 25 overlays (HEX oracle CI 4.3 to 10.0).
- `compare_fold.py` aggregates per-image mean APE then averages (nanmean over metrics for the two zero-HEX
  images), so its means differ from the metric-level MAPE by about 0.03; use it for paired deltas only.
- Uncommitted work in the main checkout and the regression-night worktree (see `git status`); nothing from
  today is committed.
