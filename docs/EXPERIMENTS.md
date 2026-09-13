# CLEAR-EC experiment ledger

Generated 2026-09-13 by `python -m experiments.run ledger` from `results/` and `code/experiments/ledger_manual.yaml`; 189 rows, machine-readable copy in `docs/experiments.csv`. Do not edit by hand: change the YAML or the results and regenerate.

Score is the equal-weight mean of the CD, CV and HEX MAPEs (percent, lower is better). Splits: `oof9000` = every labelled image scored by the fold model that did not train on it (the compass); `val892` = the original slide-disjoint val split (best-epoch numbers there are optimistic); `platform100` = the hidden Phase I test set; `test906` = the August one-shot test split; `floor` = label-noise floors, not models.

## Leaderboard

### Out-of-fold over all 9,000 labelled images

| campaign | run | model | CD | CV | HEX | mean | verdict |
|---|---|---|---:|---:|---:|---:|---|
| noise_floor_20260912 | A3 OOF ens | v2ens (V2 w2 + Tiny w1) | 6.3000 | 10.2515 | 9.9652 | 8.8389 |  |
| night_20260912 | v2ens = candidate_1 (submitted slot 1) | 5x convnextv2_tiny (w 2), 5x convnext_tiny (w 1) | 6.3000 | 10.2515 | 9.9652 | 8.8389 |  |
| noise_floor_20260912 | A3 OOF v2 | 5x ConvNeXt-V2-Tiny folds | 6.3545 | 10.3020 | 9.9301 | 8.8622 |  |
| night_20260912 | v2fold OOF over 5 folds [last+flips] |  | 6.3545 | 10.3020 | 9.9301 | 8.8622 |  |
| night_20260912 | v2fold OOF over 5 folds [last+none] |  | 6.3706 | 10.3100 | 9.9220 | 8.8676 |  |
| noise_floor_20260912 | B3 capacity: V2-Base 5 folds | b3_v2base_oof_all.csv | 6.3305 | 10.2518 | 9.9306 | 8.8708 |  |
| night_20260912 | fold OOF over 5 folds [best+flips] |  | 6.3338 | 10.3261 | 10.2098 | 8.9566 |  |
| night_20260912 | fold OOF over 5 folds [last+flips] |  | 6.3524 | 10.3516 | 10.2082 | 8.9707 |  |
| noise_floor_20260912 | A3 OOF tiny | 5x ConvNeXt-Tiny folds | 6.3524 | 10.3516 | 10.2082 | 8.9707 |  |
| night_20260912 | fold OOF over 5 folds [best+none] |  | 6.3508 | 10.3497 | 10.2253 | 8.9753 |  |
| night_20260912 | fold OOF over 5 folds [last+none] |  | 6.3730 | 10.3859 | 10.2294 | 8.9961 |  |
| smoke_runner | cv OOF over 2 folds [last+none] |  | 14.5589 | 14.4071 | 13.1888 | 14.0516 |  |

### Hidden test set (100 images)

| campaign | run | model | CD | CV | HEX | mean | verdict |
|---|---|---|---:|---:|---:|---:|---|
| platform | leaderboard snapshot |  |  |  |  | 8.7042 | Best team score at the first snapshot (idea.md); others 8.7452, 8.7720, 8.7843, 8.9145. |
| platform | Phase I slot 1 (v2ens) | 5x ConvNeXt-V2-Tiny + 5x ConvNeXt-Tiny fold models, geometric mean, flip TTA |  |  |  | 8.7583 | 4th of the Phase I leaderboard on 2026-09-12; top five 8.7042 / 8.7452 / 8.7550 / 8.7583 / 8.7720. Consistent with OOF 8.84 +/- 0.96 (no distribution shift). |

### Original 892-image val split (top 15)

| campaign | run | model | CD | CV | HEX | mean | verdict |
|---|---|---|---:|---:|---:|---:|---|
| night_20260912 | x_v2e8 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 6.1909 | 11.7213 | 9.6970 | 9.2031 |  |
| night_20260912 | x_v2base | timm:convnextv2_base.fcmae_ft_in22k_in1k | 6.1922 | 11.7371 | 9.6801 | 9.2031 | tie: OOF 8.871 vs 8.897 (CI -0.053 to -0.001) at 3.5x the cost; fails the 0.15 gate [do not rerun] |
| night_20260912 | x_v2tiny | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 6.2401 | 11.7071 | 9.6821 | 9.2098 |  |
| night_20260912 | x_highres | convnext_tiny | 6.0493 | 11.7400 | 9.9705 | 9.2533 | all within 0.05 of the baseline on 892 val (half the data costs 0.05 to 0.09) [do not rerun] |
| night_20260912 | x_reg_e20 | convnext_tiny | 6.0487 | 11.7443 | 9.9813 | 9.2581 | loses at the last epoch; V2 plateaus at epochs 7 to 8; 40 epochs is +0.046 worse than 8 [do not rerun] |
| night_20260912 | r3_seed123 [best+none] |  | 6.1023 | 11.6741 | 10.0006 | 9.2590 | loses at the last epoch; V2 plateaus at epochs 7 to 8; 40 epochs is +0.046 worse than 8 [do not rerun] |
| night_20260912 | r3_seed123 | convnext_tiny | 6.1023 | 11.6742 | 10.0007 | 9.2591 | loses at the last epoch; V2 plateaus at epochs 7 to 8; 40 epochs is +0.046 worse than 8 [do not rerun] |
| night_20260912 | r3_seed42 [best+none] |  | 6.0681 | 11.6924 | 10.0302 | 9.2635 | loses at the last epoch; V2 plateaus at epochs 7 to 8; 40 epochs is +0.046 worse than 8 [do not rerun] |
| night_20260912 | r3_seed42 | convnext_tiny | 6.0679 | 11.6924 | 10.0303 | 9.2636 | loses at the last epoch; V2 plateaus at epochs 7 to 8; 40 epochs is +0.046 worse than 8 [do not rerun] |
| night_20260912 | r3_e12 | convnext_tiny | 6.0693 | 11.7263 | 10.0226 | 9.2728 | loses at the last epoch; V2 plateaus at epochs 7 to 8; 40 epochs is +0.046 worse than 8 [do not rerun] |
| night_20260912 | x_antialias | convnext_tiny | 6.0761 | 11.7624 | 10.0072 | 9.2819 | all within 0.05 of the baseline on 892 val (half the data costs 0.05 to 0.09) [do not rerun] |
| night_20260912 | x_in12k | timm:convnext_tiny.in12k_ft_in1k | 6.2456 | 11.7872 | 9.8252 | 9.2860 | lose or tie on the 892 val split (9.29 to 12.75 vs 9.21 to 9.27) [do not rerun] |
| night_20260912 | x_wd5e-2 | convnext_tiny | 6.1071 | 11.7957 | 9.9850 | 9.2959 | all within 0.05 of the baseline on 892 val (half the data costs 0.05 to 0.09) [do not rerun] |
| night_20260912 | x_dp03 | convnext_tiny | 6.1378 | 11.7538 | 10.0068 | 9.2995 | all within 0.05 of the baseline on 892 val (half the data costs 0.05 to 0.09) [do not rerun] |
| night_20260912 | x_logspace | convnext_tiny | 6.0782 | 11.8176 | 10.0063 | 9.3007 | all within 0.05 of the baseline on 892 val (half the data costs 0.05 to 0.09) [do not rerun] |

### August one-shot test split

| campaign | run | model | CD | CV | HEX | mean | verdict |
|---|---|---|---:|---:|---:|---:|---|
| overnight | ridge_calibration (test) | ridge on Cellpose metrics | 15.9579 | 14.8776 | 13.6626 | 14.8327 | Cellpose 'cyto' outputs correlate 0.2 to 0.3 with GT CD; ridge (14.8) and the constant (14.2) beat it; superseded by direct regression [do not rerun] |
| overnight | regression_cnn seed 44 (test) | small CNN (pre-audit selection bug) | 21.7910 | 15.4725 | 12.5146 | 16.5927 | Cellpose 'cyto' outputs correlate 0.2 to 0.3 with GT CD; ridge (14.8) and the constant (14.2) beat it; superseded by direct regression [do not rerun] |
| overnight | regression_cnn seed 43 (test) | small CNN (pre-audit selection bug) | 28.6476 | 23.5387 | 16.4540 | 22.8801 | Cellpose 'cyto' outputs correlate 0.2 to 0.3 with GT CD; ridge (14.8) and the constant (14.2) beat it; superseded by direct regression [do not rerun] |
| overnight | regression_cnn seed 42 (test) | small CNN (pre-audit selection bug) | 29.0888 | 22.6753 | 18.1155 | 23.2932 | Cellpose 'cyto' outputs correlate 0.2 to 0.3 with GT CD; ridge (14.8) and the constant (14.2) beat it; superseded by direct regression [do not rerun] |
| overnight | cellpose_baseline (test) | Cellpose diameter_40 crop 0.5 | 28.6949 | 30.1826 | 22.9491 | 27.2755 | Cellpose 'cyto' outputs correlate 0.2 to 0.3 with GT CD; ridge (14.8) and the constant (14.2) beat it; superseded by direct regression [do not rerun] |

## Levers tested, with verdicts

| lever | runs | verdict | do not rerun |
|---|---|---|---|
| Label cleaning (drop top-2 percent flagged from training) | noise_floor_20260912/b1_* | fails the 0.10 gate: delta -0.028 (CI -0.087 to +0.030) on 1,802 held-out images; test labels carry the same noise | yes |
| Capacity: ConvNeXt-V2-Base, 5 folds | noise_floor_20260912/b3_*, night_20260912/x_v2base* | tie: OOF 8.871 vs 8.897 (CI -0.053 to -0.001) at 3.5x the cost; fails the 0.15 gate | yes |
| Other backbones: ConvNeXt-Small, V2-Nano, DINOv2 ViT-S, pure-MAE V2 init | night_20260912/x_small*, night_20260912/x_v2nano*, night_20260912/x_dinov2*, night_20260912/x_v2fcmae*, night_20260912/x_in12k* | lose or tie on the 892 val split (9.29 to 12.75 vs 9.21 to 9.27) | yes |
| Longer training | night_20260912/r3_*, night_20260912/x_reg_e20*, long_training_20260912_40ep/* | loses at the last epoch; V2 plateaus at epochs 7 to 8; 40 epochs is +0.046 worse than 8 | yes |
| Seeds and seed ensembles | night_20260912/v2fold0_s42*, night_20260912/r3_seed42*, night_20260912/refit_seed42* | seeds correlate 0.994; a two-seed ensemble gains 0.02; nothing to average | yes |
| Resolution, antialias, crops, log targets, weight decay, drop-path, half data | night_20260912/x_highres*, night_20260912/x_antialias*, night_20260912/x_crop06*, night_20260912/x_logspace*, night_20260912/x_wd5e-2*, night_20260912/x_dp03*, night_20260912/x_half*, convnext_20260911/* | all within 0.05 of the baseline on 892 val (half the data costs 0.05 to 0.09) | yes |
| Detector-based Voronoi readout (SAM / Cellpose-SAM) | sam_spike_20260912/* | detector recall 0.5 to 0.75, CD MAPE 21 to 35 percent vs a 6 percent gate; the only route to the CD headroom for Phase II | yes |
| Flip TTA / best-epoch selection | night_20260912/* [*+flips] | +0.02 to 0.03 / +0.02; TTA is in v2ens, best-epoch selection is not used (fixed budget) | no |
| Small-CNN normalization x seeds (round 1) | round1_20260912_002113/* | BatchNorm beats GroupNorm by 0.5; three-seed ensemble +0.09 missed the 0.2 gate; superseded by ConvNeXt | yes |
| Cellpose baseline and ridge calibration | overnight/*, audit_20260911/* | Cellpose 'cyto' outputs correlate 0.2 to 0.3 with GT CD; ridge (14.8) and the constant (14.2) beat it; superseded by direct regression | yes |

## Campaigns

### 2026-08-27 August baseline: Cellpose, ridge calibration, small CNN

`results/overnight`, report `results/overnight/REPORT.md`; 17 rows.
Slide-disjoint train/val/test protocol; frozen Cellpose config diameter_40 crop 0.5; one-shot test scores.

| run | model | seed | fold | epochs | split | n | CD | CV | HEX | mean | ci_low | ci_high | reference | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| training | small | 44 |  | /30 | val892 | 892 | 20.2609 | 17.4951 | 12.9769 | 16.9110 |  |  |  | done |
| training | small | 43 |  | /30 | val892 | 892 | 26.6344 | 25.2129 | 16.4439 | 22.7637 |  |  |  | done |
| training | small | 42 |  | /30 | val892 | 892 | 26.5552 | 24.7275 | 18.5343 | 23.2723 |  |  |  | done |
| cellpose sweep diameter_40 | Cellpose  crop 0.5 |  |  |  | val892 | 892 |  |  |  | 24.8550 |  |  |  | sweep |
| cellpose sweep cyto2 | Cellpose  crop 0.5 |  |  |  | val892 | 892 |  |  |  | 30.5321 |  |  |  | sweep |
| cellpose sweep flow_0.3 | Cellpose  crop 0.5 |  |  |  | val892 | 892 |  |  |  | 32.1113 |  |  |  | sweep |
| cellpose sweep crop_0.50 | Cellpose cyto_default crop 0.5 |  |  |  | val892 | 892 |  |  |  | 32.6781 |  |  |  | sweep |
| cellpose sweep flow_0.5 | Cellpose  crop 0.5 |  |  |  | val892 | 892 |  |  |  | 35.1056 |  |  |  | sweep |
| cellpose sweep crop_0.45 | Cellpose cyto_default crop 0.45 |  |  |  | val892 | 892 |  |  |  | 35.9182 |  |  |  | sweep |
| cellpose sweep crop_0.40 | Cellpose cyto_default crop 0.4 |  |  |  | val892 | 892 |  |  |  | 37.5196 |  |  |  | sweep |
| cellpose sweep crop_0.35 | Cellpose cyto_default crop 0.35 |  |  |  | val892 | 892 |  |  |  | 41.5395 |  |  |  | sweep |
| cellpose sweep crop_0.30 | Cellpose cyto_default crop 0.3 |  |  |  | val892 | 892 |  |  |  | 63.7475 |  |  |  | sweep |
| ridge_calibration (test) | ridge on Cellpose metrics |  |  |  | test906 | 906 | 15.9579 | 14.8776 | 13.6626 | 14.8327 | 13.9130 | 15.8969 | score CI (iid image bootstrap) | one-shot test |
| regression_cnn seed 44 (test) | small CNN (pre-audit selection bug) | 44 |  |  | test906 | 906 | 21.7910 | 15.4725 | 12.5146 | 16.5927 | 15.9699 | 17.3800 | score CI (iid image bootstrap) | one-shot test |
| regression_cnn seed 43 (test) | small CNN (pre-audit selection bug) | 43 |  |  | test906 | 906 | 28.6476 | 23.5387 | 16.4540 | 22.8801 | 22.1291 | 23.8428 | score CI (iid image bootstrap) | one-shot test |
| regression_cnn seed 42 (test) | small CNN (pre-audit selection bug) | 42 |  |  | test906 | 906 | 29.0888 | 22.6753 | 18.1155 | 23.2932 | 22.5516 | 24.2157 | score CI (iid image bootstrap) | one-shot test |
| cellpose_baseline (test) | Cellpose diameter_40 crop 0.5 |  |  |  | test906 | 906 | 28.6949 | 30.1826 | 22.9491 | 27.2755 | 23.3193 | 33.6175 | score CI (iid image bootstrap) | one-shot test |

### 2026-09-11 Readiness audit

`results/audit_20260911`; 1 rows.
Fixed checkpoint selection, cache clobbering and scoring joins (AUDIT.md); the corrected 10-epoch small CNN is the first credible model.

| run | model | seed | fold | epochs | split | n | CD | CV | HEX | mean | ci_low | ci_high | reference | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| selection_fix | small | 42 |  | /10 | val892 | 892 | 12.5034 | 13.1674 | 11.8556 | 12.5088 |  |  |  | done |

### 2026-09-11 ConvNeXt-Tiny whole-image and patch arms

`results/convnext_20260911`; 6 rows.
Six arms at seed 123, 20 epochs, best-epoch selection on the 892 val split. No REPORT.md; numbers from metrics.json.

| run | model | seed | fold | epochs | split | n | CD | CV | HEX | mean | ci_low | ci_high | reference | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| highres_relative | convnext_tiny | 123 |  | /12 | val892 | 892 | 6.4997 | 11.9446 | 10.1224 | 9.5222 |  |  |  | done |
| whole_relative | convnext_tiny | 123 |  | /20 | val892 | 892 | 6.5948 | 11.9216 | 10.1088 | 9.5417 |  |  |  | done |
| quality_patches | convnext_tiny | 123 |  | /12 | val892 | 892 | 6.8078 | 11.9315 | 9.9773 | 9.5722 |  |  |  | done |
| whole_huber | convnext_tiny | 123 |  | /20 | val892 | 892 | 6.5397 | 12.1408 | 10.1486 | 9.6097 |  |  |  | done |
| fixed_patches | convnext_tiny | 123 |  | /12 | val892 | 892 | 6.6804 | 11.9871 | 10.2018 | 9.6231 |  |  |  | done |
| lowres_relative | convnext_tiny | 123 |  | /12 | val892 | 892 | 6.6882 | 12.2270 | 10.3089 | 9.7413 |  |  |  | done |

### 2026-09-12 Long training screen (two folds, 8 to 40 epochs)

`results/long_training_20260912_40ep`, report `results/long_training_20260912_40ep/REPORT.md`; 17 rows.
Matched 8-epoch cosine prefix then constant LR; longer is slightly worse.

| run | model | seed | fold | epochs | split | n | CD | CV | HEX | mean | ci_low | ci_high | reference | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| epoch 12, both folds [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 |  | 12 | oof2fold | 3585 | 6.2408 | 10.0519 | 9.7539 | 8.6822 |  |  | epoch 8 (primary comparison) | computed |
| epoch 8, both folds [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 |  | 8 | oof2fold | 3585 | 6.2343 | 10.0475 | 9.7679 | 8.6832 |  |  | epoch 8 (primary comparison) | computed |
| epoch 20, both folds [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 |  | 20 | oof2fold | 3585 | 6.2540 | 10.0711 | 9.7592 | 8.6948 |  |  | epoch 8 (primary comparison) | computed |
| epoch 32, both folds [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 |  | 32 | oof2fold | 3585 | 6.2534 | 10.1029 | 9.7720 | 8.7095 |  |  | epoch 8 (primary comparison) | computed |
| epoch 40, both folds [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 |  | 40 | oof2fold | 3585 | 6.2641 | 10.1291 | 9.7940 | 8.7291 |  |  | epoch 8 (primary comparison) | computed |
| fold1 epoch 12 [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 1 | 12 | fold-1/5 |  | 6.0708 | 10.1235 | 9.6565 | 8.6169 |  |  |  | scored |
| fold1 epoch 8 [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 1 | 8 | fold-1/5 |  | 6.0843 | 10.0911 | 9.7010 | 8.6255 |  |  |  | scored |
| fold1 epoch 32 [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 1 | 32 | fold-1/5 |  | 6.0564 | 10.1860 | 9.6406 | 8.6277 |  |  |  | scored |
| fold1 epoch 20 [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 1 | 20 | fold-1/5 |  | 6.0677 | 10.1624 | 9.6534 | 8.6278 |  |  |  | scored |
| fold1 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 1 | 40/40 | fold-1/5 | 1783 | 6.0837 | 10.1598 | 9.6613 | 8.6349 |  |  |  | done |
| fold1 epoch 40 [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 1 | 40 | fold-1/5 |  | 6.0615 | 10.2150 | 9.6458 | 8.6408 |  |  |  | scored |
| fold0 epoch 8 [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 0 | 8 | fold-0/5 |  | 6.3826 | 10.0043 | 9.8341 | 8.7404 |  |  |  | scored |
| fold0 epoch 12 [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 0 | 12 | fold-0/5 |  | 6.4089 | 9.9810 | 9.8503 | 8.7467 |  |  |  | scored |
| fold0 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 0 | 40/40 | fold-0/5 | 1802 | 6.4296 | 9.9881 | 9.8565 | 8.7581 |  |  |  | done |
| fold0 epoch 20 [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 0 | 20 | fold-0/5 |  | 6.4384 | 9.9808 | 9.8639 | 8.7610 |  |  |  | scored |
| fold0 epoch 32 [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 0 | 32 | fold-0/5 |  | 6.4484 | 10.0207 | 9.9020 | 8.7904 |  |  |  | scored |
| fold0 epoch 40 [flips] | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 0 | 40 | fold-0/5 |  | 6.4645 | 10.0442 | 9.9407 | 8.8165 |  |  |  | scored |

### 2026-09-12 Night of 2026-09-12: stabilized recipe, folds, refits, exploration

`results/night_20260912`, report `results/night_20260912/REPORT.md`; 79 rows.
EMA + cosine + fixed budget removed the epoch swings; ConvNeXt-V2-Tiny wins every fold; ten-model ensemble OOF 8.84 became slot 1.

| run | model | seed | fold | epochs | split | n | CD | CV | HEX | mean | ci_low | ci_high | reference | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| v2ens = candidate_1 (submitted slot 1) | 5x convnextv2_tiny (w 2), 5x convnext_tiny (w 1) |  |  |  | oof9000 | 9000 | 6.3000 | 10.2515 | 9.9652 | 8.8389 |  |  |  | computed |
| v2fold OOF over 5 folds [last+flips] |  |  |  |  | oof9000 | 9000 | 6.3545 | 10.3020 | 9.9301 | 8.8622 |  |  |  | computed |
| v2fold OOF over 5 folds [last+none] |  |  |  |  | oof9000 | 9000 | 6.3706 | 10.3100 | 9.9220 | 8.8676 |  |  |  | computed |
| fold OOF over 5 folds [best+flips] |  |  |  |  | oof9000 | 9000 | 6.3338 | 10.3261 | 10.2098 | 8.9566 |  |  |  | computed |
| fold OOF over 5 folds [last+flips] |  |  |  |  | oof9000 | 9000 | 6.3524 | 10.3516 | 10.2082 | 8.9707 |  |  |  | computed |
| fold OOF over 5 folds [best+none] |  |  |  |  | oof9000 | 9000 | 6.3508 | 10.3497 | 10.2253 | 8.9753 |  |  |  | computed |
| fold OOF over 5 folds [last+none] |  |  |  |  | oof9000 | 9000 | 6.3730 | 10.3859 | 10.2294 | 8.9961 |  |  |  | computed |
| x_v2e8 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 |  | 8/8 | val892 | 892 | 6.1909 | 11.7213 | 9.6970 | 9.2031 |  |  |  | done |
| x_v2base | timm:convnextv2_base.fcmae_ft_in22k_in1k | 123 |  | 8/8 | val892 | 892 | 6.1922 | 11.7371 | 9.6801 | 9.2031 |  |  |  | done |
| x_v2tiny | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 |  | 12/12 | val892 | 892 | 6.2401 | 11.7071 | 9.6821 | 9.2098 |  |  |  | done |
| x_highres | convnext_tiny | 123 |  | 12/12 | val892 | 892 | 6.0493 | 11.7400 | 9.9705 | 9.2533 |  |  |  | done |
| x_reg_e20 | convnext_tiny | 123 |  | 20/20 | val892 | 892 | 6.0487 | 11.7443 | 9.9813 | 9.2581 |  |  |  | done |
| r3_seed123 [best+none] |  |  |  | 9 | val892 | 892 | 6.1023 | 11.6741 | 10.0006 | 9.2590 |  |  |  | scored |
| r3_seed123 | convnext_tiny | 123 |  | 20/20 | val892 | 892 | 6.1023 | 11.6742 | 10.0007 | 9.2591 |  |  |  | done |
| r3_seed42 [best+none] |  |  |  | 11 | val892 | 892 | 6.0681 | 11.6924 | 10.0302 | 9.2635 |  |  |  | scored |
| r3_seed42 | convnext_tiny | 42 |  | 20/20 | val892 | 892 | 6.0679 | 11.6924 | 10.0303 | 9.2636 |  |  |  | done |
| r3_e12 | convnext_tiny | 123 |  | 12/12 | val892 | 892 | 6.0693 | 11.7263 | 10.0226 | 9.2728 |  |  |  | done |
| x_antialias | convnext_tiny | 123 |  | 12/12 | val892 | 892 | 6.0761 | 11.7624 | 10.0072 | 9.2819 |  |  |  | done |
| x_in12k | timm:convnext_tiny.in12k_ft_in1k | 123 |  | 12/12 | val892 | 892 | 6.2456 | 11.7872 | 9.8252 | 9.2860 |  |  |  | done |
| x_wd5e-2 | convnext_tiny | 123 |  | 12/12 | val892 | 892 | 6.1071 | 11.7957 | 9.9850 | 9.2959 |  |  |  | done |
| x_dp03 | convnext_tiny | 123 |  | 12/12 | val892 | 892 | 6.1378 | 11.7538 | 10.0068 | 9.2995 |  |  |  | done |
| x_logspace | convnext_tiny | 123 |  | 12/12 | val892 | 892 | 6.0782 | 11.8176 | 10.0063 | 9.3007 |  |  |  | done |
| x_v2nano | timm:convnextv2_nano.fcmae_ft_in22k_in1k | 123 |  | 8/8 | val892 | 892 | 6.2671 | 11.9079 | 9.7400 | 9.3050 |  |  |  | done |
| x_small | convnext_small | 123 |  | 12/12 | val892 | 892 | 6.2001 | 11.7642 | 10.0542 | 9.3395 |  |  |  | done |
| x_half | convnext_tiny | 123 |  | 12/12 | val892 | 892 | 6.2465 | 11.7784 | 10.0522 | 9.3590 |  |  |  | done |
| x_crop06 | convnext_tiny | 123 |  | 12/12 | val892 | 892 | 6.2071 | 11.8338 | 10.1215 | 9.3874 |  |  |  | done |
| whole_relative [best+flips] |  |  |  | 7 | val892 | 892 | 6.5398 | 11.9372 | 10.0836 | 9.5202 |  |  |  | scored |
| whole_relative [best+none] |  |  |  | 7 | val892 | 892 | 6.5948 | 11.9217 | 10.1087 | 9.5418 |  |  |  | scored |
| r3_seed123 [last+none] |  |  |  | 20 | val892 | 892 | 6.0976 | 12.3111 | 10.3022 | 9.5703 |  |  |  | scored |
| whole_huber [best+none] |  |  |  | 8 | val892 | 892 | 6.5396 | 12.1406 | 10.1487 | 9.6096 |  |  |  | scored |
| x_dinov2 | timm:vit_small_patch14_dinov2.lvd142m | 123 |  | 12/12 | val892 | 892 | 8.3963 | 12.5452 | 10.5161 | 10.4859 |  |  |  | done |
| x_v2fcmae | timm:convnextv2_tiny.fcmae | 123 |  | 8/8 | val892 | 892 | 13.5798 | 13.1332 | 11.5321 | 12.7484 |  |  |  | done |
| v2fold3 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 3 | 8/8 | fold-3/5 | 1788 | 5.9739 | 10.0015 | 9.8369 | 8.6041 |  |  |  | done |
| v2fold3 [last+none] |  |  |  | 8 | fold-3/5 | 1788 | 5.9739 | 10.0016 | 9.8368 | 8.6041 |  |  |  | scored |
| v2fold3 [last+flips] |  |  |  | 8 | fold-3/5 | 1788 | 5.9749 | 10.0027 | 9.8643 | 8.6140 |  |  |  | scored |
| v2fold1 [last+flips] |  |  |  | 8 | fold-1/5 | 1783 | 6.0694 | 10.1246 | 9.7226 | 8.6388 |  |  |  | scored |
| v2fold1 [last+none] |  |  |  | 8 | fold-1/5 | 1783 | 6.0815 | 10.1589 | 9.7121 | 8.6508 |  |  |  | scored |
| v2fold1 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 1 | 8/8 | fold-1/5 | 1783 | 6.0816 | 10.1589 | 9.7120 | 8.6509 |  |  |  | done |
| v2fold0_s42_clip [last+flips] |  |  |  | 8 | fold-0/5 | 1802 | 6.3222 | 9.9914 | 9.8454 | 8.7197 |  |  |  | scored |
| fold3 [best+flips] |  |  |  | 10 | fold-3/5 | 1788 | 5.9894 | 10.0302 | 10.1969 | 8.7388 |  |  |  | scored |
| fold1 [best+flips] |  |  |  | 9 | fold-1/5 | 1783 | 6.1019 | 10.0808 | 10.0374 | 8.7400 |  |  |  | scored |
| fold3 [last+flips] |  |  |  | 12 | fold-3/5 | 1788 | 6.0041 | 10.0438 | 10.1931 | 8.7470 |  |  |  | scored |
| v2fold0 [last+flips] |  |  |  | 8 | fold-0/5 | 1802 | 6.3649 | 10.0428 | 9.8413 | 8.7496 |  |  |  | scored |
| v2fold0_s42_clip | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 42 | 0 | 8/8 | fold-0/5 | 1802 | 6.3766 | 10.0223 | 9.8606 | 8.7532 |  |  |  | done |
| fold1 [last+flips] |  |  |  | 12 | fold-1/5 | 1783 | 6.1248 | 10.1319 | 10.0258 | 8.7608 |  |  |  | scored |
| fold3 | convnext_tiny | 123 | 3 | 12/12 | fold-3/5 | 1788 | 5.9989 | 10.0739 | 10.2183 | 8.7637 |  |  |  | done |
| fold3 [best+none] |  |  |  | 10 | fold-3/5 | 1788 | 5.9989 | 10.0739 | 10.2183 | 8.7637 |  |  |  | scored |
| v2fold0 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 0 | 8/8 | fold-0/5 | 1802 | 6.3826 | 10.0538 | 9.8601 | 8.7655 |  |  |  | done |
| v2fold0 [last+none] |  |  |  | 8 | fold-0/5 | 1802 | 6.3827 | 10.0538 | 9.8601 | 8.7655 |  |  |  | scored |
| fold1 [best+none] |  |  |  | 9 | fold-1/5 | 1783 | 6.1603 | 10.1166 | 10.0489 | 8.7752 |  |  |  | scored |
| fold1 | convnext_tiny | 123 | 1 | 12/12 | fold-1/5 | 1783 | 6.1603 | 10.1166 | 10.0490 | 8.7753 |  |  |  | done |
| fold3 [last+none] |  |  |  | 12 | fold-3/5 | 1788 | 6.0201 | 10.1079 | 10.2116 | 8.7798 |  |  |  | scored |
| fold1 [last+none] |  |  |  | 12 | fold-1/5 | 1783 | 6.1936 | 10.1799 | 10.0424 | 8.8053 |  |  |  | scored |
| fold0 [best+flips] |  |  |  | 10 | fold-0/5 | 1802 | 6.3576 | 10.1757 | 10.1070 | 8.8801 |  |  |  | scored |
| v2fold4 [last+flips] |  |  |  | 8 | fold-4/5 | 1825 | 6.4138 | 10.2831 | 9.9945 | 8.8971 |  |  |  | scored |
| fold0 [last+flips] |  |  |  | 12 | fold-0/5 | 1802 | 6.4121 | 10.1771 | 10.1029 | 8.8974 |  |  |  | scored |
| fold0 [best+none] |  |  |  | 10 | fold-0/5 | 1802 | 6.3741 | 10.2261 | 10.1217 | 8.9073 |  |  |  | scored |
| fold0 | convnext_tiny | 123 | 0 | 12/12 | fold-0/5 | 1802 | 6.3742 | 10.2261 | 10.1218 | 8.9074 |  |  |  | done |
| v2fold4 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 4 | 8/8 | fold-4/5 | 1825 | 6.4578 | 10.2865 | 9.9993 | 8.9145 |  |  |  | done |
| v2fold4 [last+none] |  |  |  | 8 | fold-4/5 | 1825 | 6.4578 | 10.2865 | 9.9993 | 8.9145 |  |  |  | scored |
| fold4 [best+flips] |  |  |  | 11 | fold-4/5 | 1825 | 6.3289 | 10.2389 | 10.2042 | 8.9240 |  |  |  | scored |
| fold0 [last+none] |  |  |  | 12 | fold-0/5 | 1802 | 6.4137 | 10.2368 | 10.1234 | 8.9246 |  |  |  | scored |
| fold4 [last+flips] |  |  |  | 12 | fold-4/5 | 1825 | 6.3322 | 10.2421 | 10.2063 | 8.9269 |  |  |  | scored |
| fold4 [best+none] |  |  |  | 11 | fold-4/5 | 1825 | 6.3146 | 10.2374 | 10.2363 | 8.9294 |  |  |  | scored |
| fold4 | convnext_tiny | 123 | 4 | 12/12 | fold-4/5 | 1825 | 6.3148 | 10.2375 | 10.2363 | 8.9295 |  |  |  | done |
| fold4 [last+none] |  |  |  | 12 | fold-4/5 | 1825 | 6.3169 | 10.2382 | 10.2424 | 8.9325 |  |  |  | scored |
| v2fold2 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 2 | 8/8 | fold-2/5 | 1802 | 6.9708 | 11.0142 | 10.1769 | 9.3873 |  |  |  | done |
| v2fold2 [last+none] |  |  |  | 8 | fold-2/5 | 1802 | 6.9500 | 11.0457 | 10.1984 | 9.3980 |  |  |  | scored |
| v2fold2 [last+flips] |  |  |  | 8 | fold-2/5 | 1802 | 6.9430 | 11.0528 | 10.2249 | 9.4069 |  |  |  | scored |
| fold2 [best+flips] |  |  |  | 9 | fold-2/5 | 1802 | 6.8861 | 11.1010 | 10.5021 | 9.4964 |  |  |  | scored |
| fold2 [best+none] |  |  |  | 9 | fold-2/5 | 1802 | 6.9017 | 11.0914 | 10.4994 | 9.4975 |  |  |  | scored |
| fold2 | convnext_tiny | 123 | 2 | 12/12 | fold-2/5 | 1802 | 6.9017 | 11.0915 | 10.4994 | 9.4975 |  |  |  | done |
| fold2 [last+flips] |  |  |  | 12 | fold-2/5 | 1802 | 6.8839 | 11.1600 | 10.5110 | 9.5183 |  |  |  | scored |
| fold2 [last+none] |  |  |  | 12 | fold-2/5 | 1802 | 6.9169 | 11.1645 | 10.5254 | 9.5356 |  |  |  | scored |
| v2fold0_s42 [last+flips] |  |  |  | 8 | fold-0/5 | 1802 | 9.0941 | 11.4348 | 10.8562 | 10.4617 |  |  |  | scored |
| refit_seed123 | convnext_tiny | 123 |  | 12/12 | none (all data) |  |  |  |  |  |  |  |  | no-val |
| refit_seed42 | convnext_tiny | 42 |  | 12/12 | none (all data) |  |  |  |  |  |  |  |  | no-val |
| v2refit_seed123 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 |  | 8/8 | none (all data) |  |  |  |  |  |  |  |  | no-val |
| v2refit_seed7 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 7 |  | 8/8 | none (all data) |  |  |  |  |  |  |  |  | no-val |

### 2026-09-12 Label-noise floor and the last two levers

`results/noise_floor_20260912`, report `results/noise_floor_20260912/REPORT.md`; 19 rows.
Counting and process-noise floors from the 25 overlays; residual anatomy; label cleaning (B1) and V2-Base capacity (B3) both fail their gates.

| run | model | seed | fold | epochs | split | n | CD | CV | HEX | mean | ci_low | ci_high | reference | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| A3 OOF ens | v2ens (V2 w2 + Tiny w1) |  |  |  | oof9000 | 9000 | 6.3000 | 10.2515 | 9.9652 | 8.8389 |  |  |  | scored |
| A3 OOF v2 | 5x ConvNeXt-V2-Tiny folds |  |  |  | oof9000 | 9000 | 6.3545 | 10.3020 | 9.9301 | 8.8622 |  |  |  | scored |
| B3 capacity: V2-Base 5 folds | b3_v2base_oof_all.csv |  |  |  | oof9000 | 9000 | 6.3305 | 10.2518 | 9.9306 | 8.8708 | -0.0528 | -0.0014 | v2tiny_oof_all.csv = 8.8970 (paired delta -0.0262; per-image mean APE, about 0.03 off the metric-level MAPE) | paired |
| A3 OOF tiny | 5x ConvNeXt-Tiny folds |  |  |  | oof9000 | 9000 | 6.3524 | 10.3516 | 10.2082 | 8.9707 |  |  |  | scored |
| A1 counting-statistics floor (cell bootstrap, 25 overlays) | label noise floor |  |  |  | floor | 25 | 1.5200 | 5.8902 | 7.0874 | 4.8325 | 4.4612 | 5.2447 |  | floor |
| A2 oracle Voronoi readout on the annotator's clicks | reference readout |  |  |  | floor | 25 | 3.0505 |  | 6.8067 |  | 2.6142 | 3.4383 | CI is for CD; label CV = 2.0x Voronoi CV (corr 0.91) | floor |
| b3_v2base_fold3 [last+flips] |  |  |  | 8 | fold-3/5 | 1788 | 5.9558 | 9.9212 | 9.8096 | 8.5622 |  |  |  | scored |
| b3_v2base_fold3 | timm:convnextv2_base.fcmae_ft_in22k_in1k | 123 | 3 | 8/8 | fold-3/5 | 1788 | 5.9861 | 9.9537 | 9.8220 | 8.5873 |  |  |  | done |
| b3_v2base_fold1 [last+flips] |  |  |  | 8 | fold-1/5 | 1783 | 6.0616 | 10.0733 | 9.7189 | 8.6180 |  |  |  | scored |
| b3_v2base_fold1 | timm:convnextv2_base.fcmae_ft_in22k_in1k | 123 | 1 | 8/8 | fold-1/5 | 1783 | 6.1030 | 10.0806 | 9.7351 | 8.6396 |  |  |  | done |
| B1 label cleaning: v2fold0 minus top-2% flagged | b1_v2fold0_excl2_oof |  |  |  | fold-0/5 | 1802 | 6.3731 | 9.9684 | 9.8219 | 8.7212 | -0.0868 | 0.0305 | v2fold0_last_flips.csv = 8.7496 (paired delta -0.0285; per-image mean APE, about 0.03 off the metric-level MAPE) | paired |
| b1_v2fold0_excl2 [last+flips] |  |  |  | 8 | fold-0/5 | 1802 | 6.3731 | 9.9684 | 9.8219 | 8.7212 |  |  |  | scored |
| b1_v2fold0_excl2 | timm:convnextv2_tiny.fcmae_ft_in22k_in1k | 123 | 0 | 8/8 | fold-0/5 | 1802 | 6.3795 | 9.9692 | 9.8187 | 8.7225 |  |  |  | done |
| b3_v2base_fold0 [last+flips] |  |  |  | 8 | fold-0/5 | 1802 | 6.3509 | 10.0513 | 9.8497 | 8.7506 |  |  |  | scored |
| b3_v2base_fold0 | timm:convnextv2_base.fcmae_ft_in22k_in1k | 123 | 0 | 8/8 | fold-0/5 | 1802 | 6.3937 | 10.0967 | 9.8567 | 8.7824 |  |  |  | done |
| b3_v2base_fold4 [last+flips] |  |  |  | 8 | fold-4/5 | 1825 | 6.3647 | 10.2512 | 9.9843 | 8.8667 |  |  |  | scored |
| b3_v2base_fold4 | timm:convnextv2_base.fcmae_ft_in22k_in1k | 123 | 4 | 8/8 | fold-4/5 | 1825 | 6.3862 | 10.2663 | 9.9511 | 8.8679 |  |  |  | done |
| b3_v2base_fold2 | timm:convnextv2_base.fcmae_ft_in22k_in1k | 123 | 2 | 8/8 | fold-2/5 | 1802 | 6.8923 | 10.9739 | 10.2751 | 9.3804 |  |  |  | done |
| b3_v2base_fold2 [last+flips] |  |  |  | 8 | fold-2/5 | 1802 | 6.9131 | 10.9575 | 10.2873 | 9.3860 |  |  |  | scored |

### 2026-09-12 Round 1: BatchNorm vs GroupNorm x seeds (small CNN)

`results/round1_20260912_002113`, report `results/round1_20260912_002113/REPORT.md`; 15 rows.
Paired slide-clustered CIs against the audited incumbent; the ensemble gain missed the 0.2 gate.

| run | model | seed | fold | epochs | split | n | CD | CV | HEX | mean | ci_low | ci_high | reference | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| batch_ensemble (scores.csv) | small CNN batchnorm |  |  |  | val892 | 892 | 11.5690 | 12.8695 | 11.1668 | 11.8684 | -0.8582 | -0.4236 | incumbent (paired delta CI) | scored |
| batch | small | 123 |  | /10 | val892 | 892 | 11.3302 | 13.4220 | 11.1155 | 11.9559 |  |  |  | done |
| batch_123 (scores.csv) | small CNN batchnorm |  |  |  | val892 | 892 | 11.3302 | 13.4220 | 11.1155 | 11.9559 | -0.8671 | -0.2206 | incumbent (paired delta CI) | scored |
| batch | small | 456 |  | /10 | val892 | 892 | 11.4671 | 13.0463 | 11.7552 | 12.0896 |  |  |  | done |
| batch_456 (scores.csv) | small CNN batchnorm |  |  |  | val892 | 892 | 11.4671 | 13.0463 | 11.7552 | 12.0896 | -0.6060 | -0.2090 | incumbent (paired delta CI) | scored |
| incumbent (scores.csv) | small CNN (audited incumbent) |  |  |  | val892 | 892 | 12.5034 | 13.1674 | 11.8556 | 12.5088 |  |  |  | scored |
| group | small | 123 |  | /10 | val892 | 892 | 13.3812 | 13.0744 | 11.4955 | 12.6504 |  |  |  | done |
| group_123 (scores.csv) | small CNN groupnorm |  |  |  | val892 | 892 | 13.3812 | 13.0744 | 11.4955 | 12.6504 | -0.1521 | 0.4238 | incumbent (paired delta CI) | scored |
| group_ensemble (scores.csv) | small CNN groupnorm |  |  |  | val892 | 892 | 13.5064 | 13.1493 | 11.4551 | 12.7036 | -0.1056 | 0.4866 | incumbent (paired delta CI) | scored |
| batch_42 (scores.csv) | small CNN batchnorm |  |  |  | val892 | 892 | 13.4956 | 13.3185 | 11.5181 | 12.7774 | -0.0289 | 0.5514 | incumbent (paired delta CI) | scored |
| batch | small | 42 |  | /10 | val892 | 892 | 13.4956 | 13.3185 | 11.5181 | 12.7774 |  |  |  | done |
| group | small | 42 |  | /10 | val892 | 892 | 13.4895 | 13.4404 | 11.4118 | 12.7806 |  |  |  | done |
| group_42 (scores.csv) | small CNN groupnorm |  |  |  | val892 | 892 | 13.4895 | 13.4404 | 11.4118 | 12.7806 | -0.0584 | 0.5875 | incumbent (paired delta CI) | scored |
| group | small | 456 |  | /10 | val892 | 892 | 13.7696 | 13.0429 | 11.5559 | 12.7894 |  |  |  | done |
| group_456 (scores.csv) | small CNN groupnorm |  |  |  | val892 | 892 | 13.7696 | 13.0429 | 11.5559 | 12.7894 | -0.0139 | 0.5649 | incumbent (paired delta CI) | scored |

### 2026-09-12 SAM / Cellpose-SAM center-detection spike

`results/sam_spike_20260912`; 26 rows.
Detector + Voronoi readout; every pre-registered gate failed. Code in code/phase2/detector.

| run | model | seed | fold | epochs | split | n | CD | CV | HEX | mean | ci_low | ci_high | reference | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| val_cpsam_e1 | Cellpose-SAM detector + Voronoi readout |  |  |  | val300 | 300 | 24.4484 | 41.0297 | 15.7801 | 27.0861 |  |  |  | scored |
| cpsam_dNone_f0.8_c-2.0_e0 (detections, train overlays; F1 0.81) | cpsam |  |  |  | overlay19 | 19 | 18.0501 | 36.3918 | 11.4288 | 21.9569 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d28.0_f0.8_c0.0_e1 (detections, train overlays; F1 0.72) | cpsam |  |  |  | overlay19 | 19 | 21.9210 | 31.5047 | 15.2123 | 22.8793 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d28.0_f0.8_c-2.0_e1 (detections, train overlays; F1 0.79) | cpsam |  |  |  | overlay19 | 19 | 17.3806 | 37.5058 | 14.2757 | 23.0540 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d28.0_f0.8_c-2.0_e0 (detections, train overlays; F1 0.79) | cpsam |  |  |  | overlay19 | 19 | 18.8104 | 38.2540 | 12.1642 | 23.0762 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d22.0_f0.8_c-2.0_e0 (detections, train overlays; F1 0.78) | cpsam |  |  |  | overlay19 | 19 | 20.3059 | 36.8986 | 12.3898 | 23.1981 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_dNone_f0.8_c0.0_e0 (detections, train overlays; F1 0.72) | cpsam |  |  |  | overlay19 | 19 | 21.6122 | 34.4170 | 14.0580 | 23.3624 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_dNone_f0.8_c-2.0_e1 (detections, train overlays; F1 0.79) | cpsam |  |  |  | overlay19 | 19 | 17.4730 | 42.0973 | 10.6956 | 23.4220 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d22.0_f0.8_c-2.0_e1 (detections, train overlays; F1 0.77) | cpsam |  |  |  | overlay19 | 19 | 18.8972 | 38.8478 | 12.5956 | 23.4468 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d28.0_f0.8_c0.0_e0 (detections, train overlays; F1 0.69) | cpsam |  |  |  | overlay19 | 19 | 21.1224 | 32.1004 | 17.6520 | 23.6249 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d28.0_f0.4_c0.0_e1 (detections, train overlays; F1 0.64) | cpsam |  |  |  | overlay19 | 19 | 29.6139 | 26.1263 | 16.6514 | 24.1305 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d28.0_f0.4_c0.0_e0 (detections, train overlays; F1 0.66) | cpsam |  |  |  | overlay19 | 19 | 24.3938 | 31.8184 | 18.7747 | 24.9957 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d22.0_f0.8_c0.0_e1 (detections, train overlays; F1 0.70) | cpsam |  |  |  | overlay19 | 19 | 21.5253 | 38.5290 | 18.3293 | 26.1279 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d28.0_f0.4_c-2.0_e1 (detections, train overlays; F1 0.54) | cpsam |  |  |  | overlay19 | 19 | 28.8889 | 29.3165 | 21.3951 | 26.5335 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d22.0_f0.4_c-2.0_e1 (detections, train overlays; F1 0.52) | cpsam |  |  |  | overlay19 | 19 | 33.2785 | 27.1110 | 19.4528 | 26.6141 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_dNone_f0.4_c0.0_e0 (detections, train overlays; F1 0.69) | cpsam |  |  |  | overlay19 | 19 | 25.2353 | 37.2264 | 18.0476 | 26.8364 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d22.0_f0.4_c0.0_e1 (detections, train overlays; F1 0.60) | cpsam |  |  |  | overlay19 | 19 | 31.7632 | 29.5052 | 22.0197 | 27.7627 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d22.0_f0.4_c0.0_e0 (detections, train overlays; F1 0.64) | cpsam |  |  |  | overlay19 | 19 | 31.3085 | 28.8536 | 23.1951 | 27.7858 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_dNone_f0.8_c0.0_e1 (detections, train overlays; F1 0.74) | cpsam |  |  |  | overlay19 | 19 | 23.1547 | 43.6303 | 16.8826 | 27.8892 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_dNone_f0.4_c0.0_e1 (detections, train overlays; F1 0.68) | cpsam |  |  |  | overlay19 | 19 | 29.1946 | 34.9732 | 20.2600 | 28.1426 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d22.0_f0.4_c-2.0_e0 (detections, train overlays; F1 0.56) | cpsam |  |  |  | overlay19 | 19 | 31.9332 | 30.2727 | 25.9667 | 29.3909 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_dNone_f0.4_c-2.0_e1 (detections, train overlays; F1 0.58) | cpsam |  |  |  | overlay19 | 19 | 29.2943 | 36.5029 | 24.0331 | 29.9434 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d28.0_f0.4_c-2.0_e0 (detections, train overlays; F1 0.57) | cpsam |  |  |  | overlay19 | 19 | 33.5547 | 32.7208 | 26.7855 | 31.0203 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_d22.0_f0.8_c0.0_e0 (detections, train overlays; F1 0.68) | cpsam |  |  |  | overlay19 | 19 | 28.5091 | 39.9335 | 27.8005 | 32.0810 |  |  | oracle readout on clicks: CD 3.13 | scored |
| cpsam_dNone_f0.4_c-2.0_e0 (detections, train overlays; F1 0.60) | cpsam |  |  |  | overlay19 | 19 | 34.5002 | 36.0463 | 27.6741 | 32.7402 |  |  | oracle readout on clicks: CD 3.13 | scored |
| sam_vit_b_default (detections, train overlays; F1 0.26) | sam |  |  |  | overlay19 | 19 | 67.5855 | 41.4352 | 44.1329 | 51.0512 |  |  | oracle readout on clicks: CD 3.13 | scored |

### 2026-09-13 Runner smoke test

`results/smoke_runner`, report `results/smoke_runner/REPORT.md`; 7 rows.
Two-epoch small CNN through experiments/run.py; not a result.

| run | model | seed | fold | epochs | split | n | CD | CV | HEX | mean | ci_low | ci_high | reference | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| cv OOF over 2 folds [last+none] |  |  |  |  | oof9000 | 9000 | 14.5589 | 14.4071 | 13.1888 | 14.0516 |  |  |  | computed |
| split | small | 1 |  | 2/2 | val892 | 892 | 14.3815 | 14.9191 | 12.8632 | 14.0546 |  |  |  | done |
| split [last+none] |  |  |  | 2 | val892 | 892 | 14.3815 | 14.9191 | 12.8632 | 14.0546 |  |  |  | scored |
| cv0 [last+none] |  |  |  | 2 | fold-0/2 | 4463 | 14.1677 | 14.7475 | 12.9950 | 13.9701 |  |  |  | scored |
| cv0 | small | 1 | 0 | 2/2 | fold-0/2 | 4463 | 14.1677 | 14.7475 | 12.9950 | 13.9701 |  |  |  | done |
| cv1 | small | 1 | 1 | 2/2 | fold-1/2 | 4537 | 14.0932 | 14.3575 | 13.5755 | 14.0087 |  |  |  | done |
| cv1 [last+none] |  |  |  | 2 | fold-1/2 | 4537 | 14.9437 | 14.0723 | 13.3794 | 14.1318 |  |  |  | scored |

###  platform

`platform`; 2 rows.


| run | model | seed | fold | epochs | split | n | CD | CV | HEX | mean | ci_low | ci_high | reference | status |
|---|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| leaderboard snapshot |  |  |  |  | platform100 | 100 |  |  |  | 8.7042 |  |  |  | reference |
| Phase I slot 1 (v2ens) | 5x ConvNeXt-V2-Tiny + 5x ConvNeXt-Tiny fold models, geometric mean, flip TTA |  |  |  | platform100 | 100 |  |  |  | 8.7583 |  |  |  | submitted |

### 2026-08-26 Exploratory data analysis

`results/eda`; no scored rows. Label distributions, image statistics, annotated-box vs random-crop overlap (IoU about 0.16), baseline error anatomy. No model rows; see results/eda/.

## Removed artifacts

| path | size | reason | date |
|---|---|---|---|
| results/failed_run_20260827_125247 | 3.5M | aborted first overnight attempt (dev_limit 1500), superseded by results/overnight | 2026-09-13 |
| results/baseline | 4K | empty stub | 2026-09-13 |
| results/training | 8K | empty stub | 2026-09-13 |
| results/night_20260912/v2fold0_s42 | 213M | diverged seed-42 V2 run (val 9.95, spike at epoch 3); the clipped rerun v2fold0_s42_clip is kept | 2026-09-13 |
| docker image clear_ec_audit:local | 12.7G | audit-era small-CNN container, superseded | 2026-09-13 |
| docker image clear_ec_phase1:seed123 | 16.5G | small-CNN Phase I container, never submitted (tag phase1-seed123-unused keeps the commit) | 2026-09-13 |
| /home/visilant/CLEAR-EC-phase1-seed123 | 7.6G | worktree of the unused seed123 submission (container.tar and archives); commit kept by tag phase1-seed123-unused | 2026-09-13 |
| .worktrees/spike-sam | 6.2G | spike worktree (its own .venv and 358M of detector weights; re-create with code/phase2/detector/download_weights.sh); code merged, results copied to results/sam_spike_20260912 | 2026-09-13 |
| .worktrees/regression-night, .worktrees/feature/training-viewer | 2M | merged (night) or superseded (viewer copy older than code/viewer) | 2026-09-13 |
| branches fair-overnight-experiments, night/regression-levers, spike/sam-detector, feature/training-viewer, submission/phase1-seed123 |  | merged into main or preserved by tag | 2026-09-13 |

