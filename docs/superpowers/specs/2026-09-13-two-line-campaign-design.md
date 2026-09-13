# Two-line campaign, 2026-09-13 (12-hour budget)

Goal: a one-point improvement on the equal-weight MAPE mean (v2ens OOF 8.84 over 9,000; platform 8.7583).
Calibration: every ledger lever moved the score by 0.03 or less; a point needs CD 6.3 -> ~3.3, which only a
cell-centre detector can reach (Voronoi readout on true clicks: CD 3.1 percent). Off-the-shelf detectors
failed (Cellpose-SAM F1 0.48); nobody has trained one on the 19 overlays. Line S is the bet, line C the hedge.

## Line S (GPU 1): trained centre detector -> readout -> learned calibration
Data: 19 train overlays (2,950 clicks in 538x408 boxes, full-res frames 1296x972, 0.7716 um/px), 2 val
overlays (0334-23, 0587-23) for tuning, 4 test overlays (0324-23, 0329-23, 0471-23, 0586-23) scored once.
- S1 (h0-1.5): heatmap U-Net (timm ConvNeXt-V2-Nano encoder, light decoder), Gaussian sigma 3 px, loss masked
  to the box, 256-px random crops, flips/rot90/intensity aug, peaks by NMS at half the median spacing.
  Gate on the 4 test overlays: recall >= 0.90, F1 >= 0.85, |count ratio - 1| <= 0.05.
- S2 (h1.5-3): run on all 9,000, annotator-box emulation (best 538x408 window), Voronoi readout.
  Gate: CD MAPE on val 892 <= 6.3.
- S3 (h3-8): self-training on train-split frames whose readout CD is within 5 percent of the label and box
  count >= 100; retrain; re-score both gates.
- S4 (h8-12): fold-honest stack of readout features + CNN OOF -> labels; hybrid CD with CNN fallback.
  Gate: paired slide-clustered delta vs v2ens on the 1,798 val+test images >= 0.10, CI excluding 0.
Stop rule: if S1 recall on the 4 test overlays < 0.80 after one honest tuning pass on the 2 val overlays, stop.

## Line C (GPU 0): direct CNN
- C0 (h0-1): slot-2 candidate: OOF weight search (V2-Tiny, Tiny, V2-Base folds), add the two all-data refits,
  container replay on 50 cases in /home/visilant/CLEAR-EC-phase1-v2ens. Expected +0.02, refits unvalidatable.
- C1 (h1-2.5): two-fold screens (folds 0,1) of a density-map CD head (spatial sum of a 1x1-conv map) and a
  trimmed relative loss (drop top 2 percent of per-batch errors).
- C2 (h3-4): detector-distillation aux head using S2's heatmaps.
- C3 (h4-7): full-resolution 972x1296 input, only if nothing else promoted.
- C4 (h7-12): five-fold promotion of any screen with paired delta <= -0.05; ensemble; ledger + verdict rows.
Results dirs: results/detector_20260913 (S), results/cnn_20260913 (C). Spec files in code/experiments/specs/.
