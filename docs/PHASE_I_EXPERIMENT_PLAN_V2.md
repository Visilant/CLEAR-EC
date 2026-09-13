# CLEAR-EC Phase I experiment plan, version 2

Written 2026-09-12 02:30 UTC. Planning document only; nothing here has been launched. It supersedes the model-selection and schedule sections of `docs/PHASE_I_EXPERIMENT_PLAN.md` (version 1), whose incumbent has been overtaken; the protocol rules and the promotion gates in version 1 still apply. A later agent implements each experiment. Every item below says what to run, why, what it decides, and how long it should take on the two local RTX A5000 GPUs.

## 0. What changed since version 1

| Fact | Version 1 assumed | Now |
|---|---|---|
| Incumbent | Corrected small CNN, 12.51% val | ConvNeXt-Tiny whole image at 486x648, seed 123: **9.54%** (relative loss) and 9.61% (Huber); their geometric-mean ensemble **9.44%** (CD 6.43, CV 11.81, HEX 10.09) |
| Winning bar | Unknown | Phase I leaderboard snapshot (recorded in `idea.md`): top five at **8.70, 8.75, 8.77, 8.78, 8.91%** on the hidden 100 images; only each team's best submission is listed |
| Pretrained backbones | Not installed | timm and torchvision in `code/.venv`; weights cached for ConvNeXt-T (in1k and in12k), ConvNeXt-V2-T, EfficientNetV2-S, ResNet-18/50, MobileNetV3, DINOv2/v3 ViT-S; `torchvision==0.16.2` pinned in the container requirements |
| Resolution and region | Expected to be the main lever | Measured flat within noise: 243x324 finished at 9.74%, 486x648 at 9.54%, the 648x864 arm sits at 9.77% at epoch 7, and the fixed four-patch arm (384 px, 2.2x the cost) at 9.77% at epoch 8. The quality-selected patch arm is queued. Detail is not the binding constraint for CV and HEX. |
| Calibration | Expected 0.3 to 1 point | Measured on the ConvNeXt ensemble: log-affine recalibration plus a global scale changes val by +0.07 (worse); shrinkage toward the constant only hurts. Identity is the output layer. |
| Ensembling | Expected 0.5 to 1.5 | Two same-seed ConvNeXt runs have log-error correlation 0.90 (CD) and 0.98 (CV); their ensemble gains 0.10 to 0.17. Adding the small CNN at one third weight makes it worse (9.80). Gains must come from genuinely different members. |
| Where the error is | Not analysed | Lowest ground-truth quintile: CD 10.1%, CV 20.9%, HEX 17.6%; middle quintiles 4 to 8%. Family B (long IDs) 9.93% vs family A 8.98%, driven by CV 13.6 vs 10.1. Per-image error p50 7.6, p90 15.5, p99 26. |
| Container | Built for the small CNN only | A second image with torchvision installed (`clear_ec_phase1:seed123`, 16.5 GB on disk, built 01:03 UTC) exists, but no ConvNeXt bundle has been run through it offline, no parity replay has been done, and its export size and upload time are unmeasured. A Debugging Phase with sample images exists on the platform and is a free rehearsal if its submission limit is separate from Phase I's. |
| Headline numbers | Taken at face value | Every quoted val score is the best epoch out of 12 to 13, with adjacent epochs swinging 0.7 to 0.9 points (the 9.54% run's epochs 8 to 12 score 9.67 to 9.81). Treat 9.7 to 9.9% as the honest single-model number and the ensemble's 9.44% as roughly 9.6%. |

Everything else established earlier still holds and is summarised in Appendix B: the labels are a Voronoi center method on about 150 clicked cells; 19 of the 25 overlays are in train, 2 in val, 4 in test; multi-image slides are fellow eyes; the cached Cellpose masks are useless as detections; val differences under about 0.5 points are not resolvable without paired bootstrap.

## 1. Objective, bar, and clock

- **Objective:** finish in the top five of Phase I so the team advances. The snapshot bar is 8.9% on the hidden set. Local val and the hidden set are different populations, so treat the val target as "at or below 8.8% with no slice worse than 12%", and treat any val move under 0.3 points as noise unless a paired slide-clustered bootstrap says otherwise.
- **Clock:** hour 0 is 2026-09-12 02:30 UTC. The challenge page says September 14 end of day with no timezone. Plan against a hard stop at hour 57 (2026-09-14 12:00 UTC); if A1 confirms a later hour, the buffer grows, nothing else moves. Reserve hours 46 to 57 for the last two uploads, their try-outs, and recovery.
- **GPUs right now:** GPU 1 is running the fixed-patch arm with the quality-patch arm queued behind it (about 1.5 h more); GPU 0 is running the 648x864 arm (about 15 min more). Do not kill in-flight arms; they are experiment R1.
- **Debugging Phase:** the platform has a separate debug phase with sample images. If A1 confirms its limit is separate, every container goes through it before a Phase I slot is spent.
- **Agent lane:** one implementing agent is the bottleneck, not the GPUs. Every block launches GPU jobs first and fills upload waits with CPU analysis.

## 2. The approach

Stay on direct regression with ImageNet-pretrained ConvNeXt-class backbones, and spend the remaining time on the levers with measured upside, in this order:

1. **Fix the training schedule and average over its noise.** Every run so far is 12 epochs at a constant learning rate with train loss still falling and val swinging 0.7 to 0.9 points between adjacent epochs. Weight averaging (EMA or the last-k checkpoints), a cosine decay and a few more epochs attack that swing directly and cost 20 minutes a run. Then average genuinely different members: seeds, pretraining sources (in12k, ConvNeXt-V2, EfficientNetV2-S) and one capacity step (ConvNeXt-Small). Same-seed runs correlate at 0.9, so expect 0.3 to 0.6 from a five-member ensemble, not 1.5.
2. **Use more of the labels.** Training uses 7,202 of 9,000 images. Fold models on train plus val give 12% more data per model and honest out-of-fold predictions; a final refit on all 9,000 at a frozen epoch budget is the aggressive slot-3 payload.
3. **The low-quantile tail and cohort shift.** One fifth of the images contribute roughly half the error, and family B is a point worse than family A. Tail-aware targets and losses, and photometric augmentation that bridges the two families, are cheap and aim where the points are.
4. **Delivery discipline.** The ConvNeXt path is not yet container-verified. Rehearse it in the Debugging Phase as soon as A1 allows; slot 1 is the best container-verified ConvNeXt artifact by hour 12; slot 2 the frozen ensemble by hour 46; slot 3 the all-data refit or the largest ensemble by hour 52.

Resolution and region-aware inputs are demoted from "main lever" to "diversity members": three resolutions and one patch design landed within 0.25 points of each other, so a native-resolution or patch model earns its place only by decorrelating the ensemble. The cell-center detection route from version 1 is Phase II material; the only detection work left in Phase I is a stretch item using Voronoi statistics as extra features.

**Expected landing zone on val (honest, not best-epoch):** single stabilized model 9.3 to 9.6; five-member ensemble 9.0 to 9.4; with the all-data refit and tail work perhaps 8.8 to 9.2. That is at or just outside the snapshot bar. Whether it is top five on the hidden set is not knowable locally, which is why the three slots must fail differently.

## 3. Experiment catalogue

Time is wall-clock including setup by the implementing agent; GPU-h is device occupancy. Unless stated, a training run is ConvNeXt-Tiny at 486x648, batch 8, AMP, channels-last, flips, learning rate 1e-4, weight decay 1e-4, and takes about 85 s per epoch (7,202 train plus 892 val images), so 12 to 20 epochs is 17 to 28 minutes.

### Track A. Delivery

**A1. Portal and rules check** (core; 1 h agent, hour 0 to 1). Confirm the Grand Challenge account is verified and Phase I participation approved, whether a try-out consumes a slot, whether the Debugging Phase has its own submission limit, whether advancement uses the best or a designated submission, the exact deadline hour and timezone, per-case time limit and GPU type, and re-read the leaderboard (it lists only each team's best result). Email the organizers about the deadline hour. This is the only step that may need the user.

**A2. Container for the ConvNeXt path** (core; 3 to 4 h agent, 0.3 GPU-h, hours 1 to 6). Start from the existing `clear_ec_phase1:seed123` image rather than a new build. Bundle the ConvNeXt state dict (110 MB, loads with `pretrained=False`, so no download at runtime) through the existing checksum-verified bundle path; prove torchvision imports and the model runs inside the image offline; make the prediction method an explicit in-image setting that fails at build time if weights are missing; on entry write the train-fitted constants (CD 2778, CV 0.36, HEX 0.53) to the three output files, then overwrite on success; wrap every stage so the process always exits zero with three finite, physically clamped scalars; accept header pixel spacing only inside 0.6 to 1.0 um/px. Replay 50 slide-disjoint val cases through the built image at batch one and require per-metric agreement with local predictions within 0.5% and a score within 0.2 points; record cold-start and per-image runtime (about 0.3 s per image on an A5000, so under 2 s on a T4-class card) and peak memory. Keep a golden-output fixture and re-run it before every later upload. Measure the exported legacy-tar gzip size and time an upload of it: the image is 16.5 GB on disk, and at a typical uplink a multi-gigabyte archive costs 30 to 90 minutes, which sets the real slot cadence; trim the image (drop Cellpose weights and unused packages) if the export exceeds about 6 GB.

**A3. Debug-phase rehearsal, then slot 1** (core; 3 to 4 h wall including uploads and try-outs, hours 6 to 12). If A1 confirms the Debugging Phase limit is separate, upload the A2 image there first and use its try-out as the end-to-end proof; only then spend a Phase I slot. Payload: the two-run ConvNeXt geometric ensemble (9.44%) if the multi-model bundle passes A2, otherwise the single 9.54% model, otherwise the audited small-CNN bundle. Purpose: prove the delivery path with the real model family, measure the upload cycle, bank a score.

**A4. Slot 2** (core; 3 h wall, hours 40 to 46). The frozen ensemble from E1 after V2. Golden gate, build, export, upload, try-out.

**A5. Slot 3** (core; 3 h wall, hours 46 to 52). Either the aggressive arm (the R12 all-data refit if A1 confirmed best-of ranking, otherwise whichever of the fold ensemble, R4 or R8 beats slot 2 on val with paired-bootstrap P at least 0.8) or, if nothing does, a differently composed hedge of slot 2 (fewer members, no TTA) so the two slots fail differently. Nothing enters a container after hour 47.

### Track R. Regression improvements

**R1. Finish the in-flight region-aware round** (core; 0 extra GPU-h, hours 0 to 2). Let the fixed-patch and quality-patch arms complete on GPU 1. Decision at matched seed 123 against the 9.54% whole-image run: a patch arm is kept as a member if it is within 0.3 points, and becomes the base recipe for R8 only if it beats whole-image by 0.3 or more on the mean or by 0.5 on CV plus HEX combined. Record its per-image inference cost (about 3x the whole-image arm).

**R2. Seed replication of the incumbent recipe** (core; 0.7 GPU-h, hours 0 to 2 on GPU 0). Seeds 42 and 456 of the whole-image relative-loss recipe. Gives the seed standard deviation every later gate needs and two more ensemble members. Expected: seed spread 0.3 to 0.6; three-seed ensemble 9.2 to 9.4.

**R3. Training-recipe stabilization** (core; 1 GPU-h, hours 2 to 4). Two seeds of the incumbent with an exponential moving average of weights, cosine decay to a small learning rate over 20 epochs and no early stopping. The saved curves swing 9.5 to 10.6 between adjacent epochs, so the best-epoch checkpoint is a lucky draw; averaging weights should recover 0.2 to 0.4 and make every later comparison less noisy. Gate: at least 0.3 better than the R2 mean at matched seeds; if it ties, keep it anyway for its lower variance.

**R4. Native-resolution member** (optional; 1.5 GPU-h, one seed, any idle GPU after hour 8). One seed at 972x1296 (batch 4, about 5.5 min per epoch, 15 epochs) with the R3 recipe. The measured ladder (243x324 at 9.74, 486x648 at 9.54, 648x864 at about 9.77) says resolution is flat within noise, so this run is justified only as a decorrelated ensemble member and as the last check that CV and HEX are not detail-limited. Gate: enters the ensemble if within 0.3 of the incumbent with log-error correlation below 0.85 on CV, and only if its 3x inference cost fits A2's runtime budget.

**R5. Backbone pretraining variants** (core; 1.5 to 2.5 GPU-h, hours 3 to 8 on GPU 1). One seed each at 486x648 with the R3 recipe: ConvNeXt-Tiny in12k-then-in1k, ConvNeXt-V2-Tiny, EfficientNetV2-S; and ConvNeXt-Small as the one capacity step (2x cost; the train-to-val gap of about a point suggests capacity is not exhausted). Purpose is decorrelated ensemble members as much as a better single model: record log-error correlation with the incumbent for each. Gate to enter the ensemble: within 0.3 of the incumbent and log-error correlation below 0.85 on CV, or better than the incumbent outright.

**R6. Tail-targeted loss and target spaces** (core; 1 GPU-h, hours 8 to 12). Two arms, one seed each on the R3 recipe: log targets for CD and CV with a logit target for HEX under the relative loss; and an asymmetric relative loss that penalises over-prediction of low values more than under-prediction (MAPE is unbounded above for small ground truth). Read the lowest-quintile CV and HEX errors and the mean. Gate: lowest-quintile CV or HEX improves by 3 points with no mean regression beyond 0.2. Expected: 0.1 to 0.4 on the mean, concentrated where the points are.

**R7. Photometric augmentation for cohort shift** (core; 1 GPU-h, hours 8 to 12). Gamma, contrast, blur, additive noise and illumination-gradient jitter tuned to make family-A images look like family-B images and vice versa, one to two seeds. Gate: no val regression beyond 0.2 and family-B CV or the worst focus decile improves by 1 point. Even at parity this is the arm that protects the hidden-set score against a different eye-bank mix, so it is adopted unless it regresses.

**R8. Region-aware version 2** (optional; 2 to 3 GPU-h, hours 12 to 24). Only if the quality-patch arm in R1 beats whole-image by 0.3 at matched seed; on current evidence it will not, and the freed GPU goes to seeds and R12. Four 512 px patches selected by the existing quality rule plus the whole-image context, with the R3 recipe and the winning loss from R6, two seeds. Gate: 0.3 on the mean or 0.5 on CV plus HEX over the best whole-image recipe at matched seeds, and inference cost inside A2's budget.

**R9. Seed replication of the frozen recipe** (core; 1.5 to 4 GPU-h depending on resolution, hours 12 to 24). Three seeds of whatever recipe wins at the hour-12 gate (resolution, backbone, loss, augmentation), so the ensemble has at least three members of the strongest configuration plus the decorrelated R5 members.

**R10. Fold models on train plus val** (optional; 2 to 5 GPU-h across both GPUs, hours 24 to 34). Three slide-grouped folds of the frozen recipe over the combined 8,094 images, fixed epoch budget, no early stopping on the held-out fold. Purpose: 12% more training data per model, an honest out-of-fold prediction set for E2, and a fold-spread number. Gate: fold spread under 1 point. The fold models replace, not join, the train-only seeds of the same recipe in the ensemble to keep val an honest selection set for everything else.

**R11. Test-time averaging** (optional; 0.3 GPU-h, hours 32 to 36). Horizontal flip, vertical flip and 180-degree rotation (the frame is not square, so no 90-degree rotations or transposes), and for window models a fixed set of windows. Keep only settings that improve the val mean by 0.2 under the paired bootstrap and fit inside the runtime budget with a 5x margin.

**R12. All-data refit for slot 3** (core; 1 to 2 GPU-h, hours 30 to 36). Once the recipe and epoch budget are frozen from R3 and R9, retrain the strongest configuration on all 9,000 labelled images (train, val and test) with a fixed epoch count and no checkpoint selection, two seeds. It cannot be validated locally, which is exactly why it belongs in slot 3 and never in slot 2: it uses 25% more labels than the val-selected models, and best-of scoring makes that variance free if A1 confirms the ranking rule.

### Track V. Validation and robustness

**V1. Candidate table and harness** (core; 1 h CPU, hours 0 to 2). One script that scores any prediction file on val with per-metric errors, the equal-weight mean, a paired slide-clustered bootstrap against the incumbent, a 100-image subsampling simulation, and slices by ground-truth quintile, ID family, acquisition year and focus decile. Every run in track R lands in this table automatically. Exclude exact-zero labels per metric and report counts. Add a metric-weight sensitivity column (equal, CD-heavy, CV-heavy, HEX-heavy) so a candidate that wins only under one weighting is visible, since the page says equal weights but the organizers' text elsewhere says weighted.

**V2. Robustness audit of the frozen ensemble** (core; 2 to 3 h agent, 0.3 GPU-h, hours 34 to 40). Slices as in V1, plus synthetic degradations (blur, noise, gamma, illumination gradient, partial occlusion) and a leave-one-family-out check of the cheap decision rules. Output: the guard thresholds in the fallback cascade and a go or no-go on any member that collapses on a slice. Shrinkage toward the constant is measured, but on current evidence it only hurts, so identity stays unless a degradation test says otherwise.

**V3. One frozen test read** (core; 30 min, hour 40). Score the final two candidates once on the 906-image local test split, excluding the four train/test duplicates, with the disclosure that this split has had prior exploratory access. Used only to confirm the val ordering, never to pick between candidates that val could not separate.

### Track E. Ensembling and output

**E1. Ensemble assembly** (core; 1 to 2 h agent, hours 32 to 36). Geometric mean of every member that passed its gate, equal weights by default; a non-negative per-metric weight vector over at most five members only if it beats equal weights by 0.2 under the paired bootstrap on out-of-fold predictions from R10. Route per metric if it helps: the score is an equal-weight mean of three independent per-metric means, so CD may come from one member set and CV and HEX from another. Measure the per-image error correlation between the slot 2 and slot 3 candidates and prefer a slot 3 whose errors are least correlated with slot 2 at equal val score. Record the runtime of the full ensemble per image.

**E2. Output layer** (core; 30 min, hour 36). Identity by default (measured: recalibration and shrinkage do not help this model), physical clamps (CD inside the train range, CV positive, HEX in (0, 1]), and a per-metric decision to keep the model output rather than the constant, which every metric currently passes by 3 to 8 points.

### Track C. Stretch

**C1. Detector-derived features as stacking inputs** (stretch; 2 to 3 h agent, 0.5 GPU-h; only after hour 24 with a free GPU and slot 1 uploaded). A fixed-scale difference-of-Gaussians center detector tuned on the 19 train overlays, Voronoi neighbour-count fraction and area dispersion over a quality-selected cluster, appended as features to a per-metric log-space blend with the ensemble. Gate: 0.2 on the mean or 1 point on the lowest CV quintile under the paired bootstrap. Otherwise it is Phase II material.

### Dropped for Phase I

Cell-center detection as a primary route; dot-heatmap networks; self-training; Cellpose-SAM; DINOv2 probes and ViT fine-tuning; watershed segmentation; spectral CV and HEX specialists; annotator simulators; extra hand annotation; per-metric neural specialists as separate networks; dense-map distillation; cross-metric joint stackers; the small-CNN resolution ladder (superseded by the pretrained backbone); calibration and shrinkage layers as a source of gain (measured null). Appendix A records the disposition of each item from the 32-experiment critique.

## 4. Schedule

Hour 0 is 2026-09-12 02:30 UTC.

| Hours | Agent lane | GPU 0 | GPU 1 | Gate at end of block |
|---|---|---|---|---|
| 0 to 2 | A1 portal check; V1 harness and candidate table | R2 seeds 42 and 456 | R1 in-flight fixed-patch, then quality-patch arm | Seed spread known; patch arms scored; portal facts recorded |
| 2 to 4 | A2 container work starts (bundle, torchvision in image, never-throw) | R3 stabilized recipe, seed 123 then 42 | R1 finishing, then R5 backbone 1 (in12k ConvNeXt-T) | R3 verdict; base recipe chosen |
| 4 to 6 | A2 continues; 50-image container parity replay; export size and upload timing | R5 ConvNeXt-Small | R5 backbones 2 and 3 (ConvNeXt-V2-T, EfficientNetV2-S) | Container gate green; backbone verdicts in the table |
| 6 to 12 | A3 debug-phase rehearsal, then slot 1 build, export, upload, try-out; during the waits, slice analysis | R6 tail-loss arms | R7 augmentation arms | **Slot 1 uploaded by hour 12.** Hour-12 recipe freeze: resolution x backbone x loss x augmentation |
| 12 to 24 | Multi-model bundle support in the container; V1 table maintenance | R9 seeds of the frozen recipe | R4 native member, then R8 only if warranted, else more R9 seeds | Member list frozen; every member has val predictions and a log-error correlation with the incumbent |
| 24 to 34 | E1 assembly on train-only members; C1 only if a lane is free | R10 folds 1 and 2 | R10 fold 3, then R12 all-data refit seed 1 | Fold spread under 1 point; out-of-fold set exists |
| 34 to 40 | V2 robustness audit; E1 final weights; E2 output layer; V3 test read at hour 40 | R11 TTA evaluation | R12 all-data refit seed 2 | Frozen slot 2 pipeline with a golden fixture |
| 40 to 46 | A4 slot 2 build, upload, try-out | idle-fill: extra seeds of the frozen recipe, quarantined from slot 2 artifacts | idle-fill: same | **Slot 2 uploaded by hour 46** |
| 46 to 52 | A5 slot 3 decision by the pre-registered rule, build, upload, try-out | frozen | frozen | **Slot 3 uploaded by hour 52** |
| 52 to 57 | Buffer: re-run try-outs, fix platform issues, no new models | idle | idle | All three slots confirmed on the platform |

GPU budget on the critical path: about 12 to 16 GPU-hours out of roughly 90 available. The rule for idle GPU time is extra seeds of the frozen recipe or the all-data refit, written to quarantined directories that no submission artifact reads until a human-readable decision record admits them; never a new idea.

## 5. Decision rules

- **Promotion of a recipe change:** at least 0.3 points better on the val mean at matched seeds, direction consistent across at least two seeds where seeds exist, and a paired slide-clustered bootstrap interval that excludes zero. Ties go to the simpler, already verified recipe.
- **Admission of an ensemble member:** within 0.3 of the incumbent and log-error correlation with the incumbent below 0.85 on CV, or better than the incumbent outright; and the complete ensemble improves by at least 0.2 under the paired bootstrap when it is added.
- **Slot decisions:** a candidate replaces the incumbent for a slot only if P(better at n=100) is at least 0.8 by paired bootstrap; slot 3 is the aggressive arm if the leaderboard keeps the best result per team (A1), otherwise the hedge.
- **If A1 finds the account unverified or participation unapproved:** escalate immediately and continue every GPU experiment; upload the moment access exists.
- **If the ConvNeXt container path is not green by hour 8:** slot 1 ships the audited small-CNN bundle at hour 10 and A2 continues in parallel; slot 2 cannot ship a ConvNeXt until A2 is green.
- **If the quality-patch arm does not beat whole-image by 0.3 at matched seed:** skip R8; the freed GPU goes to R9 seeds, R5 ConvNeXt-Small and R12.
- **If A1 confirms best-of ranking:** slot 3 is the all-data refit ensemble (R12) or the largest ensemble, whichever has the higher val proxy among the members that can be validated; if ranking uses a designated submission, slot 3 is the hedge and R12 is not shipped.
- **If no candidate beats the two-run ensemble by 0.3 on val by hour 24:** slot 2 is still a five-member ensemble of seeds and backbones (variance reduction is worth taking even when the mean barely moves), and slot 3 is the hedge.
- **If V2 finds a member that is 5 points worse than the ensemble on any slice:** drop it from the ensemble and add the slice's guard to the fallback cascade.
- **Nothing enters a container after hour 47.**

## 6. Top risks

1. Platform access or deadline hour differ from assumption. Mitigation: A1 at hour 0, slot 1 by hour 12, ten hours of final buffer.
2. The hidden set has a different eye-bank or export mix; family B already scores a point worse than family A. Mitigation: R7 augmentation, V2 slices, an ensemble across pretraining sources, and the write-first floor in the container.
3. Val overfitting from many small comparisons. Mitigation: pre-registered gates, paired bootstrap, matched seeds, one test read after freezing.
4. Native-resolution or patch models exceed the platform runtime. Mitigation: A2 measures per-image cost with a T4 slowdown factor; members are trimmed to a 5x margin.
5. A crashed case invalidates a slot. Mitigation: write-first constants, never-throw wrapper, and failure-injection cases in A2.
6. The agent lane, not the GPUs, is the bottleneck. Mitigation: GPU jobs launched first in every block, CPU analysis in upload waits, track R optional items droppable in order R8, R10, R11, R5 ConvNeXt-Small.
7. Torchvision inside the container image fails to import or the weights are not found offline. Mitigation: A2 tests this first on the existing phase1 image, with the audited small-CNN bundle as the fallback payload for slot 1.
8. The 892 val images have now carried checkpoint, architecture, resolution, loss and ensemble selection; every headline number is a best-epoch draw. Mitigation: EMA and fixed epoch budgets replace best-epoch picking (R3), the one test read (V3) confirms ordering, and no gate below 0.3 points is acted on.
9. CV sits at 11.8 to 12.5% across every configuration tried and carries a third of the score. Mitigation: R6 tail targets, per-metric routing in E1, R5 diversity; if nothing moves CV by hour 24, accept it and spend the remaining GPU on variance reduction rather than new CV ideas.

## Appendix A. Disposition of the 32 critiqued experiments

The version-1 planning workflow produced 32 experiments (E01 to E32) that were each attacked by a feasibility critic and a validity critic; the critics re-ran the cheap measurements on the local data. Version 1 recorded their detailed critiques; this table maps them to version 2.

| ID | Idea | Version 2 disposition |
|---|---|---|
| E01 | Exact ground-truth readout operator | Appendix B records what was learned; no further Phase I work |
| E02 | Repeat-image noise floor | Dropped; multi-image slides are fellow eyes |
| E03 | Cohort-conditioned constants | Folded into A2 as the write-first floor and into V1 as a slice; not a submission |
| E04 | Correlation-to-MAPE screening gate | Replaced by V1's candidate table |
| E05 | Paired bootstrap and selection rule | V1 and the decision rules in section 5 |
| E06, E07 | Recalibration and output layer | E2, identity by default after the measured null |
| E08 to E12 | Container, fallback, logistics, parity, early submission | A1 to A3, with the ConvNeXt payload |
| E13 | Fourier density estimator | Dropped for Phase I; the regressor's CD is already 6.4% |
| E14 | Annotator-ROI study | Absorbed by the region-aware arms R1 and R8 |
| E15, E22, E23, E32 | Center detector, heatmap net, self-training, weak supervision | Phase II; C1 keeps the cheapest feature version as a stretch item |
| E16, E17, E24, E26 | Tabular features, CV/HEX sources, auxiliary inputs, dense maps | Dropped; a five-scalar feature model ties the old small CNN, not the ConvNeXt |
| E18 | Loss and target-space ablation | R6, retargeted at the low-quantile tail |
| E19 | Backbone by resolution grid | R4 and R5, one seed per cell |
| E20 | Scale-consistent augmentation | Dropped; all images share one pixel spacing |
| E21 | ROI windows and MIL | R1 and R8 without attention pooling |
| E25 | Per-metric specialists | Dropped; the target-space choice lives in R6 |
| E27 | Out-of-fold stacking | E1, equal weights unless a small blend clears 0.2 |
| E28 | Fold retrain on train plus val | R10, three folds, optional |
| E29 | Seeds, backbones, TTA | R2, R5, R9, R11 |
| E30 | Adaptive shrinkage | Dropped; uniform shrinkage already hurts |
| E31 | Robustness audit | V2 |

## Appendix B. Durable findings about the data and labels

- The 25 annotated overlays show a fixed 540x408 px box placed by the annotator on the sharpest window of the frame (the box centre sits within about 22 px of the focus-map argmax in 84% of cases) and a single contiguous cluster of 110 to 180 clicked cell centres covering 40 to 60% of the box. Metrics computed by a Voronoi center method on those centres reproduce the labels: CD from interior cells over bounded Voronoi area (MAPE 3.1%, corr 0.995), HEX as the fraction of interior cells with six neighbours (MAPE 6%, corr 0.90), CV as Voronoi-area std over mean scaled by about 2 (corr 0.92). This is why the baseline's fitted-hexagon HEX and random-crop CD measured the wrong things.
- Realistic detector noise (10% misses, 5% spurious, 3 px jitter) pushes a center-method CD to 12 to 20% and HEX to about 19%, which is why detection is not the Phase I route.
- 19 overlays are in train, 2 in val (0334-23, 0587-23) and 4 in test (0324-23, 0329-23, 0471-23, 0586-23); only train overlays may inform any rule.
- The two ID families (short `NNNN-YY` and long `YYYY-NNNN-ODCN/OSCN`) are separable from a near-black border signature at 96.6% accuracy; they differ in CV and HEX distributions and in current model error (family B is a point worse). The signature is an export artifact, not biology, so the hidden set may not follow it.
- The cached Cellpose masks hold 8 to 219 cells per frame against about 2,100 expected; they must not be used as detections.
- The train-fitted MAPE-optimal constants are CD 2778, CV 0.36, HEX 0.53 (14.17% on val). A five-scalar log-linear model of global image statistics scores 12.57%, which is what the old small CNN had learned; the ConvNeXt's 6.4% CD shows it measures cell geometry, not that shortcut.
- Hardware: 2x RTX A5000 24 GB, 16 cores, 92 GB RAM, 1.2 TB free. ConvNeXt-Tiny at 486x648 trains at about 85 s per epoch; the small CNN at about 4.5 s per epoch; Cellpose at about 0.5 s per image.
