# Region-aware regression for CLEAR-EC

## Decision and handoff

The user wants a competitive submission, aiming for roughly the top ten. The agreed research bet is **finding informative cell regions and predicting endothelial metrics from those regions**, rather than spending the next experiment budget on broad neural architecture search. This is a hypothesis worth testing, not evidence of a particular leaderboard rank.

This document is the requested handoff to a separate agent. **Documentation only: do not infer authorization to implement or launch experiments from this handoff.** The user requested discussion before further code changes. Agree on the concrete experiment with them before implementation.

## Hypothesis

The existing whole-image CNN may lose useful cell-level information through downsampling and global averaging over blurred, empty, or otherwise uninformative tissue. Selecting several cell-rich regions and retaining higher spatial resolution could improve CD, CV and HEX prediction more than simply increasing network size.

This explanation is not established. Other possible limits include label variability, mismatch between the annotated region and the selected region, and differences between development and hidden-test cohorts.

## Proposed model

Use **region-aware regression**, with a whole-image context branch and several higher-resolution patches:

1. Identify candidate regions containing visible endothelial cells, using a simple image-quality/cell-content rule or a learned selector if justified by available supervision.
2. Select several spatially distributed patches. A sharp patch is not necessarily cell-rich or representative, so avoid relying on focus alone or hard-selecting one patch.
3. Encode patches with a shared visual feature extractor. Compare simple pooling with learned patch weights only if the simpler version shows useful signal.
4. Combine regional features with a lower-resolution whole-image representation.
5. Predict three continuous outputs: CD, CV and HEX. This is regression, not classification.

Retaining whole-image context and multiple regions reduces dependence on one potentially unrepresentative crop. The whole-image baseline should remain available as a fallback.

Individual-cell segmentation is an optional supporting component. Counts, area summaries or geometry can become auxiliary features if they improve the complete predictor. Do not make perfect cell boundaries a prerequisite for the first experiment.

## Why not directly segment cells and calculate every output?

That route is plausible, but detection and boundary errors propagate into the measurements. Missed or merged cells, partial cells and incorrect region boundaries can distort density, cell-area variability and hexagonality. The exact operator behind the provided labels has not been established.

The baseline's HEX measure is fitted-hexagon IoU; a fraction of six-sided cells is a different measure. Preserve distinctions in code, naming and artifacts. Matching a geometric interpretation does not establish agreement with the challenge labels.

Only **19 of the 25 available annotated overlays belong to train**; two belong to validation and four to test. This limits supervised detector/selector development. Use only training overlays to develop region rules, extract dots or fit a detector. A result using human-selected regions or centers is an oracle diagnostic, not an end-to-end result.

Image-level labels are weak supervision for patches: a selected patch does not necessarily have the same metrics as the original annotated tissue. Prefer supervising the aggregated image prediction rather than asserting that every patch has the image's ground truth.

## Smallest informative experiment to agree on

Compare three matched predictors before adding architecture complexity:

| Arm | Input and aggregation | Question |
|---|---|---|
| Whole-image control | Full image using the selected feature extractor | Establish a matched baseline. |
| Fixed-patch control | Whole-image context plus a fixed grid of patches, simple pooling | Does extra spatial detail help without region selection? |
| Region-aware challenger | Whole-image context plus selected patches, same encoder and pooling | Does region selection add value beyond the extra detail? |

Match training budget and seeds where practical, and record inference cost. Choose a fixed patch count, size and selection recipe before evaluation; these choices are not yet agreed. Avoid conflating a better encoder with evidence that region selection works.

If the selected-patch arm wins, the next bounded comparison is simple pooling versus learned weighting. Add segmentation-derived features only after the basic region-aware approach shows value. No broad NAS, attention architecture sweep or detector self-training is part of this initial bet.

Inspect training-image patch selections for blur, borders, artifacts and spatial coverage. Validate the complete automatic pipeline; do not substitute manually selected validation regions. Retain a defined path for images with no usable candidate regions.

## Evaluation and submission decision

- Preserve the existing 7,202 train / 892 validation / 906 test partitions and deterministic cache behavior. Fit on train, select on validation, and do not use test results to tune the approach.
- Group internal splits by slide and keep known duplicate components together. The existing test partition has known train/test duplicates and prior exploratory access; it is not a pristine holdout.
- Score original labels, join by ID and enforce full coverage. Skip near-zero targets separately per metric under the existing scoring convention; do not reconstruct zeros from normalized float32 labels.
- Report CD/CV/HEX errors and their equal-weight mean, paired slide-clustered uncertainty, seed variation and inference cost. A lower overall error can include a per-metric tradeoff; document it.
- Use the current plan's practical gates: approximately 0.3 percentage points for a core replacement and 0.2 points for added complexity, with consistent evidence. These are spending rules, not guarantees of generalization.
- Freeze and verify the complete predictor, including region selection, preprocessing, patch aggregation and any fallback. Check batch-one behavior, real image inputs, offline weights and scalar JSON outputs before treating it as a submission candidate.

The target is a strong single submission, not a large collection of marginal variants. Keep the verified incumbent if the challenger fails the performance or deployment gates. Public leaderboard feedback must not become a tuning loop.

## Current evidence and artifacts

The completed first round tested the small CNN with BatchNorm versus GroupNorm at seeds 42, 123 and 456. BatchNorm remains the preferred control.

| Candidate | Local validation mean error |
|---|---:|
| Audited, container-verified incumbent | 12.509% |
| Best new BatchNorm single model, seed 123 | 11.956% |
| Equal-weight BatchNorm seed ensemble | 11.868% |
| GroupNorm seed ensemble | 12.704% |

The ensemble's 0.088-point gain over the best single model did not meet the added-component gate. The new single model has local cached-input batch-one reload parity, but has not replaced the audited submission bundle or passed final container verification. Seed-42 retraining did not exactly reproduce the audit; seed variability remains material.

- [First-round report](results/round1_20260912_002113/REPORT.md)
- [Separate provisional candidate bundle](results/round1_20260912_002113/candidate_bundle/)
- [Readiness audit](AUDIT.md)
- [Earlier experiment plan](docs/PHASE_I_EXPERIMENT_PLAN.md)

At the leaderboard check during this discussion, Phase I's top five displayed overall errors of **8.7042%, 8.7452%, 8.7720%, 8.7843% and 8.9145%**. The default debugging leaderboard's **3.3124%** leading score is from the sample-image phase. These are different evaluation sets from local validation, so their numerical differences do not establish our ranking or the improvement required. Leaderboard values are a snapshot, not a current guarantee. [Phase I leaderboard](https://clear-ec.grand-challenge.org/evaluation/phase-i-screening-phase-preliminary-evaluation/leaderboard/), [debugging leaderboard](https://clear-ec.grand-challenge.org/evaluation/debug-phase-debug-with-sample-images/leaderboard/)

## Existing partial NAS work

Before the user paused coding, partial NAS work was added in `code/src/training/nas_model.py`, `code/src/training/nas_search.py`, and the model-factory integration in `code/src/training/regression_cnn.py`. **No NAS training was launched.** That partial implementation has not been validated as a complete search system and is not the agreed next approach. It was left untouched during this documentation task, along with existing user changes.

The earlier eight-hour incremental schedule is historical context, not the implementation specification for this new region-aware bet. The next agent should discuss the concrete regional experiment and budget with the user first.
