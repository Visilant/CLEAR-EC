# Phase I submission: ConvNeXt-V2 + ConvNeXt-Tiny fold ensemble

Branch `submission/phase1-v2ens`, checked out at `/home/visilant/CLEAR-EC-phase1-v2ens`, built from
`submission/phase1-seed123` (the earlier container path) plus the night-of-2026-09-12 model code.
Full experimental record: `/home/visilant/CLEAR-EC/results/night_20260912/REPORT.md`.

## Candidate

Ten whole-image regressors, weighted geometric mean, flip TTA (identity, horizontal, vertical, both):

- five ConvNeXt-V2-Tiny (timm `convnextv2_tiny.fcmae_ft_in22k_in1k`, 486x648, EMA 0.999 + cosine, 8 fixed
  epochs), one per slide-grouped fold over all 9,000 labelled images, weight 2;
- five ConvNeXt-Tiny (torchvision, same recipe, 12 epochs), same folds, weight 1.

**Out-of-fold over all 9,000 labelled images: 8.84 mean (CD 6.30, CV 10.25, HEX 9.97).** V2 alone 8.86;
ConvNeXt-Tiny alone 8.97. The previous candidate on this platform (small CNN seed 123) scored 11.96 on the
892-image split; the V2 fold models score about 9.4 on that same subset. Physical clamps after the mean:
CD [372, 4500], CV [0.03, 1.5], HEX [0, 1].

## Container verification (image `clear_ec_phase1:v2ens`, id `sha256:d53087dbb8ad...`, torch 2.1.2 cu121, timm 1.0.29)

- 37 tests pass (`results/phase1_v2ens/tests.log`).
- Fixture MHA (val image 1201-18) through the normal entrypoint offline (`--network none`): CD 2479.28,
  CV 0.4358, HEX 0.4623, identical to the local golden ensemble to seven digits. GPU wall 18 s including
  container start; CPU at two cores 32 s (`smoke_gpu.log`, `smoke_cpu.log`).
- 50 validation MHAs replayed inside the container on GPU against the local golden ensemble: maximum relative
  difference 3.8e-7 (CD), 1.4e-7 (CV), 2.1e-7 (HEX); 6.9 s mean, 8.2 s max per case
  (`out_replay/container_replay.json`).
- Platform limits (authenticated form, 2026-09-12): 5 minutes per case, 32 GB RAM, instance type No GPU / T4
  / A10G. Configure the algorithm for **NVIDIA T4, 16 GB** (CPU-only also fits: 32 s at two cores).
- Fixed tonight before export: the CPU path once set torch threads to the host core count inside the
  container, which oversubscribed a two-core quota to 467 s; it now reads the cgroup CPU quota (capped at 8).

## Upload archive (`results/phase1_v2ens/artifacts/`)

- `clear_ec_phase1_v2ens_selfcontained.tar.gz`: the container with the ten-model bundle baked in at
  `/opt/app/model` (about 4.9 GiB), legacy Docker layout, every layer verified against the image's diff_ids
  by `scripts/export_legacy_container.py`. No separate Model upload is needed; if a Model is attached on the
  platform anyway, `/opt/ml/model` takes precedence. Verified offline with no model mount: same prediction as
  the golden ensemble (`smoke_baked.log`).
- `model_v2ens.tar.gz`: the same bundle as a standalone Model archive (about 1.0 GiB), optional.
- `SHA256SUMS`: identities of both archives.

Grand Challenge's API does not permit creating algorithm images, models or submissions with a user token
(POST returns 403; the site forms are session-only and the account uses Google login), so the upload is done
in the browser by the account holder.

## Upload steps (browser, account holder)

1. Copy the archive to the machine with the browser, e.g.
   `scp visilant@10.113.85.210:/home/visilant/CLEAR-EC-phase1-v2ens/results/phase1_v2ens/artifacts/clear_ec_phase1_v2ens_selfcontained.tar.gz ~/Downloads/`
2. Open the Algorithm [CLEAR EC Direct Regression Seed 123](https://grand-challenge.org/algorithms/clear-ec-direct-regression-seed-123/);
   in its settings choose instance type NVIDIA T4 and 16 GB memory (CPU-only also fits: 32 s per case at two cores).
3. Containers: upload the archive; wait until the import shows the image as active.
4. Try-out on one private sample image; expect three finite JSON numbers and well under 5 minutes.
5. Phase I submission form: select this Algorithm and exactly this container version, no model. Record the
   submission ID and evaluation URL in `results/phase1_v2ens/status.json` before any retry.

Nothing has been uploaded by this session; three Phase I submissions remain (per the authenticated form).
