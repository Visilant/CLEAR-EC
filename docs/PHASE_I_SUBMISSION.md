# Phase I early submission: seed 123

This branch freezes the small BatchNorm CNN candidate with **11.955926% local validation mean error**. The exact offline container replays all **892 validation images at batch size one with 11.956694% mean error**. CD is 11.332305%, CV 13.422046%, and HEX 11.115732%. No local test scoring was performed.

The branch is `submission/phase1-seed123`, checked out separately at `/home/visilant/CLEAR-EC-phase1-seed123`. The original working directory and its model bundle remain untouched. The audited runtime/protocol fixes are included; unfinished NAS implementation and the local viewer are excluded. All 37 tests passed. A real MHA input also completed the normal entrypoint offline and produced three finite scalar JSON files.

## Exact artifacts

- Image: `clear_ec_phase1:seed123` (linux/amd64).
- Candidate weights: local `code/model/best_model.pt` with `submission.json`.
- Expected checkpoint SHA-256: `a12a9709635fd8d937f00794ec289418895d3974c4fdf3f8e91ee1ca94d5421f`.
- Upload archives and checksums: local `results/phase1_submission/artifacts/`.
- Test/build/smoke/replay logs: local `results/phase1_submission/`.
- Tracked candidate identity and validation evidence: [PHASE_I_CANDIDATE.json](PHASE_I_CANDIDATE.json).

Weights, images, labels and generated archives are intentionally not committed to Git. This branch's `AUDIT.md` describes the earlier seed-42 baseline; the candidate JSON and this document identify the new seed-123 submission.

Upload `clear_ec_phase1_seed123.tar.gz` as the **container** (approximately 3.9 GiB), and `model_seed123.tar.gz` as the **model weights** (approximately 1.1 MiB). `SHA256SUMS` records their identities. The intermediate `container.tar` is not the upload artifact. The compressed container was converted to legacy Docker layout with every uncompressed layer checked against the original image's diff_id; the image configuration is unchanged.

## Account requirements and upload

The user authorized submitting this candidate to **Phase I: Screening Phase (Preliminary Evaluation)**. No further approval of that action is needed, but account authentication is not configured in this environment.

Required information:

1. A verified Grand Challenge account registered for CLEAR-EC, with a Phase I submission available (the challenge permits up to three).
2. The target Algorithm URL and editor access to manage its container and weights. If none exists, create an Algorithm under the registered account using the challenge's interface.
3. For automated upload, an API token supplied through a secure local credential file; provide its path, not its contents in chat. Alternatively, an account holder can perform the browser upload.

Upload the container archive through the Algorithm's container management page, and upload the model archive as separate model weights. Attach/activate the intended weights and wait for the container to be active. Use an authorized sample image for a private platform try-out; do not publish challenge images as example cases. Then select this algorithm and exact container/model versions in the Phase I submission form. Record the submission ID and evaluation URL before retrying any request, to avoid consuming a slot twice.

The three outputs are `cell-density.json`, `coefficient-of-variation.json` and `hexagonality.json`. Confirm the phase interface, account quota, GPU/memory limits and deadline in the authenticated form. The platform keeps the hidden test images and labels; it returns an evaluation result rather than a downloadable evaluation set.

No upload or official submission has occurred yet. Local verification does not establish platform acceptance or leaderboard rank.
