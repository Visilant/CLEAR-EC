# Cell-center detection spike (Phase II material)

Moved from branch `spike/sam-detector` (`spike/sam/`) on 2026-09-13. Self-contained: it has its
own `.venv` (see `SPEC.md`), downloads weights with `download_weights.sh`, and imports nothing
from `code/src`.

Outcome (2026-09-12, see `results/sam_spike_20260912/` and `docs/EXPERIMENTS.md`): the detectors
failed every pre-registered gate in `SPEC.md`. Held-out overlays: mean F1 0.48 vs the 0.85 gate,
count ratio 0.40; CD MAPE from detections 45% vs oracle 2.8%. On 300 val images with Cellpose-SAM:
CD MAPE 24.5% (log-corr 0.59), CV 41.0%, HEX 15.8%. The Voronoi readout on the annotator's own
clicks is fine (CD 3.1%); the detector is not.

Why it is kept: CD is deterministic given cell centers (0.8% scatter) and carries all of the
remaining headroom over the label-noise floor (about 4.6 points of CD MAPE, see
`docs/HANDOFF_2026-09-12.md` section 7). A detector with recall above 0.9 plus an annotator-box
emulation is the only known route to it, which makes this the starting point for Phase II, not a
Phase I path. Do not re-run the overlay grid; the gates are settled.

`overlays.json` (parsed click sets for the 25 annotated overlays) now lives at
`results/sam_spike_20260912/overlays.json` and is consumed by `code/scripts/noise_floor/overlay_floor.py`.
