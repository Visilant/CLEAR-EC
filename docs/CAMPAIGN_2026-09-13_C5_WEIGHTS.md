# C5: five-family weight search (2026-09-14 04:00 EDT)

Families (OOF over 9,000, last checkpoint, flip TTA): V2-Tiny 8.862, Tiny 8.971, V2-Base 8.838, V2-Tiny 648x864 (c3) 8.912,
ConvNeXt-Small (c4) 8.988. CD log-residual correlations 0.955 to 0.974 across every pair.

| ensemble | mean |
|---|---:|
| v2ens_plus fold part (V2-Tiny 2, Tiny 1, V2-Base 3) | 8.8140 |
| + highres (w1) | 8.8145 (delta +0.0005, CI -0.003 to +0.004) |
| + Small (w1) | 8.8100 (delta -0.004, CI -0.010 to +0.001) |
| best of 324 weightings (V2-Tiny 1, V2-Base 2, Small 1) | 8.8092 (delta -0.005, CI -0.012 to +0.003) |

Reading: the ensemble ceiling is reached. Two more families add at most 0.005 with confidence intervals that include zero;
the top twelve weightings sit within 0.002 of each other. Neither the highres nor the Small folds change the slot-2
recommendation (v2ens_plus). Full grid: weight_search5.csv; comparisons: weight_search5.json.
