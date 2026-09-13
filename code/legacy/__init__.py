"""Legacy Cellpose pipeline (August 2026): segmentation cache, metric sweep, ridge calibration,
and the overnight orchestrator that produced results/overnight and results/audit_20260911.

Kept runnable for provenance; no current work uses it. The container's Cellpose fallback lives in
src/ (inference.py -> src.infer_cellpose_sam), not here. Run from code/ as
`.venv/bin/python legacy/run_overnight.py --help`.
"""
