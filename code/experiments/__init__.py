"""Declarative experiment runner for CLEAR-EC: YAML spec -> GPU queue -> scored run dirs -> ledger.

Not part of the submission image (see code/.dockerignore). Run from code/ as
`python -m experiments.run ...`.
"""
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
REPO = CODE.parent
