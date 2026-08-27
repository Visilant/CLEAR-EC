#!/usr/bin/env python3
"""Compatibility wrapper for the fair overnight runner."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_overnight import main


if __name__ == "__main__":
    main()
