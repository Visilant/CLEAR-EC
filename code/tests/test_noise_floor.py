"""Tests for the noise-floor analysis helpers and the trainer's training-only exclusion filter."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _hex_lattice(nx=14, ny=14, spacing=20.0):
    pts = []
    for j in range(ny):
        for i in range(nx):
            pts.append((i * spacing + (spacing / 2 if j % 2 else 0.0), j * spacing * np.sqrt(3) / 2))
    return np.array(pts)


def test_voronoi_metrics_on_hex_lattice():
    of = _load("overlay_floor", ROOT / "code/scripts/noise_floor/overlay_floor.py")
    pts = _hex_lattice()
    m = of.voronoi_metrics(pts)
    cell_area_um2 = (np.sqrt(3) / 2 * 20.0 ** 2) * of.UM ** 2
    assert abs(m["CD"] - 1e6 / cell_area_um2) / (1e6 / cell_area_um2) < 1e-6
    assert m["CV"] < 1e-6
    assert m["HEX"] == 1.0
    assert 50 < m["n"] < 196


def test_cell_bootstrap_matches_binomial_for_hex():
    of = _load("overlay_floor", ROOT / "code/scripts/noise_floor/overlay_floor.py")
    rng = np.random.default_rng(1)
    pts = _hex_lattice() + rng.normal(0, 5.0, (196, 2))  # jitter so nsides vary and HEX < 1
    cells = of.voronoi_cells(pts)
    relsd, base = of.cell_bootstrap(cells, n_boot=1000, rng=rng)
    p, n = base["HEX"], base["n"]
    closed = np.sqrt(p * (1 - p) / n) / p
    assert 0.7 < relsd["HEX"] / closed < 1.3
    assert relsd["CD"] < relsd["CV"]


def test_apply_exclusions_drops_only_listed_indices(tmp_path):
    code_dir = ROOT / ".worktrees/regression-night/code"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    rc = _load("regression_cnn_wt", code_dir / "src/training/regression_cnn.py")
    f = tmp_path / "excl.txt"
    f.write_text("3\n5 7\n")
    train_idx = list(range(10))
    kept = rc.apply_exclusions(train_idx, str(f))
    assert kept == [0, 1, 2, 4, 6, 8, 9]
    assert rc.apply_exclusions(train_idx, "") == train_idx
