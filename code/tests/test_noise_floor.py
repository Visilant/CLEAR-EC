"""Noise-floor analysis helpers (Voronoi readout, cell bootstrap) and the training-only exclusion filter."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.noise_floor import overlay_floor as of
from src.training.data import apply_exclusions


def _hex_lattice(nx=14, ny=14, spacing=20.0):
    pts = []
    for j in range(ny):
        for i in range(nx):
            pts.append((i * spacing + (spacing / 2 if j % 2 else 0.0), j * spacing * np.sqrt(3) / 2))
    return np.array(pts)


class NoiseFloorTests(unittest.TestCase):
    def test_voronoi_metrics_on_hex_lattice(self):
        pts = _hex_lattice()
        m = of.voronoi_metrics(pts)
        cell_area_um2 = (np.sqrt(3) / 2 * 20.0 ** 2) * of.UM ** 2
        self.assertLess(abs(m["CD"] - 1e6 / cell_area_um2) / (1e6 / cell_area_um2), 1e-6)
        self.assertLess(m["CV"], 1e-6)
        self.assertEqual(m["HEX"], 1.0)
        self.assertTrue(50 < m["n"] < 196)

    def test_cell_bootstrap_matches_binomial_for_hex(self):
        rng = np.random.default_rng(1)
        pts = _hex_lattice() + rng.normal(0, 5.0, (196, 2))  # jitter so nsides vary and HEX < 1
        cells = of.voronoi_cells(pts)
        relsd, base = of.cell_bootstrap(cells, n_boot=1000, rng=rng)
        p, n = base["HEX"], base["n"]
        closed = np.sqrt(p * (1 - p) / n) / p
        self.assertTrue(0.7 < relsd["HEX"] / closed < 1.3)
        self.assertLess(relsd["CD"], relsd["CV"])

    def test_apply_exclusions_drops_only_listed_indices(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "excl.txt"
            f.write_text("3\n5 7\n")
            train_idx = list(range(10))
            self.assertEqual(apply_exclusions(train_idx, str(f)), [0, 1, 2, 4, 6, 8, 9])
            self.assertEqual(apply_exclusions(train_idx, ""), train_idx)


if __name__ == "__main__":
    unittest.main()
