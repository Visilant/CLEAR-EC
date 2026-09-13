"""Experiment specs expand to the configs that produced the recorded runs, and reject typos."""
import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

from experiments import CODE, REPO
from experiments.spec import expand_jobs, jobs_by_name, load_spec
from src.training.config import RegressionConfig, recipe_hash

SPECS = CODE / "experiments" / "specs"
NIGHT = REPO / "results" / "night_20260912"


def _recorded_config(run: str) -> dict | None:
    manifest = NIGHT / run / "manifest.json"
    if not manifest.exists():
        return None
    methods = json.loads(manifest.read_text())["methods"]
    return next(iter(methods.values()))["config"]


class SpecTests(unittest.TestCase):
    def test_night_v2_spec_expands_to_the_night_layout(self):
        spec = load_spec(SPECS / "night_v2_5fold.yaml")
        jobs = jobs_by_name(expand_jobs(spec))
        self.assertEqual(sorted(jobs), ["v2fold0", "v2fold1", "v2fold2", "v2fold3", "v2fold4", "v2refit_seed123", "v2refit_seed7"])
        self.assertEqual(jobs["v2fold3"].score_indices, "fold:3/5")
        self.assertIsNone(jobs["v2refit_seed7"].score_indices)
        self.assertEqual(jobs["v2refit_seed7"].run_dir(spec.results_dir).name, "seed_7")
        self.assertEqual(len({recipe_hash(j.config) for j in jobs.values()}), 1, "folds and refits share one recipe")

    def _assert_reproduces(self, spec_name: str, job_name: str, documented_additions: dict):
        recorded = _recorded_config(job_name)
        if recorded is None:
            self.skipTest("night results not on this machine")
        spec = load_spec(SPECS / spec_name)
        produced = asdict(jobs_by_name(expand_jobs(spec))[job_name].config)
        recorded = {k: (tuple(v) if isinstance(v, list) else v) for k, v in recorded.items()}
        common = set(recorded) & set(produced)
        self.assertEqual({k: produced[k] for k in common}, {k: recorded[k] for k in common})
        defaults = asdict(RegressionConfig())
        for key in set(produced) - set(recorded):
            expected = documented_additions.get(key, defaults[key])
            self.assertEqual(produced[key], expected, f"{key} differs from the recorded run without being documented")

    def test_v2fold0_matches_recorded_run(self):
        self._assert_reproduces("night_v2_5fold.yaml", "v2fold0", {"clip_grad": 1.0})

    def test_tiny_fold0_matches_recorded_run(self):
        self._assert_reproduces("night_tiny_5fold.yaml", "fold0", {})

    def test_smoke_spec_is_valid(self):
        jobs = expand_jobs(load_spec(SPECS / "smoke.yaml"))
        self.assertEqual([j.name for j in jobs], ["split", "cv0", "cv1"])

    def test_unknown_field_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.yaml"
            path.write_text("name: bad\nresults_dir: results/x\nbase: {learning_rate: 1}\narms: [{name: a}]\n")
            with self.assertRaises(ValueError):
                load_spec(path)
            path.write_text("name: bad\nresults_dir: results/x\narms: [{name: a, folds: [0]}]\n")
            with self.assertRaises(ValueError):
                load_spec(path)
            path.write_text("name: bad\nresults_dir: results/x\narms: [{name: a}, {name: a}]\n")
            with self.assertRaises(ValueError):
                load_spec(path)


if __name__ == "__main__":
    unittest.main()
