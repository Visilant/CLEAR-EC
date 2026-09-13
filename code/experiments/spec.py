"""Experiment specs: one YAML file = a base recipe, a list of arms, and post-processing steps.

Arm keys: name (required), seeds (default [42]), overrides (RegressionConfig fields),
n_folds + folds (slide-grouped CV), all_data (train on all labelled images, no validation).
Job names reproduce the night_20260912 layout so old and new runs look the same to the ledger:
fold arms -> <arm><k>, multi-seed arms -> <arm>_seed<s>, otherwise <arm>.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from experiments import REPO
from src.training.config import RegressionConfig, recipe_hash

CONFIG_FIELDS = {f.name for f in fields(RegressionConfig)}
SPEC_KEYS = {"name", "results_dir", "cache_dir", "labels_csv", "gpus", "concurrent_per_gpu", "base", "arms", "post"}
ARM_KEYS = {"name", "seeds", "overrides", "n_folds", "folds", "all_data"}
POST_KEYS = {"score", "ensembles", "compare"}


@dataclass
class Job:
    name: str
    arm: str
    config: RegressionConfig
    score_indices: str | None  # val | fold:k/n | None (all-data refits cannot be scored)

    @property
    def seed(self) -> int:
        return self.config.seed

    def job_dir(self, results_dir: Path) -> Path:
        return Path(results_dir) / self.name

    def run_dir(self, results_dir: Path) -> Path:
        return self.job_dir(results_dir) / "regression_cnn" / f"seed_{self.seed}"

    def is_done(self, results_dir: Path) -> bool:
        return (self.run_dir(results_dir) / "metrics.json").exists()

    def checkpoint(self, results_dir: Path, which: str = "last") -> Path:
        return self.run_dir(results_dir) / ("best_model.pt" if which == "best" else "last.pt")

    def describe(self) -> dict:
        return {"job": self.name, "arm": self.arm, "seed": self.seed, "fold": self.config.fold,
                "n_folds": self.config.n_folds, "all_data": self.config.all_data,
                "score_indices": self.score_indices, "recipe_hash": recipe_hash(self.config)}


@dataclass
class Spec:
    name: str
    path: Path
    results_dir: Path
    cache_dir: Path
    labels_csv: Path
    gpus: list[int]
    concurrent_per_gpu: int
    base: dict
    arms: list[dict]
    post: dict
    raw: str = field(repr=False, default="")


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO / path).resolve()


def _check_config_keys(mapping: dict, where: str) -> None:
    unknown = set(mapping) - CONFIG_FIELDS
    if unknown:
        raise ValueError(f"{where}: unknown RegressionConfig fields {sorted(unknown)}")
    forbidden = {"seed", "fold", "n_folds", "all_data"} & set(mapping)
    if forbidden:
        raise ValueError(f"{where}: {sorted(forbidden)} are set by the arm's seeds/folds/all_data keys")


def load_spec(path: Path | str) -> Spec:
    path = Path(path).resolve()
    raw = path.read_text()
    data = yaml.safe_load(raw) or {}
    unknown = set(data) - SPEC_KEYS
    if unknown:
        raise ValueError(f"{path.name}: unknown top-level keys {sorted(unknown)}")
    for key in ("name", "results_dir", "arms"):
        if key not in data:
            raise ValueError(f"{path.name}: missing required key '{key}'")
    base = dict(data.get("base") or {})
    _check_config_keys(base, f"{path.name} base")
    arms = list(data["arms"])
    seen = set()
    for arm in arms:
        if not isinstance(arm, dict) or "name" not in arm:
            raise ValueError(f"{path.name}: every arm needs a name")
        unknown = set(arm) - ARM_KEYS
        if unknown:
            raise ValueError(f"{path.name} arm {arm['name']}: unknown keys {sorted(unknown)}")
        if arm["name"] in seen:
            raise ValueError(f"{path.name}: duplicate arm name {arm['name']}")
        seen.add(arm["name"])
        _check_config_keys(dict(arm.get("overrides") or {}), f"{path.name} arm {arm['name']}")
        if bool(arm.get("all_data")) and arm.get("folds"):
            raise ValueError(f"{path.name} arm {arm['name']}: all_data and folds are exclusive")
        if arm.get("folds") and not arm.get("n_folds"):
            raise ValueError(f"{path.name} arm {arm['name']}: folds requires n_folds")
    post = dict(data.get("post") or {})
    unknown = set(post) - POST_KEYS
    if unknown:
        raise ValueError(f"{path.name} post: unknown keys {sorted(unknown)}")
    return Spec(
        name=str(data["name"]),
        path=path,
        results_dir=_resolve(data["results_dir"]),
        cache_dir=_resolve(data.get("cache_dir", "data/cache")),
        labels_csv=_resolve(data.get("labels_csv", "data/final_train_ids.csv")),
        gpus=[int(g) for g in data.get("gpus", [0])],
        concurrent_per_gpu=int(data.get("concurrent_per_gpu", 1)),
        base=base,
        arms=arms,
        post=post,
        raw=raw,
    )


def expand_jobs(spec: Spec) -> list[Job]:
    jobs: list[Job] = []
    for arm in spec.arms:
        name = str(arm["name"])
        seeds = [int(s) for s in arm.get("seeds", [42])]
        overrides = dict(arm.get("overrides") or {})
        n_folds = int(arm.get("n_folds") or 0)
        folds = [int(k) for k in (arm.get("folds") or [])]
        all_data = bool(arm.get("all_data", False))
        multi_seed = len(seeds) > 1
        variants: list[tuple[str, dict, str | None]] = []
        if folds:
            for k in folds:
                variants.append((f"{name}{k}", {"fold": k, "n_folds": n_folds}, f"fold:{k}/{n_folds}"))
        elif all_data:
            variants.append((name, {"all_data": True}, None))
        else:
            variants.append((name, {}, "val"))
        for job_name, extra, score_indices in variants:
            for seed in seeds:
                cfg = RegressionConfig(**{**spec.base, **overrides, **extra, "seed": seed})
                full_name = f"{job_name}_seed{seed}" if multi_seed else job_name
                jobs.append(Job(full_name, name, cfg, score_indices))
    names = [j.name for j in jobs]
    if len(names) != len(set(names)):
        raise ValueError(f"duplicate job names in {spec.name}: {sorted(n for n in names if names.count(n) > 1)}")
    return jobs


def jobs_by_name(jobs: list[Job]) -> dict[str, Job]:
    return {j.name: j for j in jobs}
