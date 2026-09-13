"""Post-training steps: score checkpoints on their held-out indices, build ensembles, compare."""
from __future__ import annotations

import glob
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.provenance import read_json, write_json
from experiments.spec import Job, Spec
from experiments.stats import compare_frames
from src.data.cache import open_image_cache
from src.training.common import METRICS, labels_for_indices, score_by_id
from src.training.folds import resolve_indices
from src.training.models import load_checkpoint_file
from src.training.predict import VIEWS, predict_frame_tta


def score_path(spec: Spec, job: Job, which: str, tta: str) -> Path:
    return spec.results_dir / "scores" / f"{job.name}_{which}_{tta}.csv"


def score_job(spec: Spec, job: Job, which: str = "last", tta: str = "flips", *, gpu: int = 0,
              batch_size: int = 16, indices: str | None = None, force: bool = False) -> dict | None:
    """Predict the job's held-out indices with the chosen checkpoint; CSV + JSON next to the night's schema."""
    spec_indices = indices or job.score_indices
    if spec_indices is None:
        return None
    out_csv = score_path(spec, job, which, tta)
    out_json = out_csv.with_suffix(".json")
    if out_json.exists() and not force:
        return read_json(out_json)
    ckpt = job.checkpoint(spec.results_dir, which)
    if not ckpt.exists():
        raise FileNotFoundError(f"{job.name}: no {ckpt.name} yet")
    device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")
    memmap, _ = open_image_cache(spec.cache_dir)
    idx = resolve_indices(spec.cache_dir, spec_indices)
    frame = labels_for_indices(spec.cache_dir, spec.labels_csv, idx)
    model, stats, cfg, epoch = load_checkpoint_file(ckpt, device)
    preds = predict_frame_tta(model, stats, cfg, memmap, frame, device, views=VIEWS[tta], batch_size=batch_size)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_csv, index=False)
    summary = {"job": job.name, "arm": job.arm, "ckpt_dir": str(ckpt.parent), "which": which, "epoch": epoch,
               "indices": spec_indices, "n": int(len(preds)), "tta": tta,
               "mape": score_by_id(preds, frame, expected_ids=frame.ID),
               "checkpoint_sha256_16": hashlib.sha256(ckpt.read_bytes()).hexdigest()[:16]}
    write_json(out_json, summary)
    del model
    return summary


def _geometric_mean(frames: list[pd.DataFrame], weights: list[float]) -> pd.DataFrame:
    index = frames[0].index
    for f in frames[1:]:
        index = index.intersection(f.index)
    if len(index) == 0:
        raise ValueError("ensemble members share no images")
    total = float(sum(weights))
    out = pd.DataFrame(index=index)
    for metric in METRICS:
        logs = sum(w * np.log(f.loc[index, metric].clip(lower=1e-6).to_numpy(float)) for f, w in zip(frames, weights))
        out[metric] = np.exp(logs / total)
    return out


def arm_predictions(spec: Spec, jobs: list[Job], arm: str, which: str, tta: str) -> pd.DataFrame:
    """One prediction per image for an arm: folds are concatenated (OOF), same-image seeds are averaged."""
    members = [j for j in jobs if j.arm == arm and j.score_indices is not None]
    if not members:
        raise ValueError(f"arm {arm} has no scorable jobs")
    per_fold: dict[str, list[pd.DataFrame]] = {}
    for job in members:
        csv = score_path(spec, job, which, tta)
        if not csv.exists():
            raise FileNotFoundError(f"{job.name} is not scored yet ({csv.name})")
        frame = pd.read_csv(csv)
        frame["ID"] = frame["ID"].astype(str).str.strip()
        per_fold.setdefault(job.score_indices, []).append(frame.set_index("ID"))
    parts = [_geometric_mean(frames, [1.0] * len(frames)).assign(idx=frames[0]["idx"]) for frames in per_fold.values()]
    combined = pd.concat(parts)
    if combined.index.duplicated().any():
        raise ValueError(f"arm {arm}: overlapping images across index sets")
    return combined


def ensemble(spec: Spec, jobs: list[Job], name: str, members: dict[str, float], which: str = "last",
             tta: str = "flips", *, force: bool = False) -> dict:
    out_csv = spec.results_dir / "ensembles" / f"{name}.csv"
    out_json = out_csv.with_suffix(".json")
    if out_json.exists() and not force:
        return read_json(out_json)
    frames, weights = [], []
    for arm, weight in members.items():
        frames.append(arm_predictions(spec, jobs, arm, which, tta))
        weights.append(float(weight))
    combined = _geometric_mean(frames, weights)
    combined["idx"] = frames[0].loc[combined.index, "idx"].astype(int)
    combined = combined.reset_index()[["idx", "ID", *METRICS]]
    labels = labels_for_indices(spec.cache_dir, spec.labels_csv, combined["idx"].tolist())
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_csv, index=False)
    summary = {"ensemble": name, "members": members, "which": which, "tta": tta, "n": int(len(combined)),
               "mape": score_by_id(combined, labels, expected_ids=labels.ID)}
    write_json(out_json, summary)
    return summary


def load_predictions(spec: Spec, jobs: list[Job], ref: str, which: str, tta: str) -> pd.DataFrame:
    """A named ensemble, a named arm, or a glob of prediction CSVs (idx, ID, CD, CV, HEX)."""
    ens = spec.results_dir / "ensembles" / f"{ref}.csv"
    if ens.exists():
        return pd.read_csv(ens)
    if any(j.arm == ref for j in jobs):
        return arm_predictions(spec, jobs, ref, which, tta).reset_index()
    pattern = ref if Path(ref).is_absolute() else str((spec.results_dir.parents[1] / ref))
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"reference {ref!r} matches no ensemble, arm or files")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def compare(spec: Spec, jobs: list[Job], candidate: str, reference: str, *, n_boot: int = 2000,
            which: str = "last", tta: str = "flips", force: bool = False) -> dict:
    tag = hashlib.sha256(reference.encode()).hexdigest()[:8]
    out_json = spec.results_dir / "compare" / f"{candidate}_vs_{tag}.json"
    if out_json.exists() and not force:
        return read_json(out_json)
    cand = load_predictions(spec, jobs, candidate, which, tta)
    ref = load_predictions(spec, jobs, reference, which, tta)
    labels = labels_for_indices(spec.cache_dir, spec.labels_csv, sorted(set(cand["idx"].astype(int))))
    result = {"candidate": candidate, "reference": reference, **compare_frames(cand, ref, labels, n_boot=n_boot)}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_json, result)
    return result
