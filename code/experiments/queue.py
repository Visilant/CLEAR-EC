"""Run training jobs across GPUs with a fixed number of concurrent workers per GPU.

Idempotent like the night drivers: a job whose run dir already holds metrics.json is skipped.
Status is written atomically to <results_dir>/status.json on every start and finish.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from experiments import CODE
from experiments.provenance import read_json, utc_now, write_json
from experiments.spec import Job, Spec
from src.training.config import recipe_hash


def worker_argv(spec: Spec, job: Job, python: str) -> list[str]:
    job_dir = job.job_dir(spec.results_dir)
    return [python, "-u", str(CODE / "scripts" / "run_training.py"),
            "--config_json", str(job_dir / "config.json"),
            "--results_dir", str(job_dir), "--gpu", "0",
            "--cache_dir", str(spec.cache_dir), "--labels_csv", str(spec.labels_csv)]


def write_job_config(spec: Spec, job: Job, git: dict) -> Path:
    job_dir = job.job_dir(spec.results_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    path = job_dir / "config.json"
    if not path.exists():
        write_json(path, {"config": asdict(job.config), "recipe_hash": recipe_hash(job.config),
                          **job.describe(), "spec": spec.name, "git": git, "created_at": utc_now()})
    return path


def _update_status(path: Path, job: str, **fields) -> None:
    status = read_json(path, {"jobs": {}})
    status.setdefault("jobs", {}).setdefault(job, {}).update(fields)
    status["updated_at"] = utc_now()
    write_json(path, status)


def run_jobs(spec: Spec, jobs: list[Job], *, gpus: list[int], per_gpu: int, python: str = sys.executable,
             git: dict | None = None, limit: int | None = None, dry_run: bool = False,
             poll_seconds: float = 5.0, log=print) -> dict[str, str]:
    """Returns {job name: 'skipped' | 'ok' | 'failed'}."""
    results_dir = spec.results_dir
    status_path = results_dir / "status.json"
    logs_dir = results_dir / "logs"
    outcome: dict[str, str] = {}
    pending: list[Job] = []
    for job in jobs:
        if job.is_done(results_dir):
            outcome[job.name] = "skipped"
        else:
            pending.append(job)
    if limit is not None:
        pending = pending[:limit]
    for job in jobs:
        state = outcome.get(job.name, "pending" if job in pending else "not-selected")
        log(f"{state:>12}  {job.name:<28} gpu-queue  {job.describe()['recipe_hash']}")
    if dry_run:
        for job in pending:
            log("  " + " ".join(worker_argv(spec, job, python)))
        return outcome
    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    git = git or {}
    slots: list[int] = [g for g in gpus for _ in range(per_gpu)]
    running: dict[int, tuple[Job, subprocess.Popen, float]] = {}  # slot index -> ...
    queue = list(pending)
    while queue or running:
        for slot_index, gpu in enumerate(slots):
            if slot_index in running or not queue:
                continue
            job = queue.pop(0)
            write_job_config(spec, job, git)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1")
            log_file = open(logs_dir / f"{job.name}.log", "a")
            process = subprocess.Popen(worker_argv(spec, job, python), cwd=CODE, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log_file, stderr=subprocess.STDOUT)
            running[slot_index] = (job, process, time.time())
            _update_status(status_path, job.name, status="running", pid=process.pid, gpu=gpu, started_at=utc_now())
            log(f"START {job.name} on gpu {gpu} (pid {process.pid})")
        for slot_index in list(running):
            job, process, started = running[slot_index]
            code = process.poll()
            if code is None:
                continue
            del running[slot_index]
            ok = code == 0 and job.is_done(results_dir)
            outcome[job.name] = "ok" if ok else "failed"
            _update_status(status_path, job.name, status=outcome[job.name], exit_code=code,
                           finished_at=utc_now(), seconds=round(time.time() - started, 1))
            log(f"END   {job.name} exit={code} {outcome[job.name]} ({time.time() - started:.0f} s)")
        if running:
            time.sleep(poll_seconds)
    return outcome
