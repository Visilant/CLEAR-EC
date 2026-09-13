"""CLI: python -m experiments.run {run,status,score,report,ledger} ...

  run SPEC [--gpus 0,1] [--limit N] [--dry-run] [--detach] [--no-post] [--results-dir DIR] [--force-post]
  status RESULTS_DIR
  score  RESULTS_DIR [--which last|best] [--tta none|flips] [--jobs GLOB] [--force] [--gpu N]
  report RESULTS_DIR
  ledger [--results DIR] [--out DIR]
"""
from __future__ import annotations

import argparse
import fnmatch
import subprocess
import sys
from pathlib import Path

import torch

from experiments import CODE, REPO
from experiments.provenance import git_state, read_json, snapshot_source, utc_now, write_json
from experiments.spec import Job, Spec, expand_jobs, load_spec


def _log(results_dir: Path):
    def log(message: str) -> None:
        line = f"[{utc_now()}] {message}"
        print(line, flush=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        with open(results_dir / "driver.log", "a") as f:
            f.write(line + "\n")
    return log


def _spec_for_results(results_dir: Path) -> tuple[Spec, list[Job]]:
    plan = read_json(Path(results_dir) / "plan.json")
    if plan is None:
        raise SystemExit(f"{results_dir} has no plan.json; is it a runner results dir?")
    spec = load_spec(plan["spec_path"])
    if Path(results_dir).resolve() != spec.results_dir:
        spec.results_dir = Path(results_dir).resolve()
    return spec, expand_jobs(spec)


def _ensure_plan(spec: Spec, jobs: list[Job], log) -> dict:
    plan_path = spec.results_dir / "plan.json"
    plan = read_json(plan_path)
    if plan is None:
        spec.results_dir.mkdir(parents=True, exist_ok=True)
        sha = snapshot_source(CODE, spec.results_dir / "source")
        plan = {"spec": spec.name, "spec_path": str(spec.path), "spec_text": spec.raw, "created_at": utc_now(),
                "git": git_state(REPO), "source_sha256": sha, "jobs": [j.describe() for j in jobs],
                "python": sys.executable, "torch": torch.__version__}
        write_json(plan_path, plan)
        log(f"plan written: {len(jobs)} jobs, git {plan['git']['commit'][:12]}{' dirty' if plan['git']['dirty'] else ''}")
    elif plan["spec"] != spec.name:
        raise SystemExit(f"{spec.results_dir} belongs to spec {plan['spec']!r}, not {spec.name!r}")
    return plan


def post_steps(spec: Spec, jobs: list[Job], *, gpu: int, force: bool, log) -> None:
    from experiments.report import write_report
    from experiments.score import compare, ensemble, score_job

    post = spec.post
    score_cfg = post.get("score")
    which = (score_cfg or {}).get("which", "last")
    tta = (score_cfg or {}).get("tta", "flips")
    if score_cfg is not None:
        for job in jobs:
            if job.score_indices is None or not job.is_done(spec.results_dir):
                continue
            summary = score_job(spec, job, which, tta, gpu=gpu, force=force)
            log(f"score {job.name} {summary['indices']} {which}/{tta}: mean {summary['mape']['mean']:.3f}")
    for ens in post.get("ensembles", []) or []:
        summary = ensemble(spec, jobs, ens["name"], dict(ens["members"]), ens.get("which", which), ens.get("tta", tta), force=force)
        log(f"ensemble {ens['name']}: n {summary['n']} mean {summary['mape']['mean']:.3f}")
    for cmp in post.get("compare", []) or []:
        result = compare(spec, jobs, cmp["candidate"], cmp["reference"], n_boot=int(cmp.get("n_boot", 2000)),
                         which=which, tta=tta, force=force)
        lo, hi = result["delta_ci95_slide_boot"]
        log(f"compare {cmp['candidate']} vs {cmp['reference']}: delta {result['delta_mean']:+.3f} CI [{lo:+.3f}, {hi:+.3f}] n {result['n']}")
    log(f"report: {write_report(spec, jobs)}")


def cmd_run(args) -> int:
    spec = load_spec(args.spec)
    if args.results_dir:
        spec.results_dir = Path(args.results_dir).resolve()
    jobs = expand_jobs(spec)
    gpus = [int(g) for g in args.gpus.split(",")] if args.gpus else spec.gpus
    if args.detach:
        argv = [sys.executable, "-m", "experiments.run", "run", str(spec.path), "--gpus", ",".join(map(str, gpus))]
        if args.results_dir:
            argv += ["--results-dir", str(spec.results_dir)]
        if args.limit is not None:
            argv += ["--limit", str(args.limit)]
        if args.no_post:
            argv.append("--no-post")
        spec.results_dir.mkdir(parents=True, exist_ok=True)
        with open(spec.results_dir / "driver.log", "a") as log_file:
            process = subprocess.Popen(argv, cwd=CODE, stdin=subprocess.DEVNULL, stdout=log_file,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        write_json(spec.results_dir / "launch.json", {"pid": process.pid, "argv": argv, "launched_at": utc_now()})
        print(f"detached pid {process.pid}; follow {spec.results_dir / 'driver.log'}")
        return 0
    log = _log(spec.results_dir) if not args.dry_run else print
    if not args.dry_run:
        plan = _ensure_plan(spec, jobs, log)
        git = plan["git"]
    else:
        git = {}
    from experiments.queue import run_jobs
    outcome = run_jobs(spec, jobs, gpus=gpus, per_gpu=spec.concurrent_per_gpu, git=git, limit=args.limit,
                       dry_run=args.dry_run, log=log)
    if args.dry_run:
        return 0
    failed = [k for k, v in outcome.items() if v == "failed"]
    log(f"training done: {sum(v == 'ok' for v in outcome.values())} ok, {sum(v == 'skipped' for v in outcome.values())} skipped, {len(failed)} failed")
    if not args.no_post:
        post_steps(spec, jobs, gpu=gpus[0], force=args.force_post, log=log)
    return 1 if failed else 0


def cmd_status(args) -> int:
    spec, jobs = _spec_for_results(Path(args.results_dir))
    status = read_json(spec.results_dir / "status.json", {"jobs": {}}).get("jobs", {})
    print(f"{spec.name}  ({spec.results_dir})")
    print(f"{'job':<28} {'status':<10} {'gpu':>3} {'s':>7}  val mean")
    for job in jobs:
        s = status.get(job.name, {})
        metrics = read_json(job.run_dir(spec.results_dir) / "metrics.json")
        state = s.get("status", "done" if metrics else "pending")
        val = metrics.get("val_mape", {}).get("mean") if metrics else None
        print(f"{job.name:<28} {state:<10} {str(s.get('gpu', '')):>3} {str(int(s['seconds'])) if s.get('seconds') else '':>7}  "
              f"{'' if val is None else f'{val:.3f}'}")
    return 0


def cmd_score(args) -> int:
    from experiments.score import score_job
    spec, jobs = _spec_for_results(Path(args.results_dir))
    log = _log(spec.results_dir)
    for job in jobs:
        if args.jobs and not fnmatch.fnmatch(job.name, args.jobs):
            continue
        if job.score_indices is None or not job.is_done(spec.results_dir):
            continue
        summary = score_job(spec, job, args.which, args.tta, gpu=args.gpu, force=args.force)
        log(f"score {job.name} {summary['indices']} {args.which}/{args.tta}: {summary['mape']}")
    return 0


def cmd_report(args) -> int:
    from experiments.report import write_report
    spec, jobs = _spec_for_results(Path(args.results_dir))
    print(write_report(spec, jobs))
    return 0


def cmd_ledger(args) -> int:
    from experiments.ledger import build_ledger
    csv_path, md_path, n = build_ledger(Path(args.results) if args.results else REPO / "results",
                                        Path(args.out) if args.out else REPO / "docs")
    print(f"{n} rows -> {csv_path}, {md_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m experiments.run", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("run", help="train every pending job of a spec, then score/ensemble/compare/report")
    p.add_argument("spec")
    p.add_argument("--gpus", default="", help="comma-separated GPU ids (default: spec.gpus)")
    p.add_argument("--limit", type=int, default=None, help="run at most N pending jobs")
    p.add_argument("--dry-run", action="store_true", help="print the expanded jobs and their commands")
    p.add_argument("--detach", action="store_true", help="launch in the background; driver.log records progress")
    p.add_argument("--no-post", action="store_true", help="skip scoring/ensembles/compare/report")
    p.add_argument("--force-post", action="store_true", help="recompute post steps (never retrains)")
    p.add_argument("--results-dir", default="", help="override the spec's results_dir")
    p.set_defaults(func=cmd_run)
    p = sub.add_parser("status", help="job table for a results dir"); p.add_argument("results_dir"); p.set_defaults(func=cmd_status)
    p = sub.add_parser("score", help="score finished jobs on their held-out indices")
    p.add_argument("results_dir"); p.add_argument("--which", choices=["last", "best"], default="last")
    p.add_argument("--tta", choices=["none", "flips"], default="flips"); p.add_argument("--jobs", default="", help="glob on job names")
    p.add_argument("--force", action="store_true"); p.add_argument("--gpu", type=int, default=0); p.set_defaults(func=cmd_score)
    p = sub.add_parser("report", help="write REPORT.md for a results dir"); p.add_argument("results_dir"); p.set_defaults(func=cmd_report)
    p = sub.add_parser("ledger", help="regenerate docs/EXPERIMENTS.md and docs/experiments.csv from every campaign")
    p.add_argument("--results", default=""); p.add_argument("--out", default=""); p.set_defaults(func=cmd_ledger)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
