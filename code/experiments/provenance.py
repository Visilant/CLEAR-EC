"""Run provenance: git state, a frozen copy of the code that ran, atomic JSON writes."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from src.artifacts import atomic_path

SNAPSHOT_DIRS = ("src", "experiments")
SNAPSHOT_FILES = ("scripts/run_training.py",)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value) -> None:
    with atomic_path(Path(path)) as temporary:
        temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")


def read_json(path: Path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def git_state(repo: Path) -> dict:
    def _run(args: list[str]) -> str:
        try:
            return subprocess.check_output(args, cwd=repo, text=True).strip()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return ""

    return {
        "branch": _run(["git", "branch", "--show-current"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
        "diff_stat": _run(["git", "diff", "--stat"]),
    }


def snapshot_source(code_root: Path, destination: Path) -> dict[str, str]:
    """Copy the code that will run into <destination> and return sha256 per file."""
    destination = Path(destination)
    if destination.exists():
        shutil.rmtree(destination)
    for directory in SNAPSHOT_DIRS:
        shutil.copytree(code_root / directory, destination / directory,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "specs"))
    for file in SNAPSHOT_FILES:
        (destination / file).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(code_root / file, destination / file)
    return {str(p.relative_to(destination)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(destination.rglob("*.py"))}
