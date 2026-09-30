"""Run bookkeeping: unique ids, exclusive directories, atomic writes, status, manifests."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
import uuid
from pathlib import Path

import numpy as np
import torch

STATES = ("running", "completed", "failed", "interrupted")


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def new_run_id(tag: str) -> str:
    return f"{tag}-{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:6]}"


def make_exclusive_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)  # never reuse a run directory
    return path


def write_atomic(path: Path, data: bytes | str) -> None:
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_bytes(data.encode() if isinstance(data, str) else data)
    os.replace(tmp, path)


def write_json(path: Path, obj: dict) -> None:
    write_atomic(path, json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


def set_status(run_dir: Path, state: str, **extra) -> None:
    assert state in STATES
    write_json(run_dir / "status.json", {"state": state, "updated": now(), **extra})


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def git_state(root: Path) -> dict:
    def run(*args: str) -> str:
        r = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else ""

    return {
        "commit": run("rev-parse", "HEAD") or None,
        "dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def sources(root: Path) -> dict:
    files = sorted((root / "src" / "floppylm").glob("*.py")) + sorted(
        (root / "experiments").glob("*.py")
    )
    return {
        "git": git_state(root),
        "files": {str(p.relative_to(root)): sha256_file(p) for p in files},
    }


def environment(threads: int) -> dict:
    cpu = ""
    try:
        cpu = next(
            line.split(":", 1)[1].strip()
            for line in open("/proc/cpuinfo")
            if line.startswith("model name")
        )
    except (OSError, StopIteration):
        pass
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cpu": cpu,
        "torch_threads": threads,
    }
