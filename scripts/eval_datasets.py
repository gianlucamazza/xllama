#!/usr/bin/env python3
"""Pinned benchmark datasets shared by the standard-benchmark lanes (#597).

A spec's "datasets" maps a name to {file, url, sha256, upstream,
upstream_licence}, and optionally "lines": the file is then the first that
many lines of what the url serves, and sha256 pins that prefix (a fixed
subset of a published file, #608). Files live in the user cache, never in
the repo; a file whose SHA256 does not match the pin is refused.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import urllib.request
from pathlib import Path

DEFAULT_CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "xllama/eval-datasets"


class EvalError(RuntimeError):
    pass


def validate_datasets(datasets: dict) -> None:
    if not datasets:
        raise EvalError("spec needs datasets")
    for name, ds in datasets.items():
        for key in ("file", "url", "sha256", "upstream", "upstream_licence"):
            if not ds.get(key):
                raise EvalError(f"dataset {name}: missing {key}")
        if not re.fullmatch(r"[0-9a-f]{64}", ds["sha256"]):
            raise EvalError(f"dataset {name}: sha256 is not 64 hex digits")
        if not ds["url"].startswith("https://"):
            raise EvalError(f"dataset {name}: url must be https")
        lines = ds.get("lines")
        if lines is not None and (type(lines) is not int or lines <= 0):
            raise EvalError(f"dataset {name}: lines must be a positive integer")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_path(spec: dict, name: str, cache: Path) -> Path:
    ds = spec["datasets"][name]
    path = cache / ds["file"]
    if not path.is_file():
        raise EvalError(f"{path} missing; run with --fetch")
    got = sha256_file(path)
    if got != ds["sha256"]:
        raise EvalError(f"{path}: sha256 {got} != pinned {ds['sha256']}")
    return path


def keep_first_lines(path: Path, lines: int, name: str) -> None:
    """Cut |path| to its first |lines| lines, line endings kept."""
    with path.open("rb") as f:
        head = [line for _, line in zip(range(lines), f)]
    if len(head) < lines:
        path.unlink()
        raise EvalError(f"{name}: the fetched file has {len(head)} lines, fewer than {lines}")
    path.write_bytes(b"".join(head))


def fetch(spec: dict, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for name, ds in spec["datasets"].items():
        path = cache / ds["file"]
        if path.is_file() and sha256_file(path) == ds["sha256"]:
            print(f"ok      {name}")
            continue
        # A private temp file per process: two concurrent fetches must not share one.
        fd, tmp_name = tempfile.mkstemp(dir=cache, prefix=ds["file"] + ".", suffix=".part")
        tmp = Path(tmp_name)
        try:
            with urllib.request.urlopen(ds["url"], timeout=120) as resp, os.fdopen(fd, "wb") as out:
                while chunk := resp.read(1 << 20):
                    out.write(chunk)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        if ds.get("lines"):
            keep_first_lines(tmp, ds["lines"], name)
        got = sha256_file(tmp)
        if got != ds["sha256"]:
            tmp.unlink()
            raise EvalError(f"{name}: fetched sha256 {got} != pinned {ds['sha256']}")
        tmp.replace(path)
        print(f"fetched {name}")
