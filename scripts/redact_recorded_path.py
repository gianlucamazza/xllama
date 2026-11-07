#!/usr/bin/env python3
"""Redact host/console absolute paths before recording evidence JSON.

Default: keep a portable placeholder (LocalState-relative, <home>, basename
prefix) so committed receipts do not embed drive letters, user profile
segments, or package-family LocalState prefixes.

Opt-in full paths: --full-paths or XLLAMA_RECORD_FULL_PATHS=1.

No console, host, or user names are hardcoded.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


FULL_PATHS_ENV = "XLLAMA_RECORD_FULL_PATHS"

# Drive-letter absolute, including JSON-escaped backslashes.
_WIN_ABS = re.compile(r"[A-Za-z]:(?:\\+|/)[^\s\"']+")
_POSIX_ABS = re.compile(r"(?:/home|/Users)/[^\s\"']+|/workspace(?:/[^\s\"']*)?")


def record_full_from_env() -> bool:
    return os.environ.get(FULL_PATHS_ENV) == "1"


def _posix_norm(path: str) -> str:
    return path.replace("\\", "/")


def redact_recorded_path(path: str, *, record_full: bool = False) -> str:
    """Return a portable stand-in for an absolute recorded path."""
    if record_full or not path:
        return path

    norm = _posix_norm(path)
    low = norm.lower()

    marker = "/localstate"
    idx = low.find(marker)
    if idx >= 0:
        after = idx + len(marker)
        if after == len(low) or low[after] == "/":
            tail = norm[after:].lstrip("/")
            return f"<console-app-data>/{tail}" if tail else "<console-app-data>"

    for prefix in ("/home/", "/Users/"):
        if low.startswith(prefix.lower()):
            rest = norm[len(prefix) :]
            parts = rest.split("/", 1)
            tail = parts[1] if len(parts) == 2 else ""
            return f"<home>/{tail}" if tail else "<home>"

    if low == "/workspace" or low.startswith("/workspace/"):
        tail = norm[len("/workspace") :].lstrip("/")
        return f"<workspace>/{tail}" if tail else "<workspace>"

    if len(norm) >= 3 and norm[0].isalpha() and norm[1] == ":" and norm[2] == "/":
        rest = norm[3:]
        rest = re.sub(r"(?i)^(?:Data/)?Users/[^/]+", "<user-profile>", rest)
        return f"<abs>/{rest}" if rest else "<abs>"

    if norm.startswith("//"):
        parts = [p for p in norm.split("/") if p]
        tail = "/".join(parts[1:]) if len(parts) > 1 else ""
        return f"<unc>/{tail}" if tail else "<unc>"

    return path


def _looks_absolute(path: str) -> bool:
    if not path:
        return False
    if path[0] == "/" or path.startswith("\\\\"):
        return True
    return len(path) >= 3 and path[0].isalpha() and path[1] == ":" and path[2] in "\\/"


def redact_json_value(value: Any, *, record_full: bool) -> Any:
    if isinstance(value, dict):
        return {k: redact_json_value(v, record_full=record_full) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_json_value(v, record_full=record_full) for v in value]
    if isinstance(value, str) and _looks_absolute(value):
        return redact_recorded_path(value, record_full=record_full)
    return value


def redact_json_text(text: str, *, record_full: bool = False) -> str:
    """Replace absolute path strings in JSON text; keep key order and layout."""
    if record_full:
        return text
    data = json.loads(text)
    pairs: list[tuple[str, str]] = []

    def collect(obj: Any) -> None:
        if isinstance(obj, dict):
            for v in obj.values():
                collect(v)
        elif isinstance(obj, list):
            for v in obj:
                collect(v)
        elif isinstance(obj, str) and _looks_absolute(obj):
            new = redact_recorded_path(obj, record_full=False)
            if new != obj:
                pairs.append((obj, new))

    collect(data)
    out = text
    for old, new in pairs:
        out = out.replace(json.dumps(old), json.dumps(new))
    return out


def redact_text(text: str, *, record_full: bool = False) -> str:
    """Replace absolute path tokens in free-form logs or JSON dumps."""
    if record_full:
        return text

    def win_repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        # JSON dumps double backslashes; normalize before classifying.
        normalized = raw.replace("\\\\", "\\")
        return redact_recorded_path(normalized, record_full=False)

    text = _WIN_ABS.sub(win_repl, text)
    text = _POSIX_ABS.sub(
        lambda m: redact_recorded_path(m.group(0), record_full=False), text
    )
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-paths",
        action="store_true",
        help=f"keep absolute paths (also {FULL_PATHS_ENV}=1)",
    )
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="redact one path and print it (repeatable)",
    )
    parser.add_argument(
        "--json",
        type=Path,
        help="redact absolute path strings in a JSON file in place",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="redact absolute path tokens on stdin, write stdout",
    )
    args = parser.parse_args(argv)
    record_full = bool(args.full_paths or record_full_from_env())

    if args.path:
        for p in args.path:
            print(redact_recorded_path(p, record_full=record_full))
        return 0
    if args.json is not None:
        original = args.json.read_text(encoding="utf-8")
        args.json.write_text(
            redact_json_text(original, record_full=record_full), encoding="utf-8"
        )
        return 0
    if args.text:
        sys.stdout.write(redact_text(sys.stdin.read(), record_full=record_full))
        return 0
    parser.error("specify --path, --json, or --text")
    return 2


if __name__ == "__main__":
    sys.exit(main())
