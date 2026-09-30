"""TinyStories: stream stories, exact dedup, deterministic hash split, byte files.

Current capabilities (roadmap S10): exact deduplication on the sha1 of the stripped story text
across all input files; deterministic split by hash (1% test, 1% val). Near-duplicate
filtering and an out-of-distribution test set are later requirements, not implemented.
Stories are joined with the separator byte 0x03, which evaluation scores as a normal target.
"""

import datetime as dt
import hashlib
import io
import json
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path

import numpy as np

EOT = "<|endoftext|>"
SEP = b"\x03"  # story separator byte; removed from story text


def stories(src: str | Iterable[str]) -> Iterator[str]:
    lines = io.StringIO(src) if isinstance(src, str) else src
    seen: set[bytes] = set()
    buf: list[str] = []

    def flush() -> Iterator[str]:
        text = "".join(buf).strip().replace("\x03", "")
        buf.clear()
        h = hashlib.sha1(text.encode()).digest()
        if text and h not in seen:
            seen.add(h)
            yield text

    for line in lines:
        if line.strip() == EOT:
            yield from flush()
        else:
            buf.append(line)
    yield from flush()


def split_of(text: str) -> str:
    h = int.from_bytes(hashlib.sha1(text.encode()).digest()[:8], "little") % 1000
    return "test" if h < 10 else ("val" if h < 20 else "train")


def prepare(raw_files: list[Path], out_dir: Path) -> dict[str, int]:
    """Merge files, dedup across them, write {train,val,test}.bin; return story counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = {s: open(out_dir / f"{s}.bin", "wb") for s in ("train", "val", "test")}
    counts = dict.fromkeys(fh, 0)

    def all_lines() -> Iterator[str]:
        for p in raw_files:
            with open(p, encoding="utf-8", errors="replace") as f:
                yield from f
                yield EOT + "\n"

    for text in stories(all_lines()):
        s = split_of(text)
        fh[s].write(text.encode() + SEP)
        counts[s] += 1
    for f in fh.values():
        f.close()
    return counts


def load(out_dir: Path, split: str) -> np.memmap:
    return np.memmap(out_dir / f"{split}.bin", dtype=np.uint8, mode="r")


SOURCES = {
    "TinyStoriesV2-GPT4-train.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt",
    "TinyStoriesV2-GPT4-valid.txt": "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def manifest(out_dir: Path, raw_dir: Path) -> dict:
    """Hashes verified now, kept apart from provenance claims that cannot be re-proven here."""
    prepared = {
        s: {"sha256": _sha256(out_dir / f"{s}.bin"), "bytes": (out_dir / f"{s}.bin").stat().st_size}
        for s in ("train", "val", "test")
    }
    raw = {}
    for name, url in SOURCES.items():
        p = raw_dir / name
        raw[name] = {"url": url, "present": p.exists()}
        if p.exists():
            raw[name] |= {
                "sha256": _sha256(p),
                "bytes": p.stat().st_size,
                "mtime_utc": dt.datetime.fromtimestamp(p.stat().st_mtime, dt.UTC).isoformat(),
            }
    repro = out_dir / "reproducibility.json"
    return {
        "reproducibility_check": json.loads(repro.read_text()) if repro.exists() else None,
        "verified_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "verified": {"prepared": prepared, "raw": raw},
        "historical": {
            "claim": "raw files downloaded from the URLs above; prepared by data.prepare",
            "demonstrable": False,
            "why": "download and preparation predate the git repository; the code revision used "
            "for preparation was not recorded. Use verify_reproducible() to re-derive.",
        },
    }


def verify_reproducible(raw_dir: Path, out_dir: Path, scratch: Path) -> dict:
    """Re-run prepare() from the raw files and compare hashes with the prepared files."""
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as tmp:
        counts = prepare([raw_dir / n for n in SOURCES], Path(tmp))
        same = {
            s: _sha256(Path(tmp) / f"{s}.bin") == _sha256(out_dir / f"{s}.bin")
            for s in ("train", "val", "test")
        }
    return {
        "reproduced": all(same.values()),
        "per_split": same,
        "story_counts": counts,
        "checked_at": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    }
