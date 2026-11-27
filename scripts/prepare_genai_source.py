"""Verify the GenAI rebuild source and materialize the reviewed #2280 DML fallback patch."""

from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vendor_source import prepare  # noqa: E402

# Microsoft.ML.OnnxRuntimeGenAI.DirectML 0.14.1 has no <repository commit=...>
# element; its nuspec records the source in <releaseNotes> ("Release Def:
# Branch: refs/heads/rel-0.14.1 / Commit: <PIN>"). Checked 2026-11-27.
VERSION = "0.14.1"
BRANCH = "rel-0.14.1"
PIN = "a30f479af016cb098688726831a9acbb8d19f0b2"
PATCH = Path(__file__).resolve().parents[1] / "patches/onnxruntime-genai-2280-dml-fallback.patch"


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--pin":
        print(PIN)
        sys.exit(0)
    source, version = sys.argv[1:]
    if version != VERSION:
        sys.exit(f"No reviewed GenAI overlay for version {version} (expected {VERSION})")
    try:
        print(f"GenAI base {PIN}, patched tree {prepare(Path(source), PIN, PATCH, 'GenAI')}")
    except (ValueError, subprocess.CalledProcessError, OSError) as exc:
        sys.exit(str(exc))
