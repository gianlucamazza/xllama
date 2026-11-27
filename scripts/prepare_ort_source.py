"""Verify the ORT rebuild source and materialize the reviewed AppContainer patch."""

from pathlib import Path
import subprocess
import sys

from vendor_source import prepare as prepare_vendor_source


# The repository.commit in the Microsoft.ML.OnnxRuntime.DirectML 1.24.4 NuGet manifest.
VERSION = "1.24.4"
PIN = "2d924974ef147392ced8409d36bd6d2e7fcc8a74"
PATCH = Path(__file__).resolve().parents[1] / "patches/onnxruntime-extdata-appcontainer.patch"


def prepare(source, pin, patch):
    return prepare_vendor_source(source, pin, patch, label="ORT")


if __name__ == "__main__":
    source, version = sys.argv[1:]
    if version != VERSION:
        sys.exit(f"No reviewed ORT overlay for version {version} (expected {VERSION})")
    try:
        print(f"ORT base {PIN}, patched tree {prepare(Path(source), PIN, PATCH)}")
    except (ValueError, subprocess.CalledProcessError, OSError) as exc:
        sys.exit(str(exc))
