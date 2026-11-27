"""Verify the ORT rebuild source and materialize the reviewed AppContainer patch."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile


# The repository.commit in the Microsoft.ML.OnnxRuntime.DirectML 1.24.4 NuGet manifest.
VERSION = "1.24.4"
PIN = "2d924974ef147392ced8409d36bd6d2e7fcc8a74"
PATCH = Path(__file__).resolve().parents[1] / "patches/onnxruntime-extdata-appcontainer.patch"


def prepare(source, pin, patch):
    def git(*args, env=None):
        return subprocess.check_output(["git", *args], cwd=source, env=env)

    if git("rev-parse", "--show-prefix").strip() or git("rev-parse", "HEAD").decode().strip() != pin:
        raise ValueError(f"ORT rebuild requires a checkout rooted at {source} at {pin}")
    if git("ls-files", "--others", "--exclude-standard", "-z"):
        raise ValueError("ORT checkout contains untracked source files; use a clean checkout")
    git("diff", "--cached", "--quiet", "HEAD", "--")

    # Derive the only acceptable result from the pinned tree, independent of
    # local edits. A reverse check alone accepts unrelated tracked changes.
    with tempfile.TemporaryDirectory() as directory:
        env = dict(os.environ, GIT_INDEX_FILE=str(Path(directory) / "index"))
        git("read-tree", pin, env=env)
        git("apply", "--cached", str(patch), env=env)
        patched_tree = git("write-tree", env=env).decode().strip()

    def matches(tree):
        result = subprocess.run(
            ["git", "diff", "--quiet", "--no-ext-diff", "--ignore-submodules=none", tree, "--"],
            cwd=source,
        )
        if result.returncode not in (0, 1):
            result.check_returncode()
        return result.returncode == 0

    if not matches(patched_tree):
        if not matches(pin):
            raise ValueError("ORT source differs from both the pinned tree and the reviewed overlay")
        git("apply", str(patch))
        if not matches(patched_tree):
            raise ValueError("ORT source verification failed after patching")
    return patched_tree


if __name__ == "__main__":
    source, version = sys.argv[1:]
    if version != VERSION:
        sys.exit(f"No reviewed ORT overlay for version {version} (expected {VERSION})")
    try:
        print(f"ORT base {PIN}, patched tree {prepare(Path(source), PIN, PATCH)}")
    except (ValueError, subprocess.CalledProcessError, OSError) as exc:
        sys.exit(str(exc))
