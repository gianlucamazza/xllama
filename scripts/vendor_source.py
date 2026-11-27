"""Bind a vendor rebuild checkout to a pinned upstream commit plus one reviewed patch.

Shared by the vendor source-rebuild lanes. The checkout may come from the
upstream repository or from any mirror: what is built is accepted only when
HEAD is the pinned commit and the working tree equals the pinned tree, or the
pinned tree with the reviewed patch applied.
"""

import os
from pathlib import Path
import subprocess
import tempfile


def prepare(source, pin, patch, label):
    """Verify `source`, apply `patch` when needed, and return the patched tree ID."""

    def git(*args, env=None):
        return subprocess.check_output(["git", *args], cwd=source, env=env)

    if git("rev-parse", "--show-prefix").strip() or git("rev-parse", "HEAD").decode().strip() != pin:
        raise ValueError(f"{label} rebuild requires a checkout rooted at {source} at {pin}")
    if git("ls-files", "--others", "--exclude-standard", "-z"):
        raise ValueError(f"{label} checkout contains untracked source files; use a clean checkout")
    # Ignored files (a previous build/, caches) are invisible to the tree
    # comparison below but can still be compiled in.
    if git("ls-files", "--others", "--ignored", "--exclude-standard", "-z"):
        raise ValueError(
            f"{label} checkout contains ignored files (old build output?); use a clean checkout"
        )
    # assume-unchanged (lowercase tag) and skip-worktree (S) entries hide local
    # edits from `git diff`, so the tree comparison would pass over them.
    for entry in git("ls-files", "-v", "-z").split(b"\0"):
        if entry and (entry[:1].islower() or entry[:1] == b"S"):
            raise ValueError(f"{label} checkout has assume-unchanged or skip-worktree entries")
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
            raise ValueError(
                f"{label} source differs from both the pinned tree and the reviewed overlay"
            )
        git("apply", str(patch))
        if not matches(patched_tree):
            raise ValueError(f"{label} source verification failed after patching")
    return patched_tree
