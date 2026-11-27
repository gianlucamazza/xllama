#!/usr/bin/env python3
"""Tests for reproducible release archive generation."""

import hashlib
import os
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseBundleTests(unittest.TestCase):
    def test_workflow_checks_tag_checkout_and_event_before_build(self):
        workflow = (ROOT / ".github/workflows/release-research.yml").read_text()
        before_build = workflow.split("      - name: Build tagged research bundle", 1)[0]
        step = before_build.split("      - name: Verify release source\n        run: |\n", 1)[1]
        script = textwrap.dedent(step.split("\n      - ", 1)[0])
        with tempfile.TemporaryDirectory() as directory:
            def git(*args):
                return subprocess.check_output(["git", *args], cwd=directory, text=True).strip()

            git("init", "-q")
            git("config", "user.name", "Release Test")
            git("config", "user.email", "test@example.org")
            (Path(directory) / "release.toml").write_text('[release]\nversion="1.2.3"\n')
            git("add", "release.toml")
            git("commit", "-qm", "tagged source")
            tagged = git("rev-parse", "HEAD")
            tag = "research-v1.2.3"
            git("tag", "-a", tag, "-m", "annotated release")

            def verify(event, requested=tag):
                return subprocess.run(
                    ["bash", "-eo", "pipefail", "-c", script],
                    cwd=directory,
                    env={**os.environ, "GITHUB_SHA": event, "RELEASE_TAG": requested},
                    capture_output=True,
                ).returncode

            self.assertEqual(verify(tagged), 0)
            self.assertNotEqual(verify(tagged, "research-v0"), 0)
            git("commit", "--allow-empty", "-qm", "different source")
            other = git("rev-parse", "HEAD")
            self.assertNotEqual(verify(other), 0)
            self.assertNotEqual(verify(tagged), 0)
            git("checkout", "--detach", "-q", tagged)
            self.assertNotEqual(verify(other), 0)

    def test_same_ref_produces_same_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            for output in (first, second):
                subprocess.run(
                    ["python3", "scripts/build-release-bundle.py", "--ref", "HEAD", "--output", str(output)],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            first_digest = hashlib.sha256(next(first.glob("*.tar.gz")).read_bytes()).digest()
            second_digest = hashlib.sha256(next(second.glob("*.tar.gz")).read_bytes()).digest()
            self.assertEqual(first_digest, second_digest)


if __name__ == "__main__":
    unittest.main()
