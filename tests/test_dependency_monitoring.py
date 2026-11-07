#!/usr/bin/env python3
"""Every tracked requirements.txt must have a Dependabot pip update entry.

This is a coverage contract, not a vulnerability scan or a dependency resolver.
Keep the check stdlib-only, like the Linux CI tooling tests.
"""

from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DependencyMonitoringTests(unittest.TestCase):
    def test_tracked_requirements_are_monitored(self):
        tracked = subprocess.check_output(
            ["git", "ls-files", "-z", "--", "requirements.txt", "**/requirements.txt"],
            cwd=ROOT,
        ).decode().split("\0")
        required = {
            "/" + str(Path(path).parent).removeprefix(".").lstrip("/")
            for path in tracked if path
        }
        self.assertTrue(required, "no tracked dependency manifests found")
        config = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
        # The repository uses one update per two-space list item. Match only
        # active pip blocks and their own directory, never comments or another
        # ecosystem's root directory. This intentionally checks the local YAML
        # convention rather than implementing a general YAML parser.
        blocks = re.split(r"(?m)^  - package-ecosystem:\s*", config)[1:]
        monitored = set()
        for block in blocks:
            ecosystem = block.splitlines()[0].strip().strip("\"'")
            if ecosystem != "pip":
                continue
            match = re.search(r"(?m)^    directory:\s*([^#\n]+)", block)
            self.assertIsNotNone(match, "pip entry missing directory")
            monitored.add(match.group(1).strip().strip("\"'"))
        self.assertFalse(
            required - monitored,
            f"requirements manifests outside pip monitoring: {sorted(required - monitored)}",
        )


if __name__ == "__main__":
    unittest.main()
