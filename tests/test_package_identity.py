#!/usr/bin/env python3
"""Tests for scripts/package_identity.py — same-version reinstall guard (#603)."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/package_identity.py"
SPEC = importlib.util.spec_from_file_location("package_identity", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

MANIFEST = """<?xml version="1.0" encoding="utf-8"?>
<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10"
         xmlns:uap="http://schemas.microsoft.com/appx/manifest/uap/windows10">
  <Identity Name="GianlucaMazza.xllama" Publisher="CN=xllama-dev" Version="{version}"
            ProcessorArchitecture="x64" />
</Package>
"""

LISTING = {
    "InstalledPackages": [
        {"PackageFullName": "Microsoft.VCLibs.140.00_14.0.33519.0_x64__8wekyb3d8bbwe"},
        {"PackageFullName": "GianlucaMazza.xllama_1.8.1.1943_x64__abcd1234efgh5"},
    ]
}


def make_msix(directory: Path, version: str, name: str = "renamed.msix") -> Path:
    path = directory / name
    with zipfile.ZipFile(path, "w") as package:
        package.writestr("AppxManifest.xml", MANIFEST.format(version=version))
    return path


class PackageIdentityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_identity_comes_from_the_manifest_not_the_file_name(self) -> None:
        path = make_msix(self.dir, "1.8.1.1943", "xllama_9.9.9.9_x64.msix")
        self.assertEqual(MODULE.msix_identity(path), ("GianlucaMazza.xllama", "1.8.1.1943"))

    def test_the_same_version_is_found_installed(self) -> None:
        self.assertEqual(
            MODULE.installed_full_name(LISTING, "GianlucaMazza.xllama", "1.8.1.1943"),
            "GianlucaMazza.xllama_1.8.1.1943_x64__abcd1234efgh5",
        )

    def test_a_newer_revision_is_not(self) -> None:
        self.assertIsNone(
            MODULE.installed_full_name(LISTING, "GianlucaMazza.xllama", "1.8.1.1956")
        )

    def test_a_version_prefix_is_not_a_match(self) -> None:
        # 1.8.1.19 must not match 1.8.1.1943: the full name has "_" after it.
        self.assertIsNone(MODULE.installed_full_name(LISTING, "GianlucaMazza.xllama", "1.8.1.19"))

    def test_another_family_with_that_version_is_not(self) -> None:
        listing = {"InstalledPackages": [{"PackageFullName": "Other.app_1.8.1.1943_x64__x"}]}
        self.assertIsNone(MODULE.installed_full_name(listing, "GianlucaMazza.xllama", "1.8.1.1943"))

    def test_an_unreadable_package_is_an_error(self) -> None:
        bad = self.dir / "bad.msix"
        bad.write_bytes(b"not a zip")
        with self.assertRaises(MODULE.IdentityError):
            MODULE.msix_identity(bad)
        empty = self.dir / "empty.msix"
        with zipfile.ZipFile(empty, "w") as package:
            package.writestr("other.txt", "x")
        with self.assertRaises(MODULE.IdentityError):
            MODULE.msix_identity(empty)

    def test_command_line(self) -> None:
        path = make_msix(self.dir, "1.8.1.1943")
        same = subprocess.run(
            [sys.executable, str(SCRIPT), "installed", str(path)],
            input=json.dumps(LISTING), capture_output=True, text=True, check=False,
        )
        self.assertEqual(same.returncode, 0)
        self.assertEqual(same.stdout.strip(), "GianlucaMazza.xllama_1.8.1.1943_x64__abcd1234efgh5")
        newer = make_msix(self.dir, "1.8.1.1956", "newer.msix")
        other = subprocess.run(
            [sys.executable, str(SCRIPT), "installed", str(newer)],
            input=json.dumps(LISTING), capture_output=True, text=True, check=False,
        )
        self.assertEqual((other.returncode, other.stdout), (0, ""))
        broken = subprocess.run(
            [sys.executable, str(SCRIPT), "installed", str(path)],
            input="{", capture_output=True, text=True, check=False,
        )
        self.assertEqual(broken.returncode, 2)


if __name__ == "__main__":
    unittest.main()
