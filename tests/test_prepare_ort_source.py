"""Exercise the actual Git operations used by the vendor rebuild without compiling ORT."""

from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_ort_source as ort


class OrtRebuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.source = Path(tmp.name) / "source"
        self.source.mkdir()
        self.git("init", "--quiet")
        self.file = self.source / "model"
        self.file.write_text("before\n")
        (self.source / "unrelated").write_text("original\n")
        self.git("add", ".")
        self.git("-c", "user.name=fixture", "-c", "user.email=fixture@example.org",
                 "commit", "-qm", "base")
        self.pin = self.git("rev-parse", "HEAD").decode().strip()
        self.file.write_text("after\n")
        self.patch = Path(tmp.name) / "change.patch"
        self.patch.write_bytes(self.git("diff"))
        self.git("restore", "--", "model")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.source)

    def prepare(self):
        return ort.prepare(self.source, self.pin, self.patch)

    def test_apply_and_verify_reused_checkout(self):
        tree = self.prepare()
        self.assertEqual(self.file.read_text(), "after\n")
        self.assertEqual(self.prepare(), tree)
        (self.source / "unrelated").write_text("changed\n")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual((self.source / "unrelated").read_text(), "changed\n")

    def test_refuse_wrong_base_and_untracked_inputs_before_applying(self):
        with self.assertRaises(ValueError):
            ort.prepare(self.source, "0" * 40, self.patch)
        (self.source / "untracked").touch()
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertEqual(self.file.read_text(), "before\n")

    def test_nuget_pin_matches_reviewed_overlay(self):
        packages = ET.parse(ROOT / "uwp/packages.config")
        package = packages.find(".//package[@id='Microsoft.ML.OnnxRuntime.DirectML']")
        self.assertEqual(package.attrib["version"], ort.VERSION)
        self.assertRegex(ort.PIN, r"^[a-f0-9]{40}$")
