"""Exercise the Git operations the GenAI vendor rebuild uses, without compiling GenAI."""

import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
import urllib.request
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import prepare_genai_source as genai  # noqa: E402
import vendor_source  # noqa: E402

PACKAGE_ID = "Microsoft.ML.OnnxRuntimeGenAI.DirectML"


def git_commit(source, message):
    subprocess.check_call(
        ["git", "-c", "user.name=fixture", "-c", "user.email=fixture@example.org",
         "commit", "-qm", message],
        cwd=source,
    )


class GenAiRebuildTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.source = Path(tmp.name) / "source"
        self.source.mkdir()
        self.git("init", "--quiet")
        self.file = self.source / "dml_helpers.cpp"
        self.file.write_text("before\n")
        (self.source / "unrelated").write_text("original\n")
        self.git("add", ".")
        git_commit(self.source, "base")
        self.pin = self.git("rev-parse", "HEAD").decode().strip()
        self.file.write_text("after\n")
        self.patch = Path(tmp.name) / "change.patch"
        self.patch.write_bytes(self.git("diff"))
        self.git("restore", "--", "dml_helpers.cpp")

    def git(self, *args):
        return subprocess.check_output(["git", *args], cwd=self.source)

    def prepare(self):
        return vendor_source.prepare(self.source, self.pin, self.patch, "GenAI")

    def test_apply_and_verify_reused_checkout(self):
        tree = self.prepare()
        self.assertEqual(self.file.read_text(), "after\n")
        self.assertEqual(self.prepare(), tree)
        (self.source / "unrelated").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "differs from both"):
            self.prepare()
        self.assertEqual((self.source / "unrelated").read_text(), "changed\n")

    def test_refuse_wrong_base_and_untracked_inputs_before_applying(self):
        with self.assertRaisesRegex(ValueError, "rebuild requires"):
            vendor_source.prepare(self.source, "0" * 40, self.patch, "GenAI")
        (self.source / "untracked").touch()
        with self.assertRaisesRegex(ValueError, "untracked"):
            self.prepare()
        self.assertEqual(self.file.read_text(), "before\n")

    def test_refuse_committed_local_change_on_top_of_pin(self):
        # A mirror (or a reused checkout) whose HEAD is not the pin is refused
        # even when its tree differs only slightly.
        (self.source / "unrelated").write_text("mirror edit\n")
        self.git("add", "unrelated")
        git_commit(self.source, "mirror edit")
        with self.assertRaisesRegex(ValueError, "rebuild requires"):
            self.prepare()

    def test_refuse_hidden_edits_and_ignored_output(self):
        self.prepare()
        for flag in ("--assume-unchanged", "--skip-worktree"):
            self.git("update-index", flag, "unrelated")
            (self.source / "unrelated").write_text("hidden\n")
            with self.assertRaisesRegex(ValueError, "assume-unchanged or skip-worktree"):
                self.prepare()
            self.git("update-index", flag.replace("--", "--no-"), "unrelated")
            self.git("restore", "--", "unrelated")
        (self.source / ".gitignore").write_text("build/\n")
        self.git("add", ".gitignore")
        git_commit(self.source, "ignore build")
        self.pin = self.git("rev-parse", "HEAD").decode().strip()
        self.git("restore", "--", "dml_helpers.cpp")
        (self.source / "build").mkdir()
        (self.source / "build" / "stale.obj").write_text("old\n")
        with self.assertRaisesRegex(ValueError, "ignored files"):
            self.prepare()

    def test_refuse_staged_change(self):
        (self.source / "unrelated").write_text("staged\n")
        self.git("add", "unrelated")
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare()


class GenAiPinTests(unittest.TestCase):
    def test_pin_matches_packages_config_version(self):
        packages = ET.parse(ROOT / "uwp/packages.config")
        package = packages.find(f".//package[@id='{PACKAGE_ID}']")
        self.assertEqual(package.attrib["version"], genai.VERSION)
        self.assertEqual(genai.BRANCH, f"rel-{genai.VERSION}")
        self.assertRegex(genai.PIN, r"^[0-9a-f]{40}$")

    def test_pin_has_one_home(self):
        # The PowerShell lane reads the pin from prepare_genai_source.py; it
        # must not carry its own copy of a commit ID.
        ps1 = (ROOT / "scripts/vendor-genai-dml-patch.ps1").read_text(encoding="utf-8")
        self.assertNotRegex(ps1, r"[0-9a-f]{40}")
        self.assertIn("prepare_genai_source.py", ps1)
        out = subprocess.check_output(
            [sys.executable, str(ROOT / "scripts/prepare_genai_source.py"), "--pin"]
        )
        self.assertEqual(out.decode().strip(), genai.PIN)
        # The licence provenance records the same revision.
        sources = (ROOT / "LICENSES/SOURCES.md").read_text(encoding="utf-8")
        self.assertIn(genai.PIN, sources)

    def test_reviewed_patch_touches_only_the_dml_fallback(self):
        stat = subprocess.check_output(
            ["git", "apply", "--numstat", str(genai.PATCH)], cwd=ROOT
        ).decode()
        self.assertEqual([line.split("\t")[2] for line in stat.splitlines()],
                         ["src/dml/dml_helpers.cpp"])

    def test_cli_refuses_other_version(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "scripts/prepare_genai_source.py"), ".", "0.0.0"],
            capture_output=True, text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("No reviewed GenAI overlay", result.stderr)

    @unittest.skipUnless(os.environ.get("XLLAMA_ONLINE_NUSPEC") == "1",
                         "set XLLAMA_ONLINE_NUSPEC=1 to re-read the nuspec from nuget.org")
    def test_pin_equals_official_nuspec_commit(self):
        lower = PACKAGE_ID.lower()
        url = (f"https://api.nuget.org/v3-flatcontainer/{lower}/{genai.VERSION}/"
               f"{lower}.{genai.VERSION}.nupkg")
        with urllib.request.urlopen(url, timeout=60) as response:
            nupkg = zipfile.ZipFile(io.BytesIO(response.read()))
        nuspec = nupkg.read(f"{PACKAGE_ID}.nuspec").decode("utf-8-sig")
        # 0.14.1 has no <repository commit=...>; the release pipeline records
        # the source in <releaseNotes>.
        notes = ET.fromstring(nuspec).find(".//{*}releaseNotes").text
        self.assertIn(f"refs/heads/{genai.BRANCH}", notes)
        self.assertEqual(re.search(r"Commit:\s*([0-9a-f]{40})", notes).group(1), genai.PIN)


if __name__ == "__main__":
    unittest.main()
