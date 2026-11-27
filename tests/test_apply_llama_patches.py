"""The patch gate must check llama.cpp itself, even in a fresh clone."""

from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LlamaPatchCheckoutTests(unittest.TestCase):
    def test_uninitialized_submodule_fails_in_both_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            subprocess.run(["git", "init", "--quiet", directory], check=True)
            (checkout / "llama.cpp").mkdir()
            (checkout / "scripts").mkdir()
            script = checkout / "scripts/apply-llama-patches.sh"
            shutil.copyfile(ROOT / "scripts" / script.name, script)
            shutil.copytree(ROOT / "patches", checkout / "patches")
            for args in ([], ["--check"]):
                with self.subTest(args=args):
                    result = subprocess.run(
                        ["bash", str(script), *args], capture_output=True, text=True
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("git submodule update --init", result.stderr)
