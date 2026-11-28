"""The sanitizer audit parses driver outputs and refuses non-Makefile trees."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_sanitizer_build", ROOT / "scripts/check-sanitizer-build.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)

FLAGS = "-fsanitize=address -fsanitize=undefined"


class OutputPathTests(unittest.TestCase):
    def test_separate_and_joined_operands(self):
        self.assertEqual(audit.output_path(["cc", "-c", "-o", "a.o", "a.c"], "a"), "a.o")
        self.assertEqual(audit.output_path(["cc", "-c", "-oa.o", "a.c"], "a"), "a.o")

    def test_missing_or_dangling_output_is_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, "expected one -o output"):
            audit.output_path(["cc", "-c", "a.c"], "a.c")
        with self.assertRaisesRegex(ValueError, "-o has no operand"):
            audit.output_path(["cc", "-c", "a.c", "-o"], "a.c")

    def test_two_outputs_are_refused(self):
        with self.assertRaisesRegex(ValueError, "found 2"):
            audit.output_path(["cc", "-o", "a.o", "-ob.o"], "a.c")


class AuditTreeTests(unittest.TestCase):
    def make_tree(self, root, generator, link_flags=FLAGS):
        (root / "CMakeCache.txt").write_text(f"CMAKE_GENERATOR:INTERNAL={generator}\n")
        target = root / "CMakeFiles/tool.dir"
        target.mkdir(parents=True)
        (target / "main.o").write_bytes(b"")
        (root / "tool").write_bytes(b"")
        (target / "link.txt").write_text(f"c++ {link_flags} CMakeFiles/tool.dir/main.o -o tool\n")
        commands = [{
            "directory": str(root),
            "file": "main.c",
            "command": f"cc {FLAGS} -o CMakeFiles/tool.dir/main.o -c main.c",
        }]
        (root / "compile_commands.json").write_text(json.dumps(commands))

    def run_audit(self, root):
        with mock.patch.object(audit, "ARCHIVES", set()), \
                mock.patch.object(audit, "LINKED", {"tool"}), \
                mock.patch("builtins.print"):
            audit.audit(root)

    def test_makefile_tree_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_tree(Path(directory), "Unix Makefiles")
            self.run_audit(Path(directory))

    def test_ninja_tree_is_refused_by_name(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_tree(Path(directory), "Ninja")
            with self.assertRaisesRegex(ValueError, "generator is 'Ninja'"):
                self.run_audit(Path(directory))

    def test_uninstrumented_link_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            self.make_tree(Path(directory), "Unix Makefiles", "-fsanitize=address")
            with self.assertRaisesRegex(ValueError, "tool link: expected ASan and UBSan"):
                self.run_audit(Path(directory))


if __name__ == "__main__":
    unittest.main()
