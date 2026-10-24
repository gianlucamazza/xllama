#!/usr/bin/env python3
"""Historical log success must never pass a later console run (#347)."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/console_run_log.py"
SPEC = importlib.util.spec_from_file_location("console_run_log", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
START = b"12:00:00.123 [xllama] App::App()\r\n"
OLD = START + b"GGUF model loaded via llama.cpp\nsession generate: n=24\n"
NEW = START + b"model load failed\n"


class ScopeTests(unittest.TestCase):
    def test_ineffective_delete_does_not_reuse_historical_success(self):
        scoped = MODULE.scope_log(OLD, OLD + NEW)
        self.assertEqual(scoped, NEW)
        self.assertNotIn(b"session generate", scoped)

    def test_absent_or_empty_baseline(self):
        self.assertEqual(MODULE.scope_log(b"", NEW), NEW)

    def test_unchanged_log_is_not_a_new_launch(self):
        with self.assertRaises(ValueError):
            MODULE.scope_log(OLD, OLD)

    def test_late_old_process_writes_do_not_count(self):
        with self.assertRaises(ValueError):
            MODULE.scope_log(OLD, OLD + b"session generate: n=99\n")
        self.assertEqual(MODULE.scope_log(OLD, OLD + b"late old write\n" + NEW), NEW)

    def test_truncation_or_replacement_fails_closed(self):
        for after in (b"", NEW, b"replaced\n" + OLD + NEW):
            with self.subTest(after=after), self.assertRaises(ValueError):
                MODULE.scope_log(OLD, after)

    def test_ambiguous_launches_fail_closed(self):
        with self.assertRaises(ValueError):
            MODULE.scope_log(OLD, OLD + NEW + NEW)

    def test_marker_must_be_exact_and_complete(self):
        for line in (b"[xllama] App::App() complete\n",
                     b"[xllama] App::App()", b"quoted [xllama] App::App()\n"):
            with self.subTest(line=line), self.assertRaises(ValueError):
                MODULE.scope_log(OLD, OLD + line)

    def test_complete_new_start_after_old_partial_line(self):
        for before in (b"old fragment", b"old success without LF", b"[xllama] App::"):
            with self.subTest(before=before):
                self.assertEqual(MODULE.scope_log(before, before + NEW), NEW)

    def test_marker_cannot_be_completed_from_historical_bytes(self):
        marker = b"[xllama] App::App()\n"
        for cut in range(1, len(marker)):
            with self.subTest(cut=cut), self.assertRaises(ValueError):
                MODULE.scope_log(marker[:cut], marker + b"old success\n")

    def test_timestamp_optional_and_binary_bytes_preserved(self):
        data = b"[xllama] App::App()\n\xff\x00\n"
        self.assertEqual(MODULE.scope_log(b"", data), data)

    def test_cli_failure_clears_previous_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            before, after, out = (Path(tmp) / name for name in ("before", "after", "out"))
            before.write_bytes(OLD)
            after.write_bytes(OLD)
            out.write_bytes(OLD)
            result = subprocess.run([sys.executable, str(SCRIPT), str(before), str(after), str(out)],
                                    capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(out.read_bytes(), b"")


class StopAppTests(unittest.TestCase):
    def test_failed_stop_cannot_leave_last_runs_snapshot_usable(self):
        for name in ("validate-console.sh", "validate-console-training.sh"):
            with self.subTest(harness=name), tempfile.TemporaryDirectory() as tmp:
                text = (ROOT / "scripts" / name).read_text()
                start = text.index("run_autopilot() {")
                end = text.index("\n}\n", start) + 3
                Path(tmp, "xllama-log-ready").write_text("ready")
                env = dict(os.environ, TMPDIR_LOCAL=tmp, SCRIPT_DIR=str(ROOT / "scripts"))
                setup = r'''
set -euo pipefail
source "$SCRIPT_DIR/console-run-log.sh"
stop_app() { return 1; }
ring_reset() { :; }
delete_file() { :; }
fetch_file() { : >"$2"; }
'''
                result = subprocess.run(
                    ["bash", "-c", setup + text[start:end] +
                     '\nif run_autopilot 1 <<<"{}"; then exit 99; fi\nfetch_log'],
                    env=env, capture_output=True)
                self.assertEqual(result.returncode, 1)
                self.assertIn(b"no verified pre-run log snapshot", result.stderr)
                self.assertEqual(result.stdout, b"")

    def test_harnesses_use_verified_stop_and_propagate_failure(self):
        for name in ("validate-console.sh", "validate-console-training.sh"):
            with self.subTest(harness=name), tempfile.TemporaryDirectory() as tmp:
                text = (ROOT / "scripts" / name).read_text()
                start = text.index("stop_app() {")
                end = text.index("\n}\n", start) + 3
                # Execute the real harness function under suppressed errexit,
                # as it is used inside command substitutions / gate conditions.
                deploy = Path(tmp) / "deploy.sh"
                deploy.write_text('#!/bin/bash\nprintf "%s\\n" "$@"\nexit 7\n')
                deploy.chmod(0o755)
                env = dict(os.environ, DEPLOY=str(deploy), PFN="test-package")
                result = subprocess.run(
                    ["bash", "-c", "sleep() { :; }; " + text[start:end] +
                     "\nif stop_app; then exit 0; else exit $?; fi"],
                    env=env, capture_output=True)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stderr, b"stop-app\ntest-package\n")


class TransportTests(unittest.TestCase):
    def run_harness(self, before=OLD, after=OLD + NEW, snapshot_code="200",
                    fetch_code="200", snapshot_rc="0", fetch_rc="0", invalidate=False):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "remote-before").write_bytes(before)
            (folder / "remote-after").write_bytes(after)
            env = dict(os.environ, TMPDIR_LOCAL=tmp, SCRIPT_DIR=str(ROOT / "scripts"),
                       SNAPSHOT_CODE=snapshot_code, FETCH_CODE=fetch_code,
                       SNAPSHOT_RC=snapshot_rc, FETCH_RC=fetch_rc,
                       INVALIDATE="1" if invalidate else "0")
            # Run the actual sourced shell helper, mocking only the portal's curl.
            script = r'''
set -euo pipefail
CURL_AUTH=()
BASE_URL=https://example.invalid
PFN=test-package
source "$SCRIPT_DIR/console-run-log.sh"
phase=before
curl() {
    local dest=""
    while (($#)); do
        if [[ "$1" == -o ]]; then dest="$2"; shift; fi
        shift
    done
    cp "$TMPDIR_LOCAL/remote-$phase" "$dest"
    if [[ "$phase" == before ]]; then
        printf '%s' "$SNAPSHOT_CODE"
        return "$SNAPSHOT_RC"
    fi
    printf '%s' "$FETCH_CODE"
    return "$FETCH_RC"
}
snapshot_run_log || exit 1
if [[ "$INVALIDATE" == 1 ]]; then
    SNAPSHOT_CODE=500
    snapshot_run_log || true
fi
phase=after
log=$(fetch_log) || exit 1
cat "$log"
'''
            return subprocess.run(["bash", "-c", script], env=env, capture_output=True)

    def test_shared_helper_extracts_current_failure_not_prior_success(self):
        result = self.run_harness()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, NEW)

    def test_failed_start_is_not_success(self):
        self.assertNotEqual(self.run_harness(after=OLD).returncode, 0)

    def test_missing_baseline_ignores_404_body(self):
        result = self.run_harness(before=b'{"error":"not found"}', after=NEW,
                                  snapshot_code="404")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, NEW)

    def test_http_and_transport_errors_fail_closed(self):
        for kwargs in ({"snapshot_code": "500"}, {"snapshot_code": "401"},
                       {"snapshot_rc": "7"}, {"fetch_code": "404"},
                       {"fetch_code": "403"}, {"fetch_rc": "18"}):
            with self.subTest(kwargs=kwargs):
                result = self.run_harness(**kwargs)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, b"")

    def test_failed_next_snapshot_invalidates_previous_success(self):
        self.assertNotEqual(self.run_harness(invalidate=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
