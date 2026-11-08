#!/usr/bin/env python3
"""This-run diffuse-result.csv rows only; no stale VAE number (#369)."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/console_taesd_result.py"
SPEC = importlib.util.spec_from_file_location("console_taesd_result", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

HEADER = "model,steps,seed,te_ms,unet_ms,unet_ms_per_step,vae_ms,total_ms,resolution,date\n"
STALE = HEADER + (
    "sd-turbo-fp16,1,42,100.0,200.0,200.0,635.6,935.6,512x512,2026-10-25T12:00:00Z\n"
)
FRESH = HEADER + (
    "sd-turbo-fp16,1,42,110.0,210.0,210.0,412.0,732.0,512x512,2026-11-01T08:00:10Z\n"
)
RUN_START = "2026-11-01T08:00:00Z"
FAMILY = "GianlucaMazza.xllama_publisher"


def row(model="sd-turbo-fp16", steps="1", seed="42", vae="635.6", date=""):
    return (
        HEADER
        + f"{model},{steps},{seed},100.0,200.0,200.0,{vae},935.6,512x512,{date}\n"
    )


class AssessTests(unittest.TestCase):
    def test_fresh_date_still_requires_expected_job(self):
        for kwargs in ({"model": "other"}, {"steps": "4"}, {"seed": "99"}):
            with self.subTest(kwargs=kwargs):
                ok, vae, _ = MODULE.assess_result(row(date="2026-11-01T08:00:10Z", **kwargs), RUN_START)
                self.assertFalse(ok)
                self.assertEqual(vae, "")

    def test_vae_duration_must_be_finite_nonnegative_number(self):
        for value in ("-1", "NaN", "inf", "1;exit 0", "word"):
            with self.subTest(value=value):
                ok, vae, _ = MODULE.assess_result(row(vae=value, date="2026-11-01T08:00:10Z"), RUN_START)
                self.assertFalse(ok)
                self.assertEqual(vae, "")


    def test_stale_last_row_is_rejected(self):
        ok, vae, msg = MODULE.assess_result(STALE, RUN_START, STALE)
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("stale", msg)
        self.assertIn("2026-10-25T12:00:00Z", msg)
        self.assertNotIn("635.6", msg.split("stale")[0])

    def test_stale_date_without_before_snapshot_is_rejected(self):
        ok, vae, msg = MODULE.assess_result(STALE, RUN_START, "")
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("2026-10-25T12:00:00Z", msg)
        self.assertIn(RUN_START, msg)

    def test_missing_date_without_snapshot_is_rejected(self):
        missing = row(date="")
        ok, vae, msg = MODULE.assess_result(missing, RUN_START, "")
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("missing or invalid date", msg)
        self.assertNotIn("635.6", msg.split("date")[0])

    def test_invalid_date_without_snapshot_is_rejected(self):
        bad = row(date="not-a-date")
        ok, vae, msg = MODULE.assess_result(bad, RUN_START, "")
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("missing or invalid date", msg)
        self.assertIn("not-a-date", msg)

    def test_invalid_run_start_without_snapshot_is_rejected(self):
        ok, vae, msg = MODULE.assess_result(FRESH, "not-a-timestamp", "")
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("run_start", msg)
        self.assertIn("not-a-timestamp", msg)

    def test_missing_date_with_rewritten_expected_job_is_accepted(self):
        missing = row(date="", vae="77.0")
        ok, vae, msg = MODULE.assess_result(missing, RUN_START, STALE)
        self.assertTrue(ok, msg)
        self.assertEqual(vae, "77.0")

    def test_invalid_run_start_with_rewritten_expected_job_is_accepted(self):
        ok, vae, msg = MODULE.assess_result(FRESH, "not-a-timestamp", STALE)
        self.assertTrue(ok, msg)
        self.assertEqual(vae, "412.0")

    def test_this_run_row_is_accepted(self):
        ok, vae, msg = MODULE.assess_result(FRESH, RUN_START, STALE)
        self.assertTrue(ok)
        self.assertEqual(vae, "412.0")
        self.assertEqual(msg, "")

    def test_clock_skew_still_accepts_a_rewritten_expected_job(self):
        # Console clock behind the host: date is before run start, but the
        # file is a rewritten taesd job versus the pre-run snapshot.
        skewed = HEADER + (
            "sd-turbo-fp16,1,42,1.0,2.0,2.0,88.0,91.0,512x512,2026-10-31T23:59:00Z\n"
        )
        ok, vae, msg = MODULE.assess_result(skewed, RUN_START, STALE)
        self.assertTrue(ok, msg)
        self.assertEqual(vae, "88.0")

    def test_rewritten_row_must_be_the_taesd_job(self):
        other = HEADER + (
            "some-other-model,4,99,1.0,2.0,2.0,88.0,91.0,512x512,2026-10-31T23:59:00Z\n"
        )
        ok, vae, msg = MODULE.assess_result(other, RUN_START, STALE)
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("not the taesd job", msg)

    def test_missing_after_file_fails(self):
        ok, vae, msg = MODULE.assess_result("", RUN_START, STALE)
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("no diffuse-result.csv row", msg)

    def test_cli_prints_vae_only_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            after = Path(tmp) / "after.csv"
            before = Path(tmp) / "before.csv"
            after.write_text(FRESH)
            before.write_text(STALE)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "assess",
                    "--after",
                    str(after),
                    "--before",
                    str(before),
                    "--run-start",
                    RUN_START,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "412.0")
            self.assertEqual(result.stderr, "")

    def test_cli_stale_is_nonzero_and_prints_no_vae(self):
        with tempfile.TemporaryDirectory() as tmp:
            after = Path(tmp) / "after.csv"
            after.write_text(STALE)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "assess",
                    "--after",
                    str(after),
                    "--run-start",
                    RUN_START,
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("stale", result.stderr)
            self.assertNotIn("ok:", result.stderr)


class PrivateWsTests(unittest.TestCase):
    def test_windows_full_package_name_matches_family(self):
        data = self.payload(PackageFullName="GianlucaMazza.xllama_1.7.0.1350_x64__publisher",
                            PrivateWorkingSet=1024 * 1024)
        self.assertEqual(MODULE.private_ws_mb(data, FAMILY), 1)

    def test_windows_full_package_name_rejects_other_publisher(self):
        data = self.payload(PackageFullName="GianlucaMazza.xllama_1.7.0.1350_x64__another",
                            PrivateWorkingSet=1024 * 1024)
        self.assertIsNone(MODULE.private_ws_mb(data, FAMILY))


    def payload(self, **proc):
        return json.dumps({"Processes": [proc]})

    def test_private_working_set_is_converted_from_bytes(self):
        data = self.payload(
            ImageName="xllama.exe",
            PackageFullName="GianlucaMazza.xllama_1.7.0.0_x64__publisher",
            WorkingSetSize=999999999,
            PrivateWorkingSet=2 * 1024 * 1024 * 1024,
        )
        self.assertEqual(MODULE.private_ws_mb(data, FAMILY), 2048)

    def test_working_set_alone_is_not_used(self):
        data = self.payload(
            ImageName="xllama.exe",
            PackageFullName="GianlucaMazza.xllama_1.7.0.0_x64__publisher",
            WorkingSetSize=3 * 1024 * 1024 * 1024,
        )
        self.assertIsNone(MODULE.private_ws_mb(data, FAMILY))

    def test_peak_keys_are_not_used(self):
        data = self.payload(
            ImageName="xllama.exe",
            PackageFullName="GianlucaMazza.xllama_1.7.0.0_x64__publisher",
            PeakWorkingSetSize=2 * 1024 * 1024 * 1024,
            PrivateWorkingSetSize=3 * 1024 * 1024 * 1024,
        )
        self.assertIsNone(MODULE.private_ws_mb(data, FAMILY))

    def test_image_name_guess_is_not_enough(self):
        data = self.payload(
            ImageName="xllama.exe",
            PrivateWorkingSet=2 * 1024 * 1024 * 1024,
        )
        self.assertIsNone(MODULE.private_ws_mb(data, FAMILY))

    def test_other_package_is_ignored(self):
        data = self.payload(
            ImageName="GameBar.exe",
            PackageFullName="Other.App_publisher",
            PrivateWorkingSet=1024 * 1024,
        )
        self.assertIsNone(MODULE.private_ws_mb(data, FAMILY))

    def test_keep_running_max(self):
        self.assertEqual(MODULE.keep_running_max(None, 10), 10)
        self.assertEqual(MODULE.keep_running_max(10, 8), 10)
        self.assertEqual(MODULE.keep_running_max(10, 12), 12)
        self.assertEqual(MODULE.keep_running_max(10, None), 10)

    def test_cli_prints_nothing_without_private_ws(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "private-ws", "--package-family", FAMILY],
            input=json.dumps(
                {"Processes": [{"ImageName": "xllama.exe", "WorkingSetSize": 1}]}
            ),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")


class ScriptContractTests(unittest.TestCase):
    def test_failed_vae_download_stops_before_any_model_swap(self):
        text = (ROOT / "scripts/validate-console.sh").read_text()
        def function(name):
            start = text.index(name + "() {")
            return text[start:text.index("\n}\n", start) + 3]
        helper = function("cache_vae_asset")
        gate = function("validate_taesd")
        for mode in ("fail", "empty"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                setup = r'''
model_provisioned() { return 0; }
upload_file() { touch "$REPO_ROOT/unexpected-upload"; }
curl() {
    local target=""
    while (($#)); do
        if [[ "$1" == -o ]]; then shift; target="$1"; fi
        shift
    done
    if [[ "$MODE" == fail ]]; then printf partial > "$target"; return 22; fi
    : > "$target"
}
VAE_RESTORE_PENDING=0
rc=0
validate_taesd || rc=$?
[[ $rc != 0 && $VAE_RESTORE_PENDING == 0 && ! -e "$REPO_ROOT/unexpected-upload" ]]
'''
                env = dict(os.environ, REPO_ROOT=tmp, MODE=mode)
                result = subprocess.run(["bash", "-c", helper + gate + setup],
                                        env=env, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(list(Path(tmp).glob(".cache-validate/*.part")), [])
                self.assertFalse((Path(tmp) / ".cache-validate/taesd_vae.onnx").exists())

    def test_taesd_gate_waits_and_rejects_stale_rows(self):
        text = (ROOT / "scripts/validate-console.sh").read_text()
        self.assertIn('{"op": "wait_idle"', text)
        self.assertIn("XLLAMA_TAESD_READY_TIMEOUT_S", text)
        self.assertIn("console_taesd_result.py", text)
        self.assertIn("250 ms", text)
        self.assertIn("max_sampled_private_ws_mb", text)
        self.assertIn("PrivateWorkingSet sampled every", text)
        self.assertIn("TAESD_SAMPLER_PID", text)
        self.assertIn("kill -0 $$", text)
        self.assertIn("--max-time", text)
        self.assertNotIn("peak_mem_mb", text)
        self.assertNotIn("rows[-1] if rows else {}", text)

    def test_ok_line_requires_autopilot_marker(self):
        text = (ROOT / "scripts/validate-console.sh").read_text()
        start = text.index("validate_taesd() {")
        block = text[start : text.index("\n}\n", start) + 3]
        self.assertIn('[[ "$marker" == "ok" ]]', block)
        # The ok print is gated on the marker, not only on a fresh CSV.
        self.assertRegex(
            block,
            r'\[\[ "\$marker" == "ok" \]\] && echo "  ok: TAESD VAE stage',
        )

    def test_wait_idle_reads_atomics_only(self):
        text = (ROOT / "uwp/MainPage.cpp").read_text()
        start = text.index('a.op == "wait_idle"')
        block = text[start : text.index('} else if (a.op == "generate_image")', start)]
        self.assertNotIn("try_acquire_compute_job", block)
        self.assertIn("kIdlePolls = 3", block)

    def test_console_exit_kills_sampler(self):
        text = (ROOT / "scripts/validate-console.sh").read_text()
        start = text.index("console_exit() {")
        fn = text[start : text.index("\n}\n", start) + 3]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            work.mkdir()
            pid_path = Path(tmp) / "sampler.pid"
            env = dict(os.environ, TMPDIR_LOCAL=str(work))
            setup = (
                "set -euo pipefail\n"
                "TAESD_SAMPLER_PID=\n"
                "VAE_RESTORE_PENDING=0\n"
                "cleanup_temp_chats() { return 0; }\n"
                "release_original_recovery_dir() { :; }\n"
                + fn
                + "\nsleep 60 &\n"
                "TAESD_SAMPLER_PID=$!\n"
                f'printf "%s\\n" "$TAESD_SAMPLER_PID" >"{pid_path}"\n'
                "console_exit 0\n"
            )
            result = subprocess.run(
                ["bash", "-c", setup], env=env, capture_output=True, timeout=10
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            pid = int(pid_path.read_text().strip())
            time.sleep(0.1)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


class WdpUploadStatusTests(unittest.TestCase):
    def upload(self, payload, curl_status=0, *, source_exists=True, http_status=200):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "model.onnx"
            if source_exists:
                source.write_bytes(b"fixture")
            setup = r"""
set -euo pipefail
source "$UPLOAD_HELPER"
CURL_AUTH=(); CSRF_TOKEN=fixture
curl() { printf '%s\n%s' "$REPLY" "$HTTP_STATUS"; return "$CURL_STATUS"; }
if wdp_upload_file_request 'https://unused.invalid/file' "$SOURCE" model.onnx; then
    exit 0
else
    exit "$?"
fi
"""
            env = dict(os.environ, UPLOAD_HELPER=str(ROOT / "scripts/wdp-file-upload.sh"),
                       REPLY=payload, CURL_STATUS=str(curl_status), HTTP_STATUS=str(http_status),
                       SOURCE=str(source))
            return subprocess.run(["bash", "-c", setup], env=env, capture_output=True,
                                  text=True, timeout=10)

    def test_legacy_empty_and_explicit_success_are_accepted(self):
        for payload in ["", " \r\n", '{"Success":true}']:
            with self.subTest(payload=payload):
                result = self.upload(payload)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_http_200_failure_and_unknown_payloads_are_rejected(self):
        for payload in ['{"Success":false,"Reason":"disk full"}', '{"Success":"true"}',
                        '{"Success":1}', '{"Reason":"unknown"}', '<html>error</html>', 'true']:
            with self.subTest(payload=payload):
                result = self.upload(payload)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("upload rejected", result.stderr)

    def test_transport_or_http_error_never_becomes_empty_body_success(self):
        for status, payload in [(7, ""), (22, '{"Success":true}'), (28, "")]:
            with self.subTest(status=status):
                result = self.upload(payload, status)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("upload request failed", result.stderr)

    def test_redirect_or_other_status_never_counts_as_an_empty_success(self):
        for status in [201, 202, 204, 302]:
            with self.subTest(status=status):
                result = self.upload("", http_status=status)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("upload returned HTTP", result.stderr)

    def test_missing_local_file_fails_before_request(self):
        result = self.upload('{"Success":true}', source_exists=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("source does not exist", result.stderr)


if __name__ == "__main__":
    unittest.main()
