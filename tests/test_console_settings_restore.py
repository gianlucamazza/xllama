#!/usr/bin/env python3
"""validate-console.sh leaves the console as found (#645).

Before the first gate the script backs up the user's settings.json; the EXIT
handler restores it on every exit and relaunches the app only when it was
running before the first gate. The real shell functions run
against a stubbed Device Portal (a directory plus an event log).
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
USER_SETTINGS = '{"schema": 2, "model": "lfm25-350m", "sampling": {"n_predict": 512}}\n'


def function(name):
    text = (ROOT / "scripts/validate-console.sh").read_text()
    start = text.index(name + "() {")
    return text[start:text.index("\n}\n", start) + 3]


STUBS = r'''
set -euo pipefail
PFN=GianlucaMazza.xllama_1.10.0.0_x64__abc
BASE_URL=https://example.invalid
CURL_AUTH=()
CHAT_JOURNAL="$TMPDIR_LOCAL/chat-cleanup.pending"
SETTINGS_BACKUP="$TMPDIR_LOCAL/settings.user.json"
SETTINGS_USER_ABSENT=0
SETTINGS_RESTORE_PENDING=0
VAE_RESTORE_PENDING=0
APP_WAS_RUNNING=0
log() { printf '%s\n' "$*" >>"$EVENTS"; }
stop_app() { log stop; [[ "${STOP_FAIL:-0}" == 0 ]]; }
start_app() { log start; }
file_absent() { [[ ! -e "$REMOTE/$1" ]]; }
fetch_file() {
    if [[ -e "$REMOTE/$1" ]]; then cp "$REMOTE/$1" "$2"; else printf '{"Code":404}' >"$2"; fi
}
upload_file() {
    log "upload $1|$2|$3"
    [[ "${UPLOAD_FAIL:-0}" == 0 ]] || return 1
    cp "$1" "$REMOTE/${3:-$(basename "$1")}"
}
delete_file() {
    log "delete $1"
    [[ "${DELETE_LIE:-0}" == 1 ]] || rm -f "$REMOTE/$1"
}
curl() { cat "$PROCESSES"; }
cleanup_temp_chats() { return 0; }
release_original_recovery_dir() { :; }
'''

FUNCS = ["app_running", "record_app_state", "backup_user_settings",
         "restore_user_settings", "console_exit"]


class SettingsRestoreTests(unittest.TestCase):
    def run_shell(self, action, remote_settings=USER_SETTINGS, processes=None, **flags):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            work, remote = base / "work", base / "remote"
            work.mkdir()
            remote.mkdir()
            if remote_settings is not None:
                (remote / "settings.json").write_text(remote_settings)
            procs = base / "processes.json"
            procs.write_text(processes if processes is not None else json.dumps({"Processes": []}))
            events = base / "events"
            env = dict(os.environ, TMPDIR_LOCAL=str(work), REMOTE=str(remote),
                       EVENTS=str(events), PROCESSES=str(procs),
                       **{k: str(v) for k, v in flags.items()})
            script = STUBS + "".join(function(f) for f in FUNCS)
            script += "\ntrap 'console_exit \"$?\"' EXIT\n" + action
            result = subprocess.run(["bash", "-c", script], env=env,
                                    capture_output=True, text=True, timeout=20)
            on_console = (remote / "settings.json").read_text() \
                if (remote / "settings.json").exists() else None
            log = events.read_text().splitlines() if events.exists() else []
            return result, on_console, log, work.exists()

    # A gate run: back up, seed the gate's own file, then restore.
    GATE = r'''
backup_user_settings
printf '{"model":"gate-seed"}' >"$REMOTE/settings.json"
restore_user_settings
[[ "$SETTINGS_RESTORE_PENDING" == 0 ]]
'''

    def test_present_backup_is_uploaded_back(self):
        result, on_console, log, work_exists = self.run_shell(self.GATE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(on_console, USER_SETTINGS)
        uploads = [e for e in log if e.startswith("upload ")]
        self.assertEqual(len(uploads), 1)
        self.assertTrue(uploads[0].endswith("settings.user.json||settings.json"), uploads[0])
        self.assertNotIn("delete settings.json", log)
        self.assertFalse(work_exists)

    def test_absent_settings_is_deleted_not_uploaded(self):
        result, on_console, log, _ = self.run_shell(self.GATE, remote_settings=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIsNone(on_console)
        self.assertIn("delete settings.json", log)
        self.assertFalse(any(e.startswith("upload ") for e in log))

    def test_restore_waits_for_a_verified_stop(self):
        action = 'SETTINGS_RESTORE_PENDING=1\nrestore_user_settings && exit 99\nexit 0'
        result, on_console, log, work_exists = self.run_shell(action, STOP_FAIL=1)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(any(e.startswith(("upload ", "delete ")) for e in log))
        self.assertTrue(work_exists)
        self.assertIn("settings.json restore still pending", result.stderr)

    def test_unusable_backup_fails_closed(self):
        for body in ['portal error', '{"Code": 500, "Reason": "x"}', '[]']:
            with self.subTest(body=body):
                action = 'backup_user_settings && exit 99\n[[ "$SETTINGS_RESTORE_PENDING" == 0 ]]'
                result, on_console, log, _ = self.run_shell(action, remote_settings=body)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(on_console, body)
                self.assertFalse(any(e.startswith(("upload ", "delete ")) for e in log))

    def test_interrupted_gate_is_restored_by_exit_handler(self):
        action = r'''
backup_user_settings
printf '{"model":"gate-seed"}' >"$REMOTE/settings.json"
kill -TERM $$
'''
        script_trap = "trap 'exit 143' TERM\n" + action
        result, on_console, _, _ = self.run_shell(script_trap)
        self.assertEqual(result.returncode, 143, result.stderr)
        self.assertEqual(on_console, USER_SETTINGS)

    def test_failed_upload_keeps_backup_and_fails(self):
        action = 'backup_user_settings\nprintf "{}" >"$REMOTE/settings.json"\nexit 0'
        result, _, _, work_exists = self.run_shell(action, UPLOAD_FAIL=1)
        self.assertEqual(result.returncode, 1)
        self.assertTrue(work_exists)
        self.assertIn("settings.user.json as LocalState", result.stderr)

    def test_delete_that_did_not_take_stays_pending(self):
        action = 'backup_user_settings\nprintf "{}" >"$REMOTE/settings.json"\nexit 0'
        result, _, _, work_exists = self.run_shell(action, remote_settings=None, DELETE_LIE=1)
        self.assertEqual(result.returncode, 1)
        self.assertTrue(work_exists)
        self.assertIn("the user had none", result.stderr)

    def test_relaunch_only_when_app_was_running(self):
        pfn = "GianlucaMazza.xllama_1.10.0.0_x64__abc"
        cases = [
            (json.dumps({"Processes": [{"ImageName": "xllama.exe", "PackageFullName": pfn}]}), True),
            (json.dumps({"Processes": [{"ImageName": "other.exe", "PackageFullName": "x"}]}), False),
            ("not json", False),
        ]
        for processes, relaunched in cases:
            with self.subTest(processes=processes):
                result, _, log, _ = self.run_shell("record_app_state\nexit 0", processes=processes)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual("start" in log, relaunched)
                if relaunched:
                    # The flag goes before the launch, so no gate is replayed.
                    self.assertLess(log.index("delete autopilot.flag"), log.index("start"))

    def test_relaunch_follows_the_settings_restore(self):
        pfn = "GianlucaMazza.xllama_1.10.0.0_x64__abc"
        processes = json.dumps({"Processes": [{"PackageFullName": pfn}]})
        action = 'record_app_state\nbackup_user_settings\nexit 1'
        result, on_console, log, _ = self.run_shell(action, processes=processes)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(on_console, USER_SETTINGS)
        upload = next(i for i, e in enumerate(log) if e.startswith("upload "))
        self.assertLess(upload, log.index("start"))


if __name__ == "__main__":
    unittest.main()
