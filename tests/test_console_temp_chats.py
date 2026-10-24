#!/usr/bin/env python3
"""Exercise the real fixture cleanup with a local, failure-injecting WDP double."""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CID = "ap-test-owned"
USERS = [{"id": "user-one", "title": "Keep me"}, {"id": "user-two", "title": "Conservar á"}]
INDEX = [USERS[0], {"id": CID, "title": "Fixture"}, USERS[1]]

MOCK = r'''
import json, os, sys
from pathlib import Path
root = Path(os.environ['REMOTE'])
mode = os.environ.get('FAIL_MODE', '')
method, name, out, *upload = sys.argv[1:]
events = root / 'events.jsonl'
with events.open('a') as f:
    f.write(json.dumps([method, name]) + '\n')
prior = [json.loads(l) for l in events.read_text().splitlines()]
if mode == 'transport':
    raise SystemExit(7)
if (mode == 'http' and method == 'GET') or (mode == 'delete500' and method == 'DELETE') or (mode == 'post500' and method == 'POST'):
    Path(out).write_text('portal error')
    print('500', end='')
    raise SystemExit(0)
p = root / name
if method == 'GET':
    if mode == 'concurrent' and name == 'index.json' and prior.count(['GET', name]) == 2:
        data = json.loads(p.read_text()); data.append({'id': 'concurrent-user'})
        p.write_text(json.dumps(data))
    Path(out).write_bytes(p.read_bytes() if p.exists() else b'not found')
    print('200' if p.exists() else '404', end='')
elif method == 'DELETE':
    existed = p.exists()
    if existed and mode != 'delete-lie':
        p.unlink()
    Path(out).write_text('')
    print('200' if existed else '404', end='')
elif method == 'POST':
    if mode != 'post-lie':
        p.write_bytes(Path(upload[0]).read_bytes())
    Path(out).write_text('')
    print('200', end='')
'''


def function(name):
    text = (ROOT / 'scripts/validate-console.sh').read_text()
    start = text.index(name + '() {')
    return text[start:text.index('\n}\n', start) + 3]


class CleanupTests(unittest.TestCase):
    def run_cleanup(self, mode='', index=INDEX, action='cleanup_temp_chats',
                    stop_failure=False, with_exit=False, optimize=None):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            work, remote = base / 'work', base / 'remote'
            work.mkdir(); remote.mkdir()
            if index is not None:
                (remote / 'index.json').write_text(json.dumps(index) if not isinstance(index, str) else index)
            (remote / f'{CID}.json').write_text('fixture')
            (remote / 'user-one.json').write_text('original one')
            (remote / 'user-two.json').write_text('original two')
            (work / 'chat-cleanup.pending').write_text(CID + '\n')
            mock = base / 'mock.py'; mock.write_text(MOCK)
            env = dict(os.environ, TMPDIR_LOCAL=str(work), REMOTE=str(remote), MOCK=str(mock),
                       SCRIPT_DIR=str(ROOT / 'scripts'), FAIL_MODE=mode, STOP_FAIL=str(int(stop_failure)))
            # Avoid inheriting an operator's recovery-journal override in tests.
            env.pop('XLLAMA_CHAT_JOURNAL', None)
            if optimize is not None:
                env['PYTHONOPTIMIZE'] = str(optimize)
            setup = r'''
set -euo pipefail
source "$SCRIPT_DIR/console-temp-chats.sh"
stop_app() {
    printf 'stop\n' >>"$REMOTE/stops"
    [[ "$STOP_FAIL" == 0 ]]
}
_chat_request() { python3 "$MOCK" "$@"; }
'''
            if with_exit:
                setup += '\nVAE_RESTORE_PENDING=0\n' + function('console_exit')
                setup += '\ntrap \'console_exit "$?"\' EXIT\ntrap \'exit 143\' TERM\n'
            result = subprocess.run(['bash', '-c', setup + '\n' + action], env=env,
                                    capture_output=True, text=True, timeout=25)
            files = {p.name: p.read_bytes() for p in remote.iterdir()}
            journal = work / 'chat-cleanup.pending'
            pending = journal.read_text() if journal.exists() else ''
            events = [json.loads(l) for l in files.get('events.jsonl', b'').splitlines()]
            return result, files, pending, events, work.exists()

    def assert_user_files(self, files):
        self.assertEqual(files['user-one.json'], b'original one')
        self.assertEqual(files['user-two.json'], b'original two')

    def test_normal_cleanup_preserves_user_chats_and_index(self):
        result, files, pending, events, _ = self.run_cleanup()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotIn(CID + '.json', files)
        self.assertEqual(json.loads(files['index.json']), USERS)
        self.assertEqual(pending, '')
        self.assert_user_files(files)
        self.assertEqual(events[0], ['GET', 'index.json'])

    def test_early_gate_failure_still_cleans_and_preserves_failure(self):
        action = r'''
SHOTS_DIR="$TMPDIR_LOCAL/shots"
GATES_NO_MIDRUN_SHOTS=""
ring_reset() { :; }
save_fail_shots() { :; }
failed_gate() { return 7; }
''' + function('run_gate') + '\nrun_gate probe failed_gate'
        result, files, pending, _, _ = self.run_cleanup(action=action)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn(CID + '.json', files)
        self.assertEqual(pending, '')

    def test_exit_and_signal_cleanup_preserve_original_status(self):
        for action, status in [('exit 0', 0), ('exit 42', 42), ('kill -TERM $$', 143)]:
            with self.subTest(action=action):
                result, files, pending, _, exists = self.run_cleanup(action=action, with_exit=True)
                self.assertEqual(result.returncode, status, result.stderr)
                self.assertNotIn(CID + '.json', files)
                self.assertEqual(pending, '')
                self.assertFalse(exists)
                self.assert_user_files(files)

    def test_http_transport_and_write_failures_remain_pending(self):
        for mode in ['http', 'transport', 'delete500', 'post500', 'delete-lie', 'post-lie']:
            with self.subTest(mode=mode):
                result, files, pending, _, exists = self.run_cleanup(mode=mode)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(CID, pending)
                self.assertTrue(exists)
                self.assert_user_files(files)

    def test_bad_index_never_becomes_empty_index(self):
        for index in ['not JSON', {'Error': 'denied'}, [1, 2]]:
            with self.subTest(index=index):
                result, files, pending, events, _ = self.run_cleanup(index=index)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(CID, pending)
                self.assertFalse(any(e[0] in ('POST', 'DELETE') for e in events))
                self.assertIn(CID + '.json', files)
                self.assert_user_files(files)

    def test_intervening_index_write_is_not_overwritten(self):
        result, files, pending, events, _ = self.run_cleanup(mode='concurrent')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(CID, pending)
        self.assertFalse(any(e[0] == 'POST' for e in events))
        self.assertEqual(json.loads(files['index.json'])[-1], {'id': 'concurrent-user'})

    def test_stop_failure_makes_no_portal_writes(self):
        result, files, pending, events, _ = self.run_cleanup(stop_failure=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(CID, pending)
        self.assertEqual(events, [])
        self.assertIn(CID + '.json', files)

    def test_failed_exit_retains_recovery_files_and_fails_gate(self):
        result, _, pending, _, exists = self.run_cleanup(mode='http', action='exit 0', with_exit=True)
        self.assertEqual(result.returncode, 1)
        self.assertIn(CID, pending)
        self.assertTrue(exists)
        self.assertIn('recovery files retained', result.stderr)

    def test_transient_failure_can_be_retried(self):
        action = 'cleanup_temp_chats && exit 99\nexport FAIL_MODE=""\ncleanup_temp_chats'
        result, files, pending, _, _ = self.run_cleanup(mode='http', action=action)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(pending, '')
        self.assertNotIn(CID + '.json', files)

    def test_unregistered_chat_is_never_deleted(self):
        result, files, pending, events, _ = self.run_cleanup(action='remove_chat user-one')
        self.assertEqual(result.returncode, 0)
        self.assertEqual(events, [])
        self.assertIn(CID, pending)
        self.assert_user_files(files)

    def test_missing_index_does_not_create_an_empty_one(self):
        result, files, pending, events, _ = self.run_cleanup(index=None)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(pending, '')
        self.assertNotIn('index.json', files)
        self.assertFalse(any(e[0] == 'POST' for e in events))

    def test_registration_is_unique_and_persisted_from_subshell(self):
        action = r'''
a=$(register_temp_chat ap-probe)
b=$(register_temp_chat ap-probe)
[[ "$a" != "$b" ]]
grep -qxF "$a" "$CHAT_JOURNAL"
grep -qxF "$b" "$CHAT_JOURNAL"
'''
        result, _, pending, events, _ = self.run_cleanup(action=action)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(pending.splitlines()), 3)
        self.assertEqual(events, [])

    def test_fixture_index_read_rejects_errors_before_upload(self):
        for mode, index in [('http', INDEX), ('transport', INDEX), ('', 'not JSON'),
                            ('', {'Error': 'denied'}), ('', [1])]:
            with self.subTest(mode=mode, index=index):
                result, files, _, events, _ = self.run_cleanup(
                    mode=mode, index=index,
                    action='fetch_chat_index "$TMPDIR_LOCAL/injection-index.json"')
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(any(e[0] in ('POST', 'DELETE') for e in events))
                self.assert_user_files(files)

    def test_runtime_checks_survive_optimized_python(self):
        for optimize in (1, 2):
            with self.subTest(optimize=optimize, mode='post-lie'):
                result, _, pending, _, _ = self.run_cleanup(mode='post-lie', optimize=optimize)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(CID, pending)
            for index in ({'Error': 'denied'}, [1]):
                with self.subTest(optimize=optimize, index=index):
                    result, _, _, events, _ = self.run_cleanup(
                        index=index, optimize=optimize,
                        action='fetch_chat_index "$TMPDIR_LOCAL/injection-index.json"')
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(any(e[0] in ('POST', 'DELETE') for e in events))

    def test_missing_fixture_index_is_known_empty_not_error_body(self):
        result, _, _, events, _ = self.run_cleanup(
            index=None,
            action='fetch_chat_index "$TMPDIR_LOCAL/injection-index.json"\ncat "$TMPDIR_LOCAL/injection-index.json"')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, '[]')
        self.assertEqual(events, [['GET', 'index.json']])

    def test_registration_does_not_run_parent_exit_cleanup_in_subshell(self):
        action = r'''
a=$(register_temp_chat ap-probe)
grep -qxF "$a" "$CHAT_JOURNAL"
[[ ! -f "$REMOTE/events.jsonl" ]]
exit 0
'''
        result, files, pending, _, exists = self.run_cleanup(action=action, with_exit=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(pending, '')
        self.assertFalse(exists)
        self.assertNotIn(CID + '.json', files)

    def test_common_exit_keeps_vae_restore_and_cleanup(self):
        action = r'''
VAE_RESTORE_PENDING=1
VAE_CACHE="$TMPDIR_LOCAL/vae-cache"
upload_file() { printf '%s\n' "$*" >"$REMOTE/vae-restored"; }
exit 0
'''
        result, files, pending, _, _ = self.run_cleanup(action=action, with_exit=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(pending, '')
        self.assertIn(b'full_vae.onnx', files['vae-restored'])
        self.assertNotIn(CID + '.json', files)


class TrainingServeCleanupTests(unittest.TestCase):
    def run_serve(self, stop_fail=False, autopilot_fail=False, skip_upload=False,
                  log_fail=False):
        """Run the actual serve gate and cleanup against local device fixtures."""
        text = (ROOT / 'scripts/validate-console-training.sh').read_text()
        functions = []
        for name in ('serve_cleanup', 'validate_serve'):
            start = text.index(name + '() {')
            functions.append(text[start:text.index('\n}\n', start) + 3])
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            remote = base / 'remote'
            remote.mkdir()
            (base / 'merged.gguf').write_text('test model')
            (remote / 'manifest.json').write_text('{"models": ["user-model"]}')
            env = dict(os.environ, TMPDIR_LOCAL=tmp, REMOTE=str(remote),
                       STOP_FAIL=str(int(stop_fail)), AP_FAIL=str(int(autopilot_fail)),
                       SKIP_UPLOAD=str(int(skip_upload)), LOG_FAIL=str(int(log_fail)))
            setup = r'''
set -euo pipefail
MERGED="$TMPDIR_LOCAL/merged.gguf"
REPO_ROOT="$TMPDIR_LOCAL"
DEPLOY=true
PFN=test-package
PINNED_SETTINGS=''
prepare_merged_q4() { :; }
stop_app() {
    if [[ "$STOP_FAIL" == 1 ]]; then
        echo 'original stop error' >&2
        return 17
    fi
}
upload_file() {
    if [[ "${3:-}" == manifest.json && "$1" != *manifest-saved.json ]]; then
        printf 'fixture override' >"$REMOTE/manifest.json"
    else
        cp "$1" "$REMOTE/${3:-$(basename "$1")}"
    fi
}
delete_file() { rm -f "$REMOTE/$1"; }
fetch_file() { cp "$REMOTE/$1" "$2"; }
list_chat_ids() { [[ ! -f "$REMOTE/current.json" ]] || echo current; }
run_autopilot() {
    cat >/dev/null
    printf '{"messages":[{"content":"xllama secret XLLAMA-LORA-OK"}]}' >"$REMOTE/current.json"
    printf 'GGUF model loaded via llama.cpp\nsession generate: n=1\n' >"$TMPDIR_LOCAL/current.log"
    [[ "$AP_FAIL" == 0 ]] && echo ok || echo 'error: inference failed'
}
fetch_log() {
    [[ "$LOG_FAIL" == 0 ]] || { echo 'original log error' >&2; return 9; }
    echo "$TMPDIR_LOCAL/current.log"
}
'''
            # Keep the same conditional call as the CLI: errexit is disabled
            # within validate_serve, so cleanup must explicitly affect verdict.
            result = subprocess.run(
                ['bash', '-c', setup + '\n' + '\n'.join(functions) +
                 '\nrc=0\nvalidate_serve || rc=1\nexit "$rc"'],
                env=env, capture_output=True, text=True, timeout=25)
            files = {p.name: p.read_bytes() for p in remote.iterdir()}
            return result, files

    def test_success_requires_cleanup(self):
        result, files = self.run_serve()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Training serve: PASS', result.stdout)
        self.assertNotIn('model.gguf', files)
        self.assertNotIn('.complete', files)
        self.assertEqual(files['manifest.json'], b'{"models": ["user-model"]}')

    def test_failed_cleanup_rejects_otherwise_valid_serve(self):
        result, files = self.run_serve(stop_fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn('Training serve: PASS', result.stdout)
        self.assertIn('Training serve: FAIL', result.stdout)
        self.assertIn('original stop error', result.stderr)
        for name in ('model.gguf', '.complete', 'manifest.json'):
            self.assertIn(name, result.stderr)
            self.assertIn(name, files)
        self.assertEqual(files['manifest.json'], b'fixture override')

    def test_original_gate_failure_is_not_hidden_by_cleanup(self):
        for stop_fail in (False, True):
            with self.subTest(stop_fail=stop_fail):
                result, _ = self.run_serve(stop_fail=stop_fail, autopilot_fail=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('error: inference failed', result.stdout)
                self.assertIn('FAIL: autopilot not ok', result.stdout)
                self.assertNotIn('Training serve: PASS', result.stdout)
                if stop_fail:
                    self.assertIn('original stop error', result.stderr)

    def test_log_failure_remains_failure_even_if_cleanup_fails(self):
        result, _ = self.run_serve(stop_fail=True, log_fail=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('original log error', result.stderr)
        self.assertIn('original stop error', result.stderr)
        self.assertNotIn('Training serve: PASS', result.stdout)

    def test_skip_upload_does_not_clean_existing_resources(self):
        result, files = self.run_serve(stop_fail=True, skip_upload=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('Training serve: PASS', result.stdout)
        self.assertNotIn('original stop error', result.stderr)
        self.assertEqual(files['manifest.json'], b'{"models": ["user-model"]}')


if __name__ == '__main__':
    unittest.main()
