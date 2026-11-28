#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT

from copy import deepcopy
from hashlib import sha256
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

DRIVER = Path(__file__).resolve().parents[1] / "scripts/console_hybrid_shifts.py"
spec = importlib.util.spec_from_file_location("console_hybrid_shifts", DRIVER)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)

PLAN = {"actions": [{"op": "set_sampling"}, {"op": "send", "text": "A"},
                    {"op": "send", "text": "B"}, {"op": "send", "text": "C"}, {"op": "quit"}]}
SHIFT = "[xllama] session: context shift — evicted 498 tokens past keep=13, kv now 1006 of 2048 (rebuilt from 13) (#169/#723)"
GEN = "[xllama] session generate: n=8 prefill=3091.1ms (1077 tok) reuse=1 drafted=0 spec_accept=0"


def send(index, lines):
    return "\n".join([f"10:52:00.001 [autopilot] action {index} send start", *lines,
                      f"10:52:00.020 [autopilot] action {index} send end"])


class RepeatedShifts(unittest.TestCase):
    def test_long_conversation_preserves_original_prompts_and_fixes_sampling(self):
        actions = driver.extended_plan(deepcopy(PLAN))["actions"]
        self.assertEqual(actions[0], {"op": "set_routing", "routing": 0})
        self.assertEqual(actions[1]["op"], "set_system_prompt")
        sampling = next(a for a in actions if a["op"] == "set_sampling")
        self.assertEqual(sampling["temperature"], 0)
        self.assertEqual(sampling["n_predict"], 64)
        self.assertEqual(sum(a["op"] == "send" for a in actions), 33)
        self.assertEqual(actions[-2:], PLAN["actions"][-2:])

    def test_shift_uses_actual_survivors_and_records_intervening_sends(self):
        text = "\n".join((SHIFT, send(1, [SHIFT, GEN]), send(2, [GEN]), send(3, [SHIFT, GEN])))
        rows = driver.measure(text, PLAN)
        self.assertEqual([r["turn"] for r in rows], [1, 3])
        self.assertEqual([r["intervening_turns"] for r in rows], [None, 1])
        self.assertEqual((rows[0]["kept_tokens"], rows[0]["new_tokens"],
                          rows[0]["restored_tokens"], rows[0]["prefill_ms"]), (1006, 84, 13, 3091.1))

    def test_missing_duplicate_unordered_or_inconsistent_observations_fail(self):
        for lines in ([SHIFT], [SHIFT, GEN, GEN], [GEN, SHIFT], [SHIFT, SHIFT, GEN],
                      [SHIFT.replace("evicted 498", "evicted 2048"), GEN],
                      [SHIFT.replace("from 13", "from 99"), GEN], [SHIFT, GEN.replace("1077", "2")]):
            with self.subTest(lines=lines), self.assertRaises(ValueError):
                driver.measure("\n".join((send(1, lines), send(2, [GEN]), send(3, [GEN]))), PLAN)
        for text in (send(1, [SHIFT, GEN]), send(2, [GEN]), "[autopilot] action 1 send end"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                driver.measure(text, PLAN)

    def test_console_campaign_persists_two_model_receipts_with_synthetic_portal(self):
        source = DRIVER.with_name("validate-console.sh").read_text()
        functions = "\n".join(re.search(rf"(?ms)^{name}\(\) \{{.*?^}}$", source)[0]
                              for name in ("model_provisioned_gguf", "validate_longchat",
                                           "validate_longchat_repeat"))
        fixture = r'''
set -euo pipefail
PFN=synthetic-package
CURL_AUTH=()
BASE_URL=fixture
DEFAULT_N_CTX=2048
EST_CHARS_PER_TOK=4
curl() {
    if [[ "$*" == */api/os/info ]]; then
        echo '{"Platform":"fixture","OsVersion":"fixture","ComputerName":"private"}'
    elif [[ "$*" == *filename=settings.json ]]; then
        python3 - "$TMPDIR_LOCAL/ap.json" <<'PY'
import json, sys
ops = {a["op"]: a for a in json.load(open(sys.argv[1]))["actions"]}
print(json.dumps({"model": ops["set_model"]["name"], "routing": 0, "kv_reuse": True,
                  "system_prompt": "You are a helpful AI assistant.",
                  "sampling": {"temperature": 0, "top_p": 1, "top_k": 1, "repetition_penalty": 1, "n_predict": 64}}))
PY
    else
        printf 'synthetic checkpoint'
    fi
}
git() { echo fixture; }
register_temp_chat() { echo synthetic-chat; }
fetch_chat_index() { echo '[]' > "$1"; }
trim_ceiling() { echo 1800; }
upload_file() { [[ -s "$1" ]]; }
remove_chat() { :; }
fetch_log() { echo "$TMPDIR_LOCAL/run.log"; }
run_autopilot() {
    cat > "$TMPDIR_LOCAL/ap.json"
    python3 - "$TMPDIR_LOCAL/ap.json" > "$TMPDIR_LOCAL/run.log" <<'PY'
import json, sys
for i, action in enumerate(json.load(open(sys.argv[1]))["actions"]):
    if action["op"] == "send":
        print(f"[autopilot] action {i} send start")
        print("[xllama] session: context shift — evicted 10 tokens past keep=2, kv now 10 of 128 (rebuilt from 2) (#169/#723)")
        print("[xllama] session generate: n=1 prefill=2.5ms (18 tok) reuse=1 drafted=0 spec_accept=0")
        print(f"[autopilot] action {i} send end")
PY
    echo ok
}
'''
        models = ("lfm25-350m", "lfm25-1.2b-instruct")
        digest = sha256(b"synthetic checkpoint").hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "uwp/models/manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"models": [{"name": name, "files": [
                {"filename": name + ".gguf", "sha256": digest}]} for name in models]}))
            out = root / "campaign"
            out.mkdir()
            env = dict(os.environ, SCRIPT_DIR=str(DRIVER.parent), REPO_ROOT=tmp, TMPDIR_LOCAL=tmp,
                       LONGCHAT_OUT=str(out))
            command = ["bash", "-c", fixture + functions + "\nvalidate_longchat_repeat"]
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual(json.loads((out / "device.json").read_text()),
                             {"Platform": "fixture", "OsVersion": "fixture"})
            for model in models:
                arm = out / model
                self.assertEqual((arm / "checkpoint.sha256").read_text(), f"{digest}  {model}.gguf\n")
                self.assertEqual((arm / "marker.txt").read_text(), "ok\n")
                self.assertEqual(json.loads((arm / "settings.json").read_text())["model"], model)
                self.assertEqual(len(json.loads((arm / "shifts.json").read_text())["shifts"]), 36)
                plan = json.loads((arm / "autopilot.json").read_text())
                self.assertEqual(next(a["name"] for a in plan["actions"] if a["op"] == "set_model"), model)

            # A different GGUF must retain its observed hash and skip that model's inference.
            manifest.write_text(manifest.read_text().replace(digest, "0" * 64, 1))
            env["LONGCHAT_OUT"] = str(root / "bad")
            (root / "bad").mkdir()
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertTrue((root / "bad" / models[0] / "checkpoint.sha256").exists())
            self.assertFalse((root / "bad" / models[0] / "autopilot.json").exists())
            self.assertTrue((root / "bad" / models[1] / "shifts.json").exists())

    def test_cli_retains_single_valid_shift_and_returns_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "plan.json").write_text(json.dumps(PLAN))
            (path / "log").write_text("\n".join((send(1, [SHIFT, GEN]), send(2, [GEN]), send(3, [GEN]))))
            run = subprocess.run([sys.executable, DRIVER, path / "plan.json", "--log", path / "log"],
                                 capture_output=True, text=True)
            self.assertEqual(run.returncode, 1)
            self.assertEqual(len(json.loads(run.stdout)["shifts"]), 1)


if __name__ == "__main__":
    unittest.main()
