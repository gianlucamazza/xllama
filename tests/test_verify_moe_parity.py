# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Exercise the MoE verifier with synthetic dumps and the real comparator."""

import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify-moe-parity.py"
STEPS = (0, 1, 2, 3, 5, 9)
CASES = ("greeting-ru", "arithmetic-it", "code-en", "prose-it")
CANDIDATE = "a" * 40


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data), encoding="utf-8")


class VerifyMoeParityTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.declaration = self.root / "declaration.json"
        self.receipt = self.root / "receipt.json"
        row = struct.pack("<12f", *range(1, 13))
        ids = [1, 5]
        continuation = [11, 9, 8, 6, 3, 7, 10, 2, 4]
        identity = {
            "checkpoint_sha256": "1" * 64,
            "template_sha256": "2" * 64,
            "tokenizer_sha256": "3" * 64,
            "vocab_sha256": "4" * 64,
        }
        template = b"fixture chat template\n"
        identity["template_sha256"] = digest(template)
        (self.root / "reference").mkdir()
        (self.root / "reference" / "chat-template.jinja").write_bytes(template)
        build = {
            "platform": "Linux x86-64", "cpu": "fixture", "compiler": "fixture",
            "cmake": {"GGML_NATIVE": "OFF"},
            "environment": {"GGML_CPU_DISABLE_FUSION": "1"},
        }
        execution = {
            "threads": 1, "context": 512, "n_batch": 512, "n_ubatch": 512,
            "prompt_tokens_max": 128, "reference_rows_after_feeding": list(STEPS),
            "cache_slots": [2, 512],
        }
        cases = []
        for case in CASES:
            prompt = f"Rendered {case}"
            cases.append({
                "id": case, "vocab_size": 12, "bos_token_id": 1, "prompt_token_ids": ids,
                "teacher_forced_token_ids": continuation,
                "rendered_prompt_sha256": digest(prompt.encode()),
                "resident_row_sha256_by_fed_tokens": {str(step): digest(row) for step in STEPS},
            })
            for arm in ("reference", "2", "512"):
                directory = self.root / arm / case
                directory.mkdir(parents=True)
                if arm == "reference":
                    (directory / "rendered.txt").write_bytes(prompt.encode())
                    write_json(directory / "tokens.json", {
                        "vocab_size": 12, "bos_token_id": 1, "prompt_token_ids": ids,
                        "teacher_forced_token_ids": continuation,
                    })
                for step in STEPS:
                    path = directory / f"row-{step}.bin"
                    path.write_bytes(row)
                    if arm == "reference":
                        continue
                    meta = {
                        **identity, "prompt": prompt, "vocab_size": 12,
                        "prompt_token_ids": ids, "fed_continuation_token_ids": continuation[:step],
                    }
                    meta.update(
                        candidate_commit=CANDIDATE, cache_slots=int(arm),
                        build_sha256=digest(arm.encode()), row_sha256=digest(row),
                    )
                    write_json(path.with_suffix(".bin.json"), meta)
        write_json(self.declaration, {
            "schema": "xllama-moe-parity-preparation-v1",
            "source": {"upstream_candidate_commit": None},
            "checkpoint": {"sha256": identity.pop("checkpoint_sha256")},
            **identity, "execution": execution, "resident_reference": build,
            "comparison": {
                "max_abs_diff": 0, "nmse": 0, "top_k": 10, "top_k_min": 1,
                "same_top_token": True, "reject_nonfinite": True,
            },
            "cases": cases,
        })
        write_json(self.receipt, {
            "schema": "xllama-moe-parity-receipt-v1",
            "candidate_commit": CANDIDATE, "declaration_sha256": digest(self.declaration.read_bytes()),
            "cpu_kernel_review": "Fixture review",
            "cpu_settings": {
                **build, **{key: execution[key] for key in ("threads", "context", "n_batch", "n_ubatch")},
                "repacking": False, "flash_attention": False,
            },
            "streamed_build_sha256": {arm: digest(arm.encode()) for arm in ("2", "512")},
        })

    def run_verifier(self, *, reference_only=False):
        args = ["--reference-only"] if reference_only else [
            "--streamed-2", str(self.root / "2"),
            "--streamed-512", str(self.root / "512"), "--receipt", str(self.receipt),
        ]
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(self.declaration), str(self.root / "reference"), *args],
            capture_output=True, text=True, check=False,
        )
        self.assertTrue(result.stdout, result.stderr)
        return result.returncode, json.loads(result.stdout)

    def row(self, arm="512", step=9):
        return self.root / arm / CASES[-1] / f"row-{step}.bin"

    def edit_sidecar(self, path, **fields):
        sidecar = path.with_suffix(".bin.json")
        meta = json.loads(sidecar.read_text())
        write_json(sidecar, {**meta, **fields})

    def assert_invalid(self, **kwargs):
        code, report = self.run_verifier(**kwargs)
        self.assertEqual(code, 2, report)
        self.assertEqual(report["status"], "INVALID_INPUT")
        self.assertEqual(report["pairs"], [])

    def test_reference_only(self):
        code, report = self.run_verifier(reference_only=True)
        self.assertEqual((code, report["status"], report["resident_rows"]), (0, "REFERENCE_VERIFIED", 24))

    def test_both_caches_and_all_positions(self):
        code, report = self.run_verifier()
        self.assertEqual((code, report["status"]), (0, "PASS"))
        self.assertEqual(report["receipt_sha256"], digest(self.receipt.read_bytes()))
        self.assertEqual(report["streamed_build_sha256"], {arm: digest(arm.encode()) for arm in ("2", "512")})
        self.assertEqual(
            {(r["case"], r["step"], r["cache_slots"]) for r in report["pairs"]},
            {(case, step, cache) for case in CASES for step in STEPS for cache in (2, 512)},
        )

    def test_numeric_comparison_uses_only_validated_logits(self):
        write_json(self.row(arm="reference").with_suffix(".bin.json"), {"top1_piece": "resident annotation"})
        self.edit_sidecar(self.row(), top1_piece="streamed annotation")
        code, report = self.run_verifier()
        self.assertEqual((code, report["status"]), (0, "PASS"))

        self.assertEqual(
            {key: report["pairs"][0][key] for key in ("max_abs_diff", "nmse", "top1_match", "top_k_overlap")},
            {"max_abs_diff": 0.0, "nmse": 0.0, "top1_match": True, "top_k_overlap": 1.0},
        )
        self.assertNotIn("comparator_output", report["pairs"][0])

    def test_zero_reference_mismatch_has_null_nmse(self):
        declaration = json.loads(self.declaration.read_text())
        raw = struct.pack("<12f", *([0.] * 12))
        self.row(arm="reference", step=0).write_bytes(raw)
        declaration["cases"][-1]["resident_row_sha256_by_fed_tokens"]["0"] = digest(raw)
        write_json(self.declaration, declaration)
        receipt = json.loads(self.receipt.read_text())
        receipt["declaration_sha256"] = digest(self.declaration.read_bytes())
        write_json(self.receipt, receipt)
        stream = self.row(arm="2", step=0)
        stream.write_bytes(raw)
        self.edit_sidecar(stream, row_sha256=digest(raw))
        code, report = self.run_verifier()
        self.assertEqual((code, report["status"], len(report["pairs"])), (1, "FAIL", 48))
        failed, = [pair for pair in report["pairs"] if pair["status"] == "FAIL"]
        self.assertIsNone(failed["nmse"])
        self.assertEqual((failed["case"], failed["step"], failed["cache_slots"]), (CASES[-1], 0, 512))

    def test_numerical_mismatch_still_evaluates_all_pairs(self):
        path = self.row(arm="2", step=0)
        data = struct.pack("<12f", *range(2, 14))
        path.write_bytes(data)
        self.edit_sidecar(path, row_sha256=digest(data))
        code, report = self.run_verifier()
        self.assertEqual((code, report["status"], len(report["pairs"])), (1, "FAIL", 48))
        self.assertEqual(
            [(r["case"], r["step"], r["cache_slots"]) for r in report["pairs"] if r["status"] == "FAIL"],
            [(CASES[-1], 0, 2)],
        )

    def test_row_errors_from_all_arms_before_comparison(self):
        self.row(arm="reference", step=1).unlink()
        self.row(arm="2", step=2).with_suffix(".bin.json").unlink()
        self.row().write_bytes(struct.pack("<12f", float("nan"), *range(11)))
        code, report = self.run_verifier()
        self.assertEqual((code, report["status"], report["pairs"]), (2, "INVALID_INPUT", []))
        self.assertEqual(report["resident_rows"], 23)
        self.assertEqual(
            [(e["arm"], e["case"], e["step"], e["status"]) for e in report["row_errors"]],
            [("resident", CASES[-1], 1, "MISSING"), ("streamed-2", CASES[-1], 2, "MISSING"),
             ("streamed-512", CASES[-1], 9, "INVALID")],
        )
        self.assertIn("nonfinite logits", report["row_errors"][-1]["error"])
        self.assert_invalid(reference_only=True)

    def test_truncated_and_nonfinite_rows_reject_campaign(self):
        for data in (b"bad", struct.pack("<12f", float("nan"), *range(11))):
            with self.subTest(data=data):
                self.row().write_bytes(data)
                self.edit_sidecar(self.row(), row_sha256=digest(data))
                self.assert_invalid()

    def test_wrong_sidecar_provenance_rejects_campaign(self):
        path = self.row()
        original = path.with_suffix(".bin.json").read_bytes()
        for fields in (
            {"candidate_commit": "b" * 40}, {"checkpoint_sha256": "0" * 64},
            {"fed_continuation_token_ids": [9, 11, 8, 6, 3, 7, 10, 2, 4]}, {"prompt_token_ids": [1, 6]},
            {"cache_slots": 2}, {"build_sha256": "0" * 64}, {"row_sha256": "0" * 64},
        ):
            with self.subTest(fields=fields):
                path.with_suffix(".bin.json").write_bytes(original)
                self.edit_sidecar(path, **fields)
                self.assert_invalid()

    def test_missing_sidecar_rejects_campaign(self):
        self.row().with_suffix(".bin.json").unlink()
        self.assert_invalid()

    def test_wrong_resident_template_or_rendered_prompt_rejects_reference(self):
        reference = self.root / "reference"
        for path in (reference / "chat-template.jinja", reference / CASES[-1] / "rendered.txt"):
            with self.subTest(path=path):
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                self.assert_invalid(reference_only=True)
                path.write_bytes(original)

    def test_wrong_resident_token_metadata_rejects_reference(self):
        path = self.root / "reference" / CASES[-1] / "tokens.json"
        original = json.loads(path.read_text())
        for fields in (
            {"bos_token_id": True}, {"prompt_token_ids": [1, 6]}, {"vocab_size": 13},
            {"teacher_forced_token_ids": [11, 9, 8, 6, 3, 7, 10, 2, 5]},
        ):
            with self.subTest(fields=fields):
                write_json(path, {**original, **fields})
                self.assert_invalid(reference_only=True)

    def test_wrong_resident_hash_rejects_reference(self):
        path = self.row(arm="reference")
        path.write_bytes(struct.pack("<12f", *range(2, 14)))
        self.assert_invalid(reference_only=True)

    def test_receipt_for_another_declaration_rejects_campaign(self):
        receipt = json.loads(self.receipt.read_text())
        receipt["declaration_sha256"] = "0" * 64
        write_json(self.receipt, receipt)
        self.assert_invalid()

    def test_unmatched_kernel_settings_reject_campaign(self):
        receipt = json.loads(self.receipt.read_text())
        receipt["cpu_settings"]["repacking"] = True
        write_json(self.receipt, receipt)
        self.assert_invalid()


if __name__ == "__main__":
    unittest.main()
