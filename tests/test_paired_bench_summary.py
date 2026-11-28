# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Preregistered paired-campaign verdict: margins, interval rule, fail-closed inputs."""
import csv
import json
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/paired-bench-summary.py"
FIELDS = ["model", "quant", "backend", "n_ctx", "n_threads", "max_length", "n_gen_tok", "run_index", "host",
          "n_prompt_tok", "prompt_tok_s", "decode_tok_s", "ttft_ms", "peak_ws_mb"]


def write_cell(root: Path, a_rows, b_rows, statuses=None):
    root.mkdir(parents=True, exist_ok=True)
    for name, rows in (("A", a_rows), ("B", b_rows)):
        with (root / f"{name}.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    lines = []
    for i in range(1, max(len(a_rows), len(b_rows)) + 1):
        order = "AB" if i % 2 else "BA"
        for arm in order:
            st = (statuses or {}).get((i, arm), "ok")
            lines.append(json.dumps({"pair": i, "order": order, "arm": arm, "status": st}))
    (root / "ledger.jsonl").write_text("\n".join(lines) + "\n")
    return root


def rows(n, prefill, decode, ttft, peak, seed, noise=0.01):
    rng = np.random.default_rng(seed)
    return [{"model": "lfm25-1.2b-instruct", "quant": "Q4_0", "backend": "d3d12",
             "n_ctx": "2048", "n_threads": "6", "max_length": "0",
             "n_prompt_tok": "298", "n_gen_tok": "128", "run_index": "2",
             "host": "xbox-series-s-declared-t6-g99-noeog",
             "prompt_tok_s": f"{prefill * (1 + noise * rng.standard_normal()):.4f}",
             "decode_tok_s": f"{decode * (1 + noise * rng.standard_normal()):.4f}",
             "ttft_ms": f"{ttft * (1 + noise * rng.standard_normal()):.4f}",
             "peak_ws_mb": f"{peak + rng.integers(-2, 3)}"} for _ in range(n)]


def bias_rows(n, prefill, decode, ttft, peak, seed, *, control=False):
    sample = rows(n, prefill, decode, ttft, peak, seed, noise=0)
    for row in sample:
        row.update(model="lfm25-1.2b-instruct" if control else "qwen25-coder-3b",
                   quant="Q4_0" if control else "Q4_K_M", backend="d3d12",
                   n_ctx="2048", n_gen_tok="128", run_index="2",
                   host="xbox-series-s-declared-u512-g99-noeog-greedy")
    return sample


class PairedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_cell(self, cell, gates="prefill,decode,ttft,peak", profile=None, report=False):
        out = self.root / "v.json"
        out.unlink(missing_ok=True)
        mode = ["--profile", profile] if profile else ["--report" if report else "--gates", gates]
        r = subprocess.run([sys.executable, str(SCRIPT), str(cell), *mode,
                            "--json-out", str(out)], capture_output=True, text=True, check=False)
        return r, (json.loads(out.read_text()) if out.exists() else None)

    def test_a_clear_speedup_with_no_cost_passes(self):
        cell = write_cell(self.root / "c", rows(20, 100, 50, 1000, 800, 1),
                          rows(20, 130, 50, 770, 805, 2))
        r, v = self.run_cell(cell)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(v["verdict"], "pass")
        self.assertAlmostEqual(v["gates"]["prefill"]["median"], 1.3, delta=0.03)

    def test_no_speedup_fails_the_prefill_margin(self):
        cell = write_cell(self.root / "c", rows(20, 100, 50, 1000, 800, 1),
                          rows(20, 100, 50, 1000, 800, 2))
        r, v = self.run_cell(cell)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertEqual(v["gates"]["prefill"]["verdict"], "fail")
        self.assertEqual(v["gates"]["decode"]["verdict"], "pass")

    def test_an_interval_across_the_margin_is_inconclusive(self):
        cell = write_cell(self.root / "c", rows(20, 100, 50, 1000, 800, 1, noise=0.0),
                          rows(20, 110, 50, 1000, 800, 2, noise=0.08))
        r, v = self.run_cell(cell, "prefill")
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertEqual(v["gates"]["prefill"]["verdict"], "inconclusive")

    def test_memory_growth_above_32_mib_fails(self):
        cell = write_cell(self.root / "c", rows(20, 100, 50, 1000, 800, 1),
                          rows(20, 130, 50, 770, 860, 2))
        _, v = self.run_cell(cell)
        self.assertEqual(v["gates"]["peak"]["verdict"], "fail")
        self.assertEqual(v["verdict"], "fail")

    def test_report_only_counts_complete_pairs(self):
        cell = write_cell(self.root / "c", rows(5, 100, 50, 1000, 800, 1),
                          rows(5, 130, 50, 770, 800, 2), statuses={(3, "B"): "failed"})
        _, v = self.run_cell(cell, "prefill", report=True)
        self.assertEqual(v["pairs"], 4)

    def test_report_gives_the_interval_without_a_verdict(self):
        # #582: an estimate, no preregistered margin. No speed-up would fail
        # --gates prefill; --report states the ratio and exits 0.
        cell = write_cell(self.root / "c", rows(5, 100, 50, 1000, 800, 1),
                          rows(5, 60, 50, 1600, 790, 2))
        out = self.root / "r.json"
        r = subprocess.run([sys.executable, str(SCRIPT), str(cell), "--report", "prefill,peak",
                            "--json-out", str(out)], capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stdout)
        v = json.loads(out.read_text())
        self.assertEqual(v["verdict"], "report")
        self.assertAlmostEqual(v["gates"]["prefill"]["median"], 0.6, delta=0.03)
        self.assertNotIn("verdict", v["gates"]["prefill"])
        self.assertNotIn("margin", v["gates"]["prefill"])

    def test_a_retried_attempt_is_not_a_pair(self):
        # bench-paired-knob.sh --retries: an attempt that wrote no row is a
        # "retry" line before the arm's ok line; the pair still counts.
        cell = write_cell(self.root / "c", rows(20, 100, 50, 1000, 800, 1),
                          rows(20, 130, 50, 770, 800, 2))
        ledger = cell / "ledger.jsonl"
        lines = ledger.read_text().splitlines()
        lines.insert(1, json.dumps({"pair": 1, "order": "AB", "arm": "B", "status": "retry"}))
        ledger.write_text("\n".join(lines) + "\n")
        _, v = self.run_cell(cell, "prefill")
        self.assertEqual(v["pairs"], 20)

    def test_gated_retry_must_refer_to_the_current_pending_arm(self):
        a = rows(20, 100, 50, 1000, 800, 1)
        for location, pair in ((0, 5), (40, 21)):
            with self.subTest(pair=pair):
                cell = write_cell(self.root / "c", a, a)
                ledger = cell / "ledger.jsonl"
                lines = ledger.read_text().splitlines()
                lines.insert(location, json.dumps({"pair": pair, "order": "AB", "arm": "A",
                                                   "status": "retry"}))
                ledger.write_text("\n".join(lines) + "\n")
                self.assertEqual(self.run_cell(cell, "prefill")[0].returncode, 2)

    def test_gated_estimate_requires_all_twenty_ordered_pairs_and_no_extra_rows(self):
        a = rows(20, 100, 50, 1000, 800, 1)
        for count in (1, 19, 21):
            with self.subTest(count=count):
                cell = write_cell(self.root / "c", (a + a)[:count], (a + a)[:count])
                self.assertEqual(self.run_cell(cell, "decode")[0].returncode, 2)
        cell = write_cell(self.root / "c", a, a)
        ledger = cell / "ledger.jsonl"
        entries = ledger.read_text().splitlines()
        entries[2], entries[3] = entries[3], entries[2]
        ledger.write_text("\n".join(entries) + "\n")
        self.assertEqual(self.run_cell(cell, "decode")[0].returncode, 2)

        cell = write_cell(self.root / "c", a, a)
        path = cell / "B.csv"
        with path.open("a") as fh:
            fh.write(path.read_text().splitlines(keepends=True)[-1])
        self.assertEqual(self.run_cell(cell, "decode")[0].returncode, 2)

    def test_gates_reject_configuration_drift_within_and_across_pairs(self):
        a = rows(20, 100, 50, 1000, 800, 1)
        for field, value in (("host", "xbox-series-x-declared-t6-g99-noeog"),
                             ("n_ctx", "4096"), ("quant", "Q4_K_M"),
                             ("n_gen_tok", "8"), ("n_threads", "4"),
                             ("max_length", "512"), ("backend", "cpu")):
            with self.subTest(field=field):
                b = [dict(row) for row in a]
                b[-1][field] = value
                cell = write_cell(self.root / "c", a, b)
                self.assertEqual(self.run_cell(cell, "decode")[0].returncode, 2)
        a[-1]["n_prompt_tok"] = "1000"
        self.assertEqual(self.run_cell(write_cell(self.root / "c", a, a), "decode")[0].returncode, 2)

    def test_candidate_invalid_measurements_cannot_win_a_lower_is_better_gate(self):
        a = rows(20, 100, 50, 1000, 800, 1)
        for column, value, gate in (("ttft_ms", "-1", "ttft"), ("ttft_ms", "0", "ttft"),
                                    ("peak_ws_mb", "-1", "peak"), ("decode_tok_s", "nan", "decode")):
            with self.subTest(column=column, value=value):
                b = [dict(row, **{column: value}) for row in a]
                cell = write_cell(self.root / "c", a, b)
                self.assertEqual(self.run_cell(cell, gate)[0].returncode, 2)

    def test_a_missing_working_set_reading_in_either_arm_is_not_a_measured_delta(self):
        a = rows(20, 100, 50, 1000, 800, 1)
        zero = [dict(row, peak_ws_mb="0") for row in a]
        for baseline, candidate in ((a, zero), (zero, a)):
            cell = write_cell(self.root / "c", baseline, candidate)
            for report in (True, False):
                result, summary = self.run_cell(cell, "peak", report=report)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIn("peak_ws_mb has a missing or zero measurement", result.stdout)
                self.assertIsNone(summary)

    def test_report_rejects_ambiguous_ledger_row_references(self):
        a = rows(3, 100, 50, 1000, 800, 1)
        for edit in ("zero", "outside", "duplicate", "detail"):
            with self.subTest(edit=edit):
                cell = write_cell(self.root / "r", a, a)
                ledger = cell / "ledger.jsonl"
                entries = [json.loads(line) for line in ledger.read_text().splitlines()]
                if edit == "duplicate":
                    entries.append(entries[-1])
                elif edit == "detail":
                    entries[0]["detail"] = "row 3"
                else:
                    for entry in entries[-2:]:
                        entry["pair"] = 0 if edit == "zero" else 5
                ledger.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
                result, summary = self.run_cell(cell, "decode", report=True)
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIsNone(summary)

    def test_published_pairs_reproduce_the_six_frozen_verdicts(self):
        declaration = ROOT / "docs/research/evidence/2026-11-16-c4-performance/manifest.json"
        for spec in json.loads(declaration.read_text())["cells"]:
            with self.subTest(cell=spec["id"]):
                cell = ROOT / "bench/results/2026-11-16-c4-paired/cells" / spec["id"]
                expected = json.loads((cell / "verdict.json").read_text())
                result, actual = self.run_cell(cell, ",".join(spec["gates"]))
                self.assertEqual(result.returncode, 0 if expected["verdict"] == "pass" else 1)
                self.assertEqual(actual["pairs"], spec["pairs"])
                self.assertEqual(actual["gates"], expected["gates"])
                self.assertEqual(actual["verdict"], expected["verdict"])

    def test_bias_add_profile_uses_preregistered_margins(self):
        a = bias_rows(20, 100, 50, 1000, 800, 1)
        b = bias_rows(20, 98, 53, 1020, 805, 2)
        cell = write_cell(self.root / "c", a, b)
        r, v = self.run_cell(cell, profile="bias-add")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        self.assertEqual(v["gates"]["decode"]["margin"], 1.03)
        self.assertEqual(v["gates"]["prefill"]["margin"], 0.97)
        self.assertEqual(v["gates"]["ttft"]["margin"], 1.03)
        self.assertEqual(v["gates"]["peak"]["margin"], 32.0)

    def test_bias_add_profile_rejects_an_incomplete_campaign(self):
        a = bias_rows(19, 100, 50, 1000, 800, 1)
        cell = write_cell(self.root / "c", a, a)
        r, _ = self.run_cell(cell, profile="bias-add")
        self.assertEqual(r.returncode, 2)
        self.assertIn("20 complete", r.stdout)

    def test_profile_requires_measured_positive_thread_count(self):
        a = bias_rows(20, 100, 50, 1000, 800, 1)
        for row in a:
            row["n_threads"] = "0"
        cell = write_cell(self.root / "c", a, a)
        self.assertEqual(self.run_cell(cell, profile="bias-add")[0].returncode, 2)

    def test_bias_add_profile_rejects_mixed_cells_and_unmatched_settings(self):
        a = bias_rows(20, 100, 50, 1000, 800, 1)
        b = bias_rows(20, 98, 53, 1020, 805, 2)
        a[-1]["model"] = b[-1]["model"] = "qwen25-coder-1.5b"
        cell = write_cell(self.root / "c", a, b)
        self.assertEqual(self.run_cell(cell, profile="bias-add")[0].returncode, 2)

        a[-1]["model"] = b[-1]["model"] = "qwen25-coder-3b"
        b[-1]["host"] = "xbox-series-s-declared-u512-g99-noeog"
        write_cell(cell, a, b)
        self.assertEqual(self.run_cell(cell, profile="bias-add")[0].returncode, 2)

    def test_bias_add_control_checks_equivalence_and_submission_counts(self):
        a = bias_rows(20, 100, 50, 1000, 800, 1, control=True)
        cell = write_cell(self.root / "c", a, a)
        log = ("[xllama] d3d12-phase phase=prefill calls=12 submissions=12 dispatches=12\n"
               "[xllama] d3d12-phase phase=decode calls=30 submissions=30 dispatches=30\n")
        for i in range(1, 21):
            for arm in "AB":
                (cell / f"pair{i}-{arm}-app.log").write_text(log + log)
        r, v = self.run_cell(cell, profile="bias-add-control")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        self.assertEqual(v["gates"]["decode"]["margin"], [0.98, 1.02])

        measured = cell / "pair7-B-app.log"
        measured.write_text(log)
        r, _ = self.run_cell(cell, profile="bias-add-control")
        self.assertEqual(r.returncode, 2, r.stderr or r.stdout)

        measured.write_text(log + log.replace("submissions=30", "submissions=31"))
        r, v = self.run_cell(cell, profile="bias-add-control")
        self.assertEqual(r.returncode, 1, r.stderr or r.stdout)
        self.assertEqual(v["gates"]["submissions"]["mismatched_pairs"], [7])

        measured.write_text(log.replace("submissions=30", "submissions=40") + log)
        r, _ = self.run_cell(cell, profile="bias-add-control")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)

        b = bias_rows(20, 100, 52, 1000, 800, 2, control=True)
        write_cell(cell, a, b)
        r, v = self.run_cell(cell, profile="bias-add-control")
        self.assertEqual(r.returncode, 1, r.stderr or r.stdout)
        self.assertEqual(v["gates"]["decode"]["verdict"], "fail")

    def write_rope_logs(self, cell, a_decode=36, b_decode=29, a_prefill=12, b_prefill=10):
        def log(decode, prefill):
            warm = (f"[xllama] d3d12-phase phase=prefill calls=1 submissions=99 dispatches=1\n"
                    f"[xllama] d3d12-phase phase=decode calls=1 submissions=99 dispatches=1\n")
            return (warm +
                    f"[xllama] d3d12-phase phase=prefill calls=1 submissions={prefill} dispatches=1\n"
                    f"[xllama] d3d12-phase phase=decode calls=1 submissions={decode} dispatches=1\n")
        for i in range(1, 21):
            (cell / f"pair{i}-A-app.log").write_text(log(a_decode, a_prefill))
            (cell / f"pair{i}-B-app.log").write_text(log(b_decode, b_prefill))

    def rope_cell(self, decode_b=53):
        a = bias_rows(20, 100, 50, 1000, 800, 1)
        b = bias_rows(20, 100, decode_b, 1000, 800, 2)
        cell = write_cell(self.root / "c", a, b)
        self.write_rope_logs(cell)
        return cell, a, b

    def test_rope_profile_shares_the_bias_add_margins_and_checks_submissions(self):
        cell, _, _ = self.rope_cell()
        r, v = self.run_cell(cell, profile="rope")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        self.assertEqual(v["gates"]["decode"]["margin"], 1.03)
        self.assertEqual(v["gates"]["peak"]["margin"], 32.0)
        self.assertEqual(v["gates"]["submissions"]["mismatched_pairs"], [])
        self.assertEqual(v["profile"], "rope")  # a bias ADD cell cannot pass as ROPE evidence

    def test_rope_profile_fails_when_the_knob_did_not_reduce_submissions(self):
        cell, _, _ = self.rope_cell()
        # A speed-up with B submitting as often as A is not the ROPE effect.
        self.write_rope_logs(cell, b_decode=36)
        r, v = self.run_cell(cell, profile="rope")
        self.assertEqual(r.returncode, 1, r.stderr or r.stdout)
        self.assertEqual(v["gates"]["submissions"]["mismatched_pairs"], list(range(1, 21)))
        self.write_rope_logs(cell, b_prefill=13)  # more prefill submissions than A
        r, _ = self.run_cell(cell, profile="rope")
        self.assertEqual(r.returncode, 1, r.stderr or r.stdout)

    def test_gpu_profiles_require_work_in_each_measured_phase(self):
        control = bias_rows(20, 100, 50, 1000, 800, 1, control=True)
        cell = write_cell(self.root / "control", control, control)
        self.write_rope_logs(cell, a_decode=0, b_decode=0, a_prefill=0, b_prefill=0)
        result, summary = self.run_cell(cell, profile="bias-add-control")
        self.assertEqual(result.returncode, 2, result.stdout)
        self.assertIsNone(summary)

        cell, _, _ = self.rope_cell()
        log = cell / "pair8-B-app.log"
        original = log.read_text().splitlines(keepends=True)
        for phase, counter in ((2, "calls"), (2, "submissions"), (3, "dispatches")):
            with self.subTest(phase=phase, counter=counter):
                lines = original.copy()
                lines[phase] = re.sub(rf"\b{counter}=\d+", f"{counter}=0", lines[phase])
                log.write_text("".join(lines))
                result, summary = self.run_cell(cell, profile="rope")
                self.assertEqual(result.returncode, 2, result.stdout)
                self.assertIsNone(summary)

    def test_rope_profile_rejects_an_incomplete_or_mixed_campaign(self):
        a = bias_rows(19, 100, 50, 1000, 800, 1)
        cell = write_cell(self.root / "i", a, a)
        self.assertEqual(self.run_cell(cell, profile="rope")[0].returncode, 2)

        cell, a, b = self.rope_cell()
        b[-1]["host"] = "xbox-series-s-declared-u512-g99-noeog"  # greedy tag missing
        write_cell(cell, a, b)
        self.assertEqual(self.run_cell(cell, profile="rope")[0].returncode, 2)

        control = bias_rows(20, 100, 50, 1000, 800, 1, control=True)
        cell = write_cell(self.root / "k", control, control)
        self.write_rope_logs(cell)
        self.assertEqual(self.run_cell(cell, profile="rope")[0].returncode, 2)

    def test_rope_profile_rejects_a_missing_measured_phase_counter(self):
        cell, _, _ = self.rope_cell()
        log = (cell / "pair9-B-app.log").read_text().splitlines(keepends=True)
        (cell / "pair9-B-app.log").write_text("".join(log[:2]))  # warm-up only
        r, _ = self.run_cell(cell, profile="rope")
        self.assertEqual(r.returncode, 2, r.stderr or r.stdout)
        (cell / "pair9-B-app.log").unlink()
        r, _ = self.run_cell(cell, profile="rope")
        self.assertEqual(r.returncode, 2, r.stderr or r.stdout)

    def test_invalid_inputs_fail_closed(self):
        good = write_cell(self.root / "g", rows(20, 100, 50, 1000, 800, 1),
                          rows(20, 130, 50, 770, 800, 2))
        self.assertEqual(self.run_cell(good, "speed")[0].returncode, 2)
        mixed = rows(20, 130, 50, 770, 800, 2)
        mixed[1]["n_prompt_tok"] = "900"
        cell = write_cell(self.root / "m", rows(20, 100, 50, 1000, 800, 1), mixed)
        self.assertEqual(self.run_cell(cell)[0].returncode, 2)
        zero = rows(20, 100, 50, 1000, 800, 1)
        zero[0]["prompt_tok_s"] = "0"
        cell = write_cell(self.root / "z", zero, rows(20, 130, 50, 770, 800, 2))
        self.assertEqual(self.run_cell(cell)[0].returncode, 2)
        (good / "ledger.jsonl").unlink()
        self.assertEqual(self.run_cell(good)[0].returncode, 2)


if __name__ == "__main__":
    unittest.main()
