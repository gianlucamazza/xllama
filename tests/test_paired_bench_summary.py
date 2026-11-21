# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Preregistered paired-campaign verdict: margins, interval rule, fail-closed inputs."""
import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/paired-bench-summary.py"
FIELDS = ["model", "n_prompt_tok", "prompt_tok_s", "decode_tok_s", "ttft_ms", "peak_ws_mb"]


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
    return [{"model": "m", "n_prompt_tok": "300",
             "prompt_tok_s": f"{prefill * (1 + noise * rng.standard_normal()):.4f}",
             "decode_tok_s": f"{decode * (1 + noise * rng.standard_normal()):.4f}",
             "ttft_ms": f"{ttft * (1 + noise * rng.standard_normal()):.4f}",
             "peak_ws_mb": f"{peak + rng.integers(-2, 3)}"} for _ in range(n)]


class PairedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def run_cell(self, cell, gates="prefill,decode,ttft,peak", profile=None):
        out = self.root / "v.json"
        mode = ["--profile", profile] if profile else ["--gates", gates]
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

    def test_only_complete_pairs_count(self):
        cell = write_cell(self.root / "c", rows(5, 100, 50, 1000, 800, 1),
                          rows(5, 130, 50, 770, 800, 2), statuses={(3, "B"): "failed"})
        _, v = self.run_cell(cell, "prefill")
        self.assertEqual(v["pairs"], 4)

    def test_bias_add_profile_uses_preregistered_margins(self):
        a = rows(20, 100, 50, 1000, 800, 1, noise=0)
        b = rows(20, 98, 53, 1020, 805, 2, noise=0)
        cell = write_cell(self.root / "c", a, b)
        r, v = self.run_cell(cell, profile="bias-add")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        self.assertEqual(v["gates"]["decode"]["margin"], 1.03)
        self.assertEqual(v["gates"]["prefill"]["margin"], 0.97)
        self.assertEqual(v["gates"]["ttft"]["margin"], 1.03)
        self.assertEqual(v["gates"]["peak"]["margin"], 32.0)

    def test_bias_add_profile_rejects_an_incomplete_campaign(self):
        a = rows(19, 100, 50, 1000, 800, 1, noise=0)
        cell = write_cell(self.root / "c", a, a)
        r, _ = self.run_cell(cell, profile="bias-add")
        self.assertEqual(r.returncode, 2)
        self.assertIn("20 complete", r.stdout)

    def test_bias_add_control_checks_equivalence_and_submission_counts(self):
        a = rows(20, 100, 50, 1000, 800, 1, noise=0)
        cell = write_cell(self.root / "c", a, a)
        log = ("[xllama] d3d12-phase phase=prefill calls=12 submissions=12 dispatches=12\n"
               "[xllama] d3d12-phase phase=decode calls=30 submissions=30 dispatches=30\n")
        for i in range(1, 21):
            for arm in "AB":
                (cell / f"pair{i}-{arm}-app.log").write_text(log)
        r, v = self.run_cell(cell, profile="bias-add-control")
        self.assertEqual(r.returncode, 0, r.stderr or r.stdout)
        self.assertEqual(v["gates"]["decode"]["margin"], [0.98, 1.02])

        (cell / "pair7-B-app.log").write_text(log.replace("submissions=30", "submissions=31"))
        r, v = self.run_cell(cell, profile="bias-add-control")
        self.assertEqual(r.returncode, 1, r.stderr or r.stdout)
        self.assertEqual(v["gates"]["submissions"]["mismatched_pairs"], [7])

        (cell / "pair7-B-app.log").write_text(log)
        b = rows(20, 100, 52, 1000, 800, 2, noise=0)
        write_cell(cell, a, b)
        r, v = self.run_cell(cell, profile="bias-add-control")
        self.assertEqual(r.returncode, 1, r.stderr or r.stdout)
        self.assertEqual(v["gates"]["decode"]["verdict"], "fail")

    def test_invalid_inputs_fail_closed(self):
        good = write_cell(self.root / "g", rows(4, 100, 50, 1000, 800, 1),
                          rows(4, 130, 50, 770, 800, 2))
        self.assertEqual(self.run_cell(good, "speed")[0].returncode, 2)
        mixed = rows(4, 130, 50, 770, 800, 2)
        mixed[1]["n_prompt_tok"] = "900"
        cell = write_cell(self.root / "m", rows(4, 100, 50, 1000, 800, 1), mixed)
        self.assertEqual(self.run_cell(cell)[0].returncode, 2)
        zero = rows(4, 100, 50, 1000, 800, 1)
        zero[0]["prompt_tok_s"] = "0"
        cell = write_cell(self.root / "z", zero, rows(4, 130, 50, 770, 800, 2))
        self.assertEqual(self.run_cell(cell)[0].returncode, 2)
        (good / "ledger.jsonl").unlink()
        self.assertEqual(self.run_cell(good)[0].returncode, 2)


if __name__ == "__main__":
    unittest.main()
