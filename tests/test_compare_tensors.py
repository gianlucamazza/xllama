# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Exercise the graph-order tensor comparator, including fail-closed inputs."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/compare-tensors.py"
NAMES = ["l_out-0", "l_out-1", "l_out-2", "result_output"]


def write_capture(path: Path, perturb=None, names=NAMES, ne0=64, skip=()):
    """perturb: {name: relative noise scale}. Same seed => same base values."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    noise = np.random.default_rng(11)
    lines = []
    for seq, name in enumerate(names):
        a = rng.standard_normal(ne0).astype(np.float32)
        scale = (perturb or {}).get(name, 0.0)
        if scale:
            a = (a + scale * noise.standard_normal(ne0)).astype(np.float32)
        f = f"{seq:06d}.f32"
        a.tofile(path / f)
        lines.append({"seq": seq, "eval": 0, "name": name, "op": "ADD", "type": "f32",
                      "ne": [ne0, 1, 1, 1], "mode": "last", "values": ne0, "file": f})
    for name in skip:
        lines.append({"eval": 0, "name": name, "op": "ADD", "type": "f32",
                      "ne": [ne0, 0, 1, 1], "skipped": "empty"})
    (path / "manifest.jsonl").write_text("".join(json.dumps(x) + "\n" for x in lines))
    return path


class CompareTensorsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ref = write_capture(self.root / "ref")

    def run_cmp(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                              capture_output=True, text=True, check=False)

    def test_identical_captures_have_no_divergence(self):
        r = self.run_cmp(self.ref, write_capture(self.root / "same"))
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_first_divergent_record_is_reported_in_graph_order(self):
        cand = write_capture(self.root / "c", {"l_out-1": 0.1, "l_out-2": 0.2, "result_output": 0.3})
        out = self.root / "out.json"
        r = self.run_cmp(self.ref, cand, "--json-out", out)
        self.assertEqual(r.returncode, 1, r.stdout)
        data = json.loads(out.read_text())
        self.assertEqual(data["first_divergence"]["name"], "l_out-1")
        self.assertEqual(data["divergent"], 3)

    def test_floor_absorbs_drift_that_accepted_paths_share(self):
        cand = write_capture(self.root / "c", {"l_out-2": 0.05})
        floor = write_capture(self.root / "f", {"l_out-2": 0.05})
        self.assertEqual(self.run_cmp(self.ref, cand, "--floor", floor).returncode, 0)
        loud = write_capture(self.root / "loud", {"l_out-2": 0.5})
        r = self.run_cmp(self.ref, loud, "--floor", floor)
        self.assertEqual(r.returncode, 1)
        self.assertIn("first divergence: eval 0 l_out-2#0", r.stdout)

    def test_repeated_names_are_matched_by_occurrence(self):
        names = ["Qcur-0", "Qcur-0", "l_out-0"]
        ref = write_capture(self.root / "r", names=names)
        cand = write_capture(self.root / "c", {"Qcur-0": 0.0}, names=names)
        self.assertEqual(self.run_cmp(ref, cand).returncode, 0)
        # Perturb only the second Qcur-0 node.
        data = np.fromfile(cand / "000001.f32", dtype=np.float32) * 1.5
        data.astype(np.float32).tofile(cand / "000001.f32")
        out = self.root / "occ.json"
        r = self.run_cmp(ref, cand, "--json-out", out)
        self.assertEqual(r.returncode, 1, r.stdout)
        first = json.loads(out.read_text())["first_divergence"]
        self.assertEqual((first["name"], first["occ"]), ("Qcur-0", 1))

    def test_mismatched_record_sets_are_invalid(self):
        cand = write_capture(self.root / "c", names=NAMES[:3])
        self.assertEqual(self.run_cmp(self.ref, cand).returncode, 2)
        skipped = write_capture(self.root / "s", skip=["l_out-9"])
        self.assertEqual(self.run_cmp(self.ref, skipped).returncode, 2)

    def test_shape_mismatch_is_invalid(self):
        cand = write_capture(self.root / "c", ne0=32)
        self.assertEqual(self.run_cmp(self.ref, cand).returncode, 2)

    def test_non_finite_and_truncated_data_are_invalid(self):
        cand = write_capture(self.root / "nan")
        data = np.fromfile(cand / "000001.f32", dtype=np.float32)
        data[3] = np.nan
        data.tofile(cand / "000001.f32")
        self.assertEqual(self.run_cmp(self.ref, cand).returncode, 2)
        short = write_capture(self.root / "short")
        (short / "000002.f32").write_bytes(b"\x00\x00\x80")
        self.assertEqual(self.run_cmp(self.ref, short).returncode, 2)

    def test_missing_or_malformed_manifest_is_invalid(self):
        self.assertEqual(self.run_cmp(self.ref, self.root / "absent").returncode, 2)
        bad = write_capture(self.root / "bad")
        (bad / "manifest.jsonl").write_text("{not json\n")
        self.assertEqual(self.run_cmp(self.ref, bad).returncode, 2)
        empty = self.root / "empty"
        empty.mkdir()
        (empty / "manifest.jsonl").write_text("")
        self.assertEqual(self.run_cmp(self.ref, empty).returncode, 2)

    def test_record_file_cannot_escape_capture_dir(self):
        evil = write_capture(self.root / "evil")
        lines = (evil / "manifest.jsonl").read_text().splitlines()
        rec = json.loads(lines[0])
        rec["file"] = "../ref/000000.f32"
        lines[0] = json.dumps(rec)
        (evil / "manifest.jsonl").write_text("\n".join(lines) + "\n")
        self.assertEqual(self.run_cmp(self.ref, evil).returncode, 2)

    def test_invalid_arguments_fail(self):
        for opt in (["--nmse", "nan"], ["--nmse", "-1"], ["--floor-ratio", "0.5"]):
            self.assertNotEqual(self.run_cmp(self.ref, self.ref, *opt).returncode, 0)


if __name__ == "__main__":
    unittest.main()
