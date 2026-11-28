# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Regression checks for DC-0 dimensions, placement, topology and CSV contracts."""

import csv
import io
import json
import runpy
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "model-distributed-inference.py"
MODEL = runpy.run_path(str(SCRIPT))
SHAPE = MODEL["SHAPES"][8]
BASE = MODEL["SCENARIOS"]["realistic"]


def case(strategy, nodes, family="dense", shape=SHAPE, cfg=BASE, **options):
    return MODEL["evaluate"](shape, cfg, family, strategy, nodes, **options)


class DistributedModelTests(unittest.TestCase):
    def test_payload_and_tcp_units(self):
        self.assertEqual(8192 * 8 / 1e9, 65.536e-6)
        cfg = replace(BASE, tcp_mbps=1000., copy_gbs=8., rtt_ms=1., sync_us=100.)
        self.assertAlmostEqual(MODEL["transport"](cfg, 1000, 3), 0.00330825)

    def test_two_worker_pipeline_owns_embeddings_and_relays(self):
        # 2 x 8192 B at the remote boundary, + one embedding, + all f32 logits.
        p = case("pipeline", 2)
        self.assertEqual(p["wire_bytes_tok"], 536576)
        self.assertEqual(case("pipeline", 2, legs=1)["wire_bytes_tok"], 528384)
        self.assertAlmostEqual(p["host_floor_gib"], (128000 * 4096 * 0.6 + 512000) / 2**30)
        self.assertEqual(case("pipeline", 1)["host_floor_gib"], 0)
        self.assertAlmostEqual(p["peak_rank_gib"],
                               ((8e9 * 0.6 / 2) + (16 * 2 * 2048 * 8 * 128 * 2)) / 2**30 + 0.65)

    def test_gqa_context_and_slots(self):
        cap = replace(BASE, cap_gib=8.)
        more_context = case("pipeline", 2, shape=replace(SHAPE, window=4096), cfg=cap)
        extra_slot = case("pipeline", 2, cfg=cap, streams=2)
        one = case("pipeline", 2, cfg=cap)
        self.assertAlmostEqual(more_context["peak_rank_gib"] - one["peak_rank_gib"], 0.125)
        self.assertAlmostEqual(more_context["peak_rank_gib"], extra_slot["peak_rank_gib"])
        self.assertEqual(extra_slot["single_tok_s"], one["single_tok_s"])
        self.assertLessEqual(extra_slot["aggregate_tok_s"], 2 * one["single_tok_s"])

    def test_replica_does_not_pool_capacity(self):
        replica = case("replica", 32)
        self.assertEqual(replica["status"], "budget")
        self.assertEqual(replica["wire_scope"], "no_tensor_payload_http_excluded")
        ample = replace(BASE, cap_gib=8.)
        one = case("replica", 1, cfg=ample)
        four = case("replica", 4, cfg=ample)
        self.assertEqual(four["peak_rank_gib"], one["peak_rank_gib"])
        self.assertEqual(four["single_tok_s"], one["single_tok_s"])
        self.assertEqual(four["aggregate_tok_s"], 4 * one["single_tok_s"])

    def test_unequal_stages_use_local_limits_and_slowest_stage(self):
        nodes = [{"layers": 1, "cap_gib": 8.}, {"layers": 31, "cap_gib": 2.}]
        self.assertEqual(case("pipeline", 2, layout=nodes)["status"], "budget")
        nodes = [{"layers": 16, "cap_gib": 4., "mem_gbs": 2.}, {"layers": 16, "cap_gib": 6.}]
        slow = case("pipeline", 2, streams=4, layout=nodes)
        even = case("pipeline", 2, streams=4)
        self.assertLess(slow["aggregate_tok_s"], even["aggregate_tok_s"])
        self.assertAlmostEqual(slow["cluster_available_gib"], 8.7)

    def test_tensor_ring_counts_egress_and_suppresses_impossible_layout(self):
        self.assertEqual(case("tensor", 2)["wire_scope"], "per_rank_tx")
        self.assertEqual(case("tensor", 2)["wire_bytes_tok"], 2 * 32 * 8192)
        bad = case("tensor", 16, shape=MODEL["SHAPES"][14])
        self.assertEqual(bad["status"], "layout")
        self.assertIsNone(bad["single_tok_s"])

    def test_expert_relay_and_locality(self):
        direct = case("expert", 2, "moe", expert_legs=1)
        relayed = case("expert", 2, "moe", expert_legs=2)
        expected = 2 * 32 * 4 * 0.5 * 8192
        self.assertEqual(direct["wire_bytes_tok"], expected)
        self.assertEqual(relayed["wire_bytes_tok"], 2 * expected)
        self.assertAlmostEqual(relayed["net_ms"], 2 * direct["net_ms"])
        self.assertEqual(case("expert", 2, "moe", remote_fraction=0.)["wire_bytes_tok"], 0)

    def test_cache_miss_units_and_complete_warm_shard(self):
        cold = case("hybrid", 2, "moe", cfg=replace(BASE, hit=0.))
        warm = case("hybrid", 2, "moe", cfg=replace(BASE, hit=0.5))
        selected = (8e9 - 2 * 128000 * 4096) * 0.6 * 0.8 * 4 / 64
        self.assertAlmostEqual(cold["ssd_bytes_tok"], selected)
        self.assertAlmostEqual(cold["ssd_ms"], selected / 400e6 * 1000 + 32 * 4 * 0.25)
        self.assertAlmostEqual(warm["ssd_bytes_tok"], selected / 2)
        self.assertEqual(case("hybrid", 2, "moe", cfg=replace(BASE, cache_gib=6.))["ssd_bytes_tok"], 0)
        self.assertEqual(case("hybrid", 2, "moe", cfg=replace(BASE, cache_gib=0.))["status"], "layout")

    def test_cost_power_and_calendar_time(self):
        cfg = replace(BASE, active_w=1000., idle_w=0., node_eur=0., eur_kwh=1., utilization=0.5)
        joules, eur = MODEL["cost"](cfg, 1, 10.)
        self.assertEqual(joules, 100.)
        self.assertAlmostEqual(eur, 1e6 / 36000)

    def test_full_grid_and_censored_baseline(self):
        result = subprocess.run([sys.executable, SCRIPT], capture_output=True, text=True, check=True)
        rows = list(csv.DictReader(io.StringIO(result.stdout)))
        self.assertEqual(len(rows), 3 * 6 * (3 + 5) * 6)
        for row in rows:
            if row["status"] != "fits":
                self.assertEqual(row["aggregate_tok_s"], "")
                self.assertEqual(row["eur_mtok"], "")
        pipeline = next(r for r in rows if r["scenario"] == "realistic" and r["family"] == "dense"
                        and r["study_class_b"] == "8" and r["nodes"] == "2" and r["strategy"] == "pipeline")
        self.assertEqual(pipeline["scaling_efficiency"], "")
        self.assertEqual(json.loads(pipeline["profile"])["legs"], 2)

    def test_bad_later_profile_emits_no_partial_csv(self):
        result = subprocess.run([sys.executable, SCRIPT, "--set", "reserve_gib=4.2"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")

    def test_invalid_shapes_and_partial_group(self):
        with self.assertRaises(ValueError):
            MODEL["validate"](replace(SHAPE, kv_heads=3), BASE, 1, 2, 2, None)
        with self.assertRaises(ValueError):
            MODEL["validate"](replace(SHAPE, vocab=10**9), BASE, 1, 2, 2, None)
        self.assertEqual(case("hybrid", 4, "moe", group=3)["status"], "layout")
        self.assertEqual(case("expert", 2, "moe", shape=replace(SHAPE, shared_fraction=0.))["status"], "layout")
        self.assertAlmostEqual(MODEL["overrides"](MODEL["SCENARIOS"]["optimistic"],
                                                  ["mem_gbs=22.5"]).mem_gbs, 22.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layout.json"
            path.write_text('[{"layers":1,"cap_gib":"4"}]')
            with self.assertRaises(ValueError):
                MODEL["read_layout"](path)


if __name__ == "__main__":
    unittest.main()
