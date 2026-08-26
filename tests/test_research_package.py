#!/usr/bin/env python3
"""Tests for the claim-to-evidence research package contract."""

import importlib.util
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_research_package", ROOT / "scripts" / "build-research-package.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ResearchPackageTests(unittest.TestCase):
    def setUp(self):
        self.claims = json.loads((ROOT / "paper/claims.json").read_text())

    def test_repeated_claim_uses_median(self):
        claim = next(c for c in self.claims["claims"] if c["id"] == "lfm25-350m-console-baseline")
        result = MODULE.resolve_claim(claim)
        self.assertEqual(result["run_count"], 3)
        self.assertAlmostEqual(result["values"]["decode_tok_s"], 89.65)

    def test_ambiguous_only_claim_is_rejected(self):
        claim = dict(self.claims["claims"][0])
        claim["selection"] = "only"
        with self.assertRaises(ValueError):
            MODULE.resolve_claim(claim)

    def test_all_current_claims_resolve(self):
        resolved = MODULE.validate_claims(self.claims)
        self.assertEqual(set(resolved), {c["id"] for c in self.claims["claims"]})

    def test_claim_digest_is_stable(self):
        self.assertEqual(len(MODULE.claims_digest(self.claims)), 64)


if __name__ == "__main__":
    unittest.main()
