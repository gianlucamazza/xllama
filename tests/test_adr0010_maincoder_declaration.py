# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Check the frozen ADR 0010 maincoder declaration."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/research/evidence/2026-11-25-adr0010-maincoder-1b"
DECLARATION = EVIDENCE / "declaration.json"
DISTRIBUTION_IDS = [f"u{i:02d}" for i in range(1, 17)]
DISTRIBUTION = {"kld_ratio_limit": 1.5, "top1_margin": 0.01, "confidence": 0.95,
                "min_positions": 4000, "bootstrap": 2000, "seed": 1,
                "prompts": DISTRIBUTION_IDS}
CAPTURES = [{"prompt": "c1", "last_layer": 31},
            {"prompt": "u03", "last_layer": 5}]


def load_module():
    spec = importlib.util.spec_from_file_location("control_parity", ROOT / "scripts/control-parity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MaincoderDeclarationTests(unittest.TestCase):
    def setUp(self):
        self.decl = json.loads(DECLARATION.read_bytes())

    def test_declaration_identity_and_hash(self):
        self.assertEqual(self.decl["schema"], "xllama.control-declaration/v2")
        self.assertEqual(self.decl["model"], "maincoder-1b")
        self.assertEqual(sha256(DECLARATION),
                         "5499e06efcaf5ead59e6b8dfe1468deb000045b99a6f746320d68259b4c3a85c")

    def test_prompt_files_and_hashes(self):
        ids = ["c1", *DISTRIBUTION_IDS]
        self.assertEqual([p["id"] for p in self.decl["prompts"]], ids)
        prompt_dir = EVIDENCE / "prompts"
        self.assertEqual({p.name for p in prompt_dir.iterdir() if p.is_file()},
                         {f"{pid}.txt" for pid in ids})
        for prompt in self.decl["prompts"]:
            with self.subTest(prompt=prompt["id"]):
                path = prompt_dir / f'{prompt["id"]}.txt'
                self.assertTrue(path.is_file())
                self.assertEqual(sha256(path), prompt["sha256"])

    def test_declared_limits(self):
        self.assertEqual(self.decl["distribution"], DISTRIBUTION)
        self.assertEqual(self.decl["reference"]["nmse_limit"], 1e-7)
        self.assertEqual(self.decl["reference"]["captures"], CAPTURES)

    def test_comparator_loads_declaration(self):
        mod = load_module()
        distribution = mod.load_declaration(DECLARATION, "distribution")
        for key, value in DISTRIBUTION.items():
            if key != "prompts":
                self.assertEqual(distribution[key], value)
        self.assertEqual(list(distribution["prompts"]), DISTRIBUTION_IDS)
        self.assertEqual(distribution["prompts"],
                         {p["id"]: p["sha256"] for p in self.decl["prompts"]
                          if p["id"] in DISTRIBUTION_IDS})
        reference = mod.load_declaration(DECLARATION, "reference")
        self.assertEqual(reference["nmse_limit"], 1e-7)
        self.assertEqual(reference["captures"], {"c1": 31, "u03": 5})

    def test_comparator_rejects_wrong_schema(self):
        mod = load_module()
        self.decl["schema"] = "xllama.control-declaration/typo"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "declaration.json"
            path.write_text(json.dumps(self.decl))
            for mode in ("distribution", "reference"):
                with self.subTest(mode=mode), self.assertRaises(mod.Invalid):
                    mod.load_declaration(path, mode)

    def test_cpu_reference_pairs(self):
        path = EVIDENCE / "cpu-reference-pairs.json"
        self.assertEqual(sha256(path),
                         "4d7fc2e59c34a4842be99fed2d0cc6501908fd8a48d82990bccbcad255a3c124")
        pairs = json.loads(path.read_bytes())
        self.assertEqual(pairs["positions"], 5650)
        self.assertEqual(pairs["model"], "maincoder-1b")


if __name__ == "__main__":
    unittest.main()
