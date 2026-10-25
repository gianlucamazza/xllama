#!/usr/bin/env python3
"""Licence-file pairing for re-hosted models-v1 catalogue entries.

Mirrors the LFM pattern: each downloadable models-v1 model lists LICENSE
(and SD-Turbo NOTICE) in files[] with a sha256 that matches the exact
bytes committed under LICENSES/. TAESD is not a catalogue entry; its MIT
file is downloaded next to the VAE overlay in MainPage.cpp.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "uwp/models/manifest.json"
LICENSES = ROOT / "LICENSES"
MAINPAGE = ROOT / "uwp/MainPage.cpp"
MODELS_V1 = "https://github.com/gianlucamazza/xllama/releases/download/models-v1"

# Catalogue id -> (local filename, LICENSES/ path, models-v1 asset name)
PAIRINGS = {
    "smollm2-360m-cpu-int4": [
        ("LICENSE", "HuggingFaceTB-SmolLM2-LICENSE", "smollm2-360m-cpu-int4_LICENSE"),
    ],
    "smollm2-360m-dml-fp16-v2": [
        ("LICENSE", "HuggingFaceTB-SmolLM2-LICENSE", "smollm2-360m-dml-fp16-v2_LICENSE"),
    ],
    "smollm2-1.7b-cpu-int4": [
        ("LICENSE", "HuggingFaceTB-SmolLM2-LICENSE", "smollm2-1.7b-cpu-int4_LICENSE"),
    ],
    "qwen35-0.8b": [
        ("LICENSE", "Qwen-Qwen3.5-0.8B-LICENSE", "qwen35-0.8b_LICENSE"),
    ],
    "sd-turbo-fp16": [
        ("LICENSE", "stabilityai-sd-turbo-LICENSE", "sd-turbo-fp16_LICENSE"),
        ("NOTICE", "stabilityai-sd-turbo-NOTICE", "sd-turbo-fp16_NOTICE"),
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _file_by_name(entry: dict, filename: str) -> dict | None:
    for f in entry.get("files", []):
        if f.get("filename") == filename:
            return f
    return None


class CatalogLicensePairingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
        cls.cat = {e["name"]: e for e in cls.manifest["models"]}

    def test_models_v1_rehosts_pin_committed_licence_files(self) -> None:
        for name, files in PAIRINGS.items():
            with self.subTest(model=name):
                entry = self.cat[name]
                self.assertEqual(entry.get("hf_base_url"), MODELS_V1)
                for filename, license_name, remote in files:
                    committed = LICENSES / license_name
                    self.assertTrue(committed.is_file(), f"missing {committed}")
                    spec = _file_by_name(entry, filename)
                    self.assertIsNotNone(spec, f"{name} missing files[] {filename}")
                    digest = _sha256(committed)
                    self.assertEqual(spec["sha256"], digest)
                    self.assertEqual(spec.get("approx_bytes"), committed.stat().st_size)
                    self.assertEqual(spec.get("remote"), remote)

    def test_every_models_v1_catalogue_entry_is_paired(self) -> None:
        hosted = [
            name
            for name, e in self.cat.items()
            if e.get("hf_base_url") == MODELS_V1
        ]
        self.assertCountEqual(hosted, list(PAIRINGS))

    def test_liquidai_lfm_entries_still_ship_license(self) -> None:
        liquid = [
            name
            for name, e in self.cat.items()
            if "huggingface.co/LiquidAI/" in e.get("hf_base_url", "")
        ]
        self.assertTrue(liquid)
        for name in liquid:
            with self.subTest(model=name):
                spec = _file_by_name(self.cat[name], "LICENSE")
                self.assertIsNotNone(spec, f"{name} lost its LFM LICENSE pairing")
                self.assertEqual(len(spec["sha256"]), 64)

    def test_qwen25_coder_3b_has_no_models_v1_licence_entry(self) -> None:
        entry = self.cat["qwen25-coder-3b"]
        self.assertNotIn("models-v1", entry.get("hf_base_url", ""))
        self.assertIsNone(_file_by_name(entry, "LICENSE"))

    def test_taesd_overlay_downloads_mit_licence(self) -> None:
        text = MAINPAGE.read_text(encoding="utf-8")
        digest = _sha256(LICENSES / "madebyollin-taesd-LICENSE")
        self.assertIn("sd-turbo-fp16_taesd_LICENSE", text)
        self.assertIn(digest, text)
        self.assertIn("TAESD_LICENSE", text)
        # Pin is a named constant above the ModelFile list (LFM-style sha256).
        self.assertIn("kTaesdLicenseSha256", text)


if __name__ == "__main__":
    unittest.main()
