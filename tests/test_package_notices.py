#!/usr/bin/env python3
"""MSIX packaging of redistributed runtime licence notices.

Linux CI cannot build the UWP package. This check asserts the project item
list that MSBuild copies into the MSIX: each notice file exists under
LICENSES/ and is DeploymentContent with TargetPath licenses\\<name>.
"""

from __future__ import annotations

import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VCXPROJ = ROOT / "uwp/xllama.vcxproj"
LICENSES = ROOT / "LICENSES"
MSBUILD_NS = "http://schemas.microsoft.com/developer/msbuild/2003"

# (LICENSES/ filename, expected TargetPath inside the MSIX)
PACKAGE_NOTICES = [
    ("onnxruntime-LICENSE", r"licenses\onnxruntime-LICENSE"),
    ("onnxruntime-ThirdPartyNotices.txt", r"licenses\onnxruntime-ThirdPartyNotices.txt"),
    ("onnxruntime-genai-LICENSE", r"licenses\onnxruntime-genai-LICENSE"),
    (
        "onnxruntime-genai-ThirdPartyNotices.txt",
        r"licenses\onnxruntime-genai-ThirdPartyNotices.txt",
    ),
    ("llama.cpp-LICENSE", r"licenses\llama.cpp-LICENSE"),
    ("DirectML-LICENSE.txt", r"licenses\DirectML-LICENSE.txt"),
    ("DirectML-ThirdPartyNotices.txt", r"licenses\DirectML-ThirdPartyNotices.txt"),
]


def _norm(path: str) -> str:
    return path.replace("/", "\\").lower()


class PackageNoticeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tree = ET.parse(VCXPROJ)
        ns = {"m": MSBUILD_NS}
        items = []
        for none in tree.getroot().findall(".//m:None", ns):
            include = none.get("Include") or ""
            dc = none.find("m:DeploymentContent", ns)
            tp = none.find("m:TargetPath", ns)
            cond = none.get("Condition")
            items.append(
                {
                    "include": include,
                    "deployment": (dc.text or "").strip() if dc is not None else "",
                    "target": (tp.text or "").strip() if tp is not None else "",
                    "condition": cond,
                }
            )
        cls.items = items

    def _item_for_source(self, filename: str) -> dict:
        want = _norm(rf"..\LICENSES\{filename}")
        matches = [it for it in self.items if _norm(it["include"]) == want]
        self.assertEqual(
            len(matches),
            1,
            f"expected one vcxproj None item for {want}, got {matches!r}",
        )
        return matches[0]

    def test_notice_source_files_exist(self) -> None:
        for filename, _target in PACKAGE_NOTICES:
            with self.subTest(file=filename):
                path = LICENSES / filename
                self.assertTrue(path.is_file(), f"missing {path}")
                self.assertGreater(path.stat().st_size, 0, f"empty {path}")

    def test_vcxproj_deploys_notices_under_licenses(self) -> None:
        for filename, target in PACKAGE_NOTICES:
            with self.subTest(file=filename):
                item = self._item_for_source(filename)
                self.assertEqual(item["deployment"].lower(), "true")
                self.assertEqual(_norm(item["target"]), _norm(target))
                self.assertTrue(
                    item["condition"] in (None, ""),
                    f"{filename} must always ship (no Condition)",
                )

    def test_onnx_and_llama_notices_are_mit(self) -> None:
        for filename in (
            "onnxruntime-LICENSE",
            "onnxruntime-genai-LICENSE",
            "llama.cpp-LICENSE",
        ):
            with self.subTest(file=filename):
                text = (LICENSES / filename).read_text(encoding="utf-8")
                self.assertIn("MIT License", text)
                self.assertIn("Permission is hereby granted", text)

    def test_directml_notice_is_microsoft_redistributable(self) -> None:
        text = (LICENSES / "DirectML-LICENSE.txt").read_text(encoding="utf-8-sig")
        self.assertIn("MICROSOFT SOFTWARE LICENSE TERMS", text)
        self.assertIn("DIRECTX MACHINE LEARNING", text)


if __name__ == "__main__":
    unittest.main()
