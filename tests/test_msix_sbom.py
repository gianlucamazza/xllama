"""The package inventory must identify the bytes that reached the signed MSIX."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import msix_sbom

CREATED = "2026-11-28T12:00:00Z"
MANIFEST = b"""<Package xmlns="http://schemas.microsoft.com/appx/manifest/foundation/windows10">
<Identity Name="Product.Name" Version="1.2.3.42"/></Package>"""


class MsixSbomTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / "renamed.msix"
        self.members = {
            "AppxManifest.xml": MANIFEST,
            "AppxSignature.p7x": b"test signing metadata",
            "a/helper.dll": b"Windows binary",
            "b/helper.dll": b"a different binary",
        }
        self.pack()

    def pack(self, compression=ZIP_STORED):
        with ZipFile(self.path, "w", compression=compression) as archive:
            for name, content in self.members.items():
                archive.writestr(name, content)

    def write(self, relative, content):
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def add_build_inputs(self):
        pins = {
            "onnxruntime": ("Microsoft.ML.OnnxRuntime.DirectML", "1.2.3"),
            "onnxruntime-genai": ("Microsoft.ML.OnnxRuntimeGenAI.DirectML", "2.3.4"),
            "DirectML": ("Microsoft.AI.DirectML", "3.4.5"),
        }
        config = "<packages>"
        for stem, (nuget, version) in pins.items():
            config += f'<package id="{nuget}" version="{version}"/>'
            filename = stem + ".dll"
            self.members[filename] = ("Test bytes for " + stem).encode()
            if stem == "DirectML":
                self.write(f"uwp/packages/{nuget}.{version}/bin/x64-win/{filename}", self.members[filename])
                notice = "DirectML-LICENSE.txt"
            else:
                digest = hashlib.sha256(self.members[filename]).hexdigest()
                label = "ORT" if stem == "onnxruntime" else "GenAI"
                self.write(f"vendor/{stem}-patched/SHA256SUMS",
                           f"# Console-validated patched {filename} ({label} {version}).\n{digest}  {filename}\n".encode())
                notice = f"{stem}-LICENSE"
            license_content = (ROOT / "LICENSES" / notice).read_bytes()
            self.members["licenses/" + notice] = license_content
            self.write("LICENSES/" + notice, license_content)
        self.write("uwp/packages.config", (config + "</packages>").encode())
        self.pack()

    def test_every_full_path_and_the_final_archive_have_independent_checksums(self):
        doc = msix_sbom.generate(self.path, CREATED)
        package, = doc["packages"]
        self.assertEqual((package["name"], package["versionInfo"]), ("Product.Name", "1.2.3.42"))
        self.assertEqual(package["licenseDeclared"], "NOASSERTION")
        self.assertEqual(package["checksums"][0]["checksumValue"], hashlib.sha256(self.path.read_bytes()).hexdigest())
        records = {f["fileName"]: f for f in doc["files"]}
        self.assertEqual(set(records), {"./" + name for name in self.members})
        for name, data in self.members.items():
            record = records["./" + name]
            self.assertEqual(
                {c["algorithm"]: c["checksumValue"] for c in record["checksums"]},
                {"SHA1": hashlib.sha1(data).hexdigest(), "SHA256": hashlib.sha256(data).hexdigest()},
            )
            self.assertIn(msix_sbom.relationship(package["SPDXID"], "CONTAINS", record["SPDXID"]), doc["relationships"])
        sha1s = "".join(sorted(hashlib.sha1(data).hexdigest() for data in self.members.values()))
        self.assertEqual(package["packageVerificationCode"]["packageVerificationCodeValue"],
                         hashlib.sha1(sha1s.encode()).hexdigest())
        self.assertNotEqual(records["./a/helper.dll"]["SPDXID"], records["./b/helper.dll"]["SPDXID"])
        self.pack(ZIP_DEFLATED)
        repacked = msix_sbom.generate(self.path, CREATED)
        self.assertEqual(repacked["files"], doc["files"])
        self.assertNotEqual(repacked["packages"][0]["checksums"], package["checksums"])
        self.assertNotEqual(repacked["documentNamespace"], doc["documentNamespace"])

    def test_source_date_epoch_reproduces_identical_sidecars(self):
        env = {**os.environ, "SOURCE_DATE_EPOCH": "1770000000"}
        command = [sys.executable, str(ROOT / "scripts/msix_sbom.py"), str(self.path)]
        subprocess.run(command, check=True, capture_output=True, env=env)
        output = self.path.with_suffix(".msix.spdx.json")
        first = output.read_bytes()
        output.unlink()
        subprocess.run(command, check=True, capture_output=True, env=env)
        self.assertEqual(output.read_bytes(), first)

    def test_ambiguous_archive_paths_fail(self):
        for alias in ("A/helper.dll", "a/../b.dll", "a\\helper.dll", "c:helper.dll",
                      "/abs.dll", "a//x.dll", "a", "B/", "a/helper.dll/child"):
            with self.subTest(alias=alias):
                self.members[alias] = b"ambiguous"
                self.pack()
                with self.assertRaisesRegex(ValueError, "ambiguous"):
                    msix_sbom.generate(self.path, CREATED)
                del self.members[alias]
        self.pack()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with ZipFile(self.path, "a") as archive:
                archive.writestr("a/helper.dll", b"duplicate member")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            msix_sbom.generate(self.path, CREATED)

    @unittest.skipIf(os.name == "nt", "open-file replacement requires POSIX rename semantics")
    def test_renamed_package_and_in_place_changes_cannot_mix_identities(self):
        original_digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        replacement = self.root / "replacement.msix"
        with ZipFile(replacement, "w") as archive:
            archive.writestr("AppxManifest.xml", MANIFEST.replace(b"Product.Name", b"Replacement"))
        read_identity = msix_sbom.manifest_identity

        def rename_then_read(manifest, path):
            replacement.replace(path)
            return read_identity(manifest, path)

        with patch.object(msix_sbom, "manifest_identity", side_effect=rename_then_read):
            doc = msix_sbom.generate(self.path, CREATED)
        self.assertEqual(doc["packages"][0]["name"], "Product.Name")
        self.assertEqual(doc["packages"][0]["checksums"][0]["checksumValue"], original_digest)

        def change_then_read(manifest, path):
            with path.open("ab") as stream:
                stream.write(b"post-inventory change")
            return read_identity(manifest, path)

        with patch.object(msix_sbom, "manifest_identity", side_effect=change_then_read):
            with self.assertRaisesRegex(ValueError, "changed while generating"):
                msix_sbom.generate(self.path, CREATED)

    @unittest.skipUnless(shutil.which("bash") and shutil.which("jq"), "requires bash and jq from the attestation runner")
    def test_trusted_workflow_rejects_a_different_or_missing_subject(self):
        workflow = (ROOT / ".github/workflows/build-uwp.yml").read_text()
        check = workflow.split("      - name: Check inventory subject against the signed package", 1)[1]
        check = textwrap.dedent(check.split("        run: |\n", 1)[1].split("\n      - ", 1)[0])
        document = msix_sbom.generate(self.path, CREATED)
        sidecar = self.path.with_suffix(".msix.spdx.json")
        env = {**os.environ, "MSIX": str(self.path)}

        def verified():
            return subprocess.run(["bash", "-eo", "pipefail", "-c", check], env=env, capture_output=True).returncode == 0

        self.assertFalse(verified())
        sidecar.write_text(json.dumps(document))
        self.assertTrue(verified())
        document["packages"][0]["checksums"][0]["checksumValue"] = "0" * 64
        sidecar.write_text(json.dumps(document))
        self.assertFalse(verified())

    def test_component_versions_and_licenses_require_matching_build_inputs(self):
        self.add_build_inputs()
        doc = msix_sbom.generate(self.path, CREATED, self.root)
        components = {p["SPDXID"]: p for p in doc["packages"][1:]}
        self.assertEqual(components["SPDXRef-onnxruntime"]["versionInfo"], "1.2.3")
        self.assertEqual(components["SPDXRef-onnxruntime-genai"]["licenseDeclared"], "MIT")
        directml = components["SPDXRef-DirectML"]["licenseDeclared"]
        extracted, = doc["hasExtractedLicensingInfos"]
        self.assertEqual(extracted["licenseId"], directml)
        self.assertEqual(extracted["extractedText"], (self.root / "LICENSES/DirectML-LICENSE.txt").read_bytes().decode("utf-8-sig"))
        self.assertEqual(len(msix_sbom.generate(self.path, CREATED)["packages"]), 1)
        for filename in ("onnxruntime.dll", "onnxruntime-genai.dll", "DirectML.dll", "licenses/onnxruntime-LICENSE"):
            with self.subTest(filename=filename):
                original = self.members[filename]
                self.members[filename] = b"changed bytes"
                self.pack()
                with self.assertRaisesRegex(ValueError, "does not match"):
                    msix_sbom.generate(self.path, CREATED, self.root)
                self.members[filename] = original
        self.pack()
        pins = self.root / "uwp/packages.config"
        pins.write_bytes(pins.read_bytes().replace(b"1.2.3", b"1.2"))
        uncertain = msix_sbom.generate(self.path, CREATED, self.root)
        ort = next(p for p in uncertain["packages"] if p["SPDXID"] == "SPDXRef-onnxruntime")
        self.assertNotIn("versionInfo", ort)
        self.assertIn("Build restore pin: Microsoft.ML.OnnxRuntime.DirectML 1.2.", ort["sourceInfo"])


if __name__ == "__main__":
    unittest.main()
