#!/usr/bin/env python3
"""Inventory the final MSIX, including signing metadata, as SPDX 2.3 JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from package_identity import IdentityError, manifest_identity

NOASSERTION = "NOASSERTION"
MSIX_ID = "SPDXRef-MSIX"
DOCUMENT_ID = "SPDXRef-DOCUMENT"


def sha256_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256_file(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def archive_files(archive):
    files = {}
    tree = {}
    explicit_paths = set()
    for entry in sorted(archive.infolist(), key=lambda item: item.filename):
        name = entry.filename.removesuffix("/")
        parts = name.split("/")
        if (
            entry.orig_filename != entry.filename
            or name.casefold() in explicit_paths
            or any(part in ("", ".", "..") for part in parts)
            or "\\" in name
            or ":" in name
        ):
            raise ValueError(f"ambiguous MSIX file path: {name!r}")
        explicit_paths.add(name.casefold())
        for length in range(1, len(parts) + 1):
            prefix = "/".join(parts[:length])
            node = (prefix, length < len(parts) or entry.is_dir())
            if tree.setdefault(prefix.casefold(), node) != node:
                raise ValueError(f"ambiguous MSIX file path: {name!r}")
        if entry.is_dir():
            continue
        sha1 = hashlib.sha1(usedforsecurity=False)  # Required for the SPDX verification code.
        sha256 = hashlib.sha256()
        with archive.open(entry) as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                sha1.update(chunk)
                sha256.update(chunk)
        record = {
            "SPDXID": f"SPDXRef-File-{len(files) + 1}",
            "fileName": f"./{name}",
            "checksums": [
                {"algorithm": "SHA1", "checksumValue": sha1.hexdigest()},
                {"algorithm": "SHA256", "checksumValue": sha256.hexdigest()},
            ],
            "licenseConcluded": NOASSERTION,
            "licenseInfoInFiles": [NOASSERTION],
            "copyrightText": NOASSERTION,
        }
        if PurePosixPath(name).suffix.lower() in {".dll", ".exe", ".winmd"}:
            record["fileTypes"] = ["BINARY"]
        files[name] = record
    return files


def package(identifier, name, version, filename, digest, files):
    # SPDX 2.3 section 7.9 hashes the sorted SHA-1 hex strings of the package's files.
    sorted_sha1 = sorted(f["checksums"][0]["checksumValue"] for f in files)
    verification = hashlib.sha1("".join(sorted_sha1).encode("ascii"), usedforsecurity=False)
    return {
        "SPDXID": identifier,
        "name": name,
        "versionInfo": version,
        "packageFileName": filename,
        "downloadLocation": NOASSERTION,
        "checksums": [{"algorithm": "SHA256", "checksumValue": digest}],
        "filesAnalyzed": True,
        "packageVerificationCode": {"packageVerificationCodeValue": verification.hexdigest()},
        "licenseConcluded": NOASSERTION,
        "licenseDeclared": NOASSERTION,
        "licenseInfoFromFiles": [NOASSERTION],
        "copyrightText": NOASSERTION,
    }


def relationship(source, kind, target):
    return {"spdxElementId": source, "relationshipType": kind, "relatedSpdxElement": target}


def add_shipping_components(document, files, root):
    versions = {p.attrib["id"]: p.attrib["version"] for p in ET.parse(root / "uwp/packages.config").getroot()}
    # The packaging step replaces both NuGet ORT DLLs before MSBuild. Compare
    # those files against the vendor pins, and DirectML against its restored DLL.
    components = (
        ("onnxruntime", "Microsoft.ML.OnnxRuntime.DirectML", "onnxruntime-LICENSE"),
        ("onnxruntime-genai", "Microsoft.ML.OnnxRuntimeGenAI.DirectML", "onnxruntime-genai-LICENSE"),
        ("DirectML", "Microsoft.AI.DirectML", "DirectML-LICENSE.txt"),
    )
    for stem, nuget, notice in components:
        filename = f"{stem}.dll"
        version = versions[nuget]
        record = files[filename]
        digest = record["checksums"][1]["checksumValue"]
        version_confirmed = True
        if stem == "DirectML":
            relative = f"uwp/packages/{nuget}.{version}/bin/x64-win/{filename}"
            expected = sha256_file(root / relative)
            name = "Microsoft DirectML x64 runtime"
        else:
            relative = f"vendor/{stem}-patched/SHA256SUMS"
            lines = (root / relative).read_text(encoding="utf-8").splitlines()
            matches = [line.split()[0] for line in lines if line.split()[1:] == [filename]]
            if len(matches) != 1:
                raise ValueError(f"expected one pin for {filename} in {relative}")
            component_label = "ORT" if stem == "onnxruntime" else "GenAI"
            declared_version = re.compile(rf"\b{component_label} {re.escape(version)}(?=[ )])")
            version_confirmed = any(
                declared_version.search(line) for line in lines if line.startswith("# Console-validated")
            )
            expected = matches[0]
            name = f"xLlama-patched {stem}"
        if digest != expected:
            raise ValueError(f"{filename} does not match {relative}")
        notice_bytes = (root / "LICENSES" / notice).read_bytes()
        if files[f"licenses/{notice}"]["checksums"][1]["checksumValue"] != hashlib.sha256(notice_bytes).hexdigest():
            raise ValueError(f"packaged notice does not match LICENSES/{notice}")
        identifier = f"SPDXRef-{stem}"
        component = package(identifier, name, version, f"./{filename}", digest, [record])
        if not version_confirmed:
            # A restore version alone cannot establish the base of a cached patched DLL.
            del component["versionInfo"]
        component["sourceInfo"] = (
            f"Shipped binary matches {relative}. Build restore pin: {nuget} {version}. "
            f"Packaged license: ./licenses/{notice} (LICENSES/SOURCES.md)."
        )
        notice_text = notice_bytes.decode("utf-8-sig")
        if stem == "DirectML":
            component["licenseDeclared"] = "LicenseRef-Microsoft-DirectML"
            document["hasExtractedLicensingInfos"] = [
                {"licenseId": component["licenseDeclared"], "extractedText": notice_text}
            ]
        elif "MIT License" in notice_text and "Permission is hereby granted" in notice_text:
            component["licenseDeclared"] = "MIT"
        document["packages"].append(component)
        document["relationships"].append(relationship(MSIX_ID, "CONTAINS", identifier))
        document["relationships"].append(relationship(identifier, "CONTAINS", record["SPDXID"]))


def generate(path, created, build_inputs=None):
    with path.open("rb") as stream:
        digest = sha256_stream(stream)
        stream.seek(0)
        with zipfile.ZipFile(stream) as archive:
            files = archive_files(archive)
            name, version = manifest_identity(archive.read("AppxManifest.xml"), path)
        stream.seek(0)
        if sha256_stream(stream) != digest:
            raise ValueError("MSIX changed while generating its inventory")
    msix = package(MSIX_ID, name, version, path.name, digest, files.values())
    msix["primaryPackagePurpose"] = "APPLICATION"
    document = {
        "spdxVersion": "SPDX-2.3",
        "dataLicense": "CC0-1.0",
        "SPDXID": DOCUMENT_ID,
        "name": path.name,
        "creationInfo": {"created": created, "creators": ["Tool: xllama-msix-sbom"]},
        "packages": [msix],
        "files": list(files.values()),
        "relationships": [relationship(DOCUMENT_ID, "DESCRIBES", MSIX_ID)]
        + [relationship(MSIX_ID, "CONTAINS", f["SPDXID"]) for f in files.values()],
    }
    if build_inputs is not None:
        add_shipping_components(document, files, build_inputs)
    # SPDX requires distinct namespaces for different documents, including
    # inventories of the same package produced with different build evidence.
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    document["documentNamespace"] = "https://spdx.org/spdxdocs/xllama-" + hashlib.sha256(canonical).hexdigest()
    return document


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("msix", type=Path)
    parser.add_argument("--build-inputs", type=Path, help="matching source checkout with restored DirectML NuGet")
    args = parser.parse_args()
    try:
        epoch = os.environ.get("SOURCE_DATE_EPOCH")
        if epoch is None:
            created = datetime.now(timezone.utc)
        else:
            created = datetime.fromtimestamp(int(epoch), timezone.utc)
        document = generate(args.msix, created.strftime("%Y-%m-%dT%H:%M:%SZ"), args.build_inputs)
        output = args.msix.with_suffix(args.msix.suffix + ".spdx.json")
        with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as temporary:
            temp_path = Path(temporary.name)
        try:
            temp_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
            temp_path.replace(output)
        finally:
            temp_path.unlink(missing_ok=True)
    except (OSError, ValueError, KeyError, ET.ParseError, zipfile.BadZipFile, IdentityError) as error:
        print(f"MSIX SBOM: {error}", file=sys.stderr)
        return 1
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
