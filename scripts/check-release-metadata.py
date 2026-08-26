#!/usr/bin/env python3
"""Fail closed when release-facing metadata drifts from release.toml."""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--require-doi", action="store_true", help="require both Zenodo DOI values")
    args = parser.parse_args()
    release = tomllib.loads((ROOT / "release.toml").read_text(encoding="utf-8"))["release"]
    version = str(release["version"])
    errors: list[str] = []
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    zenodo = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    manifest = json.loads((ROOT / "paper/research-manifest.json").read_text(encoding="utf-8"))

    if f'version: "{version}"' not in citation:
        errors.append("CITATION.cff version mismatch")
    if zenodo.get("version") != version:
        errors.append(".zenodo.json version mismatch")
    if manifest.get("report_version") != version:
        errors.append("research manifest version mismatch")
    if zenodo.get("title") != release["title"]:
        errors.append(".zenodo.json title mismatch")
    if zenodo.get("license") != release["license"]:
        errors.append(".zenodo.json license mismatch")
    if zenodo.get("keywords") != release["keywords"]:
        errors.append(".zenodo.json keywords mismatch")
    if not zenodo.get("related_identifiers") or zenodo["related_identifiers"][0].get("identifier") != release["repo_url"]:
        errors.append(".zenodo.json repository relation mismatch")
    if "<!-- XLLAMA_DOI_START -->" not in readme or "<!-- XLLAMA_DOI_END -->" not in readme:
        errors.append("README DOI marker missing")

    version_doi = str(release.get("version_doi", ""))
    concept_doi = str(release.get("concept_doi", ""))
    if args.require_doi and (not concept_doi or not version_doi):
        errors.append("release.toml must contain concept_doi and version_doi for a tagged release")
    cff_doi = re.search(r'^doi:\s*"([^"]+)"$', citation, re.MULTILINE)
    if version_doi and (not cff_doi or cff_doi.group(1) != version_doi):
        errors.append("CITATION.cff DOI mismatch")
    if not version_doi and cff_doi:
        errors.append("CITATION.cff contains a DOI but release.toml does not")

    if errors:
        print("release metadata check failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    print(f"OK release metadata: research-v{version} (DOI={'set' if version_doi else 'pending'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
