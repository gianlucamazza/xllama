#!/usr/bin/env python3
"""Synchronize release.toml into citation and Zenodo metadata files."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_release() -> dict[str, object]:
    return tomllib.loads((ROOT / "release.toml").read_text(encoding="utf-8"))["release"]


def replace_one(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise SystemExit(f"cannot update {label}")
    return updated


def main() -> int:
    release = load_release()
    version = str(release["version"])
    date = str(release.get("date", "2026-08-26"))
    concept = str(release.get("concept_doi", ""))
    version_doi = str(release.get("version_doi", ""))
    title = str(release["title"])
    repo = str(release["repo_url"])
    description = str(release["description"])
    keywords = list(release["keywords"])
    creator = {
        "name": f"{release['author_family']}, {release['author_given']}",
        "affiliation": "",
        "orcid": str(release["author_orcid"]),
    }

    citation_path = ROOT / "CITATION.cff"
    citation = citation_path.read_text(encoding="utf-8")
    citation = replace_one(citation, r'^version:\s*"[^"]+"$', f'version: "{version}"', "CFF version")
    citation = replace_one(citation, r'^date-released:\s*"?[^"]+"?$', f'date-released: "{date}"', "CFF date")
    if version_doi:
        if re.search(r'^doi:', citation, re.MULTILINE):
            citation = replace_one(citation, r'^doi:\s*"[^"]+"$', f'doi: "{version_doi}"', "CFF DOI")
        else:
            citation = citation.rstrip() + f'\ndoi: "{version_doi}"\n'
    citation_path.write_text(citation, encoding="utf-8")

    zenodo_path = ROOT / ".zenodo.json"
    zenodo = json.loads(zenodo_path.read_text(encoding="utf-8"))
    zenodo.update({
        "title": title,
        "description": description,
        "version": version,
        "publication_date": date,
        "license": str(release["license"]),
        "keywords": keywords,
        "creators": [creator],
        "related_identifiers": [{"identifier": repo, "relation": "isSupplementTo", "scheme": "url"}],
    })
    zenodo_path.write_text(json.dumps(zenodo, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest_path = ROOT / "paper" / "research-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["report_version"] = version
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    readme_path = ROOT / "README.md"
    readme = readme_path.read_text(encoding="utf-8")
    badge = (
        f"[![DOI](https://zenodo.org/badge/DOI/{concept}.svg)](https://doi.org/{concept})"
        if concept
        else "<!-- DOI badge will be added after the first Zenodo deposit. -->"
    )
    readme = re.sub(r'<!-- XLLAMA_DOI_START -->.*?<!-- XLLAMA_DOI_END -->',
                    f'<!-- XLLAMA_DOI_START -->\n{badge}\n<!-- XLLAMA_DOI_END -->',
                    readme, count=1, flags=re.DOTALL)
    readme = re.sub(r'(version\s*=\s*\{)[^}]+(\},\n\s*doi\s*=\s*\{)[^}]+(\})',
                    rf'\g<1>{version}\g<2>{version_doi or "pending"}\g<3>', readme, count=1)
    readme_path.write_text(readme, encoding="utf-8")
    print(f"OK release metadata synced: research-v{version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
