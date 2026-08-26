#!/usr/bin/env python3
"""Reserve and publish the xLlama research deposit through Zenodo's API.

Reserve before committing/tagging; upload-from-state only after the tag exists.
The local state file contains no token and is ignored by git.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / ".release_state.json"


def release() -> dict[str, object]:
    return tomllib.loads((ROOT / "release.toml").read_text(encoding="utf-8"))["release"]


def load_user_env() -> None:
    """Load the xLlama user credential file without importing other secrets."""
    path = Path.home() / ".config" / "xllama" / "zenodo.env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "ZENODO_TOKEN" and value.strip():
            os.environ.setdefault("ZENODO_TOKEN", value.strip().strip('"').strip("'"))


def request(url: str, token: str, method: str = "GET", payload: object | None = None, data: bytes | None = None):
    body = json.dumps(payload).encode() if payload is not None else data
    headers = {"Authorization": f"Bearer {token}"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            raw = response.read()
            return json.loads(raw) if raw else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise SystemExit(f"Zenodo API {exc.code}: {detail}") from exc


def save(state: dict[str, object]) -> None:
    STATE.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def reserve(base: str, token: str) -> int:
    meta = json.loads((ROOT / ".zenodo.json").read_text(encoding="utf-8"))
    meta["prereserve_doi"] = True
    current = release()
    previous = str(current.get("version_doi", ""))
    if previous:
        parent_id = previous.rsplit(".", 1)[-1]
        created = request(f"{base}/api/deposit/depositions/{parent_id}/actions/newversion", token, "POST")
        draft = request(str(created["links"]["latest_draft"]), token)
        dep_id = int(draft["id"])
        bucket = str(draft["links"]["bucket"])
        # New-version drafts inherit old files and metadata; replace metadata
        # and let Zenodo allocate a fresh version DOI.
        response = request(f"{base}/api/deposit/depositions/{dep_id}", token, "PUT", {"metadata": meta})
        draft = response
    else:
        response = request(f"{base}/api/deposit/depositions", token, "POST", {"metadata": meta})
        dep_id = int(response["id"])
        bucket = str(response["links"]["bucket"])
        draft = request(f"{base}/api/deposit/depositions/{dep_id}", token)
    reserved = str(draft.get("metadata", {}).get("prereserve_doi", {}).get("doi", ""))
    if not reserved:
        reserved = f"10.5281/zenodo.{dep_id}"
    concept = str(draft.get("conceptdoi", ""))
    state = {"version": current["version"], "tag": f"{current['tag_prefix']}{current['version']}", "deposit_id": dep_id, "bucket_url": bucket, "reserved_doi": reserved, "concept_doi": concept, "base_url": base, "zenodo_reserved": True, "zenodo_published": False}
    save(state)
    print(f"Reserved DOI: {reserved}")
    if concept:
        print(f"Concept DOI: {concept}")
    print("Copy the DOI values into release.toml, sync metadata, then commit and tag.")
    return 0


def archive(ref: str, version: str) -> Path:
    output = ROOT / ".release-upload"
    output.mkdir(exist_ok=True)
    path = output / f"xllama-research-{version}.tar.gz"
    subprocess.run(["python3", "scripts/build-release-bundle.py", "--ref", ref, "--output", str(output)], cwd=ROOT, check=True)
    return path


def paper_pdf(version: str) -> Path:
    output = ROOT / ".release-upload" / f"xllama-research-{version}.pdf"
    subprocess.run(
        ["python3", "scripts/build-paper-pdf.py", "--output", str(output)],
        cwd=ROOT,
        check=True,
    )
    return output


def clear_draft_files(base: str, deposit_id: int, token: str) -> int:
    """Remove files inherited by a new-version draft before uploading."""
    draft = request(f"{base}/api/deposit/depositions/{deposit_id}", token)
    files = draft.get("files", [])
    for file in files:
        file_url = str(file.get("links", {}).get("self", ""))
        if not file_url:
            file_id = file.get("id")
            if not file_id:
                raise SystemExit(f"Zenodo draft file has no deletion URL: {file!r}")
            file_url = f"{base}/api/deposit/depositions/{deposit_id}/files/{file_id}"
        request(file_url, token, "DELETE")
    print(f"Removed {len(files)} inherited file(s) from Zenodo draft")
    return len(files)


def publish(base: str, token: str) -> int:
    state = json.loads(STATE.read_text(encoding="utf-8"))
    if state.get("zenodo_published"):
        print("Zenodo deposit already published")
        return 0
    path = archive(str(state["tag"]), str(state["version"]))
    pdf = paper_pdf(str(state["version"]))
    clear_draft_files(base, int(state["deposit_id"]), token)
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"}
    for artifact in (path, pdf):
        req = urllib.request.Request(f"{state['bucket_url']}/{artifact.name}", data=artifact.read_bytes(), headers=headers, method="PUT")
        with urllib.request.urlopen(req, timeout=180):
            pass
    response = request(f"{base}/api/deposit/depositions/{state['deposit_id']}/actions/publish", token, "POST")
    state["zenodo_published"] = True
    state["published_doi"] = response.get("doi", state["reserved_doi"])
    save(state)
    print(f"Published DOI: {state['published_doi']}")
    return 0


def main() -> int:
    load_user_env()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--reserve-only", action="store_true")
    mode.add_argument("--upload-from-state", action="store_true")
    parser.add_argument("--base-url", default=os.environ.get("ZENODO_BASE", "https://zenodo.org"))
    args = parser.parse_args()
    token = os.environ.get("ZENODO_TOKEN", "")
    if not token:
        raise SystemExit("ZENODO_TOKEN is required")
    return reserve(args.base_url.rstrip("/"), token) if args.reserve_only else publish(args.base_url.rstrip("/"), token)


if __name__ == "__main__":
    raise SystemExit(main())
