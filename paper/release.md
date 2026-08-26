# Research release runbook

This runbook describes the release boundary for the current research release and follows the
two-phase Zenodo workflow used by the project.

## Metadata SSOT

Edit `release.toml` only. Run:

```bash
python3 scripts/sync-release-metadata.py
python3 scripts/check-release-metadata.py
```

The DOI fields are intentionally empty before the first deposit. After Zenodo
reserves a DOI, write both DOI values to `release.toml`, run the sync command,
and commit those values before tagging.

## Before tagging

Run from a clean checkout and verify:

```bash
python3 scripts/generate-benchmark-summary.py --check
python3 scripts/build-research-package.py --check
python3 -m unittest tests/test_research_package.py
python3 scripts/check-coherence.py
```

Every claim marked `publication-grade` must have a valid benchmark sidecar.
Claims without thermal or power evidence remain `historical` or
`reproducible` and must not be described as sustained production throughput or
energy measurements.

## Archive contents

The GitHub release and Zenodo record must contain the same tagged revision,
`paper/paper.md`, `paper/claims.json`, `paper/research-manifest.json`, the
generated table and figures, and the raw evidence required by the claim
registry. Each Zenodo version must contain one versioned research archive and
the companion printable PDF (`xllama-research-VERSION.pdf`). New-version drafts inherit files from the previous version, so the
publisher removes inherited files before uploading. Update `CITATION.cff` with
the Zenodo DOI only after the archive has been created.

The release does not certify the Xbox Store package, retail behaviour, customer
UAT, or cross-platform performance.

## Two-phase publication

Reserve before the release commit:

```bash
python3 scripts/zenodo-deposit.py --reserve-only
```

The command loads `ZENODO_TOKEN` from `~/.config/xllama/zenodo.env` (mode
`0600`), or from the environment if already exported.

After committing and pushing `research-v1.0.1`, upload and publish the tagged
archive:

```bash
ZENODO_TOKEN=... python3 scripts/zenodo-deposit.py --upload-from-state
```

The GitHub release is created by `.github/workflows/release-research.yml`.
Do not enable a GitHub-to-Zenodo webhook for the same repository: the API-driven
deposit is canonical and prevents duplicate records.
