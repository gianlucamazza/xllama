# xLlama research package

This directory contains the citable research report and the machine-readable
provenance needed to reproduce its claims. The report is scoped to Xbox Series
S in Dev Mode; it is not a retail certification report.

## Rebuild and validate

```bash
python3 scripts/build-research-package.py
python3 scripts/build-research-package.py --check
```

The script reads the existing benchmark SSOT and raw evidence. It never copies
or rewrites benchmark CSVs. `claims.json` maps quantitative claims to raw
results and selectors. `research-manifest.json` records the source commit and
the evidence boundary for the package. The SVG figure is generated from the
same claim registry and is checked byte-for-byte in CI.
The generated evidence table is stored at `paper/generated/benchmarks.md`.

The historical narrative in [`../docs/technical-report.md`](../docs/technical-report.md)
remains frozen as the July 2026 v1.0 snapshot.

The tag and archive procedure is documented in [`release.md`](release.md).
