# Vendor runtime lifecycle

Active removal conditions for the two patched runtime DLLs shipped by the
unified UWP package. Completed release and demo work belongs in `CHANGELOG.md`.

## Current pins

| Runtime            | NuGet pin | Local overlay                       | Tracker    | Drop condition                          |
| ------------------ | --------- | ----------------------------------- | ---------- | --------------------------------------- |
| ORT GenAI DirectML | 0.14.1    | `vendor/onnxruntime-genai-patched/` | xllama #85 | A NuGet release includes GenAI #2280    |
| ORT DirectML       | 1.24.4    | `vendor/onnxruntime-patched/`       | xllama #85 | Required AppContainer fixes reach NuGet |

On 2026-11-27 the NuGet package indices for these two DirectML variants still ended at the shipping pins. The separate Microsoft.AI.DirectML package also matched the project's 1.15.4 pin. Verify the exact package ID when comparing releases. [GenAI DirectML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntimeGenAI.DirectML) · [ORT DirectML](https://www.nuget.org/packages/Microsoft.ML.OnnxRuntime.DirectML) · [DirectML](https://www.nuget.org/packages/Microsoft.AI.DirectML).

`build-uwp.yml` verifies hashes and fails closed if a required cached DLL is
missing. Source rebuild workflows are maintenance lanes, not normal shipping
builds.

## Rebuild source

A source rebuild binds its checkout to the source commit Microsoft recorded in
the official NuGet nuspec, plus one reviewed patch, before it builds.

| Runtime            | Source pin (nuspec, checked 2026-11-27)           | Verifier                          |
| ------------------ | ------------------------------------------------- | --------------------------------- |
| ORT GenAI DirectML | `a30f479` (`<releaseNotes>`, branch `rel-0.14.1`) | `scripts/prepare_genai_source.py` |
| ORT DirectML       | `2d92497` (`<repository commit=...>`)             | `scripts/prepare_ort_source.py`   |

`github.com/microsoft/onnxruntime` and `github.com/microsoft/onnxruntime-genai`
return 404 since about 2026-11-17 (xllama #646), so the default upstream fetch
fails closed. `vendor-genai-dml-patch.ps1 -SourceUrl` takes a mirror. A mirror
or fork is acceptable only when its checkout passes the verifier unchanged:
pinned commit, pinned tree or pinned tree + reviewed patch, no untracked,
ignored or hidden local files. Until a rebuild passes that check and the
console acceptance, keep the existing DLLs and `SHA256SUMS`.

## Active upstream work

- **GenAI #2280:** merged upstream; still absent from NuGet 0.14.1. Keep
  PatchedGenAI until the package version changes and the XAML + DirectML device
  creation smoke passes.
- **ORT #28509:** AppContainer canonical-path fix merged upstream; still absent
  from ORT 1.24.4.
- **ORT #29732:** 16 MB `ReadFileIntoBuffer` chunk remains the second PatchedOrt
  requirement.
- **GenAI #2300 / ORT #29739:** tooling and driver investigation opened while
  #91 was attributed to the attention kernels. Superseded by the RMSNorm
  finding (`dml-rmsnorm-fix-runbook.md`): #29739 is to be updated with the
  `(Skip)SimplifiedLayerNormalization` root cause and single-op repro (#111 —
  fork/analyze/fix the DML operator where needed, same vendored pattern as the
  extdata and #2280 patches).
- **Local #91 experiment:** `ep.dml.disable_metacommands` vendored knob to test
  whether the broken attention is a driver metacommand
  ([dml-metacommands-runbook.md](./dml-metacommands-runbook.md)); run on
  console 2026-07-19: **FAIL**, knob-on logits bit-identical to stock — patch
  dropped per its drop condition; next angle is plan B (decomposed-attention
  text model, graph surgery).

## Refresh procedure

```bash
./scripts/check-vendor-nuget-status.sh
```

When a newer package appears:

1. update one pin at a time;
2. build without the corresponding overlay;
3. run host/CI tests;
4. validate XAML startup, external-data loading and the full console suite;
5. remove the patch, cached DLL, hash and special build flag in the same change.

## Invariants

- Never enable DML text because a DLL loads; require target-device logit parity.
- Never remove PatchedOrt after only one of its two fixes ships.
- Never silently fall back to an unverified runtime DLL.
- Keep issue trackers, rather than this document, as the live status source.
