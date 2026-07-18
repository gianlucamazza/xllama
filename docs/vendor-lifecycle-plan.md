# Vendor runtime lifecycle

Active removal conditions for the two patched runtime DLLs shipped by the
unified UWP package. Completed release and demo work belongs in `CHANGELOG.md`.

## Current pins

| Runtime                             | NuGet pin | Local overlay                                                     | Tracker        | Drop condition                                                      |
| ----------------------------------- | --------- | ----------------------------------------------------------------- | -------------- | ------------------------------------------------------------------- |
| ORT GenAI DirectML                  | 0.14.1    | `vendor/onnxruntime-genai-patched/`                               | xllama #84     | A NuGet release includes GenAI #2280                                |
| ORT DirectML                        | 1.24.4    | `vendor/onnxruntime-patched/`                                     | xllama #85/#86 | Required AppContainer fixes reach NuGet                             |
| ORT DirectML (metacommands opt-out) | 1.24.4    | same overlay, `patches/onnxruntime-dml-metacommands-optout.patch` | xllama #91     | Experiment fails, or upstream ships an equivalent knob / driver fix |

`build-uwp.yml` verifies hashes and fails closed if a required cached DLL is
missing. Source rebuild workflows are maintenance lanes, not normal shipping
builds.

## Active upstream work

- **GenAI #2280:** merged upstream; still absent from NuGet 0.14.1. Keep
  PatchedGenAI until the package version changes and the XAML + DirectML device
  creation smoke passes.
- **ORT #28509:** AppContainer canonical-path fix merged upstream; still absent
  from ORT 1.24.4.
- **ORT #29732:** 16 MB `ReadFileIntoBuffer` chunk remains the second PatchedOrt
  requirement.
- **GenAI #2300 / ORT #29739:** tooling and driver investigation for wrong Xbox
  DML text logits. These do not change the #91 product gate by themselves.
- **Local #91 experiment:** `ep.dml.disable_metacommands` vendored knob to test
  whether the broken attention is a driver metacommand
  ([dml-metacommands-runbook.md](./dml-metacommands-runbook.md)); run on
  console 2026-07-19: **FAIL**, knob-on logits bit-identical to stock — patch
  meets its drop condition.

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
