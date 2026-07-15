# vendor/onnxruntime-patched/

Patched **`onnxruntime.dll`** (DirectML) — ORT core with the AppContainer guard
on `ValidateExternalDataPath`'s `weakly_canonical()` walk
(`patches/onnxruntime-extdata-appcontainer.patch`). Unblocks loading ONNX models
with external `.onnx.data` >2 GB on the Xbox (the merge workaround caps at the
2 GB protobuf ceiling). See `docs/uwp-constraints.md §8` and
`docs/fp16-extdata-runbook.md`.

The DLL binary lives at `win-x64/onnxruntime.dll` and is **gitignored** — only
this README is tracked.

## Producing the DLL

- CI (preferred): run the **`build-uwp-ort-patched`** workflow (dispatch-only).
  It builds this DLL + the #2280 GenAI DLL from source and uploads both the MSIX
  and `onnxruntime-patched-dll` as artifacts. NB: a full ORT DirectML source
  build takes 1-3 h.
- Local: `./scripts/vendor-ort-extdata-patch.ps1 -Build` (VS2022 + Python + CMake;
  runs `tools/ci_build/build.py --use_dml --build_shared_lib` against ORT tag
  `v<version>` from `uwp/packages.config`).

Drop a console-validated DLL here at `win-x64/onnxruntime.dll` and
`./scripts/vendor-ort-extdata-patch.ps1` (no `-Build`) installs it over the
NuGet copy for a normal `build-uwp` run.
