# vendor/onnxruntime-patched/

Patched **`onnxruntime.dll`** (DirectML) — ORT core with the AppContainer
external-data fixes (`patches/onnxruntime-extdata-appcontainer.patch`):

1. `ValidateExternalDataPath` — guard `weakly_canonical()` (error_code + lexical fallback)
2. `ReadFileIntoBuffer` — chunk 1 GB → 16 MB (errcode 1450)

Unblocks loading ONNX models with external `.onnx.data` >2 GB on Xbox
AppContainer. See `docs/uwp-constraints.md §8` and
`docs/fp16-extdata-runbook.md`.

The DLL binary lives at `win-x64/onnxruntime.dll` and is **gitignored** — only
this README + `SHA256SUMS` are tracked. Shipping CI downloads the pinned
console-validated build from the [`vendor-dlls-v1`](https://github.com/gianlucamazza/xllama/releases/tag/vendor-dlls-v1)
release (hash must match `SHA256SUMS`).

## Producing / refreshing the DLL

- **CI rebuild (slow, 1–3 h):** `gh workflow run build-uwp-ort-patched.yml`
  uploads `onnxruntime-patched-dll`. After console re-validation, update the
  `vendor-dlls-v1` release asset and `SHA256SUMS`.
- **Local:** `./scripts/vendor-ort-extdata-patch.ps1 -Build` (VS2022 + Python +
  CMake; ORT tag `v<version>` from `uwp/packages.config`).

## Install over NuGet (local or CI)

```powershell
# Cached vendor path (after download or -Build):
./scripts/vendor-ort-extdata-patch.ps1

# Or via packaging:
./scripts/build-uwp.ps1 -Backend unified -PatchedGenAI -PatchedOrt
```

Pin: ORT DirectML **1.24.4** + the two AppContainer fixes.

Upstream status (as of 2026-07-16):

- **weakly_canonical / AppContainer path:** related fix on ORT `main` via
  [#28509](https://github.com/microsoft/onnxruntime/pull/28509) (not in 1.24.4).
- **ReadFile 16 MB chunk (errcode 1450):** still vendor-only; not on ORT `main`.

Remove this vendor step when an official NuGet includes both (or when xllama
bumps past them).
