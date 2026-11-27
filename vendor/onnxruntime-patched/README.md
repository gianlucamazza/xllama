# vendor/onnxruntime-patched/

Patched **`onnxruntime.dll`** (DirectML) — ORT core with the AppContainer
external-data fixes (`patches/onnxruntime-extdata-appcontainer.patch`):

1. `ValidateExternalDataPath` — guard `weakly_canonical()` (error_code + lexical fallback)
2. `ReadFileIntoBuffer` — chunk 1 GB → 16 MB (errcode 1450)

Unblocks loading ONNX models with external `.onnx.data` >2 GB on Xbox
AppContainer. See `docs/uwp-constraints.md §8` and
`docs/fp16-extdata-runbook.md`.

Patched by xllama; not an official ONNX Runtime build. MIT License, Copyright (c) Microsoft Corporation — see `LICENSES/onnxruntime-LICENSE`.

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

The rebuild verifies both fresh and reused checkouts against the source commit
from the [official NuGet manifest](https://api.nuget.org/v3-flatcontainer/microsoft.ml.onnxruntime.directml/1.24.4/microsoft.ml.onnxruntime.directml.nuspec).
It applies the canonical Git patch and prints the verified base commit and
patched tree ID. Preserve both with rebuild logs. A checkout with unrelated
source changes or an older locally transformed patch must be replaced with a
clean checkout before rebuilding. Ignored files (a previous `build/`) and
`assume-unchanged` or `skip-worktree` entries are refused too, because the tree
comparison cannot see them; the script removes its own `build/` first.

**Upstream source unavailable (2026-11-27).** `github.com/microsoft/onnxruntime`
has returned 404 since about 2026-11-17, with no notice from Microsoft, so the
clone step fails and neither rebuild lane can run. The shipped DLL and
`SHA256SUMS` are unaffected, and so is the NuGet package
(`repository commit="2d924974ef147392ced8409d36bd6d2e7fcc8a74"` in its nuspec,
checked on that date). Because the verification binds the checkout to that
commit and the reviewed tree, a rebuild from any other copy of the source is
acceptable only when it passes `prepare_ort_source.py` unchanged. Keep the existing DLL and digest until the
candidate passes Windows packaging and console external-data acceptance.

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
