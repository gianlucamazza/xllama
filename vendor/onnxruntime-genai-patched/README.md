# Patched onnxruntime-genai.dll (#2280)

Patched **`onnxruntime-genai.dll`** — DML Agility `CreateDevice` fallback for
XAML hosts (`887A0036`). Upstream
[microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280)
**merged on `main` 2026-07-13**; **not** in NuGet GenAI **0.14.1**.

The DLL binary lives at `win-x64/onnxruntime-genai.dll` and is **gitignored** —
only this README + `SHA256SUMS` are tracked. Shipping CI downloads the pinned
DLL from the [`vendor-dlls-v1`](https://github.com/gianlucamazza/xllama/releases/tag/vendor-dlls-v1)
release (hash must match `SHA256SUMS`).

## Producing / refreshing the DLL

- **CI rebuild:** `gh workflow run build-uwp-patched.yml` uploads
  `onnxruntime-genai-patched-dll`. After re-validation, update the
  `vendor-dlls-v1` release asset and `SHA256SUMS`.
- **Local:** `./scripts/vendor-genai-dml-patch.ps1 -Build` (VS2022 + NuGet
  restore; GenAI branch `rel-0.14.1` pin `a30f479` +
  `patches/onnxruntime-genai-2280-dml-fallback.patch`).

## Install over NuGet (local or CI)

```powershell
# Cached vendor path (after download or -Build):
./scripts/vendor-genai-dml-patch.ps1

# Or via packaging:
./scripts/build-uwp.ps1 -Backend unified -PatchedGenAI -PatchedOrt
```

Pin: ORT GenAI **0.14.1** + #2280. **Remove this vendor step** when Microsoft
ships #2280 in an official NuGet package (track ROADMAP “drop PatchedGenAI”).
