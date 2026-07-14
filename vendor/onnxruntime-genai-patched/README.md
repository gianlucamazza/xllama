# Patched onnxruntime-genai.dll (#2280)

Place a console-validated DLL here:

```
vendor/onnxruntime-genai-patched/win-x64/onnxruntime-genai.dll
```

Build or install:

```powershell
# From repo root (Windows + VS2022 + prior 'nuget restore' in uwp/)
./scripts/vendor-genai-dml-patch.ps1 -Build   # always rebuilds; ignores this cache
./scripts/build-uwp.ps1 -PatchedGenAI          # installs the cached DLL over NuGet
```

No Windows machine: download the default **`xllama-appx`** artifact from the
latest green `build-uwp` run on `main` (unified + PatchedGenAI #2280). Manual
fallback: `gh workflow run build-uwp-patched.yml`.

The binary is gitignored. Pin: ORT GenAI **0.14.1** — upstream ships it as
branch `rel-0.14.1` (there is no `v0.14.1` tag), pinned to commit `a30f479`
in `scripts/vendor-genai-dml-patch.ps1` — plus
[`patches/onnxruntime-genai-2280-dml-fallback.patch`](../../patches/onnxruntime-genai-2280-dml-fallback.patch),
built against the restored `Microsoft.ML.OnnxRuntime.DirectML` NuGet
(`ORT_HOME`), so the DLL links the exact onnxruntime the MSIX ships.

Remove this vendor step when the fix ships in the official NuGet package.
