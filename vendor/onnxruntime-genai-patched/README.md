# Patched onnxruntime-genai.dll (#2280)

Place a console-validated DLL here:

```
vendor/onnxruntime-genai-patched/win-x64/onnxruntime-genai.dll
```

Build or install:

```powershell
# From repo root (Windows + VS2022)
./scripts/vendor-genai-dml-patch.ps1 -Build
./scripts/build-uwp.ps1 -PatchedGenAI
```

The binary is gitignored. Pin: ORT GenAI **0.14.1** + patch from
[`patches/onnxruntime-genai-2280-dml-fallback.patch`](../../patches/onnxruntime-genai-2280-dml-fallback.patch).

Remove this vendor step when the fix ships in the official NuGet package.