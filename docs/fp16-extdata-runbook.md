# External ONNX data on Xbox AppContainer

Current operational note for the patched ORT external-data path. The original
fp16-GPU feasibility campaign is complete; historical experiments remain in Git
history and `CHANGELOG.md`.

## Measured outcome

- **Loading fixed:** the patched `onnxruntime.dll` guards AppContainer path
  canonicalisation and reads large external data in 16 MB chunks. A 1.86 GB
  external-data int4 model completed 6/6 console restarts.
- **Large fp16 GPU inference rejected:** a native-DML 1B fp16 model loads but
  inference OOMs inside the measured 3801 MB GPU budget. The external-data fix
  enables large CPU/int4 models, not larger GPU text models.

These patches ship through the hash-pinned `vendor-dlls-v1` artifact:

- `patches/onnxruntime-extdata-appcontainer.patch`
- `vendor/onnxruntime-patched/SHA256SUMS`

## Rebuild and validate

Rebuild only when refreshing the ORT pin:

```powershell
.\scripts\vendor-ort-extdata-patch.ps1 -Build
```

The long Windows CI lane is `.github/workflows/build-uwp-ort-patched.yml`. After
promoting a DLL, verify its SHA-256 pin, build the unified package, and run the
console validation suite plus a large external-data model restart test.

## Drop conditions

Keep PatchedOrt until both AppContainer fixes are available in the pinned NuGet
release:

1. `WeaklyCanonicalPath` AppContainer access fix (ORT #28509, merged upstream).
2. `ReadFileIntoBuffer` 16 MB chunk (ORT #29732, pending upstream/release).

Track the decision in [vendor-lifecycle-plan.md](vendor-lifecycle-plan.md) and
check current package versions with:

```bash
./scripts/check-vendor-nuget-status.sh
```

Do not reopen the >2 GB fp16-on-GPU campaign without a materially larger measured
GPU budget or a runtime change that reduces the inference working set.
