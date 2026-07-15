# patches/

UWP/AppContainer patches against the pinned `llama.cpp` submodule, used by the
`llamacpp` build variant (`uwp/ggml-uwp.vcxproj`, `XLLAMA_USE_ORT=0`).

## Active patches

| File                                        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0001-uwp-appcontainer-guards.patch`        | Guard the three Win32 desktop-only APIs with `WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)`: the registry CPU-name lookup (`ggml-cpu.cpp`), `SetThreadAffinityMask` (`ggml-cpu.c` — AppContainer falls back to the system scheduler), and `SetThreadInformation(ThreadPowerThrottling)` (`ggml-cpu.c`).                                                                                                                                                                                                                                                                                                                                                            |
| `onnxruntime-genai-2280-dml-fallback.patch` | ORT GenAI `CreateDmlObjects`: fall back to system D3D12 when Agility `CreateDevice` fails with `887A0036` (XAML + DML). Applied via `scripts/vendor-genai-dml-patch.ps1` onto NuGet 0.14.1. Upstream: [microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280).                                                                                                                                                                                                                                                                                                                                                                     |
| `onnxruntime-extdata-appcontainer.patch`    | ORT core `ValidateExternalDataPath` (`tensorprotoutils.cc`): guard `weakly_canonical()` (error_code overload + `lexically_normal` fallback) so loading a model with external `.onnx.data` does not crash on the AppContainer walk over `Q:\Users\UserMgr0` (`docs/uwp-constraints.md §8`). Unblocks fp16 models >2 GB on the GPU (the merge workaround caps at the 2 GB protobuf ceiling). Applied via `scripts/vendor-ort-extdata-patch.ps1 -Build` onto onnxruntime `v1.24.4`. **Fase 1 only** — used if the zero-code USB spike (`docs/fp16-extdata-runbook.md`) fails. Reference diff; the script falls back to a context-tolerant transform if `git apply` drifts. |

Regenerated 2026-07-08 against submodule `9a532ae4b` (the three earlier per-file
patches previously referenced here were stale and no longer existed on disk; the
WinEvt / `dl_load_library` issue is avoided at build level instead — the UWP
variant compiles the CPU backend statically, no dynamic backend loading).

## Applying

```bash
./scripts/apply-uwp-patches.sh   # idempotent; used by the llamacpp CI variant
```

## Rebasing after a submodule bump

The patch is a plain `git diff`. If it no longer applies, redo the three guards
by hand (they are one-liners plus a small `#if` block — see the patch body) and
regenerate with `git -C llama.cpp diff > patches/0001-uwp-appcontainer-guards.patch`.
