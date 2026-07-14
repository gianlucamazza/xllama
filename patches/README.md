# patches/

UWP/AppContainer patches against the pinned `llama.cpp` submodule, used by the
`llamacpp` build variant (`uwp/ggml-uwp.vcxproj`, `XLLAMA_USE_ORT=0`).

## Active patches

| File                                        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0001-uwp-appcontainer-guards.patch`        | (a) Guard desktop-only Win32 APIs with `WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)`: registry CPU-name lookup (`ggml-cpu.cpp`), `SetThreadAffinityMask` + `ThreadPowerThrottling` (`ggml-cpu.c`), `LoadLibrary`/`SetErrorMode` (`ggml-backend-dl.cpp`), `VirtualLock` mlock (`llama-mmap.cpp`, kept disabled on AppContainer). (b) **Enable file mmap on the AppContainer**: the `_WIN32` mmap impl uses `CreateFileMappingFromApp`/`MapViewOfFileFromApp` on the app partition (`llama-mmap.cpp`), with a `llama_model_loader::init_mappings` try/catch that falls back to a buffered read if the mapping is denied — so worst case matches the previous buffered-load behaviour, best case avoids the multi-GB heap copy. |
| `onnxruntime-genai-2280-dml-fallback.patch` | ORT GenAI `CreateDmlObjects`: fall back to system D3D12 when Agility `CreateDevice` fails with `887A0036` (XAML + DML). Applied via `scripts/vendor-genai-dml-patch.ps1` onto NuGet 0.14.1. Upstream: [microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280).                                                                                                                                                                                                                                                                                                                                                                                                                                |

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
