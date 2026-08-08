# patches/

UWP/AppContainer patches against the pinned `llama.cpp` submodule, applied to
both build variants that compile ggml (`uwp/ggml-uwp.vcxproj`): the shipping
`unified` variant (`XLLAMA_USE_ORT=1` + `XLLAMA_USE_LLAMA=1`) and the bench-only
`llamacpp` variant. CI runs `scripts/apply-uwp-patches.sh` for both.

## Active patches

| File                                        | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `0001-uwp-appcontainer-guards.patch`        | Guard Win32 desktop-only APIs across **4 files** via **`WINAPI_FAMILY_PARTITION(DESKTOP)` only** (VS + uwp-crossbuild both set `WINAPI_FAMILY=APP` for UWP). Stubs registry CPU-name (`ggml-cpu.cpp`), `SetThreadAffinityMask` / power throttling (`ggml-cpu.c`), packaged `LoadPackagedLibrary` (`ggml-backend-dl.cpp`), and desktop mmap/mlock (`llama-mmap.cpp`). No dual `XLLAMA_UWP` workaround — the family partition is the contract. Always run `./scripts/apply-uwp-patches.sh` before a Linux UWP build; gate the PE with `uwp-crossbuild/scripts/pe-import-audit.sh`. |
| `onnxruntime-genai-2280-dml-fallback.patch` | ORT GenAI `CreateDmlObjects`: fall back to system D3D12 when Agility `CreateDevice` fails with `887A0036` (XAML + DML). **Upstream MERGED** [microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280) (2026-07-13 on `main`); **not** in NuGet 0.14.1. Shipping CI installs the pinned DLL from `vendor-dlls-v1` (hash in `vendor/onnxruntime-genai-patched/SHA256SUMS`); rebuild via `scripts/vendor-genai-dml-patch.ps1 -Build` / `build-uwp-patched.yml`.                                                                                                                                                                         |
| `onnxruntime-extdata-appcontainer.patch`    | ORT core, two AppContainer external-data fixes (`docs/fp16-extdata-runbook.md`, console-validated 2026-07-15, **shipping since 1.1.8.0**): **(1)** `tensorprotoutils.cc` `ValidateExternalDataPath` — guard `weakly_canonical()` (related upstream on ORT `main`: [#28509](https://github.com/microsoft/onnxruntime/pull/28509), **not** in NuGet 1.24.4); **(2)** `env.cc` `ReadFileIntoBuffer` — 1 GB→16 MB chunk (`errcode 1450`, **still open on ORT main**). Shipping CI installs the pinned DLL from `vendor-dlls-v1` (hash in `vendor/onnxruntime-patched/SHA256SUMS`); rebuild via `scripts/vendor-ort-extdata-patch.ps1 -Build` / `build-uwp-ort-patched.yml`. |

Regenerated 2026-07-08 against submodule `9a532ae4b`; still applies clean on the
current pin `6d5a910c5` (`git apply --check`, 2026-07-23). (The three earlier per-file
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
