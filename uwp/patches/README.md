# UWP patches for llama.cpp submodule

These patches guard APIs that are WINAPI_PARTITION_DESKTOP-only (not available
in the UWP app-container sandbox used by Xbox Dev Mode).

## Why patches instead of editing the submodule

The submodule is pinned at a specific SHA and kept pristine in git. Patches live
here and are applied in CI before the MSBuild step via `scripts/apply-uwp-patches.sh`.
When bumping the llama.cpp SHA, rebase the patches with:

```bash
cd llama.cpp
git apply --check ../uwp/patches/llama.cpp/000*.patch 2>&1
```

## Patch list (apply in order)

| File | What it fixes |
|---|---|
| 0001-uwp-ggml-backend-dl-remove-winevt.patch | Removes `#include <winevt.h>` (upstream bug — unused, desktop-only) and adds UWP-safe no-op deleter |
| 0002-uwp-ggml-cpu-cpp-no-registry.patch | Guards `RegOpenKeyEx`/`RegQueryValueExA`/`RegCloseKey` (advapi32, desktop-only) with `WINAPI_PARTITION_DESKTOP`; falls back to static CPU string on UWP |
| 0003-uwp-ggml-cpu-c-no-thread-affinity-throttling.patch | Guards `SetThreadAffinityMask` and `SetThreadInformation(ThreadPowerThrottling)` (both desktop-only); no-ops on UWP |

## How to apply manually

```bash
cd llama.cpp
for p in ../uwp/patches/llama.cpp/000*.patch; do
    git apply "$p"
done
```

## APIs confirmed available in WINAPI_PARTITION_APP (no patch needed)

- `CreateThread`, `WaitForSingleObject`, `CloseHandle`, `Sleep` — kernel32 PARTITION_APP since 10.0.14393
- `FreeLibrary` — kernel32 PARTITION_APP
- `GlobalMemoryStatusEx` — kernel32 PARTITION_APP since 10.0.15063
- `GetModuleFileNameW` — kernel32 PARTITION_APP
- `SetThreadPriority`, `GetCurrentThread` — kernel32 PARTITION_APP
- SRWLOCK, CONDITION_VARIABLE, Interlocked* — kernel32 PARTITION_APP
- `QueryPerformanceCounter/Frequency`, `_aligned_malloc/_free` — available
