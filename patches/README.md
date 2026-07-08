# patches/

UWP/AppContainer patches against the pinned `llama.cpp` submodule, used by the
`llamacpp` build variant (`uwp/ggml-uwp.vcxproj`, `XLLAMA_USE_ORT=0`).

## Active patches

| File                                 | Description                                                                                                                                                                                                                                                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `0001-uwp-appcontainer-guards.patch` | Guard the three Win32 desktop-only APIs with `WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)`: the registry CPU-name lookup (`ggml-cpu.cpp`), `SetThreadAffinityMask` (`ggml-cpu.c` — AppContainer falls back to the system scheduler), and `SetThreadInformation(ThreadPowerThrottling)` (`ggml-cpu.c`). |

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
