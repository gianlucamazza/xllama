# patches/

UWP-specific patches against the pinned `llama.cpp` submodule (`b9222`).

## Applying patches

```bash
cd llama.cpp
git apply ../patches/*.patch
```

## Patch naming convention

```
NNNN-short-description.patch
```

Example: `0001-disable-posix-mmap-for-uwp.patch`

## Current patches

None yet — Phase 1 will add patches as UWP incompatibilities are identified.

Expected patches for Phase 1:
- Disable `POSIX mmap` and route through `mmap_replacement()` in `llama-bridge.cpp`
- Remove `dlopen`-based backend discovery (compile-time backend selection only)
- Replace `getenv` calls blocked in UWP sandbox with compile-time defaults
