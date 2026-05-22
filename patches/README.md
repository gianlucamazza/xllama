# patches/

Root-level placeholder. UWP-specific patches against the pinned `llama.cpp` submodule live in `uwp/patches/llama.cpp/`.

## Active patches

Three patches are maintained in `uwp/patches/llama.cpp/`:

| File | Description |
|------|-------------|
| `0001-uwp-ggml-backend-dl-remove-winevt.patch` | Remove WinEvt dependency from ggml backend dynamic loading |
| `0002-uwp-ggml-cpu-cpp-no-registry.patch` | Remove registry access in ggml CPU implementation |
| `0003-uwp-ggml-cpu-c-no-thread-affinity-throttling.patch` | Remove thread affinity / throttling APIs blocked in UWP sandbox |

## Applying

```bash
./scripts/apply-uwp-patches.sh
```

Or manually:
```bash
cd llama.cpp
git apply ../uwp/patches/llama.cpp/*.patch
```

## Status

These patches are **not applied for the current UWP build** (which uses ORT GenAI, not llama.cpp). They are kept for any future evaluation of the llama.cpp path on UWP.

The patches apply cleanly against the pinned submodule commit. If `llama.cpp` is updated, rebase them with `git am --rebase`.
