# UWP Constraints

This document maps the UWP sandbox constraints relevant to running `llama.cpp` on Xbox Dev Mode, and describes how xllama addresses each one.

## 1. No POSIX `mmap`

**Problem**: `llama.cpp` defaults to memory-mapping model weight files via `mmap()`. This is unavailable in the UWP sandbox.

**Workaround**: `mparams.use_mmap = false` causes llama.cpp to fall back to a full read into heap memory. This costs extra peak RAM at load time but works correctly.

**Phase 1 improvement**: Implement `CreateFileMappingFromApp` + `MapViewOfFileFromApp` in `llama-bridge.cpp` to get zero-copy loading back. These Win32 functions are available in UWP unlike the POSIX equivalents.

## 2. Sandboxed Filesystem

**Problem**: UWP apps can only read/write their `LocalFolder` (`ApplicationData::Current::LocalFolder`), plus locations explicitly granted by the user via file pickers. Arbitrary path access (e.g. `C:\Users\...`) is blocked.

**Workaround**: Models must be copied to the app's `LocalFolder` before inference. This can be done:
- Via Device Portal file browser (`https://<ip>:11443/#fileExplorer`)
- Via USB mass storage (with Dev Mode enabled)
- Programmatically using the Device Portal REST API

**Phase 1 plan**: accept a relative model path interpreted against `LocalFolder`. Document the transfer workflow.

## 3. No `dlopen` / Dynamic Backend Loading

**Problem**: `llama.cpp` can dynamically load GPU backends (CUDA, Metal, Vulkan) at runtime via `dlopen`/`LoadLibrary`. UWP does not allow loading unsigned DLLs.

**Workaround**: Compile-time backend selection only. Phase 1 is CPU-only (`GGML_USE_STATIC_BACKEND=1`, no dynamic loading). Phase 2 will statically link the Mesa Vulkan driver.

## 4. No JIT Compilation

**Problem**: UWP blocks executable memory allocation (`VirtualAlloc` with `PAGE_EXECUTE_*`), preventing JIT-compiled kernels.

**Impact**: Minimal for llama.cpp — its GGML kernels are pre-compiled C/C++/AVX2. Only ggml-jit (experimental) is affected, and it is not used here.

## 5. DirectML Not Available in Dev Mode

**Problem**: The `Windows.AI.MachineLearning` API and DirectML acceleration are not exposed to Dev Mode UWP apps. The Xbox GPU's INT8/INT4 hardware is not directly addressable.

**Workaround**: Phase 2 targets the Mesa Vulkan driver, which exposes the GPU through the standard Vulkan compute API. INT8 quantization via `VK_EXT_shader_integer_dot_product` is a stretch goal.

## 6. Limited Thread Count

**Problem**: Dev Mode apps share CPU resources with the system. The Xbox Series S has 8 Zen 2 cores; typically 6–7 are available to a Dev Mode app.

**Workaround**: `detect_threads()` in `src/bridge/platform.cpp` reads `hardware_concurrency()` at runtime and caps threads accordingly. The default llama.cpp thread heuristic also works well.
