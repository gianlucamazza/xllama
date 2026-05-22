# UWP Constraints

Platform limitations relevant to running LLM inference on Xbox Dev Mode, and how xllama addresses each one.

## 1. No POSIX `mmap`

**Problem**: `mmap()` is unavailable in the UWP sandbox.

**Status**: Not an issue for the current ORT GenAI path. ORT loads the model internally using Win32 file APIs compatible with UWP. The Linux llama.cpp path uses `mparams.use_mmap = false` to fall back to heap reads.

## 2. Sandboxed Filesystem

**Problem**: UWP apps can only read/write their `LocalFolder` (`ApplicationData::Current::LocalFolder`) and locations explicitly granted by the user. Arbitrary path access (e.g. `C:\Users\...`) is denied.

**Workaround**: The bundled model is included as `DeploymentContent` in the MSIX and placed under `Package.InstalledPath\models\`. On first launch, `resolve_model_path` (`src/bridge/path_utils.cpp`) copies it to `LocalState\models\` so subsequent writes (log, bench CSV) land in the writable sandbox. Device Portal is used for one-off file transfers in dev.

## 3. No `dlopen` / Dynamic Backend Loading

**Problem**: UWP does not allow loading unsigned DLLs at runtime.

**Workaround**: ORT GenAI and its dependencies (`onnxruntime-genai.dll`, `onnxruntime.dll`, `DirectML.dll`) are NuGet packages restored at build time and included in the MSIX as **app-local** DLLs with `<DeploymentContent>true</DeploymentContent>`. No system-wide DLL loading. The backend (`XLLAMA_USE_ORT`) is selected at compile time.

## 4. No JIT Compilation

**Problem**: UWP blocks `VirtualAlloc` with `PAGE_EXECUTE_*`, preventing JIT-compiled kernels.

**Impact**: Minimal. GGML and ORT GenAI kernels are pre-compiled C/C++. Only ggml-jit (experimental, unused here) is affected.

## 5. DirectML Available but GPU Pool Too Small

**Problem**: The UWP GPU memory pool on Xbox Series S is approximately **768 MB** (128 MB dedicated + 640 MB shared pool). This limit applies to any "Game" category Dev Mode app regardless of the console's 10 GB unified memory.

**Effect**: `OgaCreateModel` with the DirectML execution provider crashes with SEH `0xC0000005` (STATUS_ACCESS_VIOLATION — null-deref in the DML allocator when it hits OOM) for any LLM whose on-device weights exceed the pool. Tested:

| Model | Variant | On-disk | Result |
|-------|---------|---------|--------|
| Phi-3.5-mini | GPU INT4 AWQ | ~2.2 GB | Crash (GPU OOM) |
| SmolLM2-1.7B | INT4 CPU | 1.4 GB | Above disk budget |
| SmolLM2-360M | INT4 CPU | 403 MB | ✅ Works (CPU EP) |

Note: DirectML itself *is* available in Dev Mode (NuGet `Microsoft.AI.DirectML 1.15.4`). The constraint is the memory pool, not the API.

**Current approach**: CPU EP (`"provider_options": []` in `genai_config.json`). See §7.

## 6. Limited Thread Count

**Problem**: Dev Mode apps share CPU resources. Xbox Series S has 8 Zen 2 cores; typically 6–7 are available.

**Workaround**: `detect_threads()` in `src/bridge/platform.cpp` reads `hardware_concurrency()` at runtime and ORT GenAI respects the system thread pool. Thread count can be overridden via `InferenceParams`.

## 7. GPU Memory Pool — Detail

The UWP "Game" process category on Xbox Series S receives a GPU memory budget of approximately 768 MB. This is separate from the ~8 GB of CPU-accessible RAM and cannot be expanded from a Dev Mode application.

When `OgaCreateModel` initialises the DirectML execution provider, the DML allocator attempts to reserve GPU memory for model weights. If the model's total weight size exceeds the available pool, the allocator returns a null pointer; subsequent use of that pointer produces a STATUS_ACCESS_VIOLATION fault.

The fault manifests before any inference call — at model load time. There is no recovery path short of using a smaller model or switching to CPU EP.

**Diagnosis**: SEH `0xC0000005` in `OgaCreateModel`. WDP minidump (`type=2`) and the `xllama.log` entry `OgaCreateModel failed: ...` confirm the cause.

## 8. AppContainer Filesystem Walk (`weakly_canonical`)

**Problem**: ORT Runtime 1.24.4 calls `std::filesystem::weakly_canonical()` in `ValidateExternalDataPath()` for models that have a separate `.onnx.data` file. MSVC STL implements `weakly_canonical` by walking path segments from the root: `Q:\` → `Q:\Users` → `Q:\Users\UserMgr0` → ... The intermediate segment `UserMgr0` (Xbox AppContainer user manager) is not accessible from the AppContainer → `ACCESS_DENIED` → exception → crash.

The `\\?\` long-path prefix does not help: it bypasses MAX_PATH but not the access check.

**Fix**: before MSIX packaging, merge `model.onnx.data` into `model.onnx` to produce a self-contained model file. With no external data file, `ValidateExternalDataPath` is never called and `weakly_canonical` is never invoked.

Tool: `scripts/merge_onnx_external_data.py`. CI runs this automatically as part of `build-uwp`.

**Diagnosis**: Win32 probes on the model path — `GetFileAttributesW` and `CreateFile2` with `GENERIC_READ` succeed on `model_dir\model.onnx`, but the crash occurs inside the ORT segment-walking loop. Confirmed by matching the call site to `onnxruntime/core/framework/tensorprotoutils.cc` L337/338/346.
