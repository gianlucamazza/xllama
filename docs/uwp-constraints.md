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

## 5. DirectML: GPU Pool Limits Larger Models (360M Fits)

**Background**: The UWP GPU-accessible memory pool on Xbox Series S is approximately **768 MB** (inferred from Phi-3.5-mini OOM). This is separate from CPU RAM.

**Effect on DirectML EP**: `OgaCreateModel` with the DirectML execution provider crashes with SEH `0xC0000005` (STATUS_ACCESS_VIOLATION — null-deref in the DML allocator when it hits OOM) for any LLM whose on-device weights exceed the pool. Smaller models that fit within the pool load without error.

DirectML EP test results (Series S, xllama v0.3.1, 2026-05-23):

| Model        | Variant          | On-disk | Result                                          |
| ------------ | ---------------- | ------- | ----------------------------------------------- |
| Phi-3.5-mini | GPU INT4 AWQ     | ~2.2 GB | GPU OOM (`0xC0000005`) — exceeds 768 MB pool    |
| SmolLM2-360M | INT4, DML config | 403 MB  | ✅ Loads without OOM; 71.7 tok/s ≈ CPU baseline |

**Interpretation note** (updated 2026-07-07): the profiled GPU-truth run on
console settled this — **the DirectML EP does not initialise at all**.
`OgaCreateModel` throws `887A0036` "The desired element already exists" at
`dml_helpers.cpp(140)` before any kernel runs (details in §7). Exp 1's earlier
"~71 tok/s ≈ CPU baseline" was therefore a **silent CPU fallback**, never real
DML execution. The CPU EP is the only working backend.

**Effect on disk**: models too large to fit the Dev Mode partition also fail before reaching `OgaCreateModel`. This is a distinct failure mode — see §9.

Disk budget failures (deploy-time or LocalState copy):

| Model        | Variant  | On-disk | Failure           |
| ------------ | -------- | ------- | ----------------- |
| Phi-3.5-mini | INT4 CPU | ~2.7 GB | Above disk budget |
| SmolLM2-1.7B | INT4 CPU | 1.4 GB  | Above disk budget |
| SmolLM2-360M | INT4 CPU | 403 MB  | ✅ Works (CPU EP) |

Note: DirectML itself _is_ available in Dev Mode (NuGet `Microsoft.AI.DirectML 1.15.4`). The memory pool constraint applies to model weight size, not to the API itself.

**Current approach**: CPU EP (`"provider_options": []` in `genai_config.json`) — chosen for deterministic behaviour. GPU EP research with proper D3D profiling is a future work item. See §7 for GPU pool detail and §9 for disk budget.

## 6. Limited Thread Count

**Problem**: Dev Mode apps share CPU resources. Xbox Series S has 8 Zen 2 cores; typically 6–7 are available.

**Workaround**: `detect_threads()` in `src/bridge/platform.cpp` reads `hardware_concurrency()` at runtime and ORT GenAI respects the system thread pool. Thread count can be overridden via `InferenceParams`.

## 7. GPU Memory Pool — Detail

The UWP sandbox on Xbox Series S provides approximately **768 MB of GPU-accessible memory**, as observed through `OgaCreateModel` OOM behavior in this project. This is separate from CPU-accessible RAM.

When `OgaCreateModel` initialises the DirectML execution provider, the DML allocator attempts to reserve GPU memory for model weights. If the model's total weight size exceeds the available pool, the allocator returns a null pointer; subsequent use of that pointer produces a STATUS_ACCESS_VIOLATION fault.

The fault manifests before any inference call — at model load time. There is no recovery path short of using a smaller model or switching to CPU EP.

**Distinct failure mode — DML EP init throws even for models that fit the pool**
(2026-07-07, GPU-truth run, ORT GenAI 0.13.2 / ORT 1.24.4 / DirectML 1.15.4):
SmolLM2-360M INT4 (403 MB, well within the ~768 MB pool) does **not** OOM — it
fails earlier, at DirectML EP device creation. `OgaCreateModel` throws
`887A0036` "The desired element already exists" at
`onnxruntime-genai .../dml/dml_helpers.cpp(140)`, before any kernel runs.
Reproduced 3× (profiling and plain DML configs; with and without our
`gpu_mem_info` pre-load probe — the probe is not the cause). Not OOM
(`avail_phys` 5.0 GB, `budget` 3801 MB). GPU telemetry stays flat (only the
display engine active), confirming no GPU execution.

Likely root cause: a Direct3D 12 device is a singleton per adapter — a UWP XAML
app already holds a D3D12 device (the compositor) on adapter 0 before
`OgaCreateModel`, and the DML EP's own device/element creation collides with it.
This makes the DirectML EP **not viable** on this ORT GenAI build in the Xbox
UWP sandbox regardless of model size; the CPU EP is the only working backend
(70.9 tok/s on SmolLM2-360M). Escaping it would need a different ORT GenAI
version, a GDK (non-UWP) path, or creating the DML device before the XAML
compositor claims the adapter.

**Diagnosis**: SEH `0xC0000005` in `OgaCreateModel`. WDP minidump (`type=2`) and the `xllama.log` entry `OgaCreateModel failed: ...` confirm the cause.

**Source note**: the ~768 MB figure is the total observed at OOM in this project's tests. We do not document the underlying Xbox OS memory partition layout — treat any claim about the internal platform architecture as informed inference, not authoritative fact, unless backed by a Microsoft source.

## 8. AppContainer Filesystem Walk (`weakly_canonical`)

**Problem**: ORT Runtime 1.24.4 calls `std::filesystem::weakly_canonical()` in `ValidateExternalDataPath()` for models that have a separate `.onnx.data` file. MSVC STL implements `weakly_canonical` by walking path segments from the root: `Q:\` → `Q:\Users` → `Q:\Users\UserMgr0` → ... The intermediate segment `UserMgr0` (Xbox AppContainer user manager) is not accessible from the AppContainer → `ACCESS_DENIED` → exception → crash.

The `\\?\` long-path prefix does not help: it bypasses MAX_PATH but not the access check.

**Fix**: before MSIX packaging, merge `model.onnx.data` into `model.onnx` to produce a self-contained model file. With no external data file, `ValidateExternalDataPath` is never called and `weakly_canonical` is never invoked.

Tool: `scripts/merge_onnx_external_data.py`. CI runs this automatically as part of `build-uwp`.

**Diagnosis**: Win32 probes on the model path — `GetFileAttributesW` and `CreateFile2` with `GENERIC_READ` succeed on `model_dir\model.onnx`, but the crash occurs inside the ORT segment-walking loop. Confirmed by matching the call site to `onnxruntime/core/framework/tensorprotoutils.cc` L337/338/346.

## 9. Disk Budget (Dev Mode Partition)

**Observed**: the Xbox Series S Dev Mode partition (`Q:\`) provides approximately **2.2–2.5 GB of free space** after a clean Dev Mode activation, before any sideloaded package.

**Effect on model selection**: peak disk usage during deploy is approximately **2× the MSIX size**, because the package is uploaded to a staging area before installation completes. A 400 MB MSIX requires ~800 MB free during deploy, and ~400 MB residual after.

**Working budgets** (empirical, Series S Dev Mode, with xllama installed):

| Budget                      | Conservative | Borderline | Over budget |
| --------------------------- | ------------ | ---------- | ----------- |
| MSIX size                   | < 600 MB     | 600–800 MB | > 800 MB    |
| Model on-disk (merged ONNX) | < 400 MB     | 400–600 MB | > 600 MB    |

**Failure mode**: deploy fails with `0x80070070` (ERROR_DISK_FULL) if free space drops below the staging requirement. Alternatively, the first-launch copy from `InstalledPath` to `LocalState` fails silently, and `resolve_model_path` falls through to the default path — resulting in a model-not-found error at runtime rather than a visible install error.

**Source**: repeated `deploy.sh` runs against `https://<XBOX_IP>:11443/api/devices/file/usage`. Series X Dev Mode has the same partition layout but available free space was not measured by this project.

For models above the 600 MB on-disk budget that cannot be bundled in the MSIX, see Exp 3 (USB NTFS fallback at `E:\xllama\models\<name>`) in `docs/phase1-runbook.md` and `src/bridge/path_utils.cpp`.

See also `docs/model-selection.md` for a consolidated checklist.

## 10. Win32 APIs Available in WINAPI_PARTITION_APP (Xbox Dev Mode)

Reference for future work — APIs that do **not** require a desktop-only guard on Xbox UWP:

| API                                                           | Header   | Notes                          |
| ------------------------------------------------------------- | -------- | ------------------------------ |
| `CreateThread`, `WaitForSingleObject`, `CloseHandle`, `Sleep` | kernel32 | PARTITION_APP since 10.0.14393 |
| `FreeLibrary`                                                 | kernel32 | PARTITION_APP                  |
| `GlobalMemoryStatusEx`                                        | kernel32 | PARTITION_APP since 10.0.15063 |
| `GetModuleFileNameW`                                          | kernel32 | PARTITION_APP                  |
| `SetThreadPriority`, `GetCurrentThread`                       | kernel32 | PARTITION_APP                  |
| SRWLOCK, CONDITION_VARIABLE, Interlocked\*                    | kernel32 | PARTITION_APP                  |
| `QueryPerformanceCounter/Frequency`                           | kernel32 | PARTITION_APP                  |
| `_aligned_malloc` / `_aligned_free`                           | CRT      | Available                      |

APIs that are **desktop-only** and require `#if WINAPI_PARTITION_DESKTOP` guards: `RegOpenKeyEx`, `RegQueryValueExA`, `SetThreadAffinityMask`, `SetThreadInformation(ThreadPowerThrottling)`, `<winevt.h>` includes.

## 11. GPU Truth — EP Attribution Without PIX

PIX for Xbox is GDK tooling gated behind the managed partner program; it is **not available for Dev Mode UWP**. GPU-vs-CPU execution truth is instead established by converging three surfaces (all verified against primary sources, ORT GenAI 0.13.2 / ORT 1.24.4):

1. **ORT profiling JSON (primary, definitive)**. `genai_config.json` → `session_options` accepts `enable_profiling` (a string: the profile file _path prefix_, producing `<prefix>_<timestamp>.json`) and `log_severity_level` (0 = VERBOSE). Every `<node>_kernel_time` event in the trace carries `args.provider` — literally `"DmlExecutionProvider"` or `"CPUExecutionProvider"`. Heavy kernels (MatMul/Attention) tagged CPU = silent fallback. This works regardless of ORT build flavor and log routing. Tooling: `scripts/profile-dml-run.sh` (config swap + run + fetch) and `scripts/analyze_ort_profile.py` (per-provider summary + greppable `VERDICT:` line).
   - _Profile location ladder_: the relative prefix resolves against the process CWD, which in AppContainer may be the read-only install root (ORT's profiler ofstream then fails silently). Step 1: the fetch script checks LocalState root **and** `models\<name>\`. Step 2: `--absolute-prefix` renders `genai_config-dml-profile.tpl.json` with an absolute LocalState path. Step 3 (definitive): `set_cwd_to_local_folder()` pins CWD to LocalState at bench startup (v0.3.2+ MSIX).
2. **Device Portal telemetry (corroborating)**. `GET /api/resourcemanager/systemperf` exists on the Xbox device family and reports `GPUData.AvailableAdapters[]` with `EnginesUtilization[]` (0–1 per engine) and `DedicatedMemoryUsed`. System-wide, ~1 Hz — run a control pass with the CPU config to calibrate background noise. Tooling: `scripts/xbox-gpu-sample.sh`, integrated as `--gpu-sample` in the bench/profile scripts.
3. **In-app GPU memory (corroborating)**. `IDXGIAdapter3::QueryVideoMemoryInfo(LOCAL)` is callable from the AppContainer and is _per-process_: `CurrentUsage` climbing toward the model size after `OgaCreateModel` means the weights are resident on the GPU; `Budget` is the OS-granted ceiling (~768 MB App-mode Series S — trust this value over the hard-coded constant). Implemented as `gpu_mem_info()` in `src/bridge/platform.cpp`, logged pre-load/post-load/post-decode and exported as `gpu_mem_mb,gpu_budget_mb` bench CSV columns.

**Node-placement log caveat**: at `log_severity_level: 0` ORT emits "Node placements" lines from `session_state.cc`, but (a) only in full (non-minimal) ORT builds, and (b) ORT-core session logs may not route through the `OgaSetLogCallback` sink into `xllama.log`. Absence of the lines is not evidence — the profiling JSON is the primary probe.

**Fallbacks if the above is inconclusive** (not implemented; documentation only):

- Op inventory of the model graph (python + `onnx`, count `node.op_type` incl. `com.microsoft.*` domains) cross-referenced against the profiler's per-op CPU list to identify which ops force fallback.
- Full-vs-minimal ORT build probe: `strings onnxruntime.dll | grep "Node placements"` on the NuGet-restored DLL (affects only the log probe, not profiling).
- D3D12 debug layer: not viable in Dev Mode UWP (`DML_CREATE_DEVICE_FLAG_DEBUG` needs `DirectML.Debug.dll` and ORT creates the DML device internally); DXGI HRESULTs already surface through the SEH translator.
