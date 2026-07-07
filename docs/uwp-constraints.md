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

## 5. DirectML: Works on GPU (Headless) but Loses to CPU on Small Models

**Background — pool estimate corrected (2026-07-07)**: the GPU budget measured
in-app (`QueryVideoMemoryInfo(LOCAL).Budget`) is **3801 MB** in App-mode on
Series S — the earlier "~768 MB pool" was a coarse inference from the
Phi-3.5-mini OOM bracketing and is superseded. The operative constraint for
model sizing is the **Dev Mode disk budget** (`Q:\` ~2.2–2.5 GB free), not GPU
memory.

**Effect on DirectML EP**: with a DML model variant and the headless path
(§7), the EP loads and executes on the GPU. Very large models can still OOM
(`0xC0000005`, null-deref in the DML allocator).

DirectML EP test results (Series S):

| Model        | Variant              | On-disk | Result                                                                                              |
| ------------ | -------------------- | ------- | --------------------------------------------------------------------------------------------------- |
| Phi-3.5-mini | GPU INT4 AWQ         | ~2.2 GB | GPU OOM (`0xC0000005`), v0.3.1 (2026-05-23)                                                         |
| SmolLM2-360M | INT4 CPU, DML config | 403 MB  | v0.3.1: silent CPU fallback (71.7 tok/s); v0.3.4: DML fused node fails (`80070057`, CPU-int4 graph) |
| SmolLM2-360M | INT4 **DML build**   | 285 MB  | ✅ **GPU execution, decode completes — 8.8 tok/s** (v0.3.4 headless, 2026-07-07)                    |

**Interpretation note** (updated 2026-07-07, evening): settled — **the DML EP
executes on the GPU** when the process is D3D12-clean. In headless bench mode
(v0.3.4, no XAML compositor) the profiled run yields **`VERDICT: GPU`** (fused
DML node on `DmlExecutionProvider`, 96% of kernel time) with 411 MB of weights
resident on the GPU (`gpu-mem post-load`). The earlier `887A0036` init failure
was the Agility-factory vs XAML-compositor device conflict (§7). Exp 1's
"~71 tok/s ≈ CPU baseline" was a silent CPU fallback on a pre-614 OS.

**Final performance verdict** (SmolLM2-360M INT4 DML build, 285 MB, decode
completes end-to-end): **8.8 tok/s on GPU vs 70.9 tok/s on CPU** — the Zen 2
CPU is ~8× faster. Autoregressive decode of a 360M-parameter model is
dominated by per-token DML dispatch overhead, while `MatMulNBits` on AVX2 is
highly optimised. **CPU EP remains the production backend**; DML is proven
functional but not competitive at this model scale.

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

**Measured budget (2026-07-07, in-app `QueryVideoMemoryInfo(LOCAL).Budget`):
3801 MB** in App-mode on Series S. The "~768 MB" figure previously documented
here was inferred from OOM bracketing and is superseded by this direct
measurement.

When `OgaCreateModel` initialises the DirectML execution provider, the DML allocator attempts to reserve GPU memory for model weights. If the model's total weight size exceeds the available pool, the allocator returns a null pointer; subsequent use of that pointer produces a STATUS_ACCESS_VIOLATION fault.

The fault manifests before any inference call — at model load time. There is no recovery path short of using a smaller model or switching to CPU EP.

**Distinct failure mode — DML EP init `887A0036` in XAML apps** (2026-07-07,
GPU-truth run, ORT GenAI 0.13.2 / ORT 1.24.4 / DirectML 1.15.4) — **root cause
found at the exact source line and fixed architecturally in v0.3.4**:

`OgaCreateModel` threw `887A0036 DXGI_ERROR_ALREADY_EXISTS` at
`onnxruntime-genai/src/dml/dml_helpers.cpp(140)` (`CreateDmlObjects`): ORT GenAI
creates its D3D12 device through the **Agility SDK device factory** —
`ID3D12SDKConfiguration1::CreateDeviceFactory(614, module_path)` succeeds on
Xbox OS 26100 via the in-box runtime ≥ 614 (no app-local `D3D12Core.dll`
needed; verified absent from the MSIX), and the factory's `CreateDevice`
collides with the process-wide D3D12 device the **XAML compositor**
(D3D11on12) created at `Window.Activate()`. Two different D3D12 runtimes cannot
share a process. Not OOM, not the profiling config, not our telemetry
(reproduced 3× including after removing the `gpu_mem_info` pre-load probe).
Exp 1 (May) passed because the then-OS had in-box < 614, so ORT fell back to
plain `D3D12CreateDevice` (line 144) which coexists with the compositor device
— the OS update flipped the branch. No upstream fix on `main` (v0.14.0
identical); upstream is missing a fallback when the factory returns
`ALREADY_EXISTS`.

**Fix (v0.3.4): headless bench mode** — with `bench.flag` present, `wWinMain`
skips `Application::Start` entirely and runs `main_loop()` under a minimal
`CoreApplication` `IFrameworkView` (CoreWindow activated for the PLM watchdog,
no compositor, no in-process D3D12 device). Result: DML EP initialises, weights
load onto the GPU (411 MB), profiled kernels run on `DmlExecutionProvider`
(**`VERDICT: GPU`**). Config prerequisite: DML graph capture requires
`past_present_share_buffer: true` in `genai_config.json`.

**Upstream fix (validated on console, 2026-07-07)**: we patched
`CreateDmlObjects` to fall back to the system D3D12 runtime when the Agility
device factory cannot create a device (upstream PR
[microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280)).
Validated with a test MSIX (patched DLL, XAML path): the same XAML + DML
scenario that threw `887A0036` loads in 886 ms and completes decode at
8.8 tok/s; CPU path unaffected (67.2 tok/s). Once the fix ships in an ORT
GenAI release, the interactive (XAML) app can use DML without the headless
path — practically relevant only if a larger model ever makes DML
competitive (CPU is 8× faster at 360M scale).

**Diagnosis**: SEH `0xC0000005` in `OgaCreateModel`. WDP minidump (`type=2`) and the `xllama.log` entry `OgaCreateModel failed: ...` confirm the cause.

**Source note**: the GPU budget (3801 MB) is measured per-process via
`QueryVideoMemoryInfo(LOCAL).Budget` in App-mode. The historical "~768 MB"
estimate came from OOM bracketing (Phi-3.5-mini vs SmolLM2-360M) and proved to
be a strong underestimate. We do not document the underlying Xbox OS memory
partition layout — treat any claim about the internal platform architecture as
informed inference unless backed by a Microsoft source.

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
3. **In-app GPU memory (corroborating)**. `IDXGIAdapter3::QueryVideoMemoryInfo(LOCAL)` is callable from the AppContainer and is _per-process_: `CurrentUsage` climbing toward the model size after `OgaCreateModel` means the weights are resident on the GPU; `Budget` is the OS-granted ceiling (measured **3801 MB** App-mode Series S — trust this value over any hard-coded constant). Implemented as `gpu_mem_info()` in `src/bridge/platform.cpp`, logged pre-load/post-load/post-decode and exported as `gpu_mem_mb,gpu_budget_mb` bench CSV columns.

**Node-placement log caveat**: at `log_severity_level: 0` ORT emits "Node placements" lines from `session_state.cc`, but (a) only in full (non-minimal) ORT builds, and (b) ORT-core session logs may not route through the `OgaSetLogCallback` sink into `xllama.log`. Absence of the lines is not evidence — the profiling JSON is the primary probe.

**Fallbacks if the above is inconclusive** (not implemented; documentation only):

- Op inventory of the model graph (python + `onnx`, count `node.op_type` incl. `com.microsoft.*` domains) cross-referenced against the profiler's per-op CPU list to identify which ops force fallback.
- Full-vs-minimal ORT build probe: `strings onnxruntime.dll | grep "Node placements"` on the NuGet-restored DLL (affects only the log probe, not profiling).
- D3D12 debug layer: not viable in Dev Mode UWP (`DML_CREATE_DEVICE_FLAG_DEBUG` needs `DirectML.Debug.dll` and ORT creates the DML device internally); DXGI HRESULTs already surface through the SEH translator.
