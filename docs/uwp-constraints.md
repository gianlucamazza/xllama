# UWP Constraints

> **SSOT for UWP/AppContainer constraints** (numbered §1–§12): the measured GPU
> budget (3801 MB), the 2 GB per-file limit, disk budget, no-mmap, `887A0036`,
> `weakly_canonical`, thread cap, and the DirectML low-bit GEMM analysis. Other
> docs link to a `§n` here rather than restating these.

Platform limitations relevant to running LLM inference on Xbox Dev Mode, and how xllama addresses each one.

## 1. No POSIX `mmap`

**Problem**: `mmap()` is unavailable in the UWP sandbox.

**Status**: Not an issue for the ORT GenAI path — ORT loads the model internally using Win32 file APIs compatible with UWP. For the GGUF/llama.cpp path the `patches/0001` guards disable the desktop-only `_WIN32` mmap on the AppContainer, so GGUF weights are read into the heap (buffered). Enabling an AppContainer mmap via `CreateFileMappingFromApp`/`MapViewOfFileFromApp` was **tried and reverted (2026-07-14): no benefit** — the CPU GGUF load is dominated by the AVX2 tensor repack, not the file read (full analysis in [`benchmarks.md`](benchmarks.md) → root-cause notes).

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
in-app (`QueryVideoMemoryInfo(LOCAL).Budget`) is **3801 MB** (package designated Game — verified 2026-07-08, see §5) on
Series S — the earlier "~768 MB pool" was a coarse inference from the
Phi-3.5-mini OOM bracketing and is superseded. Disk was then the operative
sizing constraint, but the Dev Mode allocation was raised to **90 GB** on
2026-07-08 (§9, superseded note) — the binding limits now are the **2 GB ONNX
protobuf / per-file ceiling** (§8; §9 caveat) and the RAM/GPU budget.

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

**Hardware utilization matrix** (2026-07-07, v0.3.6 with separate
prefill/decode timing; SmolLM2-360M, 2 runs each, median-ish first run shown):

| Variant  | prefill 285 tok | prefill ~1050 tok | decode (short ctx) | decode (long ctx) |
| -------- | --------------- | ----------------- | ------------------ | ----------------- |
| CPU int4 | **220 tok/s**   | 198 tok/s         | **68.0 tok/s**     | **50.9 tok/s**    |
| GPU int4 | 152 tok/s       | 334 tok/s         | 8.8 tok/s          | 8.3 tok/s         |
| GPU fp16 | 169 tok/s       | **354 tok/s**     | 46.8 tok/s         | 36.5 tok/s        |

Readings:

1. **Prefill: the GPU scales with batch size, the CPU does not.** The
   crossover sits between ~285 and ~1050 prompt tokens; at 1k tokens the GPU
   is **1.8× faster** (TTFT ~3.0 s vs ~5.3 s). For prompt-heavy workloads
   (RAG, long context) DML fp16 is already the better choice.
2. **The int4 decode collapse is a non-fused DML kernel, not a missing/CPU one**
   (corrected 2026-07-08 — see §12). The `MatMulNBits` op **is** present in the
   graph (225 nodes, `bits=4 block_size=32`, verified) and **does run on the GPU**:
   the profiled run shows the whole model as one `DmlFusedNode_0_0` on
   `DmlExecutionProvider` (96% of kernel time), with no `MatMulNBits` on CPU — so
   it is neither absent nor a CPU fallback. But DirectML's `MatMulNBits` kernel
   (`DmlOperatorMatMulNBits.cpp`) is a **non-fused** `DML_DEQUANTIZE`
   (int4→fp16 on GPU) + full `DML_GEMM`: it materialises the fp16 weight tensor,
   so int4 decode moves **more** bandwidth than plain fp16 (read 4-bit, write
   fp16, read fp16). That is why int4-DML (8.8) is _slower_ than fp16-DML (46.8).
   The builder also targets DML with `int4_accuracy_level=0` (fp16 compute) while
   CPU gets `=4` (fused int8 MLAS `SQNBitGemm`) — the reason CPU int4 reaches 68.
   There is **no fused low-bit GPU GEMM** on DirectML through 1.15.x, so the
   ~180 tok/s "bandwidth ceiling" is **not** reachable on DML by any config we
   control; fp16 is the best DML decode config, and it still loses to CPU int4.
3. CPU decode degrades ~25% from short to ~1 k context; GPU fp16 similarly.

**Verdict**: CPU int4 remains the decode winner (68 vs fp16-DML 46.8 vs int4-DML
8.8); GPU fp16 is superior only for prompt-heavy prefill. The int4-DML gap is a
DirectML kernel-_design_ limit (non-fused low-bit GEMM), not a hardware limit and
not fixable by our quantization config — see §12 for the full analysis and the
config tests that confirm it. Effective bandwidth: CPU ~13 GB/s, GPU fp16
~34 GB/s, against a ~224 GB/s theoretical bus.

**Effect on disk**: models too large to fit the Dev Mode partition also fail before reaching `OgaCreateModel`. This is a distinct failure mode — see §9.

Disk budget failures (deploy-time or LocalState copy):

| Model        | Variant  | On-disk | Failure           |
| ------------ | -------- | ------- | ----------------- |
| Phi-3.5-mini | INT4 CPU | ~2.7 GB | Above disk budget |
| SmolLM2-1.7B | INT4 CPU | 1.4 GB  | Above disk budget |
| SmolLM2-360M | INT4 CPU | 403 MB  | ✅ Works (CPU EP) |

Note: DirectML itself _is_ available in Dev Mode (NuGet `Microsoft.AI.DirectML 1.15.4`). The memory pool constraint applies to model weight size, not to the API itself.

**Current approach**: CPU EP (`"provider_options": []` in `genai_config.json`) remains the default for the interactive app (decode-heavy chat). DML fp16 is measured-viable for prompt-heavy workloads and profiling tooling exists (§11). The remaining GPU/CPU unlock levers — larger models via no-bundle deploy,
llama.cpp CPU kernel A/B, per-workload routing — are tracked in `ROADMAP.md`
Phase 3.5. (int4-on-DML decode is **not** a lever: it is blocked by DirectML's
non-fused low-bit kernel, see §12.) See §7 for GPU pool detail and §9 for disk budget.

**Platform lever — App vs Game designation (settled 2026-07-08)**: Dev Home
can flip a sideloaded package between **App** and **Game** (tile → View
details → App type); Game grants Game OS resources (full GPU access, more
RAM). Checked on console: **xllama is already designated Game** — the
designation persists per package family across forward upgrades. Therefore
the measured figures in this document (3801 MB GPU budget, the utilization
matrix, the per-token DML dispatch overhead) are **Game-mode numbers**, and
the earlier assumption labelling them "App-mode" was wrong. Consequence: the
GPU decode gap cannot be blamed on App-mode scheduling — it is a DML/kernel
issue, consistent with the fused-int4 analysis above. Operational note:
re-check the designation after any package reinstall (it can reset to App).

## 6. Limited Thread Count

**Problem**: Dev Mode apps share CPU resources. Xbox Series S has 8 Zen 2 cores; typically 6–7 are available.

**Workaround**: `detect_threads()` in `src/bridge/platform.cpp` reads `hardware_concurrency()` at runtime and ORT GenAI respects the system thread pool. Thread count can be overridden via `InferenceParams`.

## 7. GPU Memory Pool — Detail

**Measured budget (2026-07-07, in-app `QueryVideoMemoryInfo(LOCAL).Budget`):
3801 MB** on Series S (package designated Game, verified 2026-07-08 — see §5). The "~768 MB" figure previously documented
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

**Plain ORT DML coexists with the XAML compositor — ✅ MEASURED 2026-07-09.**
The `887A0036` conflict was measured with **ORT GenAI**, whose DML EP goes
through the Agility device factory. The diffusion pipeline (`uwp/diffuse.cpp`)
uses **plain ORT DirectML** (`OrtSessionOptionsAppendExecutionProvider_DML`,
ORT 1.24.4), and the headless requirement was _inherited_ from the GenAI finding,
never falsified for plain ORT. `diffuse-inproc.flag` (`App.cpp`) runs
`run_diffuse()` on a background MTA thread **inside the live XAML process**;
on console it ran the full SD-Turbo pipeline to a coherent 512×512 PNG **with
the compositor alive** — no `887A0036`, no OOM (te 1006 ms, UNet 2991 ms, VAE
1576 ms, **total 5.57 s**, actually faster than the 6.9 s headless run, warm GPU).
**Conclusion**: the device conflict is specific to GenAI's Agility factory;
plain ORT DML shares the compositor's device fine. Image generation can run
**in-app without the restart flow** — the headless `diffuse.flag` path is no
longer required for diffusion (kept for bench parity). GPU headroom held:
compositor + sequential-lifetime weights (~2.4 GB) inside the 3801 MB budget.

**Diagnosis**: SEH `0xC0000005` in `OgaCreateModel`. WDP minidump (`type=2`) and the `xllama.log` entry `OgaCreateModel failed: ...` confirm the cause.

**Source note**: the GPU budget (3801 MB) is measured per-process via
`QueryVideoMemoryInfo(LOCAL).Budget` with the package designated Game. The historical "~768 MB"
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

> **Superseded (2026-07-08)**: the Dev Mode storage allocation was raised to
> **90 GB** via Dev Home → Manage Dev Storage. The figures below describe the
> default allocation and remain valid as the baseline for a fresh Dev Mode
> activation; with the enlarged allocation, disk is no longer the binding
> constraint for model sizing (GPU budget and RAM are). One caveat to verify:
> community reports a ~2 GB per-file limit in Dev Mode, relevant for merged
> `model.onnx` files above that size.

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
3. **In-app GPU memory (corroborating)**. `IDXGIAdapter3::QueryVideoMemoryInfo(LOCAL)` is callable from the AppContainer and is _per-process_: `CurrentUsage` climbing toward the model size after `OgaCreateModel` means the weights are resident on the GPU; `Budget` is the OS-granted ceiling (measured **3801 MB** on Series S with the package designated Game — trust this value over any hard-coded constant). Implemented as `gpu_mem_info()` in `src/bridge/platform.cpp`, logged pre-load/post-load/post-decode and exported as `gpu_mem_mb,gpu_budget_mb` bench CSV columns.

**Node-placement log caveat**: at `log_severity_level: 0` ORT emits "Node placements" lines from `session_state.cc`, but (a) only in full (non-minimal) ORT builds, and (b) ORT-core session logs may not route through the `OgaSetLogCallback` sink into `xllama.log`. Absence of the lines is not evidence — the profiling JSON is the primary probe.

**Fallbacks if the above is inconclusive** (not implemented; documentation only):

- Op inventory of the model graph (python + `onnx`, count `node.op_type` incl. `com.microsoft.*` domains) cross-referenced against the profiler's per-op CPU list to identify which ops force fallback.
- Full-vs-minimal ORT build probe: `strings onnxruntime.dll | grep "Node placements"` on the NuGet-restored DLL (affects only the log probe, not profiling).
- D3D12 debug layer: not viable in Dev Mode UWP (`DML_CREATE_DEVICE_FLAG_DEBUG` needs `DirectML.Debug.dll` and ORT creates the DML device internally); DXGI HRESULTs already surface through the SEH translator.

## 12. Why int4 Decode Is Slow on DirectML (not a missing kernel)

Desk-check dated 2026-07-08. Corrects the earlier wording that the int4 decode
collapse (8.8 tok/s) was a "missing fused int4 DML kernel" — the op is present
and GPU-executed; the limit is that DirectML's kernel is **non-fused**.

**Ground truth (verified locally)**:

- The ORT GenAI model builder `-p int4 -e dml` emits **225
  `com.microsoft::MatMulNBits`** nodes (`bits=4, block_size=32`) — a fused-op
  graph, not `DequantizeLinear`+`MatMul`. (`onnx` op inventory of the built
  `model.onnx`.)
- The profiled on-console run
  (`bench/results/profiles/20260707T144203Z/ort_profile_*.json`) shows the entire
  model as a single **`DmlFusedNode_0_0` on `DmlExecutionProvider`** (1549 ms,
  96% of kernel time); the only CPU kernels are `Gather` + `Cast` (0.1 ms). No
  `MatMulNBits` runs on CPU → **not a silent CPU fallback**.

**Root cause (verified against ORT source)**:

- DirectML **does** register `MatMulNBits`
  (`onnxruntime/core/providers/dml/.../Operators/OperatorRegistration.cpp`), but
  `DmlOperatorMatMulNBits.cpp` implements it as **`DML_DEQUANTIZE` (int4→fp16 on
  the GPU) + a full `DML_GEMM`** — it materialises the fp16 weight tensor. There
  is **no fused low-bit GPU GEMM**. In memory-bound decode (M=1) this reads the
  4-bit weights, writes a full fp16 weight matrix to VRAM, and reads it back —
  strictly **more** bandwidth than plain fp16, which is exactly why int4-DML
  (8.8) is _slower_ than fp16-DML (46.8).
- The GenAI builder defaults `int4_accuracy_level = 4` for the CPU EP but **`0`
  for non-CPU EPs**. CPU's level 4 activates MLAS's fused int8 low-bit GEMM
  (`SQNBitGemm`) — the reason CPU int4 hits 68 tok/s. DML gets level 0 (fp16
  compute) and its kernel has no int8 fast path to switch into.
- ORT 1.19/1.20 int4 improvements ("QDQ INT4", int4 embeddings) were added to the
  **CPU and CUDA EPs only**; no DirectML fused low-bit kernel has shipped through
  DirectML 1.15.x. Published DirectML-GenAI INT4 models (Phi-3, Llama-3.1) note
  compute runs at fp16/fp32 accuracy — consistent with level 0.

**Consequence for xllama**: int4-on-DML has **no path to beat fp16-on-DML** for
decode by any quantization config we control, and fp16-on-DML (46.8) still loses
to CPU int4 (68). So **CPU int4 stays the decode default**, and the GPU's real
win is prefill (§5, reading 1) and larger-model bandwidth. A genuinely fused
low-bit GPU GEMM would be a DirectML-team feature, not an ORT-side PR; treat GPU
int4 decode as blocked upstream, not a local TODO.

**Config tests that confirm the dead-end** (built, pending one console bench —
expected ≈ 8.8 tok/s, a fast negative): `int4_block_size=128` and
`int4_accuracy_level=4` variants of SmolLM2-360M. If either materially beat 8.8
it would refute the above; the kernel structure predicts they will not.
