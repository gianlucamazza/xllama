# Runbook — Unblock fp16 models >2 GB on the GPU (DirectML)

Operational entrypoint for the ">2 GB fp16 on GPU" plan. Goal: load an fp16 ONNX
GenAI model with **external `.onnx.data`** on the Xbox, which today crashes in
`ValidateExternalDataPath` → `weakly_canonical` on the AppContainer walk over
`Q:\Users\UserMgr0` (`docs/uwp-constraints.md §8`). The merge workaround
(`scripts/merge_onnx_external_data.py`) caps at the 2 GB protobuf single-file
ceiling, so fp16 models >2 GB cannot ship — this is the last local GPU lever
(DirectML itself is in maintenance mode; the int4-DML decode wall in §12 is
permanent upstream).

## Outcome (2026-07-15, console-validated)

Two conclusions, one positive and one negative — both measured on Xbox Series S:

1. ✅ **External-data loading on the AppContainer is unblocked** by the two ORT
   patches (weakly_canonical guard + ReadFile 16 MB chunk). A 1.86 GB-extdata int4
   model that crashed pre-patch now loads and runs, 6/6 clean restarts, 0× 1450.
   This is the shipping win: **large external-data models (int4/CPU) work**.
2. ❌ **A 1B fp16 model cannot run _inference_ on the GPU** within the 3801 MB
   budget — the crossover this plan chased is **not reachable on this hardware**.
   A correctly built native-DML Llama-3.2-1B fp16 (2.49 GB weights, GQA) **loads**
   (`gpu-mem post-load: 2878/3801 MB`) but every inference call OOMs
   (`AppendTokenSequences … DmlCommittedResourceAllocator … 8007000E Not enough
memory`), even at `context_length=2048`. Weights fit; the ~923 MB headroom is
   too small for the DML inference working set (graph-capture allocators + prefill
   activations + KV). Any fp16 >2 GB is ≥~1B → same wall. **So the unblock and the
   fp16-GPU goal don't overlap on Series S**: the practical DML-fp16 window stays at
   ~360-500 M (e.g. `smollm2-360m-dml-fp16`, ~700 MB weights, ~3 GB headroom), which
   don't need the >2 GB unblock. The unblock's real value is CPU-side (large
   external-data int4 models), not bigger fp16 on the GPU.

## Feasibility gate (superseded by the Outcome above)

- GPU budget is **3801 MB** per-process (`docs/uwp-constraints.md §7`) — and the
  binding limit is the **inference working set**, not just the weights: a 2.49 GB
  fp16 model loads but OOMs mid-inference (measured, see Outcome #2).
- The `weakly_canonical` check runs at model **parse**, before GPU weight
  allocation — so external-data loading is answerable even for over-budget models.
- DML-fp16 builds are scarce on HF (mostly int4-DML or fp16-cuda); a native DML
  build (`onnxruntime_genai.models.builder … -p fp16 -e dml`) is required — a
  cuda-fp16 re-host loads/partitions to DML but **fails DML inference at runtime**.

## Prereqs

- Host: `python3` with `onnxruntime-genai torch transformers onnx` (a venv is fine).
- Console: Xbox Series S/X in Dev Mode with xllama installed; `source ~/.config/xllama/xbox-env`
  (`XBOX_IP`/`XBOX_USER`/`XBOX_PASS`).
- An NTFS USB stick.

## Fase 0 — zero-code USB spike (REFUTED 2026-07-15)

**Result: the zero-code USB path does NOT work — Fase 1 is required.** Ran with
`onnx-community/Llama-3.2-1B-Instruct-GENAI-ONNX` cpu-int4 (external `.onnx.data`
1.86 GB) staged on USB. The console log:

```
[xllama] USB probe: E:
[xllama] USB model found on E:
[xllama] USB copy: model.onnx.data ...
[xllama] USB model copy complete                     ← app copies USB → LocalState
[xllama] pre-OgaCreateModel: ... model=Q:\Users\UserMgr0\...\LocalState\models\llama32-1b-int4-extdata
[xllama] inference error: OgaCreateModel: ... weakly_canonical: Access is denied.
```

`EnsureModelNamedAsync` treats USB as a **provisioning source**: it copies the
model into `LocalState` and loads it from there, so it lands back under
`Q:\Users\UserMgr0` and hits the same `weakly_canonical` crash. USB is never the
load location. This also confirms the crash is universal to any external-data
model in LocalState (catalogue-downloaded fp16 >2 GB would crash identically), not
USB-specific. → **Do Fase 1.** (An alternative — make the app load USB models
in-place from `E:\` instead of copying — is also a code change and may hit
AppContainer USB read limits; the ORT patch is cleaner and fixes all external-data
loads.)

---

Original hypothesis (refuted above): the USB path
(`<usb>\xllama\models\<name>\`, resolved via the USB fallback in
`src/bridge/path_utils.cpp:167-203`) does not traverse `UserMgr0`, so the
external-data model may load without any code change.

```bash
# 1. Build the fp16-DML model (external data left UNMERGED). Smoke first (fast):
./scripts/build-fp16-dml-model.sh --smoke        # Qwen2.5-0.5B, code-path test
#    then the real target once the path is confirmed:
./scripts/build-fp16-dml-model.sh -m meta-llama/Llama-3.2-1B-Instruct

# 2. Copy dist/fp16-dml/<name>/ onto the USB at  <usb>\xllama\models\<name>\
#    then plug the stick into the console.

# 3. Drive the load + prefill bench and classify the verdict automatically:
source ~/.config/xllama/xbox-env
./scripts/spike-fp16-extdata-usb.sh <name>
```

`spike-fp16-extdata-usb.sh` exit codes / verdicts:

| Code | Verdict                         | Meaning / next step                                                                                                                                                                                                    |
| ---: | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|    0 | **LOADED**                      | Spike PASS — unblocked at zero cost. Route large fp16 to USB in `uwp/models/manifest.json` + `docs/model-selection.md`; record the prefill crossover in `docs/benchmarks.md`; close `ROADMAP.md:163-167`. Skip Fase 1. |
|    2 | **CRASH (weakly_canonical)**    | Walk fails even from USB → do Fase 1.                                                                                                                                                                                  |
|    3 | **OTHER (OOM/budget/DML init)** | Got past parse (hypothesis may hold) but too big — retry with `--smoke`; check `deploy.sh list-dumps`.                                                                                                                 |
|    4 | **INCONCLUSIVE**                | No bench row, no clear log signal — inspect the log + dumps.                                                                                                                                                           |

## Fase 1 — CONSOLE-VALIDATED 2026-07-15 (patch works; a second obstacle surfaced)

Built via the `build-uwp-ort-patched` lane (run 29405016840 → MSIX 1.1.7.1) and
deployed. Re-tested the exact model that crashed pre-patch
(`llama32-1b-int4-extdata`, external `.onnx.data` 1.86 GB, in LocalState):

- ✅ **`weakly_canonical: Access is denied` is GONE.** Two clean bench runs
  produced valid rows — prompt ~79 tok/s, decode ~35 tok/s, peak 2275 MB, load
  ~6.5 s (`bench/results/phase6-fp16-extdata.csv`). The external-data model now
  loads from LocalState and generates. The ORT `ValidateExternalDataPath` guard
  works.
- ✅ **Second obstacle `errcode 1450` — root-caused and FIXED, console-validated
  2026-07-15** (MSIX 1.1.7.2). Stress test: 6 consecutive restarts of the
  1.86 GB-extdata model all loaded clean — **0× errcode 1450, 0× weakly_canonical,
  0 crash dumps**, 6/6 valid bench rows (prefill ~79, decode ~34 tok/s). Before the
  fix it was intermittent on repeated restarts.
  The failing tensor is `model.embed_tokens.weight` (Llama-3.2 vocab 128k × hidden,
  **not** int4-quantized → ~0.5-1 GB fp16/fp32). ORT's `ReadFileIntoBuffer`
  (`onnxruntime/core/platform/windows/env.cc`) reads it in one `ReadFile` of up to
  `k_max_bytes_to_read = 1 << 30` (**1 GB**); a synchronous `ReadFile` locks all the
  destination pages into physical memory (MDL) for the transfer, and locking
  ~0.5-1 GB under AppContainer pressure exceeds resources → 1450 (intermittent:
  cold runs succeed, fragmented ones fail). mmap can't save it — ORT's
  `MapFileIntoMemory` uses the non-`-FromApp` Win32 APIs, blocked in AppContainer,
  so it always falls back to this ReadFile. **Fix (in the same patched DLL):**
  `patches/onnxruntime-extdata-appcontainer.patch` now also shrinks
  `k_max_bytes_to_read` to `1 << 24` (**16 MB**) — few pages locked per read, always
  within resources; this is what actually lets fp16 `.onnx.data` >2 GB load. Distinct
  from the (already-fixed) weakly_canonical blocker.

## Fase 1 — patch ORT core (only if Fase 0 returns code 2)

Builds a patched `onnxruntime.dll` (DirectML) that guards the `weakly_canonical`
walk and shrinks `ReadFile` chunks (errcode 1450). Mirrors the GenAI #2280
patched-DLL doctrine (both DLLs are now **hash-pinned** on `vendor-dlls-v1`).

**Shipping (v1.1.8.0+):** `build-uwp.yml` installs the pinned, console-validated
DLL from the [`vendor-dlls-v1`](https://github.com/gianlucamazza/xllama/releases/tag/vendor-dlls-v1)
release (SHA256 in `vendor/onnxruntime-patched/SHA256SUMS`) via
`vendor-ort-extdata-patch.ps1` / `-PatchedOrt`. No full ORT rebuild on every PR.

**Rebuild lane (pin refresh):** run **`build-uwp-ort-patched`**
(`.github/workflows/build-uwp-ort-patched.yml`). It builds both patched DLLs
(this + #2280 GenAI), packages a unified MSIX, and uploads the MSIX +
`onnxruntime-patched-dll`. After console re-validation, update the
`vendor-dlls-v1` release asset and `SHA256SUMS`.

```powershell
# Local alternative (VS2022 + Python + CMake):
nuget restore uwp/xllama.sln
./scripts/vendor-ort-extdata-patch.ps1 -Build   # clone v1.24.4, apply guard, build, vendor
./scripts/build-uwp.ps1 -Configuration Release -Platform x64 -PatchedGenAI -PatchedOrt
```

- Patch: `patches/onnxruntime-extdata-appcontainer.patch` (reference diff; the
  script falls back to a context-tolerant transform if `git apply` drifts, and
  verifies both fixes are present before the build).
- **Cost of rebuild**: a full ORT DirectML source build (1-3 h). Shipping CI
  never pays this — it downloads the pin.
- Follow-up: with the guard in place, `merge_onnx_external_data.py` is no longer a
  hard prerequisite for load; keep it for models that still fit under 2 GB
  protobuf when a single-file layout is preferred.

## Verification (both fasi)

1. **Load**: fp16 model with external data loads on console — no `weakly_canonical`/
   `ACCESS_DENIED`, no WDP minidump (`deploy.sh list-dumps`).
2. **Budget**: weights within 3801 MB — `bench-xbox-ort.sh <name> --gpu-sample`.
3. **Payoff**: prefill fp16-DML on `bench/prompts/long-1k.txt` vs CPU int4 of the
   same model → record the fp16-at-scale crossover in `docs/benchmarks.md`.
4. **Non-regression**: CPU int4 decode and diffusion unchanged (decode still routes
   to CPU).
