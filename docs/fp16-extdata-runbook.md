# Runbook — Unblock fp16 models >2 GB on the GPU (DirectML)

Operational entrypoint for the ">2 GB fp16 on GPU" plan. Goal: load an fp16 ONNX
GenAI model with **external `.onnx.data`** on the Xbox, which today crashes in
`ValidateExternalDataPath` → `weakly_canonical` on the AppContainer walk over
`Q:\Users\UserMgr0` (`docs/uwp-constraints.md §8`). The merge workaround
(`scripts/merge_onnx_external_data.py`) caps at the 2 GB protobuf single-file
ceiling, so fp16 models >2 GB cannot ship — this is the last local GPU lever
(DirectML itself is in maintenance mode; the int4-DML decode wall in §12 is
permanent upstream).

## Feasibility gate (read first)

- GPU budget is **3801 MB** per-process (`docs/uwp-constraints.md §7`).
- Useful window: fp16 **~1B / 2–2.5 GB** — big enough to need the fix (>2 GB),
  small enough to run (≤3801 MB with KV/activations). 1.7B fp16 (~3.4 GB) is
  out of budget; even Llama-3.2-1B fp16 (~3.0 GB) is borderline.
- DML-fp16 builds in that window are scarce on HF (mostly int4-DML or fp16-cuda),
  so the target model is **built locally** with the ORT GenAI model builder.
- The `weakly_canonical` check runs at model **parse**, before GPU weight
  allocation — so the spike question ("does it get past ValidateExternalDataPath?")
  is answerable even with a slightly over-budget model.

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
- ⚠️ **Second obstacle: intermittent `errcode 1450` (ERROR_NO_SYSTEM_RESOURCES)**
  on `ReadFile model.onnx.data` — root-caused and fixed (pending re-validation).
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
walk. Mirrors the #2280 GenAI patched-DLL doctrine.

**Preferred — CI (no local Windows box):** run the **`build-uwp-ort-patched`**
workflow (dispatch-only, `.github/workflows/build-uwp-ort-patched.yml`). It builds
both patched DLLs (this + #2280 GenAI), packages a unified MSIX, and uploads the
MSIX (`xllama-appx-ort-patched`) + the ORT DLL (`onnxruntime-patched-dll`). Then
deploy the MSIX and re-run the load test — but this time with a **catalogue or
LocalState** model (the fix is not USB-specific; the USB→LocalState copy that
refuted Fase 0 is now harmless).

```powershell
# Local alternative (VS2022 + Python + CMake):
nuget restore uwp/xllama.sln
./scripts/vendor-ort-extdata-patch.ps1 -Build   # clone v1.24.4, apply guard, build, vendor
./scripts/build-uwp.ps1 -Configuration Release -Platform x64 -PatchedGenAI
```

- Patch: `patches/onnxruntime-extdata-appcontainer.patch` (reference diff; the
  script falls back to a context-tolerant transform if `git apply` drifts, and
  verifies the guard is present before the build).
- **Cost**: a full ORT DirectML source build (1-3 h) — materially heavier than the
  GenAI DLL build. Kept in a dispatch-only lane; `build-uwp.yml` still ships the
  vanilla ORT until this patch is console-validated (same doctrine as #2280).
- Follow-up: with the guard in place, `merge_onnx_external_data.py` is no longer a
  hard prerequisite — relax `scripts/package-catalogue-ort-model.sh` accordingly.

## Verification (both fasi)

1. **Load**: fp16 model with external data loads on console — no `weakly_canonical`/
   `ACCESS_DENIED`, no WDP minidump (`deploy.sh list-dumps`).
2. **Budget**: weights within 3801 MB — `bench-xbox-ort.sh <name> --gpu-sample`.
3. **Payoff**: prefill fp16-DML on `bench/prompts/long-1k.txt` vs CPU int4 of the
   same model → record the fp16-at-scale crossover in `docs/benchmarks.md`.
4. **Non-regression**: CPU int4 decode and diffusion unchanged (decode still routes
   to CPU).
