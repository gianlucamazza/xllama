# xllama Roadmap

## Phase 1 — CPU Baseline ✅ DONE

**Goal**: working UWP build, ORT GenAI CPU EP inference on Xbox Series S, MSIX self-contained.

Milestones:

- [x] Linux build green (CMake presets + CI `build-linux`)
- [x] Modular C++ bridge with RAII wrappers, CLI parser, unit tests (doctest)
- [x] Pivot from llama.cpp to ONNX Runtime GenAI + DirectML (`XLLAMA_USE_ORT`)
- [x] XAML-free programmatic UI (`MainPageController`, no WMC9999 workarounds)
- [x] GPU EP evaluated and ruled out: UWP pool ~768 MB → OOM on any usable LLM
- [x] CPU EP adopted: SmolLM2-360M-Instruct INT4 (403 MB), Zen 2 compatible
- [x] ONNX external data merged into self-contained `model.onnx` (AppContainer `weakly_canonical` fix)
- [x] SmolLM2-360M bundled inside MSIX as `DeploymentContent`; CI `build-uwp` green
- [x] ChatML prompt template applied for SmolLM2 Instruct

## Phase 2 — GPU Acceleration ✅ COMPLETE — GPU proven; verdict is per-workload (CPU decode, GPU prefill)

**Goal**: DirectML EP inference using Xbox RDNA 2 hardware.

**Status (2026-07-07, xllama v0.3.4)** — end-to-end DML bench completed:

**`VERDICT: GPU` + full decode** with a purpose-built SmolLM2-360M INT4 **DML**
variant (285 MB, ORT GenAI model builder `-p int4 -e dml`): weights resident on
GPU (`gpu-mem post-load: 307 MB`), GPU compute engine ~88% saturated during
decode, 739 tokens generated. **Result: 8.83 tok/s on GPU vs 70.9 tok/s on CPU
— the Zen 2 CPU is ~8× faster** (median of 3 runs, `phase2-dml.csv`).
Autoregressive decode of a 360M model is dominated by per-token DML dispatch
overhead; `MatMulNBits` on AVX2 wins decisively at this scale. The bundled
CPU-int4 variant is not DML-compatible (`80070057` in the fused node).

_Root cause history_ (see `CHANGELOG.md` [0.3.3]/[0.3.4],
`docs/uwp-constraints.md §7`): `887A0036` at `dml_helpers.cpp(140)` was the
**Agility SDK device factory** (`CreateDeviceFactory(614)`, in-box runtime)
colliding with the process-wide D3D12 device the XAML compositor creates at
`Window.Activate()`. Fixed architecturally in 0.3.4: **headless bench mode**
(`bench.flag` → no XAML, D3D12-clean process). Exp 1's "71.7 tok/s ≈ CPU"
(May) was a silent CPU fallback on a pre-614 OS. Config note: DML graph
capture requires `past_present_share_buffer: true` (all `genai_config-dml-*`
updated).

**Conclusion — Phase 2 closed, refined by the utilization matrix (v0.3.6)**:
DML EP works on GPU in the Xbox UWP sandbox (headless path). **CPU int4 stays
the default for decode-heavy chat** (68 tok/s vs GPU fp16 46.8), but the
picture is workload-dependent:

- **Prefill: GPU wins at scale** — 354 vs 198 tok/s at ~1k prompt tokens
  (1.8×, TTFT 3.0 s vs 5.3 s). DML fp16 is the better choice for prompt-heavy
  workloads (long context / RAG) already today.
- **The int4 GPU decode collapse (8.8 tok/s) is DirectML's non-fused low-bit
  kernel, not a missing/CPU one** (desk-check 2026-07-08, `docs/uwp-constraints.md
§12`): `MatMulNBits` IS registered and runs on the DML GPU (profile: one
  `DmlFusedNode`, 96%), but DML implements it as `DML_DEQUANTIZE`→fp16 + full
  `DML_GEMM` — materialising fp16 weights, so int4 moves _more_ bandwidth than
  fp16 (hence 8.8 < fp16's 46.8). The builder also gives DML `accuracy_level=0`
  vs CPU's `=4` (fused int8 MLAS → CPU's 68). No config we control fixes it; a
  fused low-bit GPU GEMM is a DirectML-team feature. **CPU int4 stays the decode
  winner; GPU's win is prefill.**

GPU pool estimate corrected: measured budget **3801 MB** (was "~768 MB");
disk is the real constraint. Upstream fix for the `887A0036` init failure
contributed and validated on console:
[microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280).
Future options: DML int4-AWQ variants (e.g. block-128) as a fused-kernel
proxy; larger models (1B+) where GPU compute amortises further — blocked
today by disk budget, not GPU memory.

Milestones:

- [x] Exp 1: test DML provider_options on SmolLM2-360M — loads without OOM; later found to be silent CPU fallback (see Status)
- [x] GPU-truth toolkit in place of PIX (unavailable in Dev Mode): ORT profiler with per-kernel EP attribution (`profile-dml-run.sh` + `analyze_ort_profile.py`), WDP `systemperf` GPU counters (`xbox-gpu-sample.sh`), in-app `QueryVideoMemoryInfo` (`gpu_mem_mb` CSV columns) — see `docs/uwp-constraints.md §11`
- [x] Run the profiled DML experiment on console — ✅ **`VERDICT: GPU`** via headless bench mode (0.3.4); weights on GPU (411 MB); CPU control run 70.9 tok/s
- [x] Root-cause and fix DML EP init failure — `887A0036` = Agility-factory vs XAML-compositor device conflict → headless bench mode (`uwp/App.cpp`, 0.3.4)
- [x] Evaluate Qwen2.5-0.5B INT4 ONNX as GPU EP candidate — ❌ CPU-int4 is ~822 MB (not ~200 MB); only the DML int4-awq variant (~507 MB) borderline fits the pool (see `docs/model-selection.md`)
- [x] Procure a DML-compatible model variant and run the end-to-end DML bench — ✅ SmolLM2-360M INT4 DML build (285 MB): decode completes, **8.83 tok/s GPU vs 70.9 CPU** (`phase2-dml.csv`)
- [x] Stage B (only if DML tok/s beats CPU) — ❌ dropped: DML is 8× slower than CPU at this scale; interactive app stays on CPU EP
- [x] Measure GPU vs CPU tok/s for the same model — ✅ v0.3.6 utilization matrix (same SmolLM2-360M, 3 variants × 2 prompts): CPU int4 decode 68.0 vs GPU fp16 46.8 vs GPU int4 8.8 tok/s; prefill inverts at ~1k tok (GPU fp16 354 vs CPU 198)

## Phase 3 — Benchmarks + Model Exploration ✅ DONE

**Goal**: reproducible results, n_threads tuning, evaluate alternative compact models.

Milestones:

- [x] Populate `bench/results/phase1-cpu.csv` — SmolLM2-360M INT4 at t=4,6,8; optimal: **t=4 → 71.4 tok/s**
- [x] Tune `n_threads` on Zen 2 Xbox: t=4 optimal; t=8 causes severe regression (28 tok/s — memory bandwidth saturation)
- [x] `bench-xbox-ort.sh` + `bench_threads.txt` mechanism for automated multi-thread bench runs
- [x] Evaluate Qwen2.5-0.5B INT4 ONNX — ❌ ~822 MB real (vocab embedding dominates); exceeds disk borderline and GPU pool
- [x] Evaluate Llama-3.2-1B INT4 ONNX CPU — ❌ ~1.77 GB real; USB-only, same class as SmolLM2-1.7B
- [x] `load_ms` in bench CSV: column existed but ORT path never measured it (always 0) — `run_inference` now times `OgaCreateModel`; baseline measured 2026-07-08 (`phase35-014-*.csv`, `phase35-1b-cpu.csv`): CPU int4 360M **1593 ms**, DML fp16 2830, DML int4 837, 1.7B CPU 6179

## Phase 3.5 — Hardware Ceiling ✅ CONSOLE-VALIDATED 2026-07-08 (residuals tracked in Phase 5)

**Goal**: close the gap between measured utilization (CPU ~13 GB/s, GPU ~34 GB/s
effective vs ~224 GB/s bus) and what the Series S can realistically deliver.
Levers ordered by cost/leverage; the first two are near-free and change the
denominators for everything else, so they go first.

**Software perf track** (merged to main, **console-validated 2026-07-08** —
v0.4.0.0 on Xbox; CSVs `bench/results/phase35-*.csv`):

- [x] **ORT GenAI 0.13.2 → 0.14.1** (Stage 1, v0.3.7): reduced per-token CPU
      overhead + prereq for continuous decoding. **Console: decode flat vs v0.3.6**
      (CPU int4 66.3 vs 68.0; DML fp16 46.8 = 46.8) — the bump is a prereq/overhead
      win, not a decode-rate win at this scale.
- [x] **KV-cache reuse across chat turns** (Stage 2, v0.3.8): persistent
      generator, append-only delta per turn. **Console: turn-2 prefill 4.87× faster**
      with reuse (103.7 ms / 22-tok delta vs 505.2 ms / 114-tok cold re-prefill).
      Correctness-guarded fallback; `kv_reuse` toggle (default on).
- [x] **Per-conversation CPU/GPU routing** (Stage 3, v0.3.9, default off): route
      long-prompt conversations to DML fp16, chat to CPU int4; sticky per
      conversation. Machinery done + dml-fp16 model on device; **interactive A/B
      still pending** (XAML UI — needs a person at the console).

- [x] **Image-generation spike** (v0.4.0, flagship hypothesis): **CONFIRMED on
      console 2026-07-08** — on a compute-bound fp16 batch (309 GFLOP), DirectML is
      **11.1× faster than CPU** (2403 vs 216 GFLOP/s). The inverse of text decode;
      diffusion is the GPU's workload. → C++ pipeline console-validated 2026-07-08:
      SD-Turbo fp16 512×512 in 6.9 s (`bench/results/phase5-diffuse.csv`).

Milestones:

- [x] **Game-mode designation** — ✅ settled 2026-07-08: checked in Dev Home,
      the package is **already designated Game**. All measured figures
      (3801 MB budget, v0.3.6 matrix, per-token dispatch overhead) are
      Game-mode numbers; the platform lever is already exhausted and the GPU
      decode gap is a DML/kernel issue, not App-mode scheduling. Optional
      residue: a reverse A/B (flip to App, one run, flip back) to quantify the
      App/Game delta for the record. Re-check the designation after every
      package reinstall (it can reset).
- [x] **Disk unblocked** — ✅ 2026-07-08: Dev Mode storage allocation raised to
      **90 GB** (Dev Home → Manage Dev Storage). The Q:\ ~2.2–2.5 GB budget in
      `docs/uwp-constraints.md §9` is superseded; 1B+ (and fp16 1.7B ~3.4 GB)
      variants can now be uploaded to LocalState. Caveat to verify on first
      big upload: community reports a ~2 GB per-file limit in Dev Mode
      (relevant for merged `model.onnx` > 2 GB). Exp 2 nobundle (Phase 4)
      remains useful for its own sake but is no longer a disk prerequisite.
- [~] **1B+ scale bench**: SmolLM2-1.7B. **CPU int4 measured on console
  2026-07-08: 20.6 tok/s decode, peak 2423 MB** (`phase35-1b-cpu.csv`).
  **fp16-DML blocked** (found 2026-07-08): a 1.7B fp16
  `model.onnx` is ~3.4 GB and **exceeds the 2 GB protobuf serialization
  limit**, so it cannot be merged self-contained; keeping external data
  re-triggers the `weakly_canonical` AppContainer crash (§8). So the
  pure-bandwidth fp16-at-scale test is **not deployable as-is** — it needs a
  sub-2 GB model, an upstream `weakly_canonical` fix, or ORT's external-data
  path made AppContainer-safe. int4-DML 1.7B is mergeable (<2 GB) but is the
  dead kernel (§12). Net: at 1.7B we can bench CPU int4 vs int4-DML, but not
  the fp16 bandwidth crossover — the interesting question stays blocked by
  the serialization/AppContainer constraint, not the GPU.
- [x] **Desk check upstream int4 status** — ✅ done 2026-07-08
      (`docs/uwp-constraints.md §12`). Verdict: `MatMulNBits` is present and runs
      on the DML GPU (not missing, not CPU fallback); DirectML implements it
      **non-fused** (`DML_DEQUANTIZE`→fp16 + `DML_GEMM`), and the builder gives
      DML `accuracy_level=0` vs CPU's `=4`. int4-on-DML decode cannot beat
      fp16-on-DML by any config we control — it's a DirectML kernel-design limit,
      not "weeks of HLSL" we could contribute. **This closes GPU int4 decode as a
      local lever.**
- [~] **int4 DML config confirmation** (fast negative): SmolLM2-360M
  `int4_block_size=128` and `int4_accuracy_level=4` variants are **built**
  (rebuilt 2026-07-09, merged self-contained, in
  `~/.cache/xllama-diffusion/int4-variants/`); one console bench each to
  confirm they stay ≈ 8.8 tok/s (the
  kernel structure predicts no material gain). If either beats fp16-DML it
  would refute §12 — worth the ~5 min. Not a path forward, just closure.
- [x] **llama.cpp CPU A/B** — ✅ **MEASURED 2026-07-08, hypothesis FALSIFIED.**
      The lane was built (uwp/ggml-uwp.vcxproj static ggml+llama, `patches/0001`
      AppContainer guards for 5 desktop-only APIs, CI `build (llamacpp)` variant,
      `-Backend llamacpp`) and llama.cpp **runs on the console in AppContainer**.
      SmolLM2-360M **Q4_K_M** decode scaling (standard-512, ChatML,
      `bench/results/phase35-llamacpp-scaling.csv`): t1 **19.9**, t4 **51.5**,
      t6 **62.9** tok/s; **t7/t8 livelock** (ggml spin-wait threadpool
      oversubscribes the ~6 cores Dev Mode leaves the app — no thread affinity in
      AppContainer). Versus ORT int4 @8t (66.3): **parity, not 2×** — Q4*K_M does
      not extract more bandwidth than ORT's `MatMulNBits` on this machine; both
      saturate ~13 GB/s effective. Prefill is \_worse* (141 vs 220 tok/s).
      **Verdict: ORT GenAI stays the text backend** (better prefill, KV-reuse,
      routing, DML); the llamacpp lane remains in CI as a bench-only variant.
      Fixed on the way (all real bugs): `#ifdef XLLAMA_USE_ORT` vs `=0`,
      tokenize size-query sign, no_perf-hidden timings, obj-name collisions
      (ggml.c/.cpp), 128 `src/models/*.cpp` in the static lib.
- [x] **Per-workload routing in the app** — ✅ implemented (Stage 3, see software
      perf track above); pending on-console A/B with the DML fp16 model present.
- [ ] (deprioritised by §12) **Fused low-bit GPU GEMM for DirectML** — the real
      unlock for GPU int4 decode, but it lives in **DirectML itself**
      (`DmlOperatorMatMulNBits` currently dequantises to fp16), not in an ORT-side
      patch we can carry via the PR #2280 pipeline. Track as an upstream
      DirectML feature request, not a local contribution.
- [ ] (optional) in-app memory-bandwidth micro-bench (`membw.flag`) to fix the
      CPU ceiling denominator precisely.

## Phase 4 — In-App Download + Publication 🔮 FUTURE

**Goal**: remove bundled-model constraint; demo video; technical write-up.

Milestones:

- [x] `ModelDownloader` with chunked `HttpClient` download from HF endpoint (Exp 2, `uwp/model-downloader.cpp`)
- [x] `EnsureModelAsync()` bootstrap: LocalState → InstalledPath → HF download fallback chain
- [x] USB external drive fallback at `E:\xllama\models\<name>` in `resolve_model_path` (Exp 3)
- [x] No-bundle build variant to unblock Exp 2: `build-uwp.ps1 -NoBundledModel` (MSBuild `XllamaNoBundledModel=true` excludes the model ItemGroup); CI publishes `xllama-appx-nobundle` alongside `xllama-appx`
- [x] Validate Exp 2 on console — ✅ 2026-07-08: the nobundle app downloaded the full model (417 MB merged) from the GitHub Release catalogue inside the AppContainer, byte-exact, `.complete` written. (The upstream HF repo turned out to ship a non-merged model.onnx + a file list with a nonexistent entry — the download had been broken from the start; distribution moved to Release assets.)
- [x] Remove model bundle from MSIX — ✅ 2026-07-08 (ItemGroup deleted; CI matrix simplified to default+llamacpp; `xllama-appx` is now the 19 MB no-model package)
- [x] `model-manifest.json` — ✅ 2026-07-08 (`uwp/models/manifest.json` catalogue + LocalState override; ComboBox and downloader de-hardcoded)
- [ ] Demo video: model loaded and running on Xbox hardware
- [x] Technical report — ✅ 2026-07-08 draft written (`docs/technical-report.md`,
      the measured story 0.3.x→1.0); publication venue (GitHub Discussions vs
      arXiv) still to pick
- [x] Tagged v1.0.0 release — ✅ 2026-07-08 (`gh release view v1.0.0`: 19 MB
      MSIX + `.cer` + VCLibs x64; models on the `models-v1` release)

## Phase 5 — Post-1.0 improvements 🚧 IN PROGRESS (2026-07-09)

- [x] **In-process diffusion experiment** (`diffuse-inproc.flag`) — ✅ **PASS
      2026-07-09**: plain ORT DML coexists with the XAML compositor (full
      pipeline in-process, 5.57 s, coherent PNG, no 887A0036). Image generation
      no longer needs the restart flow (runbook §7b, `docs/uwp-constraints.md`
      §7). **Follow-up**: wire the in-app Generate (no restart).
- [x] Diffusion progress/cancel plumbing (`diffuse-progress.txt`,
      `diffuse-cancel.flag`) — PR #20
- [x] **Runtime backend dispatch** (PR #27) — ORT GenAI + llama.cpp compile
      into one binary; `unified` MSIX variant CI-green. Unblocks modern
      GGUF-only models (Qwen3.5, LFM2). Fase 2 (UI `kind:gguf`) next.
- [x] **Modern-model survey** — Qwen3.5-0.8B / LFM2.5-350M load via llama.cpp;
      Qwen3-0.6B builds via ORT (969 MB); TAESD decoder validated (`docs/model-selection.md`).
- [ ] Interactive validations at the pad: §2 routing A/B + Image dialog flow
- [~] Closure benches: int4 `block_size=128` / `accuracy_level=4` (§12
  confirm/refute) — running on console 2026-07-09; `load_ms` baseline done
- [x] Fase 2: catalogue `kind:gguf` → `sp.backend`, gate KV-reuse/routing off
      for GGUF (plumbing complete 2026-07-09 via PR #30 + layout-aware Auto,
      resolve support, bench guard, tests; asset promotion + console benches
      remain Fase 2b).
- [ ] If §2 shows `887A0036` on the GPU turn in XAML: vendor the patched GenAI
      DLL (PR microsoft/onnxruntime-genai#2280, console-validated) until the
      fix ships upstream
- [ ] `diffusion/requirements.txt` toolchain bump + re-validation (dependabot:
      27 dev-only alerts; pins are a coherent export set — bump requires
      re-running export → convert → validate_pipeline)
