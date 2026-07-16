# xllama Roadmap

**Shipping (2026-07-15):** MSIX **1.1.8.x** — `xllama-appx` from `build-uwp.yml`
(unified ORT + llama.cpp + **PatchedGenAI #2280** + **PatchedOrt** extdata;
Major.Minor.Build from `AppxManifest.xml`, Revision auto-stamped from the CI run
number). First-launch default chat model on unified: **`lfm25-350m`** (~94 tok/s).
**1.1.8.0** promotes the console-validated AppContainer external-data
`onnxruntime.dll` into the default shipping package (cached from the
`vendor-dlls-v1` release; hash pin in `vendor/onnxruntime-patched/SHA256SUMS` —
no 1–3 h ORT rebuild on every PR). Carries 1.1.7.0: GGUF **KV-reuse** (4.07×),
**quant auto-upgrade**, **membw**, **SmolLM2-1.7B** on catalogue, Gemma family.
Full numbers: `docs/benchmarks.md`; architecture: `docs/architecture.md`.
Console gates: `validate-console.sh all` → **ALL PASS** (2026-07-14).
**GitHub Release [v1.1.8.0](https://github.com/gianlucamazza/xllama/releases/tag/v1.1.8.0)**
published 2026-07-16 (MSIX + cert + VCLibs). Field smoke of the combined 1.1.8
package was **pending at release** (Device Portal unreachable from the release
host); re-run `./scripts/install-latest-build.sh` + a short LFM chat when the
console is online. See
[`docs/benchmarks.md`](docs/benchmarks.md),
[`docs/recommended-config.md`](docs/recommended-config.md) and
[`docs/console-validation-runbook.md`](docs/console-validation-runbook.md).

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

## Phase 3.5 — Hardware Ceiling ✅ CONSOLE-VALIDATED 2026-07-08 / 2026-07-14

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
      conversation. **Console-validated 2026-07-14** via `validate-console.sh
routing` on unified 1.1.3.0 + patched GenAI: long turn auto→GPU (959 tok),
      new short chat auto→CPU, no `887A0036`.

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
- [x] **1B+ scale bench (CPU int4)** — ✅ SmolLM2-1.7B **cpu-int4 measured
      2026-07-08: 20.6 tok/s decode, peak 2423 MB** (`phase35-1b-cpu.csv`).
      Catalogue entry + `package-catalogue-ort-model.sh` ready; **1.4 GB asset
      upload to `models-v1` optional** (WDP works today). **fp16-DML at scale —
      resolved 2026-07-15**: external-data loading >2 GB is now **unblocked** by the
      patched ORT DLL (§8 Fix B: `weakly_canonical` guard + `ReadFile` 16 MB chunk,
      console-validated with a 1.86 GB-extdata model). But the fp16-at-scale prefill
      crossover is **not reachable on Series S for other reasons**: a native-DML
      1B fp16 (2.49 GB) loads (2878/3801 MB) yet OOMs _inference_
      (`AppendTokenSequences … 8007000E`, §7) — the GPU budget can't hold weights +
      the DML inference working set at ≥1 B. int4-DML 1.7B is mergeable but hits the
      §12 non-fused low-bit GEMM. Net: the >2 GB unblock helps **CPU** external-data
      models, not bigger fp16 on the GPU (budget wall). See `docs/fp16-extdata-runbook.md`.
- [x] **Desk check upstream int4 status** — ✅ done 2026-07-08
      (`docs/uwp-constraints.md §12`). Verdict: `MatMulNBits` is present and runs
      on the DML GPU (not missing, not CPU fallback); DirectML implements it
      **non-fused** (`DML_DEQUANTIZE`→fp16 + `DML_GEMM`), and the builder gives
      DML `accuracy_level=0` vs CPU's `=4`. int4-on-DML decode cannot beat
      fp16-on-DML by any config we control — it's a DirectML kernel-design limit,
      not "weeks of HLSL" we could contribute. **This closes GPU int4 decode as a
      local lever.**
- [x] **int4 DML config confirmation** — ✅ **closed inconclusive 2026-07-09**
      (runbook §5b): block128 fails DML partition; acc4 not bench-promoted.
      §12 desk-check stands — not a local lever.
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
- [x] **Per-workload routing in the app** — ✅ implemented + console-validated
      2026-07-14 (`validate-console.sh routing`, see software perf track above).
- [x] **#2280 patched GenAI in shipping MSIX** — unified+XAML routing GPU
      validated 2026-07-14 (no `887A0036`). Upstream merge into NuGet TBD.
- [ ] (deprioritised — see Phase 6) **Fused low-bit GPU GEMM for DirectML** (upstream)

## Phase 4 — In-App Download + Publication ✅ DONE (demo video open)

**Goal**: remove bundled-model constraint; technical write-up; publication assets.

Milestones:

- [x] `ModelDownloader` with chunked `HttpClient` download from HF endpoint (Exp 2, `uwp/model-downloader.cpp`)
- [x] `EnsureModelAsync()` bootstrap: LocalState → InstalledPath → HF download fallback chain
- [x] USB external drive fallback at `E:\xllama\models\<name>` in `resolve_model_path` (Exp 3)
- [x] No-bundle build variant to unblock Exp 2: `build-uwp.ps1 -NoBundledModel` (MSBuild `XllamaNoBundledModel=true` excludes the model ItemGroup); CI publishes `xllama-appx-nobundle` alongside `xllama-appx`
- [x] Validate Exp 2 on console — ✅ 2026-07-08: the nobundle app downloaded the full model (417 MB merged) from the GitHub Release catalogue inside the AppContainer, byte-exact, `.complete` written. (The upstream HF repo turned out to ship a non-merged model.onnx + a file list with a nonexistent entry — the download had been broken from the start; distribution moved to Release assets.)
- [x] Remove model bundle from MSIX — ✅ 2026-07-08 (ItemGroup deleted; CI matrix simplified to default+llamacpp; `xllama-appx` is now the 19 MB no-model package)
- [x] `model-manifest.json` — ✅ 2026-07-08 (`uwp/models/manifest.json` catalogue + LocalState override; ComboBox and downloader de-hardcoded)
- [ ] Demo video: model loaded and running on Xbox hardware — **tracked under Phase 6**
- [x] Technical report — ✅ 2026-07-08 draft (`docs/technical-report.md`); published
      as [Discussion #76](https://github.com/gianlucamazza/xllama/discussions/76)
      (2026-07-15). arXiv only if a formal citation is needed later.
- [x] Tagged v1.0.0 release — ✅ 2026-07-08 (`gh release view v1.0.0`: 19 MB
      MSIX + `.cer` + VCLibs x64; models on the `models-v1` release)

## Phase 5 — Post-1.0 improvements ✅ DONE (2026-07-14)

**Goal**: unified shipping build, modern GGUF models, in-process diffusion, console
validation gates, patched GenAI in XAML.

- [x] **In-process diffusion experiment** (`diffuse-inproc.flag`) — ✅ **PASS
      2026-07-09**: plain ORT DML coexists with the XAML compositor (full
      pipeline in-process, 5.57 s, coherent PNG, no 887A0036). Image generation
      no longer needs the restart flow (runbook §7b, `docs/uwp-constraints.md`
      §7).
- [x] **In-app Generate (no restart)** — Image dialog calls `run_diffuse()` on a
      background MTA thread; progress/cancel via `diffuse-progress.txt` /
      `diffuse-cancel.flag`.
- [x] Diffusion progress/cancel plumbing (`diffuse-progress.txt`,
      `diffuse-cancel.flag`) — PR #20
- [x] **Runtime backend dispatch** (PR #27) — ORT GenAI + llama.cpp compile
      into one binary; `unified` MSIX variant CI-green. UI `kind:gguf` complete
      (picker, KV/routing gates, bench path, console-validated 2026-07-10/14).
- [x] **Modern-model survey** — Qwen3.5-0.8B / LFM2.5-350M load via llama.cpp;
      Qwen3-0.6B builds via ORT (969 MB); TAESD decoder validated (`docs/model-selection.md`).
- [x] **TAESD UI + asset pipeline** — Image dialog toggle, `diffuse_taesd_vae` in
      settings.json, `scripts/export-taesd-asset.sh`; **console-validated 2026-07-14**
      (runbook §7c, VAE **593–625 ms**).
- [x] **Patched GenAI DLL pipeline** — `patches/onnxruntime-genai-2280-dml-fallback.patch`,
      `scripts/vendor-genai-dml-patch.ps1`, `build-uwp.ps1 -PatchedGenAI`.
- [x] **Recommended config doc** — `docs/recommended-config.md` +
      `bench/configs/settings-modern.json`.
- [x] **Interactive validations** — autopilot + `validate-console.sh all` →
      **ALL PASS** on console 2026-07-14 (routing, GGUF `lfm25-350m`, TAESD §7c).
      Shipping MSIX at the time was **1.1.5.0** (unified + PatchedGenAI), CI-green
      and deployed; current shipping version is in the header above.
- [x] Closure benches: int4 `block_size=128` / `accuracy_level=4` (§12
      confirm/refute) — closed inconclusive (PR #29, §12 stands)
- [x] Fase 2: catalogue `kind:gguf` → `sp.backend`, gate KV-reuse/routing off
      for GGUF (plumbing complete 2026-07-09 via PR #30 + layout-aware Auto,
      resolve support, bench guard, tests).
- [x] Fase 2b (assets): Qwen3.5-0.8B + LFM2.5-350M Q4_K_M published on
      `models-v1` with catalogue entries (LFM license redistributed per §4a);
      host smoke test OK. **Console benches measured 2026-07-10** (unified
      1.1.1.0, t6, `phase5-gguf.csv`): Qwen3.5-0.8B decode **35.1** tok/s
      (prefill 98.1), LFM2.5-350M decode **94.2** tok/s (prefill 241.4, ~42%
      over the ORT 360M baseline). Found + fixed on the way: GGUF dir load in
      the bench path, llama default-threads livelock (capped at 6 on UWP),
      unified CSV labels.
- [x] **Promotion of `unified` + PatchedGenAI to default CI** — `build-uwp.yml`
      ships `xllama-appx` as unified+#2280 (2026-07-14); `build-uwp-patched.yml`
      kept as manual fallback lane.
- [x] `diffusion/requirements.txt` toolchain bump + re-validation (2026-07-10):
      torch 2.9.1 / optimum-onnx 0.1.0 / transformers 4.57.6 / diffusers 0.39.0.
      Full recipe re-run green: export (diff 0.0102, warning-level precedent),
      fp16 convert (3× <2 GB, LOAD OK EXTENDED), validate_pipeline std=51.9 +
      visual, golden vectors byte-identical outside `reference`, ctest 100%,
      TAESD export + swap OK. Residual alerts documented in requirements.txt
      (kernels not installed; Trainer unused). New-generation exports declare
      scalar UNet `timestep` — `diffuse.cpp` is shape-aware since 2026-07-10;
      console gate **closed** (runbook §7 measured 2026-07-08, §7b PASS
      2026-07-09, §7c TAESD 2026-07-14).
- [x] **Model provisioning layer (F1)** — `IsModelProvisioned`,
      `EnsureModelNamedAsync`, background `gpu_model` download when
      `routing≠0`, routing guards, Settings model-change re-provision
      (`f3d733e`).
- [x] **Catalogue distribution (F2)** — `hf_base_url` + prefixed remotes for
      `smollm2-360m-dml-fp16` and `smollm2-1.7b-cpu-int4`;
      `scripts/package-catalogue-ort-model.sh`; **dml-fp16 published on
      `models-v1`** (691 MB merged, built ORT GenAI `-p fp16 -e dml`).
- [x] **`routing_policy.h` (F5)** — extracted 600-tok threshold + GGUF/routing
      capability gates; unit tests; `MainPage` uses shared policy.
- [x] **Default modern settings (F6)** — `bench/configs/settings-modern.json`
      default chat **LFM2.5-350M** (GGUF) on unified builds; routing GPU
      unchanged (`smollm2-360m-dml-fp16`).
- [x] **`validate-console.sh` hardening** — `model_provisioned` WDP basename fix;
      `upload_file` mkdirs remote dirs (chats, nested model paths).

## Phase 6 — Publication + polish 🔮 NEXT

Open items only — everything above is measured or shipped.

- [x] **Perf + architecture backlog** (2026-07-14, PRs #53–#60). CI: `src/models`
      wildcard + drift-check (no more per-bump LNK2001); unified stop-sequence
      helper; bench `.done` verified-delete. Perf: **GGUF KV-reuse** (persistent
      `llama_context`) — turn-2 prefill **4.07×** on gemma3-270m
      (`phase6-gemma-kv.csv`); **gemma4-e2b → Q3_K_S** (fixes the 2-bit EOG,
      15.3 tok/s). Investigated + reverted: AppContainer mmap (load is repack-bound,
      not file-read-bound — no benefit). See `docs/benchmarks.md`.
- [x] **Per-architecture chat template** — hard-coded ChatML replaced by a
      data-driven `ChatFormat` (`src/bridge/chat_prompt.cpp`, `chat_format_for()`);
      byte-identical ChatML preserved, Gemma (`<start_of_turn>…<end_of_turn>`)
      added, KV-reuse invariant unit-tested. UWP + bench call one abstraction.
- [x] **Gemma family — console-validated** (MSIX 1.1.6.x, `phase6-gemma.csv`).
      Vendored `llama.cpp` carries `gemma3` + `gemma4`; catalogue entries
      `gemma3-270m` and `gemma4-e2b`. On Xbox Series S: **gemma3-270m** 76.8 tok/s
      decode (368 MB); **gemma4-e2b** (2.45 GB Q3_K_S) loads at 2742 MB RAM and
      decodes 15.3 tok/s. **The "Gemma-4 too big" verdict is overturned** — the
      ~2 GB Dev Mode per-file limit does not apply to GGUF (a >2 GB single file
      loads). See `docs/benchmarks.md` (incl. root-cause notes on the negative
      performers).
- [x] **Benchmark report + comparative charts** — `docs/benchmarks.md` +
      self-contained `docs/benchmarks-charts.html` consolidate every tested model
      (decode/prefill/RAM across 3 backends) from `bench/results/`.
- [x] **Automated MSIX versioning** — CI stamps the Revision from `run_number`
      (`build-uwp.ps1 -BuildRevision`); Major.Minor.Build stays the semantic
      version. Console in-place updates never hit the same-identity block.
- [x] **In-app HF download verified on-console** (2026-07-15) — the app's own
      downloader pulled the 2.29 GB single `.gguf` from HF and loaded+generated it,
      confirming the >2 GB self-download path (see `docs/benchmarks.md`).
- [x] **`IsModelProvisioned` quant-upgrade** (2026-07-15, PR #64) — expected-aware
      overload + dir reconcile: a stale-quant dir is now re-downloaded to the
      manifest's file. Verified on-console — `gemma4-e2b` with a stale IQ2_M +
      `.complete` marker → removed IQ2_M → downloaded Q3_K_S (2.45 GB) → loaded.
      13 host doctest cases. See `docs/benchmarks.md`.
- [ ] **Demo video** — model loaded and running on Xbox hardware (Phase 4 carry-over).
      **Checklist:** (1) deploy 1.1.8 shipping MSIX, (2) first-launch LFM download,
      (3) chat short + long (routing), (4) Image Generate one frame, (5) 60–90 s
      capture via capture card / Game Bar if available. No code blocker.
- [x] **Publication venue** — GitHub Discussions; **posted**
      [Discussion #76](https://github.com/gianlucamazza/xllama/discussions/76)
      (technical-report entrypoint + SSOT links). arXiv only if a formal citation
      is needed later. Report stays a dated v1.0 snapshot; live numbers stay in
      `docs/benchmarks.md`.
- [x] **`smollm2-1.7b-cpu-int4` on `models-v1`** (2026-07-15) — the 4 catalogue
      assets (incl. the 1.47 GB `model.onnx`) uploaded to the `models-v1` release;
      verified on-console: in-app download of all 4 files → load → generate (63
      decode tokens). Enables the 1.7B chat model (~20.6 tok/s) via the picker.
- [x] **`install-latest-build.sh`** — no longer leaves `bench.flag` by default;
      headless bench mode is now `--bench` opt-in (a normal install launches
      straight into the UI). See `scripts/install-latest-build.sh`.
- [x] (optional) **membw.flag** micro-bench — pins the CPU bandwidth denominator
      behind decode. `measure_membw` (STREAM read/copy/triad), `xllama-cli --membw`
      (host) + `membw.flag` (console → `membw-result.csv`). **Console-measured
      2026-07-15**: Xbox Zen 2 read 12.35 GB/s @1t / 30.29 @8t — the single-thread
      read matches the deduced ~13 GB/s GEMV denominator. See `docs/benchmarks.md`.
- [x] **External-data ONNX loading unblocked >2 GB** (2026-07-15, PRs #67/#68/#71/#72/#73)
      — patched ORT DLL (`patches/onnxruntime-extdata-appcontainer.patch`): 
      `weakly_canonical` guard + `ReadFileIntoBuffer` 16 MB chunk. Console-validated
      with a 1.86 GB-extdata int4 model — **6/6 restarts, 0 crashes**. **Closed
      negative:** 1B fp16 GPU inference OOM (§7 budget wall). See
      `docs/fp16-extdata-runbook.md`.
- [x] **Promote the patched ORT DLL into the shipping build** (2026-07-15, v1.1.8.0)
      — `build-uwp.yml` downloads the pinned DLL from `vendor-dlls-v1`, verifies
      `vendor/onnxruntime-patched/SHA256SUMS`, installs over NuGet
      (`-PatchedOrt`). Full source rebuild stays in `build-uwp-ort-patched.yml`
      (refresh the release asset after re-validation). Optional catalogue entry
      for a public >2 GB external-data int4 model remains open (USB/custom load
      already works with the shipping DLL).
- [ ] (optional) **Catalogue entry for a >2 GB external-data int4 model** — make
      the PatchedOrt capability visible in the picker without USB/WDP staging.
- [ ] (upstream, deprioritised) **Fused low-bit GPU GEMM in DirectML** — §12;
      not a local contribution via #2280
