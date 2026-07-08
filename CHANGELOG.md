# Changelog

All notable changes to xllama are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Corrected — int4 DML decode diagnosis (desk-check 2026-07-08)

The [0.3.6] entry below (and the docs at the time) attributed the int4 GPU decode
collapse (8.8 tok/s) to a "missing fused int4 DML kernel" that, if added, would
imply ~180 tok/s. A source-level desk-check **refines that**: `MatMulNBits` is
present in the graph (225 nodes) and **runs on the DML GPU** — the profiled run
shows one `DmlFusedNode` on `DmlExecutionProvider` (96%), no CPU fallback. The
real limit is that DirectML's `MatMulNBits` is **non-fused**
(`DML_DEQUANTIZE`→fp16 + full `DML_GEMM`, materialising fp16 weights), so int4
moves more bandwidth than fp16 and cannot beat it on DML; the builder also gives
DML `accuracy_level=0` vs CPU's fused-int8 `=4`. Full analysis + config tests in
`docs/uwp-constraints.md §12`; `ROADMAP.md` Phase 3.5 updated. Net: GPU int4
decode is blocked by a DirectML kernel-design limit (out of our control), not a
kernel we could contribute — CPU int4 stays the decode default.

### Added — diffusion model toolchain (image generation, `diffusion/`)

Model-side toolchain for running SD-Turbo (1-step distilled diffusion) on the
console GPU — the workload that plays to DirectML's strength (compute-bound fp16
batch), unlike LLM decode.

- **Validated on the owned layer (2026-07-08)**: SD-Turbo exported to ONNX and
  run through **ONNX Runtime (CPU EP)** generates a coherent 512×512 image in
  ~13 s / 1 step — the same artifact + runtime that runs on Xbox via the
  DirectML EP.
- `diffusion/export_sd_turbo.sh` + `requirements.txt`: reproducible, **pinned**
  toolchain (Python 3.10, torch 2.4.1 legacy ONNX tracer, optimum 1.23.3,
  transformers 4.46.3, diffusers 0.31.0). The pins are load-bearing: newer torch
  uses the dynamo/onnxscript exporter that optimum 1.23 can't consume (external-
  data-naming + LayerNormalization opset-downgrade bugs).
- `diffusion/generate_onnx.py`: validates the exported ONNX through ORT.
- `diffusion/convert_fp16.py`: fp16 conversion for the 3801 MB GPU budget. SD
  components are each < 2 GB (UNet fp16 ~1.7 GB), so all save self-contained and
  avoid the AppContainer `weakly_canonical` crash (§8) — unlike a 1.7B LLM fp16
  blob. (A cleaner fp16 export uses `optimum-cli --fp16 --device cuda` on a GPU.)
- Next: the C++ pipeline (3 ORT DirectML sessions + scheduler + CLIP tokenizer)
  behind a `diffuse.flag` headless mode, on the plain-ORT DirectML foundation
  already proven by the image spike (PR #3).

## [0.4.0.0] - 2026-07-08

### Added — image-generation spike (plain ORT DirectML, new model axis)

First step toward image/vision models on the console. The desk-check (§12) showed
the GPU loses at text decode because that workload is M=1, dispatch-bound, and
int4-DML is non-fused — the GPU's _worst_ case. Image generation (diffusion) is
the opposite: compute-bound fp16 batch, the case where DML already wins (prefill).
This spike tests that hypothesis cheaply before building a full diffusion pipeline.

- `uwp/image-spike.cpp`: runs a compute-bound conv model through the **plain
  ONNX Runtime DirectML** EP (not ORT GenAI — `onnxruntime.dll` was already a
  dependency) plus a CPU EP control, measuring forward-pass latency + GFLOP/s.
  Model `imgspike.onnx` (`scripts/gen_imgspike_model.py`, deterministic): 17
  Conv(3×3,64ch)+Relu on 1×3×512×512 fp16, ~309 GFLOP/forward — a faithful proxy
  for one diffusion UNet step.
- `uwp/App.cpp`: generalised the headless flag dispatch — `image.flag` →
  `run_image_spike()`, `bench.flag` → `main_loop()` (shared `HeadlessView`, same
  D3D12-clean host that avoids the 887A0036 compositor conflict).
- `uwp/xllama.vcxproj`: added the ORT DirectML NuGet include dir + the new source.
  `onnxruntime.lib` was already linked.
- Writes `imgspike-result.csv` (+ `.done`): DML vs CPU ms, GFLOP/s, GPU speedup.
- On-console validation pending: upload `imgspike.onnx` to LocalState + drop
  `image.flag`, fetch the CSV. **Hypothesis**: DML ≫ CPU here (compute-bound), the
  inverse of text decode — if confirmed, a distilled SD-Turbo/LCM diffusion
  pipeline becomes the flagship GPU workload (Phase 5).

## [0.3.9.0] - 2026-07-08

### Added — per-conversation CPU/GPU routing (Stage 3, default off)

The v0.3.6 matrix showed the EP choice is per-workload: DML fp16 wins prompt
prefill at scale (1.8× at ~1k tokens), CPU int4 wins decode. The app can now
route between them.

- Settings `routing` (0 = CPU only / default = unchanged behaviour, 1 = GPU
  only, 2 = auto: route first prompts over ~500 est. tokens to GPU) + `gpu_model`
  (DML fp16 model dir, default `smollm2-360m-dml-fp16`); a routing ComboBox in
  the Settings dialog.
- The decision is made once at a conversation's first turn and **sticky** for its
  lifetime (the KV cache is per-EP): `m_active_model` holds the routed dir,
  cleared on new/loaded chat so each conversation re-decides. Reuses the existing
  single-slot `EnsureSession` (one model resident at a time) — switching
  conversations may reload, which is memory-safe on the 3801 MB budget.
- Default routing = CPU, so behaviour is identical until a user opts in. GPU
  routing requires the DML fp16 model present on device (LocalState/USB).
- On-console validation pending: confirm auto-routing picks GPU for long prompts
  and that TTFT improves for prompt-heavy conversations.

## [0.3.8.0] - 2026-07-08

### Added — KV-cache reuse across chat turns (continuous decoding)

The interactive app re-prefilled the **entire** ChatML history on every turn
(`BuildChatMLPrompt` + a fresh `OgaGenerator` per `generate()`), so turn-N TTFT
paid to re-process ~all prior tokens (~1.8k at budget → seconds). `OrtSession`
now keeps its generator alive across turns and appends only the new turn's
tokens (`OgaGenerator_AppendTokenSequences` on the persistent generator), so the
per-turn prefill covers just the delta.

- `GenerateParams` gains `reuse_kv` / `reset_kv`; `InferenceResult` gains
  `ended_with_stop` and now populates prefill telemetry (`n_p_eval`/`t_p_eval_ms`)
  on the interactive path too (previously bench-only). Decode timing excludes
  prefill, matching the bench convention.
- `OrtSession` holds `m_chat_gen`/`m_chat_params`/`m_chat_stream` + the bound
  sampling signature; the stateless path is preserved unchanged and still used
  when `reuse_kv` is false.
- `MainPageController::BuildDeltaPrompt` builds the incremental turn; the KV is
  reused only when valid and no context turn was evicted (RewindTo truncates the
  tail, not the head → eviction forces a full re-prefill). Reuse is invalidated
  on new/loaded chat, settings change, abort, and any generator failure.
- **Correctness guard**: a continuation that fails before emitting a token
  auto-falls back to a full re-prefill (no UI double-streaming) — worst case is
  the previous behaviour, never a wrong result.
- Settings toggle `kv_reuse` (ToggleSwitch + `settings.json`, default on) so the
  win can be A/B'd on console.
- On-console validation pending: measure turn-2 TTFT with reuse on vs off
  (Stage 2b bench) and confirm multi-turn coherence before trusting it as default.

### Added — multi-turn TTFT bench (Stage 2b)

- Headless bench gains a KV-reuse measurement: when `bench_turns.txt` is present
  (turn-2 user prompt; `prompt.txt` supplies turn 1), `main_loop` runs both turns
  on one persistent `Session` and measures turn-2 prefill **with reuse** (append
  only the delta) vs the **cold** baseline (full re-prefill of the 2-turn
  context), writing `bench-kv-result.csv` (+ `.done`) with a `speedup` column and
  logging the numbers. This measures the Stage 2 win on console instead of
  assuming it.

## [0.3.7.0] - 2026-07-08

### Changed — ONNX Runtime GenAI 0.13.2 → 0.14.1

- Bumped `Microsoft.ML.OnnxRuntimeGenAI.DirectML` NuGet from 0.13.2 to **0.14.1**
  (`uwp/packages.config`, `uwp/xllama.vcxproj` — 8 package-path references).
  0.14.x reduces CPU-side per-token overhead in `GenerateNextToken`/`SampleTopP`
  (directly relevant to the decode bottleneck measured in the v0.3.6 matrix) and
  is the prerequisite for continuous decoding / KV-cache reuse (`RewindTo`,
  generator reuse — Stage 2). No breaking C-API changes vs 0.13; existing model
  directories (built with model builder 0.14.1) load unchanged.
- On-console validation pending: rerun the CPU + DML bench matrix and compare
  decode tok/s to the v0.3.6 baselines to quantify the per-token overhead win.

### Docs

- `ROADMAP.md`: new Phase 3.5 — Hardware Ceiling with the ordered unlock
  levers (Game-mode designation, 1B+ models via no-bundle, int4-AWQ proxy,
  upstream `MatMulNBits` desk check, llama.cpp CPU A/B, per-workload routing,
  optional upstream kernel contribution); Phase 2 heading refined to the
  per-workload verdict; "GPU vs CPU same model" milestone closed by the
  v0.3.6 matrix.
- `docs/uwp-constraints.md` §5: stale "future work" paragraph replaced with
  the measured per-workload approach; documented the App-vs-Game designation
  lever — **settled 2026-07-08**: the package was found already designated
  Game, so all measured figures are Game-mode numbers (the interim "all
  numbers are App-mode" assumption was wrong and has been rectified in
  §5/§7/§11); the GPU decode gap is a DML/kernel issue, not platform
  scheduling.
- Environment change (2026-07-08): Dev Mode storage allocation raised to
  90 GB — the Q:\ ~2.2–2.5 GB disk budget (§9) is superseded as the binding
  constraint for model sizing; ROADMAP Phase 3.5 updated accordingly.
- `docs/device-portal.md`: new "Lifecycle gotchas" section — LocalState purge
  semantics (only forward upgrades preserve it), LocalState absent before
  first launch, WDP file APIs returning HTTP 200 with `"Success": false`,
  stale PFN staging window, portal unreachable while a game is running.

## [0.3.6] - 2026-07-07

### Measured — Hardware utilization matrix (prefill vs decode, CPU vs GPU)

SmolLM2-360M, 3 variants × 2 prompts (285 / ~1050 tok), v0.3.6 timing:

| Variant  | prefill 285 | prefill ~1050 | decode short | decode long |
| -------- | ----------- | ------------- | ------------ | ----------- |
| CPU int4 | **220**     | 198           | **68.0**     | **50.9**    |
| GPU int4 | 152         | 334           | 8.8          | 8.3         |
| GPU fp16 | 169         | **354**       | 46.8         | 36.5        |

- **Prefill crossover**: GPU scales with batch, CPU doesn't — at ~1k prompt
  tokens the GPU is 1.8× faster (TTFT 3.0 s vs 5.3 s). DML fp16 wins
  prompt-heavy workloads today.
- **int4 GPU decode collapse = missing fused int4 kernel**, not dispatch: fp16
  decode is 5.3× int4 on DML. Effective bandwidth: GPU fp16 ~34 GB/s vs CPU
  ~13 GB/s (bus ~224 GB/s) — the GPU exploits memory 2.6× better; a
  `MatMulNBits`-class DML kernel would imply ~180 tok/s (2.5× CPU).
- Verdict in `docs/uwp-constraints.md §5`; matrix rows appended to
  `bench/results/phase2-dml.csv`. New model on-device:
  `smollm2-360m-dml-fp16` (builder `-p fp16 -e dml`, 691 MB merged).

### Fixed

- Prefill timing (0.3.5) started the clock after `AppendTokenSequences`, but
  on-console measurement showed the prompt prefill runs **inside**
  `AppendTokenSequences` in ORT GenAI 0.13.2 (the first `GenerateNextToken`
  returns in ~40 µs) — `prompt_tok_s` came out in the millions. The clock now
  starts before `AppendTokenSequences`, covering the prefill under either
  implementation. Decode numbers were unaffected (measured after prefill in
  both layouts).

## [0.3.5] - 2026-07-07

### Added

**Prefill measurement in the ORT path** (`prompt_tok_s` was always 0.00):

- `src/bridge/inference.cpp`: ORT GenAI runs the prompt prefill inside the
  _first_ `GenerateNextToken` call; it was never timed (`n_p_eval`/`t_p_eval_ms`
  were only populated by the legacy llama.cpp path). Now the first loop
  iteration is timestamped: `prompt_tok_s` lands in the bench CSV and the log
  gains `prefill=N tok/s (M tok, T ms)`.
- `decode_tok_s` is now prefill-free (rate over `n_generated-1` tokens from the
  end of the first iteration) — previous decode numbers were slightly
  underestimated because they included prefill time.
- `bench/prompts/long-1k.txt`: ~1k-token prompt to exercise prefill in the
  CPU-vs-GPU utilization matrix.

### Upstream fix validated on hardware (2026-07-07, evening)

The `887A0036` DML init failure was fixed at the source and validated on
console before contributing upstream:

- Patch to `onnxruntime-genai` `CreateDmlObjects`: fall back to the system
  D3D12 runtime when the Agility SDK device factory cannot create a device
  (upstream PR [microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280)).
- Validated with a throwaway 0.3.9 test MSIX (branch
  `test/dml-fallback-validation`: patched DLL injected over the NuGet one,
  headless detection disabled to force the XAML repro path):
  - XAML + DML + vanilla DLL: `887A0036` at `OgaCreateModel` (known repro)
  - XAML + DML + **patched DLL: loads in 886 ms, 315 MB on GPU, full decode
    8.8 tok/s** — identical to the headless/Agility path
  - XAML + CPU + patched DLL: 67.2 tok/s — no regression
- Console restored to the official 0.3.4 MSIX afterwards.

## [0.3.4] - 2026-07-07

### Added

**Headless bench mode — unblocks DML EP init** (root cause of the 0.3.3
`887A0036` finding, now identified at the exact source line):

- Root cause: ORT GenAI 0.13.2 creates its D3D12 device through the **Agility
  SDK device factory** (`dml_helpers.cpp` `CreateDmlObjects`:
  `CreateDeviceFactory(614, module_path)` succeeds on Xbox OS 26100 via the
  in-box runtime ≥ 614, no app-local `D3D12Core.dll` needed — verified absent
  from our MSIX). The factory's `CreateDevice` then collides with the
  process-wide D3D12 device the **XAML compositor** (D3D11on12) created at
  `Window.Activate()` — two different D3D12 runtimes cannot share a process →
  `DXGI_ERROR_ALREADY_EXISTS`. This also reconciles Exp 1 (May): the then-OS
  had in-box < 614, so ORT fell back to plain `D3D12CreateDevice` (line 144)
  which coexists with the compositor device — the OS update flipped the branch.
  No upstream fix on `main`; v0.14.0 has identical code.
- `uwp/App.cpp`: when `LocalFolder\bench.flag` exists at startup, `wWinMain`
  skips `Application::Start` entirely and runs a minimal `IFrameworkView`
  (`BenchView`) via `CoreApplication::Run` — CoreWindow activated (PLM watchdog
  satisfied) but **no XAML compositor, no in-process D3D12 device** — then runs
  `xllama::bridge::main_loop()` on an MTA thread and exits via
  `CoreApplication::Exit()`. Interactive path unchanged (flag absent →
  `uninit_apartment` + `Application::Start` as before); `CheckBenchMode` in
  MainPage stays as fallback if the early detection throws.

### Fixed

- `bench/configs/genai_config-dml-{test,profile,profile.tpl}.json`:
  `past_present_share_buffer` `false` → `true` — the DML EP uses graph capture,
  which rejects `false` at `OgaCreateGenerator` ("Graph capture is not
  supported with past_present_share_buffer set to false"). The `false` was an
  Exp 1 leftover that only ever ran on the silent-CPU-fallback path.

### Measured — GPU-truth verdict on console (2026-07-07, headless bench mode)

**`VERDICT: GPU`** — first proven DML GPU execution in the Xbox UWP sandbox:

- ORT profiler: `DmlExecutionProvider` 10.7 ms kernel time (fused node 9.7 ms),
  `CPUExecutionProvider` 0.4 ms → 96% DML.
- In-app probe: `gpu-mem post-load: current=411MB` ≈ model size (403 MB) —
  weights resident on the RDNA 2 GPU. CPU control run: `current=0MB`,
  69.7 tok/s.
- The bundled CPU-int4 `MatMulNBits` model fails inside the fused DML node
  (`80070057`) — a DML model variant is required for full decode (below).

### Measured — Phase 2 end-to-end DML bench (2026-07-07, evening)

SmolLM2-360M INT4 **DML** variant, built with the ORT GenAI model builder
(0.14.1, `-p int4 -e dml`, 285 MB merged), uploaded to
`LocalState\models\smollm2-360m-dml-int4`:

| Backend            | decode tok/s | load ms | gpu_mem MB | peak WS MB |
| ------------------ | ------------ | ------- | ---------- | ---------- |
| DML EP (RDNA 2)    | **8.83**     | 927     | 307        | 999        |
| CPU EP (Zen 2 t=8) | **70.9**     | 1518    | 8          | 771        |

- Full decode completes on GPU (739 tokens, `VERDICT: GPU`, GPU compute engine
  ~88% saturated via `systemperf`) — but the **CPU is ~8× faster**: per-token
  DML dispatch overhead dominates a 360M model while `MatMulNBits`/AVX2 is
  highly optimised. **Phase 2 closed: CPU EP stays the production backend.**
- GPU pool estimate corrected: measured `Budget` is **3801 MB** (the "~768 MB
  pool" was OOM-bracketing inference). Disk (`Q:\` ~2.2–2.5 GB) is the real
  model-size constraint.
- Known cosmetic issue: bench CSV `backend` column says `ort-genai-cpu` even on
  DML runs (label hardcoded in `bench.cpp`, does not detect the configured EP).

## [0.3.3] - 2026-07-07

### Changed

- `src/bridge/inference.cpp`: dropped the `gpu-mem pre-load` probe that ran
  _before_ `OgaCreateModel`. `gpu_mem_info()` opens and caches an
  `IDXGIAdapter3` on adapter 0; not holding that handle open across model load
  is cleaner. GPU memory is still sampled `post-load`/`post-decode`, after the
  session exists — where the GPU-truth signal (`current` ≈ model size) matters.
  **Note:** this was first hypothesised to be the cause of the DML init failure
  below, but the rebuild (0.3.3) disproved it — see _Investigated_. The change
  is kept as harmless cleanup, not a fix.

### Investigated — GPU-truth DML experiment on console (2026-07-07)

First profiled DML run on Xbox Series S, using the 0.3.2 GPU-truth toolkit.
**Result: the DirectML EP does not initialise at all** on this stack — it is
neither GPU execution nor a CPU fallback.

- `OgaCreateModel` throws `887A0036` "The desired element already exists" at
  `onnxruntime-genai .../dml/dml_helpers.cpp(140)`, **before any kernel runs**,
  so no `ort_profile_*.json` is produced and the profiler emits no `VERDICT:`.
- Reproduced 3×: profiling config (0.3.2), plain `dml-test` config (0.3.2), and
  after the pre-load-probe removal (0.3.3) — identical signature each time.
- **Not OOM** (`avail_phys` 5.0 GB, `budget` 3801 MB), **not the profiling
  config** (plain config fails identically), **not our telemetry** (0.3.3 fix
  changes nothing). Cause is inside the ORT GenAI DML EP device creation.
- Likely a D3D12 single-device-per-adapter conflict: a UWP XAML app already
  holds a D3D12 device (compositor) on adapter 0 before `OgaCreateModel`.
- Corroborated by telemetry: GPU engines flat (only the display engine ~99%),
  `gpu_dedicated` ~100 MB (Dev Home noise), no compute spike.
- Reconciles Exp 1 (2026-05-23): its "loads without OOM, 71.7 tok/s ≈ CPU
  baseline, GPU execution unconfirmed" was almost certainly a **silent CPU
  fallback**, never real DML execution. Same ORT GenAI 0.13.2, same config.
- **Conclusion:** DML EP is not viable on ORT GenAI 0.13.2 in the Xbox UWP
  sandbox; the CPU EP is the only working backend (70.9 tok/s control run, same
  session). Supersedes the earlier "GPU EP ruled out via OOM" with a precise
  init-failure signature.

## [0.3.2] - 2026-07-07

MSIX carrying the GPU-truth toolkit + in-app `gpu_mem` telemetry (CWD pin to
LocalState) for the first profiled DML run on console.

### Added

**Bench infrastructure — ORT GenAI thread tuning** (`feat(bench): ORT GenAI bench infrastructure`):

- `scripts/bench-xbox-ort.sh`: new bench orchestrator for ORT GenAI models already on device (no model upload). Supports `--threads N`, `--runs N`, `--prompt file`. Appends median row to `bench/results/phase1-cpu.csv`. Drops warmup run automatically.
- `bench/configs/genai_config-threads-{4,6,8}.json`: `genai_config.json` variants with `intra_op_num_threads` set for Zen 2 thread-count tuning.
- `uwp/inference-bridge.cpp`: reads optional `bench_threads.txt` (uploaded per bench variant) to set `params.n_threads` for CSV tracking and suffix host_label (`xbox-series-s-tN`).

**GPU-truth debug toolkit — DML EP attribution without PIX** (see `docs/uwp-constraints.md §11`):

- `bench/configs/genai_config-dml-profile.json` (+ `.tpl.json` absolute-prefix variant): DML EP config with ORT `enable_profiling` + `log_severity_level: 0`.
- `scripts/profile-dml-run.sh`: one profiled DML run — config swap, bench run, fetch `ort_profile_*.json` + log tail into `bench/results/profiles/<ts>/`, restore config, analyze.
- `scripts/analyze_ort_profile.py`: per-provider kernel-time summary from the ORT profiling JSON; greppable `VERDICT: GPU | MIXED | CPU-FALLBACK` line; tolerates truncated traces.
- `scripts/xbox-gpu-sample.sh`: WDP `systemperf` sampler (per-engine GPU utilization + VRAM used → CSV + max/mean summary); `--gpu-sample` integration in `profile-dml-run.sh` and `bench-xbox-ort.sh`.
- `gpu_mem_info()` (`src/bridge/platform.cpp`): per-process `IDXGIAdapter3::QueryVideoMemoryInfo` (LOCAL); logged pre-load/post-load/post-decode; `dxgi.lib` linked in `uwp/xllama.vcxproj`.
- `set_cwd_to_local_folder()`: bench mode pins CWD to LocalState so the relative ORT profiling prefix lands in a writable, WDP-fetchable location.
- Bench CSV schema: new `gpu_mem_mb,gpu_budget_mb` columns before `host,date` (header updated in `bench.cpp`, both bench scripts, `bench/README.md`; existing `phase1-cpu.csv` rows backfilled with `0,0`); `--out FILE` flag in `bench-xbox-ort.sh` (DML runs → `bench/results/phase2-dml.csv`).
- CI (`build-linux`): smoke step for the analyzer (fixtures in `tests/fixtures/`) and the sampler parser.

**No-bundle MSIX build variant — unblocks Exp 2 validation**:

- `uwp/xllama.vcxproj`: model `ItemGroup` now also conditioned on `'$(XllamaNoBundledModel)' != 'true'`.
- `scripts/build-uwp.ps1 -NoBundledModel`: builds an MSIX without the bundled SmolLM2 model, so `EnsureModelAsync` reaches the USB/HF-download fallbacks on console.
- `.github/workflows/build-uwp.yml`: matrix `variant: [bundled, nobundle]`; the `nobundle` job skips model download/merge and uploads artifact `xllama-appx-nobundle`.

### Fixed

- `src/bridge/bench.cpp`: backend label was `"directml"` even on CPU EP → corrected to `"ort-genai-cpu"`; quant `"int4-awq"` → `"int4"`.
- `src/bridge/inference.cpp` (ORT path): `load_ms` was always 0 in the bench CSV — `run_inference` never measured model load. Now times `OgaCreateModel` wall-clock and logs `ORT model loaded in N ms`.
- Docs drift: `bench/README.md` results table said "pending" for the populated `phase1-cpu.csv`; `bench/README.md` + `docs/phase1-runbook.md` still documented the old `directml` backend label.

### Measured — Phase 1 bench results (Xbox Series S Zen 2, 2026-05-23)

SmolLM2-360M-Instruct INT4 CPU, ORT GenAI 0.13.2, n=990:

| n_threads          | decode tok/s | peak RAM MB | notes                                           |
| ------------------ | ------------ | ----------- | ----------------------------------------------- |
| auto (ORT default) | 66.9         | 704         | baseline, no `intra_op_num_threads`             |
| 4 (explicit)       | **71.4**     | 771         | **best**                                        |
| 6 (explicit)       | 68.0         | 772         |                                                 |
| 8 (explicit)       | 28.2         | 771         | severe regression — memory bandwidth saturation |

**Recommendation**: use `intra_op_num_threads: 4` in `genai_config.json` for SmolLM2-360M on Zen 2.

### Investigated — Exp 1 DirectML (2026-05-23)

- DML `genai_config.json` (provider: dml, `enable_cpu_mem_arena=0`, `enable_mem_pattern=0`) loads without SEH 0xC0000005 on SmolLM2-360M INT4.
- Performance: 71.7 tok/s — indistinguishable from CPU baseline.
- Conclusion: SmolLM2-360M INT4 (~200 MB ONNX) likely fits within the 768 MB UWP GPU pool. Cannot confirm GPU execution vs CPU fallback without D3D profiling tools. Phase 2 "blocked" status revised: **360M model fits; larger models still blocked**.
- Exp 2 (HF in-app download): unreachable with current bundled MSIX. `EnsureModelAsync` checks InstalledPath before HF download; model is always found there. Requires a separate build without bundled model to validate — now available via `build-uwp.ps1 -NoBundledModel` / CI artifact `xllama-appx-nobundle`.

### Investigated — model candidates via HF Hub file sizes (2026-07-02)

- Qwen2.5-0.5B INT4 CPU: ~822 MB `model.onnx.data` (rtn-block-32) — the ~200 MB estimate was wrong (151k-vocab embedding not INT4-quantized). Ruled out for disk and GPU pool. DML int4-awq variant ~507 MB: borderline GPU-pool fit, possible DML retry via USB.
- Llama-3.2-1B INT4 CPU: ~1.77 GB — USB-only, same class as SmolLM2-1.7B. Details in `docs/model-selection.md`.

---

## [0.3.0] — 2026-05-23

### Added

**Settings dialog — sampling parameters** (`feat(uwp): Settings dialog — sampling params`):

- `ShowSettings` now exposes `temperature` (Slider 0–2, default 0.8), `top_p` (Slider 0–1, default 0.9), `top_k` (NumberBox 1–200, default 40), `repetition_penalty` (Slider 1–2, default 1.1), and `n_predict` (NumberBox 16–2048, default 512).
- New `MainPageController` members `m_temperature`, `m_top_p`, `m_top_k`, `m_repetition_penalty`, `m_n_predict` wired into `StartInference` → `GenerateParams`.
- `settings.json` schema extended with a nested `"sampling"` object; back-compat preserved for existing 0.2.x files.

**Settings dialog — model selection ComboBox** (`feat(uwp): Settings dialog — model selection ComboBox`):

- ComboBox with three entries: SmolLM2-360M (bundled MSIX), SmolLM2-1.7B (USB `E:\xllama\models\`), SmolLM2-360M (HF download in `LocalState`).
- Selected model persisted to `settings.json`; `EnsureSession` detects model change at next `StartInference` and rebuilds transparently.
- `LoadModelName` now reads `m_model_filename` from `settings.json`; falls back to `LocalState/model.txt` for 0.2.x installations.

**History dialog enhancements** (`feat(uwp): History dialog — delete, clear all, timestamps`):

- Per-item ✕ Delete button: click closes the dialog and opens a confirmation ContentDialog; on confirm calls `ChatHistory::Delete(id)`. If deleted entry was the active conversation, `NewChat()` is called.
- Clear all (Secondary button): confirmation ContentDialog → `ChatHistory::Clear()` → `NewChat()`.
- Current conversation indicator: ● prefix on the active history entry.
- Relative timestamps via `FormatRelativeTs` helper: "today HH:MM", "yesterday HH:MM", "DD Mon HH:MM".
- Index refreshed from disk on every `ShowHistory` call.
- Empty-state ContentDialog with placeholder TextBlock instead of silent no-op.

**ChatHistory::Delete / Clear** (`feat(uwp/chat-history): add Delete and Clear methods`):

- `Delete(id)` removes `<id>.json` from `LocalState/chats/` and updates the in-memory index + `index.json`.
- `Clear()` removes all conversation files and writes an empty `index.json`.

**Tests** (`test: add ChatHistory helpers and TitleFrom smoke tests`):

- `tests/test_chat_history.cpp`: Linux CI tests exercise `TitleFrom` logic (truncation, newline stop, empty fallback). UWP `#ifdef` branch tests `Save`/`Load` roundtrip, `Delete`, and `Clear` with a temp directory.

### Fixed

- **Newline rendering**: `AppendOutput` now splits text on `\n` and inserts `LineBreak` inlines; previously `\n` in a `Run` was rendered as a space by WinUI `RichTextBlock`.
- **Prompt not cleared after Run**: `OnRunClick` clears `m_promptInput.Text(L"")` before handing off to `StartInference`.
- **NewChat did not clear prompt**: `NewChat()` now resets `m_promptInput.Text(L"")`.
- **Double FocusEngagement on Xbox**: removed `IsFocusEngagementEnabled(true)` from `m_outputScroll`; only the TextBox retains it, eliminating the extra A-press required to engage text input.
- **Focus not returned after generation**: `SetRunning(false)` now calls `m_promptInput.Focus(FocusState::Programmatic)` so the cursor returns to the input after inference completes.
- **Smart autoscroll**: `AppendOutput` auto-scrolls only when the user is already at the bottom (within 24 px of `ScrollableHeight`). A `ViewChanged` handler tracks `m_at_bottom`; `SetRunning(false)` resets it to `true`.
- **Status shows "Loading model…" at startup**: `BuildUI` initialises `m_statusText` to `L"Loading model..."` and disables the Run button. `EnsureModelAsync` sets `L"Ready"` and re-enables Run when the model is confirmed loaded.
- **Context trim overflow**: `kMaxEstimatedTokens` lowered from 3500 to 1800 to stay within `n_ctx = 2048` (leaves ~250 token generation headroom). Trim events now surface a `"Context trimmed: N turns dropped"` status message.
- **Partial save on cancel**: `StartInference` completion path sets `ChatMessage::partial = true` when the user pressed Cancel (`m_abort.load() == true`) before saving the conversation.

---

## [0.2.1] — 2026-05-23

### Added

- ChatML stop sequence `<|im_end|>` in UI inference path (`uwp/MainPage.cpp`). SmolLM2-360M does not always emit EOS naturally; without this the model would continue generating filler or hallucinate the next user turn up to `n_predict=512`. Bench path unchanged.
- `tests/test_session.cpp`: smoke tests for `Session::create` error paths (non-existent path, empty path) — covers the Linux/llama.cpp path in CI.

### Fixed

- `CHANGELOG.md` 0.2.0 section: collapsed duplicate `### Added` blocks; removed stale empty `[Unreleased]` header.

---

## [0.2.0] — 2026-05-22

### Added

**Persistent inference session** (`feat(uwp): integrate xllama::Session into MainPage`):

- `MainPageController` now keeps an `xllama::Session` alive across chat turns; subsequent turns skip model reload entirely (~1–2 s overhead eliminated after first turn).
- `EnsureSession()` private helper: lazy-build on first turn, transparent rebuild on model change (Settings), free-then-alloc to avoid 2× RAM during transitions.
- Bench mode (`inference-bridge.cpp`) unchanged — continues to call `run_inference()` for cold-load measurement.

**Bench diagnostics** (`bench(inference): log prompt token count + bump bench n_predict`):

- `inference.cpp`: logs `[xllama] prompt=N tok, max_length=M (new≤K)` after tokenisation — makes `n` in bench CSV self-explanatory.
- `inference-bridge.cpp`: bench `n_predict` raised `128 → 512` (effective `max_length` 640 → 1024 total tokens); gives SmolLM2-360M room to show natural generation length while 1.7B still exits at EOS.

**Multi-turn Session API** (`include/xllama/session.h`, `src/bridge/session.cpp`):

- `xllama::Session::create(SessionParams)` — loads model + tokenizer once; subsequent `generate(GenerateParams)` calls reuse them, eliminating the ~1-2s per-call reload overhead of `run_inference()`.
- `GenerateParams` exposes `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop_sequences` (substring-checked against accumulated output; matching sequence stripped on hit).
- Implemented for both UWP (ORT GenAI: `OrtSession` keeps `OgaModel` + `OgaTokenizer` alive) and Linux (llama.cpp: `LlamaSession` keeps `llama_model` alive; `llama_context` rebuilt per call to respect context parameters).
- `run_inference()` is unchanged — fully back-compatible.
- Consumer: `xbox_faraday` game (FARADAY) for per-turn dialogue generation.

**Workaround experiments for Xbox UWP constraints** (see `docs/uwp-constraints.md §7, §9`):

- `uwp/model-downloader.cpp/h`: in-app Hugging Face download via `HttpClient` chunked streaming (Exp 2). `EnsureModelAsync()` in `MainPage` implements a three-step bootstrap: LocalState `.complete` marker → InstalledPath bundle → HF download. Reduces peak disk usage from ~1.4 GB to ~480 MB; frees ~900 MB on Dev Mode partition Q:\ enabling models up to ~1 GB. `internetClient` capability already present in manifest.
- `src/bridge/path_utils.cpp`: third fallback `E:\xllama\models\<name>` for NTFS USB stick (Exp 3). No UWP capability required; probe via `GetFileAttributesW`. Enables models up to 2 GB single-file (Xbox Dev Mode USB limit). Zero cost if USB absent.
- `scripts/test-dml-config.sh`: uploads DML provider_options config to Xbox via Device Portal without MSIX rebuild (Exp 1). Backs up original `genai_config.json`; `--restore` reverts.
- `bench/configs/genai_config-dml-test.json`: DML EP test config — `enable_cpu_mem_arena=0`, `enable_mem_pattern=0`, `past_present_share_buffer=false` (reduces up-front KV-cache GPU allocation; may allow SmolLM2-360M to fit ~768 MB pool).
- `docs/model-selection.md`: consolidated model evaluation checklist — hard limits, 9-step selection sequence, tested/candidate models, conservative/borderline/over-budget tables.
- `docs/uwp-constraints.md §9`: Disk Budget — ~2.2–2.5 GB Dev Mode free space, 2× peak-install rule, working budget table (empirical, Series S).
- `scripts/merge_onnx_external_data.py`: NOTE stderr (>400 MB) and WARNING stderr (>600 MB) budget thresholds post-merge.
- `docs/uwp-constraints.md §5`: split GPU OOM and disk-budget failure modes into separate tables (previously mixed under a single "Result" column).

**UX improvements** (commits `3a12bda`–`42741e1`):

- Multi-turn chat: `uwp/chat-history.cpp/h`, conversation persistence in `LocalState/chats/` (JSON, indexed by timestamp), history browser overlay, new-chat button.
- System prompt editable via settings overlay (persisted to `LocalState/settings.json`).
- Live metrics: real-time tok/s updated every flush cycle; `StatusKind` enum (`Info`, `Working`, `Success`, `Error`) for colour-coded status bar.
- `RichTextBlock` streaming: `Paragraph::Inlines` append (O(1) per token); flush timer batches token appends every 80 ms to avoid layout thrash.
- Xbox UX: TV safe-area margins (48/27 px), dark theme on Xbox hardware, B-button cancels inference, gamepad Y jumps to prompt, Reveal focus visual, `ElementSoundPlayer::On`.
- `AppxManifest.xml`: `xbox:DefaultTile`, `xbox:SplashScreen`, dark splash background `#0E1116`.
- ChatML prompt template applied for SmolLM2-360M-Instruct (system / user / assistant turns).

**Documentation:**

- `README.md`: "About the name" section — disambiguates xllama from llama.cpp engine.
- Full docs realignment: `README.md`, `ROADMAP.md`, `AGENTS.md`, `docs/phase1-runbook.md`, `docs/uwp-constraints.md`, `docs/device-portal.md`, `patches/README.md` all updated to reflect ORT GenAI CPU EP as the active path.
- `docs/windows-dev-vm.md` (new): end-to-end Windows VM build guide.
- `scripts/setup-windows-uwp-dev.ps1` (new): Windows VM setup via `winget` (VS2022 BuildTools + UWP workload).
- `scripts/check-uwp-host.sh` (new): Arch Linux host preflight (KVM, qemu, libvirt groups).
- `docs/uwp-constraints.md §7`: removed unverified architectural claims ("Game process category", "128 MB dedicated + 640 MB shared"); replaced with observed-behaviour framing and source note.
- `ROADMAP.md`: Phase 4 milestones updated — `ModelDownloader` (Exp 2) and USB fallback (Exp 3) marked done; next: validate Exp 2 on console, remove MSIX model bundle.

**CI:**

- `build-uwp` CI step: downloads model from HF (`homen3/SmolLM2-360M-Instruct-ort-genai-int4-cpu`), merges ONNX external data, then `nuget restore` + `build-uwp.ps1`. Cache key: `smollm2-360m-ort-genai-int4-cpu-embedded-v1`.

### Changed

- ORT GenAI bumped `0.8.3 → 0.13.2`, ORT `1.22.0 → 1.24.4` (`uwp/packages.config`).
- `bench.cpp`: backend field = `directml` when `XLLAMA_USE_ORT` (define-time; CPU EP is active runtime on Series S).
- `uwp/pch.h`: added `Windows.Web.Http`, `Windows.Web.Http.Filters` for model downloader.
- `ROADMAP.md Phase 2`: corrected GPU pool description to observed-behaviour framing.

### Removed

- `uwp/llama-bridge.cpp`, `uwp/llama-bridge.h`: legacy files not compiled since ORT GenAI pivot.

### Fixed

- `fix(bridge): OrtModelPtr → OgaModelPtr` typo in `OrtSession` UWP build (MSVC `C2065`; GCC/clang skip the `XLLAMA_USE_ORT` block on Linux).
- ASCII-safe status strings: removed em-dash and ellipsis Unicode literals that caused MSVC `C4566` warnings.
- `XYFocusKeyboardNavigationMode` removed from `MainPage.cpp` (unresolvable symbol in MSVC UWP context).
- `weakly_canonical: Access is denied` crash (`OgaCreateModel`, status `0xC0000005`): ORT runtime walks path segments of the model directory to validate external data; `Q:\Users\UserMgr0\...` is inaccessible from UWP AppContainer. Fix: merge external data into monolithic `model.onnx` so `ValidateExternalDataPath` is never invoked. Confirmed via Win32 probes (`GetFileAttributesW`, `CreateFile2 GENERIC_READ`, `CreateFile2 FILE_READ_ATTRIBUTES|SYNCHRONIZE`).

---

## [Pivot: SmolLM2-360M + CPU EP] — commit `14e6a14`

### Added

- SmolLM2-360M-Instruct INT4 CPU as the bundled model (403 MB on-disk, ONNX opset 21, IR version 10).
- Model included as `DeploymentContent` in `uwp/xllama.vcxproj`; deployed to `Package.InstalledPath\models\smollm2-360m-cpu-int4\`.
- `resolve_model_path` (`src/bridge/path_utils.cpp`): checks `LocalState\models\<name>` first (runtime override), falls back to `Package.InstalledPath\models\<name>` with copy-on-first-launch to LocalState.

### Changed

- Default model in `uwp/inference-bridge.cpp` and `uwp/MainPage.cpp` changed from Phi-3.5 to `smollm2-360m-cpu-int4`.
- `genai_config.json` uses `"provider_options": []` → CPU EP active (no DirectML).

### Notes

- **GPU EP ruled out**: Xbox Series S UWP GPU pool is ~768 MB. `OgaCreateModel` with DirectML EP on any tested model (Phi-3.5-mini GPU INT4 ~2.2 GB, SmolLM2-1.7B ~1.4 GB) crashes with null-deref in DML allocator on OOM. See `docs/uwp-constraints.md §7`.

---

## [Pivot: ONNX Runtime GenAI + DirectML] — commit `385cb07`

### Added

- `XLLAMA_USE_ORT=1` preprocessor flag in `uwp/xllama.vcxproj`; enables ORT GenAI path in `src/bridge/inference.cpp`.
- `include/xllama/ort_raii.h`: RAII `unique_ptr` wrappers for `OgaModel`, `OgaTokenizer`, `OgaTokenizerStream`, `OgaGeneratorParams`, `OgaGenerator`, `OgaSequences`.
- `uwp/inference-bridge.cpp` / `inference-bridge.h`: replaces `llama-bridge.cpp`; thin UWP glue around `xllama::bridge::run_inference`.
- NuGet packages (`uwp/packages.config`): `Microsoft.AI.DirectML 1.15.4`, `Microsoft.ML.OnnxRuntime.DirectML`, `Microsoft.ML.OnnxRuntimeGenAI.DirectML`.
- `src/bridge/platform.cpp`: `log_output` now writes to `LocalState/xllama.log` in UWP (previously `OutputDebugStringA` only).

### Changed

- `src/bridge/inference.cpp`: `#ifdef XLLAMA_USE_ORT` path uses `OgaGenerator` loop; `#else` path retains `llama_decode` for Linux.
- Linux CI (`build-linux.yml`): `submodules: false` for UWP; llama.cpp submodule only for Linux.
- `deploy.sh`: `upload-file` auto-creates subdirectory; new subcommands `mkdir-localstate`, `upload-dir`.

### Notes

- llama.cpp submodule retained for Linux path (`CMakeLists.txt`). Three UWP patches (`uwp/patches/llama.cpp/`) kept but not applied for this build.

---

## [Pivot: XAML-free UI] — commits `77a651a`, `3f7a950`, `385cb07`

### Removed

- `App.xaml`, `MainPage.xaml`, `XamlTypeInfo_impl.cpp`: eliminated to avoid WMC9999 (`XamlC.exe` crash during `MarkupCompilePass2` in SDK 22621/26100 for C++/WinRT projects).
- `runtimeclass MainPage` from IDL: `MainPageController` is now a plain C++ class.

### Added

- `MainPageController` (`uwp/MainPage.cpp`): programmatic UI built via `Windows.UI.Xaml.Controls.*` API. Uses `enable_shared_from_this`; `shared_from_this()` must not be called from the constructor — use `Init()` post-construction.
- `runtimeclass App` retained (required by `Application::Start`).

### Notes

- Root cause of WMC9999: without MarkupCompilePass2, `XamlTypeInfoProvider::CreateXamlType` cannot provide correct metadata for `xllama.MainPage`; parser fast-fails when `LoadComponent` tries to validate the binding. No workaround existed; XAML-free was the correct fix.

---

## [Baseline: llama.cpp + Linux CI] — initial commits

### Added

- Linux build via CMake presets (`linux-release`, `linux-test`).
- Modular bridge: `src/bridge/inference.cpp`, `bench.cpp`, `platform.cpp`, `path_utils.cpp`, `utf8_utils.cpp`, `cli.cpp`.
- Shared headers under `include/xllama/` (inference, CLI, RAII, platform, path utils).
- Unit tests with doctest (target: `xllama-tests`, preset: `linux-test`).
- `xllama-cli` binary: `src/main.cpp` with `getopt_long`.
- `scripts/deploy.sh`: Device Portal REST API wrapper (deploy, install-cert, get-log, list-localstate, list-dumps, start-app, stop-app, diagnose-startup).
- `scripts/bench-xbox.sh`: automated benchmark runner (upload, trigger, poll, fetch CSV, compute median).
- `docs/`: `device-portal.md`, `uwp-constraints.md`, `phase1-runbook.md`.
- `bench/`: methodology README, config JSONs, `prompts/standard-512.txt`, `prompts/short-32.txt`.
- `.github/workflows/build-linux.yml`: clang-format, shellcheck, cmake, ctest, UWP gate.
