# AGENTS.md — xllama

Quick reference for AI agents and new contributors.

## Conventions

- **Language**: English for code, comments, filenames, commit messages.
- **C++ standard**: C++17 (UWP / SDK 22621 + MSVC).
- **Style**: concise, no over-engineering. Prefer RAII and `unique_ptr`.
- **Formatting**: auto-format on save if possible; otherwise follow existing
  style. Both formatters are pinned and gated in CI — `clang-format` 22.1.5 for
  C++, `prettier` 3.9.6 for every tracked `*.md` / `*.markdown` (`.prettierrc` /
  `.prettierignore`).

## Directory structure

```
xllama/
├── include/xllama/          # Shared public headers (WinRT-free, host-testable)
│   ├── inference_params.h   # InferenceParams / InferenceResult
│   ├── inference.h          # run_inference, write_bench_csv
│   ├── training_params.h    # TrainingJob / TrainingCapability (training pillar)
│   ├── training.h           # validate/load job, capability matrix, stage names
│   ├── device_train.h       # Lane B run_device_train_job + progress callbacks
│   ├── personalize.h        # Phase 11: last-block filter, job builder, sample count
│   ├── preference_capture.h # preference JSONL (UI rate + POST /v1/preferences)
│   ├── routing_policy.h     # routing decision + prompt budget (threshold must stay under it)
│   ├── speculative.h        # Phase 15 W2: prompt_lookup_draft (pure, host-testable)
│   ├── gpubw.h              # Phase 15 W3: GPU STREAM probe helpers + kill gate (#211)
│   ├── gpugemv.h            # Phase 15 H6.2: wave32 Q4_K GEMV density probe (#228)
│   ├── sampling.h           # sampling defaults shared by CLI/bench and GUI/API
│   ├── session.h            # xllama::Session API (persistent model across turns)
│   ├── session_hub.h        # SessionHub: the ONE process-wide resident-Session owner (GUI+API)
│   ├── ort_raii.h           # RAII unique_ptr for OGA* types (UWP/ORT GenAI path)
│   ├── llama_raii.h         # RAII unique_ptr for llama_* types (Linux path)
│   ├── cli.h                # parse_cli_args (Linux)
│   ├── platform.h           # log_output, detect_threads(_llama), peak_working_set_mb
│   ├── path_utils.h         # resolve_model_path, first_gguf_in_dir, model_uses_llama_backend
│   ├── utf8_utils.h         # utf8 <-> wstring (Windows)
│   ├── chat_prompt.h        # ChatFormat, chat_format_for, apply_stop_sequences
│   ├── model_provision.h    # dir_satisfies_expected_files, normalize_model_path
│   ├── manifest_merge.h     # merge_manifest_entries (per-entry catalogue override)
│   ├── autopilot.h          # AutopilotAction, validate_autopilot_script (console gates)
│   ├── prompt_budget.h      # fit_prompt — exact token-budget trimmer
│   ├── json_utils.h         # json_escape, json_read_string (canonical JSON helpers)
│   ├── cancel_policy.h      # CancelTarget — which running job a cancel request targets
│   ├── kv_store.h           # KvStore — on-disk KV snapshot pool (3 files / 192 MB, LRU)
│   ├── logit_dump.h         # Logit-parity harness: float32 dump + JSON sidecar
│   ├── ramceil.h            # probe_ram_ceiling — heap ceiling probe
│   ├── diskbw.h             # measure_diskbw — NVMe/disk bandwidth probe
│   ├── d3d12_dyn.h          # Dynamic d3d12.dll entry-point resolve (AppContainer PE hygiene)
│   ├── membw.h              # measure_membw — STREAM-style CPU bandwidth probe
│   └── diffusion/           # Diffusion sub-pipeline (CLIP tokenizer, Euler scheduler, PNG writer)
├── src/bridge/              # Shared implementation (Linux + UWP)
│   ├── inference.cpp        # ORT GenAI and/or llama_decode (unified: runtime dispatch)
│   ├── sampler_chain.h      # add_sampler_stages — the one llama.cpp sampler chain (#125)
│   ├── ort_sampling.h       # apply_ort_sampling — the ORT twin, greedy guard shared (#141)
│   ├── decode_loop.h        # shared prefill/decode; W2 prompt-lookup verify (opt-in)
│   ├── decode_loop_ort.h    # consolidated ORT GenAI decode loop (stop sequences on stateless path)
│   ├── ort_common.h         # Shared ORT setup: SEH translator + OgaSetLogCallback
│   ├── session.cpp          # xllama::Session (OrtSession UWP + LlamaSession Linux)
│   ├── training.cpp         # TrainingJob validate/parse (host + UWP linkable)
│   ├── device_train.cpp     # Lane B engine: prepare → train → export → evaluate
│   ├── personalize.cpp      # Phase 11 pure helpers
│   ├── preference_capture.cpp
│   ├── chat_prompt.cpp      # ChatFormat, chat_format_for, apply_stop_sequences
│   ├── bench.cpp            # bench CSV writer (incl. run_index)
│   ├── platform.cpp         # log_output (writes xllama.log in UWP)
│   ├── path_utils.cpp       # resolve_model_path: LocalState\models\ + InstalledPath fallback
│   ├── utf8_utils.cpp
│   ├── cli.cpp
│   ├── json_utils.cpp       # json_escape, json_read_string implementation
│   ├── prompt_budget.cpp    # fit_prompt implementation
│   ├── autopilot.cpp        # ApRun driver
│   ├── kv_store.cpp         # KvStore with LRU eviction
│   ├── membw.cpp            # STREAM-style CPU bandwidth probe
│   ├── diskbw.cpp           # NVMe disk bandwidth probe
│   ├── ramceil.cpp          # Heap ceiling probe
│   ├── gpubw.cpp            # GPU STREAM probe D3D12 driver
│   └── gpugemv.cpp          # Q4_K GEMV density probe D3D12 driver
├── src/main.cpp             # Linux entry point (getopt_long; --train-job)
├── training/                # Training pillar ops: jobs/, manifest-overrides/, datasets, host PEFT
├── docs/                    # SSOT map in docs/README.md
│   ├── architecture.md      # System structure SSOT
│   ├── training-architecture.md  # Training SSOT (RE + capability matrix + §11 UI arc)
│   └── api-endpoint.md      # LAN protocol (chat + prefs + train status + images)
├── uwp/                     # C++/WinRT UWP app
│   ├── App.cpp / App.h      # Application::OnLaunched
│   ├── MainPage.cpp / .h    # MainPageController; personalize UI; autopilot
│   ├── inference-bridge.cpp / .h   # main_loop, run_train_job_localized, headless flags
│   ├── api-server.cpp / .h  # opt-in LAN endpoint
│   ├── chat-history.cpp / .h
│   ├── model-downloader.cpp / .h   # catalogue download + LoadModelManifest
│   ├── packages.config      # NuGet pins (versions: packages.config; lifecycle: docs/vendor-lifecycle-plan.md)
│   └── xllama.sln / .vcxproj
├── scripts/
│   ├── deploy.sh                      # Device Portal: deploy, logs, bench trigger
│   ├── build-uwp.ps1                  # Windows UWP packaging script
│   ├── bench-xbox-ort.sh              # benchmark runner (run_index, multi-run)
│   ├── validate-console.sh            # autopilot: the 10 console gates (docs/console-validation-runbook.md)
│   ├── validate-console-training.sh   # rate / serve / device-train
│   ├── validate-api.sh                # LAN: spike|chat|prefs|train|all
│   ├── generate-benchmark-summary.py  # raw results → docs table + dashboard
│   ├── install-latest-build.sh        # fetch + deploy latest CI artifact
│   └── …
├── tests/                   # Unit tests (doctest; incl. test_personalize)
├── shaders/                 # AOT HLSL + generated DXIL (Phase 15 W3 gpubw)
├── bench/                   # configs, raw results, summary policy
├── demo/                    # demo-script.json — what the capture records, reviewable in a PR
├── diffusion/               # SD-Turbo → ONNX host toolchain (not shipped in the MSIX)
├── patches/                 # AppContainer / runtime patches applied at build time
├── cmake/
└── .github/workflows/       # build-linux.yml + build-uwp.yml
```

**Doc ownership (do not invent a second SSOT):** see `docs/README.md`. Structure →
`architecture.md` (incl. catalogue `n_ctx`/`role`, ChatFormat, deferred surfaces);
training → `training-architecture.md`; inventory/status → `model-matrix.md`;
numbers → `bench/results` + generated `benchmarks.md`; UI steps → `using-the-app.md`;
Phase 15 RE/opt (W2 findings, default-OFF) → `docs/phase15-re-opt.md`; Phase 16
model scouting (candidate funnel, validation ladder, WS cards) →
`docs/phase16-model-scouting.md`; Linux→Xbox
pack path → `docs/crossbuild-console.md` (launch = CI MSVC).

**Catalogue policy:** optional `n_ctx` and `role` (`coding`) are session knobs only
— not a second backend. Gate: host Release smoke → console bench → then manifest.
Measured ≠ shipped. No Settings magic for system prompts.

## Build

### Linux (development + tests)

```bash
# Release
cmake --preset linux-release
cmake --build build/linux-release -j$(nproc)

# Debug with tests
cmake --preset linux-test
cmake --build build/linux-test -j$(nproc)
ctest --test-dir build/linux-test --output-on-failure

# Smoke test
./build/linux-release/bin/xllama-cli --help
```

### UWP (Windows / CI)

Recommended: push to `main` and download the `xllama-appx` artifact from the
`build-uwp` GitHub Actions workflow. **That CI MSVC package is the shipping
and measurement path on Series S.** Linux uwp-crossbuild (≥ 0.5.1) also
produces a package whose launch was observed on device 2026-08-08 (see
`docs/crossbuild-console.md`; ORT/GenAI, first boot and uptime unproven
there) — use CI for console benches and shipping claims.

For local builds (requires a Windows VM — see `docs/windows-dev-vm.md`):

```powershell
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64
```

Use `-ForceNewCert` only to regenerate the test signing certificate.

**Versioning**: `Major.Minor.Build` in `uwp/AppxManifest.xml` is the semantic
version — bump it manually per release (with `CHANGELOG.md`). The `Revision` (4th
component) is stamped automatically in CI to the workflow run number
(`build-uwp.ps1 -BuildRevision`, wired from `github.run_number` in the
workflows), so every CI package is uniquely and monotonically versioned and the
console always takes an **in-place update** — no manual per-build bump, and never
the "same identity, different contents" install block. Local builds leave `.0`.
Exception: **1.5.0.0 changed the `Identity Name` itself** (`VenereLabs.xllama`
→ `GianlucaMazza.xllama`) — across that boundary there is no in-place update
(new app, fresh LocalState; `deploy.sh` keeps `APP_ID_LEGACY` for the
transition; migration steps in `docs/install-release.md`).

## Tests

- Framework: **doctest** (header-only, fetched via CMake).
- Target: `xllama-tests`.
- Command: `ctest --test-dir build/linux-test --output-on-failure`.
- Add tests in `tests/test_*.cpp`.

## Critical notes

- **Never commit `.env`** (credentials) or **`.pfx` / `.cer`** (gitignored).
- **ORT path**: UWP inference under `#ifdef XLLAMA_USE_ORT` in `src/bridge/inference.cpp`;
  RAII in `ort_raii.h`. Linux `#else` is llama.cpp.
- **No model in the MSIX** (~19 MB): first launch downloads default chat from
  `models-v1` (`lfm25-350m` unified / `smollm2-360m-cpu-int4` ORT-only). Console
  gate: `./scripts/validate-console.sh all`.
- **Shipping CI**: unified + PatchedGenAI + PatchedOrt from `vendor-dlls-v1`
  (hashes under `vendor/*/SHA256SUMS`). Pin lifecycle SSOT:
  `docs/vendor-lifecycle-plan.md`. Poll: `scripts/check-vendor-nuget-status.sh`.
- **External data / AppContainer**: merge `.onnx.data` with
  `scripts/merge_onnx_external_data.py` before packaging (CI does this). Details:
  `docs/fp16-extdata-runbook.md`, `docs/uwp-constraints.md` §8.
- **app-local DLLs**: `DirectML.dll`, `onnxruntime.dll`, `onnxruntime-genai.dll`
  need `<DeploymentContent>true</DeploymentContent>` or the MSIX omits them.
- **UWP constraints list**: `docs/uwp-constraints.md` (no mmap/dlopen/registry/…).
