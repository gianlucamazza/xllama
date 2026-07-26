# Recommended configurations (Xbox Series S, 2026)

Operational reference for correct, modern xllama settings. Every workload verdict
is **measured** unless marked "host-only" or "pending console". See
[technical-report.md](./technical-report.md) for the full story.

## Runtime pins (do not drift)

| Package               | Version    | File                  |
| --------------------- | ---------- | --------------------- |
| ORT GenAI DirectML    | **0.14.1** | `uwp/packages.config` |
| ONNX Runtime DirectML | **1.24.4** | same                  |
| DirectML              | **1.15.4** | same                  |

Chat GPU inside XAML requires the **#2280** patched `onnxruntime-genai.dll`
(merged on Microsoft GenAI `main`; **not** in NuGet **0.14.1**). Shipping CI
installs the hash-pinned DLL from `vendor-dlls-v1`. Local:

```powershell
./scripts/vendor-genai-dml-patch.ps1   # install cached/vendor-dlls pin over NuGet
./scripts/build-uwp.ps1 -PatchedGenAI -PatchedOrt
```

Upstream: [microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280).

## UWP build variants

| Variant                                 | CI artifact / command              | When to use                                                      |
| --------------------------------------- | ---------------------------------- | ---------------------------------------------------------------- |
| **unified+#2280+PatchedOrt** (shipping) | `xllama-appx` from `build-uwp.yml` | GGUF + ORT, external-data ORT (GPU text routing via `-v2` asset) |
| **llamacpp**                            | `xllama-appx-llamacpp`             | Bench A/B only — not for end users                               |

Both UWP lanes compile ggml with `GGML_USE_CPU_REPACK` since PR #155 (the
repacked-weight GEMM path was silently dead code before — enabling it raised
GGUF prefill ~62%). On Linux, `XLLAMA_NATIVE_OPT=ON` opts into host-tuned
`-march=native` builds; default is portable AVX2.

## Chat models

The decode figures below are rough guidance; the authoritative, disambiguated
tables are in [benchmarks.md](benchmarks.md) (the perf SSOT).

| Use case              | Catalogue `name`           | Backend               | Measured decode                                                                                                                                                                                                                                         |
| --------------------- | -------------------------- | --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Default (unified)** | `lfm25-350m`               | llama.cpp (`unified`) | **93.0 tok/s** (recommended)                                                                                                                                                                                                                            |
| **Balanced chat**     | `lfm25-1.2b-instruct`      | llama.cpp (`unified`) | **37.9 tok/s**, 811 MB peak; H9 6/8                                                                                                                                                                                                                     |
| **Quality chat**      | `lfm2-2.6b`                | llama.cpp (`unified`) | **18.4 tok/s**, 1623 MB peak; H9 7/8                                                                                                                                                                                                                    |
| **ORT default**       | `smollm2-360m-cpu-int4`    | ORT CPU int4          | **74.8** tok/s (262.4 prefill — shipped t6 asset, `t6-shipped-confirm.csv`)                                                                                                                                                                             |
| **Routing GPU**       | `smollm2-360m-dml-fp16-v2` | ORT DML fp16          | 44.4 tok/s decode; 236.7 tok/s prefill (cold-process **bench** figure — in-app turns run warm since the load warm-up + pre-load, §5e: first request ~873 tok/s prefill) — RMSNorm-decomposed graph, #91 parity-validated (`dml-rmsnorm-fix-runbook.md`) |
| **Larger chat**       | `smollm2-1.7b-cpu-int4`    | ORT CPU int4          | **20.6** tok/s (in-app `models-v1` download)                                                                                                                                                                                                            |
| **Modern GGUF**       | `qwen35-0.8b`              | llama.cpp (`unified`) | 35.1 tok/s (98.1 prefill, t6 — pre-repack figure, not re-measured)                                                                                                                                                                                      |
| **Fast modern GGUF**  | `lfm25-350m`               | llama.cpp (`unified`) | 93.0 tok/s (394.8 prefill, t6 — post-repack, PR #155: prefill was 241.4 before)                                                                                                                                                                         |

GGUF thread default: llama.cpp auto-detect is capped at **6** on console
(`detect_threads_llama` — t6 measured optimum, t7/t8 livelock); explicit
`--threads` on `bench-xbox-ort.sh` still wins. Caveat (#168): the "t6" in the
GGUF rows above describes **decode** — until PR #177 the app never set
`n_threads_batch`, so every published GGUF **prefill** figure ran on the
llama.cpp default of 4 prefill threads. Re-measure pending
(`uwp-constraints.md` §5f, 2026-07-26 note).

### Do not use

- DML **int4** for decode (~8.8 tok/s — DirectML kernel limit)
- CPU-int4 SmolLM graph with DML `genai_config` (`80070057`)
- Qwen2.5-0.5B ONNX CPU (~822 MB vocab embedding)
- llama.cpp for a model that also has an ORT build (same-model A/B: decode
  parity, worse prefill vs ORT) — the `unified` lane is for GGUF-only models
  (Qwen3.5, LFM2.5), where it earns its keep (LFM2.5 beats the ORT 360M
  baseline by ~42%)
- llama.cpp auto threads above 6 on console (ggml spin-wait livelock)

## `genai_config.json` (on-device model dir)

**CPU (bundled default)** — copy from [`bench/configs/genai_config-threads-6.json`](../bench/configs/genai_config-threads-6.json):

- `provider_options: []`
- `intra_op_num_threads: 6` — **corrected 2026-07-21** (§5f). The previous 4 came
  from a decode-only sweep; measuring prefill as well puts 6 ahead by **+8.5%**
  on prefill at 1380 tokens with no decode cost (46.5 vs 47.3, inside noise).
  Under a streaming UI prefill is what the user waits for, so 6 wins.
  **t=8 is a trap**: it collapses decode to ~10 tok/s _and_ prefill to ~87 tok/s
  — not the bandwidth saturation it was recorded as.
  ✅ **SHIPPED 2026-07-25.** The ship condition (≥3-length sweep, 3 runs,
  closing control) was met by `bench/results/phase12b-threads-sweep.csv` — t6
  prefill **+4.4/+4.7/+6.1%** at P=39/285/960, decode neutral within the
  closing-control drift, t4 ≈ unset — and the 1.5.0.0 identity migration
  forced a full re-provision, i.e. exactly the "next models-v1 republish"
  the condition asked to bundle with. The `models-v1` release's
  `genai_config.json` now matches
  [`genai_config-threads-6.json`](../bench/configs/genai_config-threads-6.json)
  (pristine + the one key); any device provisions it from now on. Note the
  gain is the multi-length +4-6%, smaller than §5f's single-length +8.5%.
  **On-device confirmation** (1.5.0.0 migration, 3 recorded runs,
  `bench/results/t6-shipped-confirm.csv`): prefill 262.4 median vs 244.8
  unset (+7.2% at P=285), decode 74.7 vs 74.3 (parity) — the shipped config
  is live and performs as measured.
- `past_present_share_buffer: true` (required for KV reuse)

**DML fp16 (routing)** — [`bench/configs/genai_config-dml-test.json`](../bench/configs/genai_config-dml-test.json):

```json
"provider_options": [{ "dml": {
  "enable_cpu_mem_arena": "0",
  "enable_mem_pattern": "0"
}}]
```

Swap without MSIX rebuild: [`scripts/test-dml-config.sh`](../scripts/test-dml-config.sh).

Models must be **self-contained** `model.onnx` (< 2 GB merged) — run
[`scripts/merge_onnx_external_data.py`](../scripts/merge_onnx_external_data.py) before upload.

## App settings (`LocalState/settings.json`)

Copy [`bench/configs/settings-modern.json`](../bench/configs/settings-modern.json) via Device Portal, or use the UI.

| Key                    | Recommended                       | Notes                                                                    |
| ---------------------- | --------------------------------- | ------------------------------------------------------------------------ |
| `model`                | `lfm25-350m`                      | First-launch default on **unified** shipping                             |
| `kv_reuse`             | `true`                            | 4.87× ORT; 4.07–20.02× GGUF turn-2 prefill depending on model (measured) |
| `routing`              | `2` (Auto) on unified             | GPU above 1550 tok with the `-v2` asset provisioned; GGUF skips          |
| `gpu_model`            | `smollm2-360m-dml-fp16-v2`        | parity-validated asset (#91 lifted); auto-download when routing ≠ 0      |
| `diffuse_taesd_vae`    | `true` after asset on `models-v1` | ~4.5 s/image target                                                      |
| `sampling.temperature` | `0.8`                             | UI default                                                               |
| `sampling.n_predict`   | `512`                             | UI default                                                               |

Factory defaults match [`settings-modern.json`](../bench/configs/settings-modern.json)
on unified builds (`DefaultChatModelId()` → `lfm25-350m`). ORT-only builds still
default to `smollm2-360m-cpu-int4`.

Routing Auto uses **1550 tokens** (real tokenizer count), sticky per conversation
— the `dml_text_model_ok` allowlist (#91 postmortem) precedes the threshold: a
non-validated `gpu_model` resolves every text turn to the CPU model regardless
of length.

Two caveats on that number, both live:

- It sits under the context trimmer's budget by design (#133). The trimmer runs
  first and drops turns over its own limit, so a threshold above that ceiling
  makes auto routing unreachable for every input — which is what happened
  between the 600 → 1550 retune and its fix. The usable band is roughly 1550 to
  1685 real tokens, about 135 wide.
- It was calibrated against a DirectML slowdown believed to track prompt length.
  It tracks `max_length` instead (#130, `docs/uwp-constraints.md` §5c). The
  re-derivation (§5d) concluded there is **no correct single prompt-length
  threshold** — see the next point — so 1550 is left as-is with its rationale
  corrected, not swept for a better number.
- It optimizes the **first turn only** (§5d). The GPU buys a faster first token
  on a long prompt; from the second turn the CPU wins at every reachable length,
  because DirectML cannot reuse a KV cache. So the value is a product judgement —
  whether Auto is worth leaving on depends on how single-turn your long-prompt
  conversations are, which is not measurable from a bench.

## Image generation

| Setting     | Modern                                                      | Obsolete                     |
| ----------- | ----------------------------------------------------------- | ---------------------------- |
| Trigger     | `[*] Image` → **Generate** (in-process)                     | App restart + `diffuse.flag` |
| Model dir   | `sd-turbo-fp16`                                             | —                            |
| VAE         | TAESD toggle on                                             | Full VAE only (~2.6 s stage) |
| Steps       | `1` (SD-Turbo)                                              | —                            |
| TAESD asset | `sd-turbo-fp16_taesd_vae_decoder_model.onnx` on `models-v1` | —                            |

Export host-side: [`scripts/export-taesd-asset.sh`](../scripts/export-taesd-asset.sh).

`diffuse.flag` remains for **headless bench/WDP** only.

## Linux development

```bash
cmake --preset linux-test
cmake --build build/linux-test -j$(nproc)
ctest --test-dir build/linux-test --output-on-failure
```

## Bench protocol

- Prompts: `bench/prompts/standard-512.txt`, `long-1k.txt`
- 2+ runs; drop run 1; median runs 2–3
- DML truth: `scripts/profile-dml-run.sh --gpu-sample` → `VERDICT: GPU`

## Obsolete assumptions (do not reuse)

| Myth                                | Reality                                                   |
| ----------------------------------- | --------------------------------------------------------- |
| GPU pool ~768 MB                    | **3801 MB** (Game designation)                            |
| Dev Mode disk ~2.5 GB cap           | Raised to **90 GB** (Dev Home)                            |
| Diffusion needs headless (887A0036) | Conflict is **ORT GenAI** only; plain ORT DML in-proc OK  |
| MSIX bundles model                  | **~19 MB** MSIX; catalogue download                       |
| `intra_op_num_threads: 8`           | Collapses BOTH decode (~10 tok/s) and prefill (~87) — §5f |

## Validation checklist

1. Deploy the default **`xllama-appx`** CI artifact (unified + PatchedGenAI #2280; version is `Major.Minor.Build` from `uwp/AppxManifest.xml` with the **Revision** auto-stamped from the CI run number — see `CHANGELOG.md`).
   ⚠️ **Coming from ≤1.4.x**: 1.5.0.0 changed the package identity
   (`VenereLabs.xllama` → `GianlucaMazza.xllama`) — it installs as a NEW app,
   LocalState does not carry over, and the old app should be uninstalled
   (see [install-release.md](./install-release.md)). Launch the new app once
   before provisioning (LocalState does not exist until first launch —
   runbook preflight).
2. Provision models AFTER the install — `install-latest-build.sh` always
   uninstalls first, wiping LocalState: `./scripts/provision-models.sh --all-test`
   (seeds `lfm25-350m`, `smollm2-360m-cpu-int4`, the parity-validated
   `smollm2-360m-dml-fp16-v2` routing target and `sd-turbo-fp16`).
3. Remove `bench.flag` from LocalState if `install-latest-build.sh --bench` left it behind
4. Run the automated gate:

   ```bash
   source ~/.config/xllama/xbox-env
   ./scripts/validate-console.sh all   # routing + settings (9 values) + GGUF + TAESD → ALL PASS on 1.5.0.698 (2026-07-25, post-identity-migration; API gates also ALL PASS)
   ```

5. Manual/debug path: [console-validation-runbook.md](./console-validation-runbook.md) per §
