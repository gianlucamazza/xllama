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

## Chat models

The decode figures below are rough guidance; the authoritative, disambiguated
tables are in [benchmarks.md](benchmarks.md) (the perf SSOT).

| Use case              | Catalogue `name`           | Backend               | Measured decode                                                                                                        |
| --------------------- | -------------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Default (unified)** | `lfm25-350m`               | llama.cpp (`unified`) | **94.2 tok/s** (recommended)                                                                                           |
| **Balanced chat**     | `lfm25-1.2b-instruct`      | llama.cpp (`unified`) | **37.9 tok/s**, 811 MB peak; H9 6/8                                                                                    |
| **Quality chat**      | `lfm2-2.6b`                | llama.cpp (`unified`) | **18.4 tok/s**, 1623 MB peak; H9 7/8                                                                                   |
| **ORT default**       | `smollm2-360m-cpu-int4`    | ORT CPU int4          | ~66 tok/s                                                                                                              |
| **Routing GPU**       | `smollm2-360m-dml-fp16-v2` | ORT DML fp16          | 44.4 tok/s decode; 236.7 tok/s prefill — RMSNorm-decomposed graph, #91 parity-validated (`dml-rmsnorm-fix-runbook.md`) |
| **Larger chat**       | `smollm2-1.7b-cpu-int4`    | ORT CPU int4          | ~21 tok/s (in-app `models-v1` download)                                                                                |
| **Modern GGUF**       | `qwen35-0.8b`              | llama.cpp (`unified`) | 35.1 tok/s (98.1 prefill, t6)                                                                                          |
| **Fast modern GGUF**  | `lfm25-350m`               | llama.cpp (`unified`) | 94.2 tok/s (241.4 prefill, t6)                                                                                         |

GGUF thread default: llama.cpp auto-detect is capped at **6** on console
(`detect_threads_llama` — t6 measured optimum, t7/t8 livelock); explicit
`--threads` on `bench-xbox-ort.sh` still wins.

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
  ⚠️ The shipped `smollm2-360m-cpu-int4` asset currently sets **no**
  `intra_op_num_threads` at all, so neither this value nor the previous one is
  in production. Applying it means republishing the asset on models-v1.
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
  It tracks `max_length` instead (#130, `docs/uwp-constraints.md` §5c), so the
  value is provisional and under re-derivation.
- It optimizes the **first turn only** (§5d). The GPU buys a faster first token
  on a long prompt; from the second turn the CPU wins at every reachable length,
  because DirectML cannot reuse a KV cache. Whether Auto is worth leaving on
  depends on how single-turn your long-prompt conversations are.

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

1. Deploy the default **`xllama-appx`** CI artifact (unified + PatchedGenAI #2280; version is `Major.Minor.Build` from `uwp/AppxManifest.xml` with the **Revision** auto-stamped from the CI run number — see `CHANGELOG.md`)
2. Provision models AFTER the install — `install-latest-build.sh` always
   uninstalls first, wiping LocalState: `./scripts/provision-models.sh --all-test`
   (seeds `lfm25-350m`, `smollm2-360m-cpu-int4`, the parity-validated
   `smollm2-360m-dml-fp16-v2` routing target and `sd-turbo-fp16`).
3. Remove `bench.flag` from LocalState if `install-latest-build.sh --bench` left it behind
4. Run the automated gate:

   ```bash
   source ~/.config/xllama/xbox-env
   ./scripts/validate-console.sh all   # routing + settings (9 values) + GGUF + TAESD → routing+settings PASS on 1.4.0.641, GGUF PASS on 1.4.0.633 (2026-07-22); TAESD not re-run
   ```

5. Manual/debug path: [console-validation-runbook.md](./console-validation-runbook.md) per §
