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
| **unified+#2280+PatchedOrt** (shipping) | `xllama-appx` from `build-uwp.yml` | GGUF + ORT, external-data ORT (GPU text routing suspended — #91) |
| **llamacpp**                            | `xllama-appx-llamacpp`             | Bench A/B only — not for end users                               |

## Chat models

The decode figures below are rough guidance; the authoritative, disambiguated
tables are in [benchmarks.md](benchmarks.md) (the perf SSOT).

| Use case                       | Catalogue `name`        | Backend               | Measured decode                                                                                                        |
| ------------------------------ | ----------------------- | --------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| **Default (unified)**          | `lfm25-350m`            | llama.cpp (`unified`) | **94.2 tok/s** (recommended)                                                                                           |
| **ORT default**                | `smollm2-360m-cpu-int4` | ORT CPU int4          | ~66 tok/s                                                                                                              |
| **Routing GPU** — disabled #91 | `smollm2-360m-dml-fp16` | ORT DML fp16          | ~47 tok/s decode; ~354 tok/s prefill @1k — **text logits wrong on DML (#91): not routable, not auto-downloaded (#95)** |
| **Larger chat**                | `smollm2-1.7b-cpu-int4` | ORT CPU int4          | ~21 tok/s (in-app `models-v1` download)                                                                                |
| **Modern GGUF**                | `qwen35-0.8b`           | llama.cpp (`unified`) | 35.1 tok/s (98.1 prefill, t6)                                                                                          |
| **Fast modern GGUF**           | `lfm25-350m`            | llama.cpp (`unified`) | 94.2 tok/s (241.4 prefill, t6)                                                                                         |

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

**CPU (bundled default)** — copy from [`bench/configs/genai_config-threads-4.json`](../bench/configs/genai_config-threads-4.json):

- `provider_options: []`
- `intra_op_num_threads: 4` (t=8 regresses to ~28 tok/s on Series S)
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

| Key                    | Recommended                       | Notes                                                                             |
| ---------------------- | --------------------------------- | --------------------------------------------------------------------------------- |
| `model`                | `lfm25-350m`                      | First-launch default on **unified** shipping                                      |
| `kv_reuse`             | `true`                            | 4.87× ORT / 4.07× GGUF turn-2 prefill (measured)                                  |
| `routing`              | `2` (Auto) on unified             | inert for text while #91 holds (always CPU); GGUF skips                           |
| `gpu_model`            | `smollm2-360m-dml-fp16`           | NOT auto-downloaded while #91 holds (#95); 725 MB, WDP/`provision-models.sh` only |
| `diffuse_taesd_vae`    | `true` after asset on `models-v1` | ~4.5 s/image target                                                               |
| `sampling.temperature` | `0.8`                             | UI default                                                                        |
| `sampling.n_predict`   | `512`                             | UI default                                                                        |

Factory defaults match [`settings-modern.json`](../bench/configs/settings-modern.json)
on unified builds (`DefaultChatModelId()` → `lfm25-350m`). ORT-only builds still
default to `smollm2-360m-cpu-int4`.

Routing Auto uses **600 tokens** (real tokenizer count), sticky per conversation
— but the #91 gate (`kDmlTextLogitsBroken`) precedes the threshold: while it
holds, every text turn resolves to the CPU model regardless of length.

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

| Myth                                | Reality                                                  |
| ----------------------------------- | -------------------------------------------------------- |
| GPU pool ~768 MB                    | **3801 MB** (Game designation)                           |
| Dev Mode disk ~2.5 GB cap           | Raised to **90 GB** (Dev Home)                           |
| Diffusion needs headless (887A0036) | Conflict is **ORT GenAI** only; plain ORT DML in-proc OK |
| MSIX bundles model                  | **~19 MB** MSIX; catalogue download                      |
| `intra_op_num_threads: 8`           | Bandwidth saturation on Series S                         |

## Validation checklist

1. Deploy the default **`xllama-appx`** CI artifact (unified + PatchedGenAI #2280; version is `Major.Minor.Build` from `uwp/AppxManifest.xml` with the **Revision** auto-stamped from the CI run number — see `CHANGELOG.md`)
2. Provision models AFTER the install — `install-latest-build.sh` always
   uninstalls first, wiping LocalState: `./scripts/provision-models.sh --all-test`
   (seeds `lfm25-350m`, `smollm2-360m-cpu-int4` for routing/parity, `sd-turbo-fp16`;
   the gated `dml-fp16` is intentionally excluded — #91/#95).
3. Remove `bench.flag` from LocalState if `install-latest-build.sh --bench` left it behind
4. Run the automated gate:

   ```bash
   source ~/.config/xllama/xbox-env
   ./scripts/validate-console.sh all   # routing + GGUF + TAESD → ALL PASS (2026-07-16, 1.2.0.534)
   ```

5. Manual/debug path: [console-validation-runbook.md](./console-validation-runbook.md) per §
