# Model Selection Checklist

Operational criteria for choosing or evaluating an ONNX GenAI model for the
xllama UWP build (Xbox Series S Dev Mode, CPU EP).

## Hard limits (must pass)

| Constraint                              | Limit                                                                       | Source                       |
| --------------------------------------- | --------------------------------------------------------------------------- | ---------------------------- |
| Format                                  | ONNX GenAI directory (`genai_config.json` + `model.onnx` + tokenizer files) | ORT GenAI 0.14.1 requirement |
| External data files                     | Must be merged into a self-contained `model.onnx` before distribution       | `uwp-constraints.md §8`      |
| On-disk size (merged ONNX)              | Dev Mode disk budget: ~2.2–2.5 GB free total across all models              | `uwp-constraints.md §9`      |
| GPU EP weights (if attempting DirectML) | < ~300 MB                                                                   | `uwp-constraints.md §5`, §7  |

## Selection sequence

1. Identify a candidate (Hugging Face, ONNX zoo, etc.). Confirm format is ONNX GenAI
   (directory with `genai_config.json`), not a plain `.onnx` or a GGUF file.

2. Download locally:

   ```bash
   huggingface-cli download <repo-id> --local-dir models/<name>
   ```

3. Merge external data if present:

   ```bash
   python3 scripts/merge_onnx_external_data.py models/<name>
   ```

   The script prints a `NOTE` or `WARNING` if the merged size exceeds budget thresholds.

4. Check on-disk size against the Dev Mode disk budget (~2.2–2.5 GB free total):

   ```bash
   du -sm models/<name>
   ```

5. Provision it on the console — either add a catalogue entry (see
   "Add your own model" below) and let the app download it, or upload directly:

   ```bash
   source ~/.config/xllama/xbox-env
   PFN=$(./scripts/deploy.sh pfn)
   ./scripts/deploy.sh upload-dir models/<name>/ "$PFN" "models\\<name>"
   ```

   If the upload fails for space (`0x80070070` ERROR_DISK_FULL), the model
   exceeds the available Dev Mode partition space.

6. Launch and check the log:

   ```bash
   ./scripts/deploy.sh get-log
   ```

   If `OgaCreateModel failed` appears, see `phase1-runbook.md §8` for diagnosis.

7. Benchmark and compare against the current baseline:
   ```bash
   ./scripts/bench-xbox-ort.sh <model-name> --runs 3 --out bench/results/phase1-cpu.csv
   ```
   Results land in `bench/results/phase1-cpu.csv`.

## Reference: tested models

| Model                          | On-disk (merged) | CPU EP                                  | DirectML EP                                      | Notes                                                                                                     |
| ------------------------------ | ---------------- | --------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| SmolLM2-360M-Instruct INT4 CPU | 417 MB           | ✅ Active baseline (66.3 tok/s, 0.14.1) | ❌ `80070057` (CPU-int4 graph in DML fused node) | Default; downloaded from the `models-v1` Release catalogue                                                |
| SmolLM2-360M-Instruct INT4 DML | 285 MB           | —                                       | ✅ **8.8 tok/s** (headless v0.3.4)               | Built with ORT GenAI model builder (`-p int4 -e dml`); CPU ~8× faster — DML not competitive at this scale |
| SmolLM2-1.7B-Instruct INT4 CPU | 1.4 GB           | ✅ via USB / LocalState                 | —                                                | Console: 20.6 tok/s decode, peak 2423 MB (`phase35-1b-cpu.csv`)                                           |
| Phi-3.5-mini INT4 CPU          | ~2.7 GB          | ❌ Disk budget                          | —                                                | Not attempted                                                                                             |
| Phi-3.5-mini GPU INT4 AWQ      | ~2.2 GB          | —                                       | ❌ GPU OOM + disk                                | Not viable                                                                                                |

## Candidates evaluated (HF Hub file sizes, 2026-07-02)

Sizes below are the published `model.onnx.data` file sizes on Hugging Face
(merged size adds the small `model.onnx` graph + tokenizer files, ~10–20 MB).

- **Qwen2.5-0.5B INT4 ONNX CPU** — ❌ **ruled out**. Real size ~822 MB
  (`cpu-int4-rtn-block-32` in `xiaoyao9184/Qwen2.5-0.5B-Instruct-onnx-genai` and
  `hazemmabbas/Qwen2.5-0.5B-int4-block-32-acc-3-Instruct-onnx-cpu`; AMD's
  int4-float16 variant is ~780 MB). The earlier ~200 MB estimate was wrong: the
  151k-token vocab embedding dominates and is not INT4-quantized. Exceeds the
  600 MB disk borderline (note: the "~768 MB GPU pool" cited at evaluation time
  proved wrong — measured budget is 3801 MB; disk is the real constraint).
- **Qwen2.5-0.5B INT4 AWQ DirectML** — ⚠️ borderline. `dml-int4-awq-block-128`
  variant (same xiaoyao9184 repo) is ~507 MB: below the GPU pool on paper, but
  KV cache + activations leave little headroom. Possible DML retry candidate
  via USB/LocalState (not MSIX bundling).
- **Llama-3.2-1B INT4 ONNX CPU** — ❌ ruled out for MSIX. Real size ~1.77 GB
  (`onnx-community/Llama-3.2-1B-Instruct-GENAI-ONNX`
  `cpu-int4-rtn-block-32-acc-level-4`; `patdev/Llama-3.2-1B-Instruct-int4-cpu-onnx`
  identical; `aigdat` AWQ-uint4 ~1.66 GB). USB-only path, same class as
  SmolLM2-1.7B (1.4 GB, 20.6 tok/s on console).
- **Gemma-2-2B INT4 ONNX CPU**: estimated above 1 GB merged — unlikely to fit.

Verify with `merge_onnx_external_data.py` output before committing to a build.

**Future — BitNet / INT2 models**: Microsoft BitNet b1.58 (1-bit/1.58-bit quantization)
could bring a 1.7B–3B model under 400 MB, fitting both the disk and GPU budgets.
ORT GenAI does not natively support INT2 as of 0.14.1. Re-evaluate when
`microsoft/onnxruntime-genai` adds a stable INT2/BitNet execution path.

## Why these limits

The disk and GPU limits derive from the UWP sandbox on Xbox Series S in Dev Mode.
This document records only observed behavior at the application boundary. See
`uwp-constraints.md §7` (GPU pool) and `uwp-constraints.md §9` (disk budget) for
what we measure. Do not treat any claim about the internal Xbox OS partition layout
as authoritative unless backed by a Microsoft source.

## Add your own model (manifest override, no reinstall)

The Settings ComboBox is populated from the model catalogue. A
`LocalState\manifest.json` uploaded via Device Portal is **merged per entry**
into the bundled catalogue (`uwp/model-downloader.cpp`, `LoadModelManifest`):
a same-name entry replaces the bundled one, new names are appended, and
bundled entries you don't mention stay available — so a minimal override file
with just your model is enough, and it keeps working when the bundled
catalogue grows:

1. Write a manifest with just your entry (or copy `uwp/models/manifest.json`):

   ```json
   {
     "name": "my-model-dir",
     "display": "My Model (CPU int4)",
     "kind": "ort-genai",
     "hf_base_url": "https://example.com/base/url",
     "files": [{ "filename": "model.onnx", "approx_bytes": 123456789 }]
   }
   ```

   With `hf_base_url` set, the app downloads `<hf_base_url>/<remote or filename>`
   for each file on selection. Without it, the entry expects the directory at
   `LocalState\models\<name>` (Device Portal upload) or USB `E:\xllama\models\<name>`.
   Optional fields: `kind` (`"ort-genai"` default; `"diffusion"` entries feed the
   Image dialog and are hidden from the chat picker) and per-file `remote` (the
   flat asset name in the release when `filename` carries a subpath, e.g.
   `"filename": "unet/model.onnx"` + `"remote": "sd-turbo-fp16_unet_model.onnx"`).

2. Upload the override and (if needed) the model files:

   ```bash
   source ~/.config/xllama/xbox-env
   PFN=$(./scripts/deploy.sh pfn)
   ./scripts/deploy.sh upload-file my-manifest.json "$PFN" ""   # rename to manifest.json first
   ./scripts/deploy.sh upload-dir ./my-model-dir/ "$PFN" "models\\my-model-dir"
   ```

3. Restart the app: the ComboBox now shows the new entry.

Constraints: `model.onnx` must be **self-contained** (< 2 GB, external data merged —
`uwp-constraints.md §8`) and fit the Dev Mode disk budget. For diffusion models the
contract is different (three components + CLIP assets): see `diffusion/README.md`.

## Modern models (2026 survey, runtime-backend era)

The ORT GenAI model builder is frozen at the Qwen3 / Gemma3 architectures.
Newer small models (Qwen3.5, LFM2, Gemma-4) ship as GGUF and run **only via
llama.cpp** — reachable once the `unified` backend build (PR #27) is the default
and the catalogue carries `kind: gguf` entries. Host-validated 2026-07-09;
on-console decode/prefill benches pending.

| Model             | Path      | Size (Q4_K_M / int4) | Status                                                                |
| ----------------- | --------- | -------------------- | --------------------------------------------------------------------- |
| Qwen3.5-0.8B      | llama.cpp | 507 MB               | ✅ loads+generates via submodule (`qwen35`); modern default candidate |
| LFM2.5-350M       | llama.cpp | 218 MB               | ✅ loads via submodule; hybrid edge arch                              |
| Qwen3-0.6B        | ORT GenAI | 969 MB merged        | ✅ builds; heavy (151k-vocab embedding dominates)                     |
| Gemma-3-270M      | ORT GenAI | ~300 MB (est.)       | ⛔ gated on HF — needs an access token to build                       |
| Gemma-4 (E2B/E4B) | —         | ≥2B effective        | ⛔ too big + arch not in builder                                      |

Backend selection is by `SessionParams::backend` (explicit values take
precedence in dual-backend builds). `Auto` (default) uses either a `.gguf`
suffix or on-disk layout inspection via the public helper
`model_uses_llama_backend()` (bare catalogue names or directories containing a
`.gguf` file → llama.cpp; otherwise ORT GenAI). Catalogue entries `qwen35-0.8b`
and `lfm25-350m` download in-app from `models-v1` (Fase 2b, 2026-07-10).

Redistribution licensing (verified 2026-07-10): the Qwen quant
(`unsloth/Qwen3.5-0.8B-GGUF`) is Apache-2.0 ✅. LFM Open License v1.0 §4
permits redistribution provided recipients get a copy of the license —
`LFM2.5-350M_LICENSE.txt` is published on the release and listed in the
catalogue entry so it lands next to the model on-device. Note §5: commercial
use is limited to entities under $10M revenue (xllama is non-commercial
research). Gemma Terms — still verify before ever hosting.

### TAESD — a faster diffusion VAE decoder

`madebyollin/taesd` (MIT) is a **4.9 MB** tiny autoencoder that drop-in replaces
SD-Turbo's 94 MB VAE decoder (same `latent_sample [1,4,64,64] → sample
[1,3,512,512]` contract). On console the VAE stage is 2.6 s of the 6.9 s total,
so TAESD targets **~4.5 s/image** and frees ~90 MB of GPU. Export with
`diffusion/export_taesd.py` (the diffusers `AutoencoderTiny` decoder already
emits SD `[-1,1]` — the [0,1]→remap assumption was falsified); validate the swap
with `validate_pipeline.py <sd_dir> out.png "<prompt>" 1 42 <taesd_dir>`.
