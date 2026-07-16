# Model Selection Checklist

> **SSOT for the model catalogue and backend selection.** The catalogue data
> lives in [`uwp/models/manifest.json`](../uwp/models/manifest.json); this file
> is the narrative source of truth for limits, licensing, backend routing, and
> "add your own model". Performance numbers live in [`benchmarks.md`](benchmarks.md).

Operational criteria for choosing or evaluating a model for the xllama UWP build
(Xbox Series S Dev Mode). Covers both backends — ONNX Runtime GenAI (CPU/DirectML)
and llama.cpp (GGUF, CPU).

## Hard limits (must pass)

| Constraint                              | Limit                                                                       | Source                       |
| --------------------------------------- | --------------------------------------------------------------------------- | ---------------------------- |
| Format                                  | ONNX GenAI directory (`genai_config.json` + `model.onnx` + tokenizer files) | ORT GenAI 0.14.1 requirement |
| External data files                     | Must be merged into a self-contained `model.onnx` before distribution       | `uwp-constraints.md §8`      |
| On-disk size (merged ONNX)              | ~2.2–2.5 GB free by default (**expandable to 90 GB**, so rarely binding)    | `uwp-constraints.md §9`      |
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

4. Check on-disk size against the Dev Mode disk budget (~2.2–2.5 GB free by
   default, expandable to 90 GB — `uwp-constraints.md §9`):

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

   If model creation fails, use `deploy.sh diagnose-startup` and the current
   troubleshooting guidance in `device-portal.md`.

7. Benchmark and compare against the current baseline:
   ```bash
   ./scripts/bench-xbox-ort.sh <model-name> --runs 3 --out bench/results/phase1-cpu.csv
   ```
   Store results in a campaign-specific CSV under `bench/results/`; do not append
   unrelated models to the historical Phase 1 file.

## Reference: tested models

| Model                          | On-disk (merged) | CPU EP                                  | DirectML EP                                      | Notes                                                                                                                                 |
| ------------------------------ | ---------------- | --------------------------------------- | ------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| SmolLM2-360M-Instruct INT4 CPU | 417 MB           | ✅ ORT baseline (66.3 tok/s, 0.14.1)   | ❌ `80070057` (CPU-int4 graph in DML fused node) | ORT-only default; downloaded from the `models-v1` Release catalogue                                                                   |
| SmolLM2-360M-Instruct INT4 DML | 285 MB           | —                                       | ⚠️ **8.8 tok/s** but wrong logits (#91)          | Built with ORT GenAI model builder (`-p int4 -e dml`); CPU ~8× faster — and DML text output is numerically wrong on this device (#91) |
| SmolLM2-1.7B-Instruct INT4 CPU | 1.4 GB           | ✅ in-app (`models-v1` catalogue)       | —                                                | Console: 20.6 tok/s decode, peak 2423 MB (`phase35-1b-cpu.csv`); also USB/LocalState                                                  |
| Phi-3.5-mini Q3_K_S (GGUF)     | 1.68 GB          | ✅ H4 A/B (11.3 tok/s, 2453 MB)         | —                                                | Loses speed+RAM vs Llama-3.2-3B Q3; **not** catalogue (`phase7-scale.csv`)                                                            |

## Historical ONNX candidate survey

The July 2 candidate survey used an early disk estimate that was later
superseded by the measured 90 GB Dev Mode allocation and the patched external-
data path. Its individual size-based rejections are not current selection
policy. Historical details remain in Git history and `CHANGELOG.md`; evaluate
new candidates against the measured limits in `uwp-constraints.md` and the
sequence above.

**External data >2 GB (un-mergeable) — unblocked for loading, but not for fp16-GPU
(2026-07-15).** The 2 GB protobuf merge ceiling no longer blocks _loading_: the
patched ORT DLL (`uwp-constraints.md §8` Fix B) loads external-`.onnx.data` models
directly, console-validated with a 1.86 GB int4 model. **But this does not make big
fp16 models viable on the GPU**: a native-DML 1B fp16 (2.49 GB) loads yet OOMs
inference within the 3801 MB budget (`uwp-constraints.md §7`). Practical takeaway:
the >2 GB path is a **CPU/int4** enabler; DML-fp16 stays ≤~360-500 M
(`smollm2-360m-dml-fp16` — whose text output is anyway wrong on this device,
`#91`: the model is kept only as the parity-gate probe). Build DML models
natively (`builder … -p fp16 -e dml`) — a cuda-fp16 re-host loads but fails DML
inference.

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

## Publishing ORT model assets (models-v1) — logit-parity gate

**No ORT text-model asset reaches the `models-v1` release without passing a
logit-parity check.** Institutionalized after #91: the DML fp16/int4 assets
shipped for weeks producing numerically wrong logits — only tok/s was ever
measured. Healthy cross-quant parity looks like NMSE ≈ 0.09 / top-10 overlap
≈ 0.9; broken looks like NMSE ≈ 0.98 / top-10 ≈ 0.3 — the rank metrics
(top-1 + top-10) are the decisive gate, not the raw diff.

Runbook, in order:

1. **Golden** (once per model family): dump the llama.cpp reference from the
   equivalent GGUF on the host —
   `xllama-cli -m <model>.gguf -p "The capital of France is" -n 1 --greedy
--dump-logits tests/golden/logits-<model>-short.bin` — and commit
   `.bin` + `.json` sidecar (see `tests/golden/logits-smol-short.bin`).
   Templated prompts must tokenize with `parse_special` (the CLI does this
   when a chat template is active).
2. **Host CPU-EP check** (every asset, minimum bar): run the ORT asset through
   the bridge CLI with `--greedy --dump-logits`, then
   `scripts/compare-logits.py <golden.bin> <ort.bin>` — PASS requires top-1
   agreement and top-10 overlap within thresholds (defaults in the script;
   cross-quant gate used on-device: `MAX_ABS_DIFF=8.0`, `NMSE=0.2`).
3. **On-device check** (every DML asset): provision with
   `scripts/provision-models.sh`, then
   `MODEL=<catalogue-name> ./scripts/validate-logit-parity.sh [golden.bin]`.
   A DML text asset that fails here must not ship — and
   `routing_policy.h kDmlTextLogitsBroken` stays set until one passes.
4. **Upload**: `gh release upload models-v1 <files> --clobber`, then update
   `approx_bytes` in `uwp/models/manifest.json` with the **exact** byte sizes
   (a mismatch re-triggers the download loop fixed in #90).

`tests/test_logit_parity.cpp` runs the golden regression in CI when
`XLLAMA_TEST_MODEL` is set (skips cleanly otherwise).

## Modern models (2026 survey, runtime-backend era)

The ORT GenAI model builder is frozen at the Qwen3 / Gemma3 architectures.
Newer small models (Qwen3.5, LFM2, Gemma-4) ship as GGUF and run **only via
llama.cpp** — reachable via the shipping `unified` backend build with
`kind: gguf` catalogue entries. All are console-measured — **decode/prefill/RAM
numbers live in [`benchmarks.md`](benchmarks.md)** (the perf SSOT); this table is
the catalogue status only.

| Model            | Path      | Disk size        | Status                                                             |
| ---------------- | --------- | ---------------- | ------------------------------------------------------------------ |
| Qwen3.5-0.8B     | llama.cpp | 507 MB           | ✅ `qwen35-0.8b` — optional modern GGUF                            |
| LFM2.5-350M      | llama.cpp | 218 MB           | ✅ `lfm25-350m` — hybrid edge arch; default chat on unified builds |
| LFM2.5-1.2B      | llama.cpp | 697 MB           | ✅ `lfm25-1.2b-instruct` — balanced; 37.9 tok/s, H9 6/8            |
| LFM2-2.6B        | llama.cpp | 1.46 GB          | ✅ `lfm2-2.6b` — quality; 18.4 tok/s, H9 7/8                       |
| Qwen3-0.6B       | ORT GenAI | 969 MB merged    | ✅ builds; heavy (151k-vocab embedding dominates)                  |
| Gemma-3-270M     | llama.cpp | 253 MB           | ✅ `gemma3-270m` — fast, tiny, fits easily                         |
| Gemma-4-E2B      | llama.cpp | 2.45 GB (Q3_K_S) | ✅ **console-validated** `gemma4-e2b` (see verdict below)          |
| Gemma-4 E4B/12B+ | llama.cpp | ≥4.5 GB          | ⛔ too big / too slow for the console                              |

**Gemma chat template**: the ORT GenAI _builder_ is frozen at Gemma3, but the
vendored `llama.cpp` (`9a532ae4b`) already carries `LLM_ARCH_GEMMA3` **and**
`LLM_ARCH_GEMMA4` — both load and generate (verified via `xllama-cli`,
`general.architecture = gemma3`/`gemma4`). What was missing was the prompt
format: the app hard-coded ChatML. `chat_format_for()` (`src/bridge/chat_prompt.cpp`)
now selects the Gemma template (`<start_of_turn>…<end_of_turn>`, no system role,
stop `<end_of_turn>`, `<bos>` via `add_bos`) by model id, so any `gemma*` GGUF
gets the right template with zero per-model code.

**Gemma-4-E2B verdict** (console-validated 2026-07-14): the "too big" call is
**overturned**. E2B is ~5B raw params (2B effective, MatFormer). Key findings
(full numbers in [`benchmarks.md`](benchmarks.md)):

- ✅ **The ~2 GB Dev Mode per-file limit does not apply to GGUF** — a >2 GB single
  `.gguf` loads with no OOM under the Game budget.
- ✅ **Generates coherently** and stops on `<end_of_turn>`.
- ⚠️ **Load is slow** (a few seconds; the CPU load is repack-bound, not
  file-read-bound — mmap does not help, see `benchmarks.md`).
- The catalogue default is **Q3_K_S (2.45 GB)**, not the smaller 2-bit UD-IQ2_M:
  IQ2_M collapsed to an immediate EOG on long declarative prompts, which Q3_K_S
  fixes (and it decodes faster). Details in `benchmarks.md` root-cause notes.

Disk was never the real constraint (Dev Mode is now 90 GB, `uwp-constraints.md §9`).
Catalogue entry `gemma4-e2b` downloads from HF (2.29 GB > the GitHub release 2 GB
asset limit). See `docs/benchmarks.md` for the full comparison.

Backend selection is by `SessionParams::backend` (explicit values take
precedence in dual-backend builds). `Auto` (default) uses either a `.gguf`
suffix or on-disk layout inspection via the public helper
`model_uses_llama_backend()` (bare catalogue names or directories containing a
`.gguf` file → llama.cpp; otherwise ORT GenAI). Catalogue entries `qwen35-0.8b`
and `lfm25-350m` download in-app from `models-v1`. The larger LFM entries
download directly from LiquidAI so their upstream `LICENSE` file accompanies
each GGUF.

Redistribution licensing (verified 2026-07-10): the Qwen quant
(`unsloth/Qwen3.5-0.8B-GGUF`) is Apache-2.0 ✅. LFM Open License v1.0 §4
permits redistribution provided recipients get a copy of the license —
`LFM2.5-350M_LICENSE.txt` is published on the release and listed in the
catalogue entry so it lands next to the model on-device. Note §5: commercial
use is limited to entities under $10M revenue (xllama is non-commercial
research). **Gemma** is under the Gemma Terms of Use (not Apache/MIT): the
`gemma3-270m` catalogue entry therefore downloads straight from the original
Hugging Face repo (`hf_base_url` → `unsloth/gemma-3-270m-it-GGUF/resolve/main`)
rather than re-hosting the weights on `models-v1` — verify the Terms permit
redistribution before ever mirroring them onto our release.

### TAESD — a faster diffusion VAE decoder

`madebyollin/taesd` (MIT) is a **4.9 MB** tiny autoencoder that drop-in replaces
SD-Turbo's 94 MB VAE decoder (same `latent_sample [1,4,64,64] → sample
[1,3,512,512]` contract). On console the VAE stage is 2.6 s of the 6.9 s total,
so TAESD targets **~4.5 s/image** and frees ~90 MB of GPU. Export with
`diffusion/export_taesd.py` (the diffusers `AutoencoderTiny` decoder already
emits SD `[-1,1]` — the [0,1]→remap assumption was falsified); validate the swap
with `validate_pipeline.py <sd_dir> out.png "<prompt>" 1 42 <taesd_dir>`.
