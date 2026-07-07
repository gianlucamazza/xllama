# Model Selection Checklist

Operational criteria for choosing or evaluating an ONNX GenAI model for the
xllama UWP build (Xbox Series S Dev Mode, CPU EP).

## Hard limits (must pass)

| Constraint                              | Limit                                                                       | Source                       |
| --------------------------------------- | --------------------------------------------------------------------------- | ---------------------------- |
| Format                                  | ONNX GenAI directory (`genai_config.json` + `model.onnx` + tokenizer files) | ORT GenAI 0.13.2 requirement |
| External data files                     | Must be merged before MSIX packaging                                        | `uwp-constraints.md §8`      |
| On-disk size (merged ONNX)              | < 400 MB recommended, < 600 MB borderline                                   | `uwp-constraints.md §9`      |
| MSIX size                               | < 600 MB recommended, < 800 MB borderline                                   | `uwp-constraints.md §9`      |
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

4. Check on-disk size. Reject if > 600 MB:

   ```bash
   du -sm models/<name>
   ```

5. Update `uwp/xllama.vcxproj` to point to the new model directory. Build MSIX:

   ```bash
   # From Windows VM or via CI push:
   .\scripts\build-uwp.ps1 -Configuration Release -Platform x64
   ```

6. Inspect MSIX size. Reject if > 800 MB:

   ```bash
   ls -lh uwp/AppPackages/xllama/*.msix
   ```

7. Deploy:

   ```bash
   source ~/.config/xllama/xbox-env
   ./scripts/deploy.sh path/to/xllama_*.msix
   ```

   If install fails with `0x80070070` (ERROR_DISK_FULL), the model exceeds the
   available Dev Mode partition space.

8. Launch and check the log:

   ```bash
   ./scripts/deploy.sh get-log
   ```

   If `OgaCreateModel failed` appears, see `phase1-runbook.md §8` for diagnosis.

9. Benchmark and compare against the current baseline:
   ```bash
   ./scripts/bench-xbox.sh <model-name> bench/config/phase1-smollm2-360m.json
   ```
   Results land in `bench/results/phase1-cpu.csv`.

## Reference: tested models

| Model                          | On-disk (merged) | CPU EP                                | DirectML EP                                      | Notes                                                                                                     |
| ------------------------------ | ---------------- | ------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| SmolLM2-360M-Instruct INT4 CPU | 403 MB           | ✅ Active baseline (70.9 tok/s)       | ❌ `80070057` (CPU-int4 graph in DML fused node) | Bundled in MSIX                                                                                           |
| SmolLM2-360M-Instruct INT4 DML | 285 MB           | —                                     | ✅ **8.8 tok/s** (headless v0.3.4)               | Built with ORT GenAI model builder (`-p int4 -e dml`); CPU ~8× faster — DML not competitive at this scale |
| SmolLM2-1.7B-Instruct INT4 CPU | 1.4 GB           | ❌ MSIX bundling / ✅ via USB (Exp 3) | —                                                | 23.6 tok/s, n=191, peak 2195 MB                                                                           |
| Phi-3.5-mini INT4 CPU          | ~2.7 GB          | ❌ Disk budget                        | —                                                | Not attempted                                                                                             |
| Phi-3.5-mini GPU INT4 AWQ      | ~2.2 GB          | —                                     | ❌ GPU OOM + disk                                | Not viable                                                                                                |

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
  SmolLM2-1.7B (1.4 GB, 23.6 tok/s).
- **Gemma-2-2B INT4 ONNX CPU**: estimated above 1 GB merged — unlikely to fit.

Verify with `merge_onnx_external_data.py` output before committing to a build.

**Future — BitNet / INT2 models**: Microsoft BitNet b1.58 (1-bit/1.58-bit quantization)
could bring a 1.7B–3B model under 400 MB, fitting both the disk and GPU budgets.
ORT GenAI does not natively support INT2 as of 0.13.2. Re-evaluate when
`microsoft/onnxruntime-genai` adds a stable INT2/BitNet execution path.

## Why these limits

The disk and GPU limits derive from the UWP sandbox on Xbox Series S in Dev Mode.
This document records only observed behavior at the application boundary. See
`uwp-constraints.md §7` (GPU pool) and `uwp-constraints.md §9` (disk budget) for
what we measure. Do not treat any claim about the internal Xbox OS partition layout
as authoritative unless backed by a Microsoft source.
