# Model Selection Checklist

Operational criteria for choosing or evaluating an ONNX GenAI model for the
xllama UWP build (Xbox Series S Dev Mode, CPU EP).

## Hard limits (must pass)

| Constraint | Limit | Source |
|------------|-------|--------|
| Format | ONNX GenAI directory (`genai_config.json` + `model.onnx` + tokenizer files) | ORT GenAI 0.13.2 requirement |
| External data files | Must be merged before MSIX packaging | `uwp-constraints.md §8` |
| On-disk size (merged ONNX) | < 400 MB recommended, < 600 MB borderline | `uwp-constraints.md §9` |
| MSIX size | < 600 MB recommended, < 800 MB borderline | `uwp-constraints.md §9` |
| GPU EP weights (if attempting DirectML) | < ~300 MB | `uwp-constraints.md §5`, §7 |

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

| Model | On-disk (merged) | CPU EP | DirectML EP | Notes |
|-------|-----------------|--------|-------------|-------|
| SmolLM2-360M-Instruct INT4 CPU | 403 MB | ✅ Active baseline | ❌ GPU OOM | Bundled in MSIX |
| SmolLM2-1.7B-Instruct INT4 CPU | 1.4 GB | ❌ Disk budget | — | Not attempted |
| Phi-3.5-mini INT4 CPU | ~2.7 GB | ❌ Disk budget | — | Not attempted |
| Phi-3.5-mini GPU INT4 AWQ | ~2.2 GB | — | ❌ GPU OOM + disk | Not viable |

## Candidates worth evaluating

- **Qwen2.5-0.5B INT4 ONNX** (~200 MB merged estimate): would fit both disk and GPU
  pool budgets — potential DirectML EP retry candidate.
- **Llama-3.2-1B INT4 ONNX CPU** (~700 MB merged estimate): borderline disk budget;
  worth checking merged size before committing to a build.
- **Gemma-2-2B INT4 ONNX CPU**: estimated above 1 GB merged — unlikely to fit.

Size estimates are from comparable INT4 quantizations on Hugging Face; verify with
`merge_onnx_external_data.py` output before proceeding.

## Why these limits

The disk and GPU limits derive from the UWP sandbox on Xbox Series S in Dev Mode.
This document records only observed behavior at the application boundary. See
`uwp-constraints.md §7` (GPU pool) and `uwp-constraints.md §9` (disk budget) for
what we measure. Do not treat any claim about the internal Xbox OS partition layout
as authoritative unless backed by a Microsoft source.
