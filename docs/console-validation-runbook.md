# Console Validation Runbook — pending on-console checks (v0.4.0.0)

Three PRs landed on `main` **CI-green but console-pending** (per the merge-on-CI-green
policy, 2026-07-08): #2 (ORT GenAI 0.14.1 + KV-cache reuse + CPU/GPU routing), #3 (plain
ORT DirectML image spike), #4 (diffusion toolchain, model-side only). Every code path is
additive and flag-gated, so the default behaviour is unchanged until a flag/setting is
used — but the perf wins and the image-spike hypothesis are **assumed, not measured** until
run on the Xbox.

This runbook batches all of them into **one Xbox session**. Each step writes a CSV/artifact
fetched via Device Portal, with an explicit sanity gate. Mechanics (deploy, WDP quirks,
model upload, GPU telemetry, troubleshooting) are in [`phase1-runbook.md`](./phase1-runbook.md);
this file is the ordered checklist + the "what am I looking for" per step.

## Convention (applies to every bench step)

- **2 runs minimum** per config; the harness drops run 1 (cold). Report the median row.
- **Prefill and decode are separate columns — never read the aggregate** (the aggregate
  hides the hardware crossover; GPU wins prefill at scale, CPU wins decode).
- **Sanity per row**: `prefill_toks ≫ decode_toks` (tok/s), and `gpu_mem_mb ≈ model weight
size` on any DML row (≈ 0 ⇒ silent CPU fallback — see `phase1-runbook.md §9`).
- Record each result CSV under `bench/results/`, commit with a one-line "Measured" note in
  `CHANGELOG.md` (flip the matching "On-console validation pending" line to the number).

## 0. One-time — deploy main (v0.4.0.0)

`AppxManifest.xml` is at `0.4.0.0` (> any previously deployed 0.3.x), so this is a
**forward upgrade** and LocalState is preserved (existing uploaded models/settings survive).

```bash
source ~/.config/xllama/xbox-env
# Get the xllama-appx artifact from the latest green build-uwp run on main, then:
./scripts/deploy.sh path/to/xllama_0.4.0.0_*.msix
PFN=$(./scripts/deploy.sh pfn)
./scripts/deploy.sh get-log        # confirm clean startup (App::App → Window activated)
```

> Re-verify the **Game** designation in Dev Home after any reinstall (it can reset; all
> measured figures assume Game-mode — `docs/uwp-constraints.md §5`).

## 1. PR #2 — KV-cache reuse (Stage 2b bench) → `bench-kv-result.csv`

**Validates**: turn-2 TTFT with KV reuse (append only the delta) vs the cold full
re-prefill. Trigger is `bench_turns.txt` present (turn-2 prompt; `prompt.txt` = turn 1).

```bash
source ~/.config/xllama/xbox-env
PFN=$(./scripts/deploy.sh pfn)
./scripts/deploy.sh upload-file bench/prompts/standard-512.txt "$PFN" ""   # → prompt.txt (turn 1)
printf 'And summarise that in one sentence.' > /tmp/bench_turns.txt
./scripts/deploy.sh upload-file /tmp/bench_turns.txt "$PFN" ""
printf 'bench' > /tmp/bench.flag
./scripts/deploy.sh upload-file /tmp/bench.flag "$PFN" ""
# launch from Dev Home; wait; fetch bench-kv-result.csv (see phase1-runbook §6)
```

**Looking for**: `speedup` column > 1 (reuse prefill ≪ cold prefill). Expected large —
turn-2 cold re-prefills the full 2-turn context, reuse only the ~10-token delta. Also
confirm multi-turn **coherence** (read the log's decoded turn-2 output) before trusting
`kv_reuse` as the interactive default.

## 2. PR #2 — CPU/GPU routing (Stage 3, interactive)

**Validates**: the `routing=2` (auto) path picks GPU (DML fp16) for long prompts and CPU
for decode, sticky per conversation.

Prereq: the DML fp16 model must be on device. Upload the 360M DML fp16 dir:

```bash
python3 scripts/merge_onnx_external_data.py ~/.cache/.../smollm2-360m-dml-fp16   # if external data
./scripts/deploy.sh upload-dir <dir>/ "$PFN" "models\\smollm2-360m-dml-fp16"
```

In-app: ⚙ Settings → routing = **auto** → send a long (>~500 tok) prompt, then a short
follow-up. **Looking for**: log shows the first turn routed to `smollm2-360m-dml-fp16`
(GPU) and TTFT improved vs CPU-only on the long prompt; the conversation stays on its
routed EP (sticky). New chat re-decides.

## 3. PR #2 — 0.14.1 decode overhead → refresh the v0.3.6 matrix

**Validates**: the 0.13.2 → 0.14.1 bump reduced CPU-side per-token overhead.

```bash
./scripts/bench-xbox-ort.sh smollm2-360m-cpu-int4 --runs 3 --out bench/results/phase35-014-cpu.csv
./scripts/bench-xbox-ort.sh smollm2-360m-dml-fp16  --runs 3 --out bench/results/phase35-014-dml.csv --gpu-sample
```

**Looking for**: decode tok/s vs the v0.3.6 baselines (CPU int4 68.0 short / 50.9 long;
DML fp16 46.8 / 36.5). Any uplift is the 0.14.x win; flat is also a valid (recorded) result.

## 4. PR #3 — image spike → `imgspike-result.csv`

**Validates the flagship hypothesis**: on a compute-bound fp16 batch (one diffusion UNet
step proxy), the RDNA 2 GPU **beats** the CPU — the inverse of text decode.

```bash
# Generate the deterministic proxy model (host) — writes bench/models/imgspike.onnx
python3 scripts/gen_imgspike_model.py
./scripts/deploy.sh upload-file bench/models/imgspike.onnx "$PFN" ""
printf 'image' > /tmp/image.flag
./scripts/deploy.sh upload-file /tmp/image.flag "$PFN" ""
# launch; wait; fetch imgspike-result.csv (DML vs CPU ms, GFLOP/s, speedup)
```

**Looking for**: `speedup` (DML/CPU) **≫ 1** (compute-bound → GPU wins). If confirmed, the
C++ diffusion pipeline (Fase 3) is greenlit as the flagship GPU workload. If DML ≈ CPU or
worse, re-examine the hypothesis before building the pipeline.

## 5. int4 DML — confirm/falsify the §12 desk-check

Two distinct levers; keep them separate. The desk-check (§12) predicts **neither** moves
the 8.8 tok/s decode floor, because the limit is the **non-fused** `MatMulNBits` kernel
(`DML_DEQUANTIZE`→fp16 + full GEMM), independent of block size or EP session options.

**5a — EP session-options swap (no rebuild).** `test-dml-config.sh` swaps in
`bench/configs/genai_config-dml-test.json` (DML EP with `enable_cpu_mem_arena=0`,
`enable_mem_pattern=0`) on an existing DML int4 model dir. This tests **runtime EP options
only** — not quantization params.

```bash
./scripts/test-dml-config.sh --model <dml-int4-model-dir>
# launch, bench, read decode tok/s; then restore:
./scripts/test-dml-config.sh --model <dml-int4-model-dir> --restore
```

**5b — block_size / accuracy_level (requires a model REBUILD, host-side).** `block_size`
and `accuracy_level` are baked into the `MatMulNBits` nodes at build time — they are **not**
a runtime `genai_config.json` setting, so a config swap cannot change them. To test
block128/acc4, rebuild with the ORT GenAI model builder (`-p int4 -e dml` +
`--extra_options`), then upload the new dir and bench it like §6.

**Looking for**: decode still ≈ 8.8 tok/s in both ⇒ desk-check **confirmed** (kernel-design
limit, CPU int4 stays the decode default); a jump ⇒ the lever exists and §12 needs
revising. Either outcome is a recorded verdict.

## 6. 1.7B scale bench — does the GPU advantage grow with model size?

**Validates**: GPU int4 1.7B decode vs CPU int4 1.7B, and prefill crossover at scale. The
variants are already built in `~/.cache/xllama-1b-build/` (`smollm2-1.7b-cpu-int4`,
`smollm2-1.7b-dml-fp16`).

```bash
for m in smollm2-1.7b-cpu-int4 smollm2-1.7b-dml-fp16; do
  python3 scripts/merge_onnx_external_data.py ~/.cache/xllama-1b-build/$m   # if external data
  ./scripts/deploy.sh upload-dir ~/.cache/xllama-1b-build/$m/ "$PFN" "models\\$m"
done
./scripts/bench-xbox-ort.sh smollm2-1.7b-cpu-int4 --runs 3 --out bench/results/phase35-1b-cpu.csv
./scripts/bench-xbox-ort.sh smollm2-1.7b-dml-fp16  --runs 3 --out bench/results/phase35-1b-dml.csv --gpu-sample
```

**Looking for**: (a) does DML fp16 1.7B **load** inside the 3801 MB budget (fp16 1.7B ≈
3.4 GB weights — expect borderline/OOM; a clean OOM is an informative result); (b) prefill
GPU-vs-CPU advantage at ~1k tokens (expected to grow with scale); (c) decode GPU vs CPU
(expected CPU still wins per §12). Sanity: `gpu_mem_mb ≈ 3.4 GB` on the DML row if it loads.

## 7. Diffusion pipeline → `diffuse-out.png`

**Validates the flagship GPU workload end to end**: the C++ pipeline (`uwp/diffuse.cpp`
— tokenize → text_encoder → 1× UNet denoise → VAE decode → PNG) runs three ORT DirectML
sessions on the console. The correctness-critical logic (CLIP tokenizer, Euler scheduler,
fp16, PNG) is already host-validated against the diffusers reference
(`tests/test_diffusion.cpp`, 638 assertions) — this step validates the ORT DirectML
orchestration + the model on real hardware.

**Model contract** (see `uwp/diffuse.cpp` header): an **fp16** SD-Turbo-class ONNX model,
each component self-contained (< 2 GB, external data merged). fp16 SD-Turbo needs a GPU
export (`optimum-cli --fp16 --device cuda`) or Olive — see `diffusion/README.md`. The fp32
`sd-turbo-onnx` validates quality but its UNet (~3.4 GB) exceeds the budget + 2 GB protobuf
limit, so it is not deployable; a fp16 export is required for the console.

```bash
source ~/.config/xllama/xbox-env
PFN=$(./scripts/deploy.sh pfn)
DIR=sd-turbo-fp16   # LocalState\models\<DIR>\{text_encoder,unet,vae_decoder}\model.onnx

# 1. Model components (merge external data first so each is self-contained).
for comp in text_encoder unet vae_decoder; do
  python3 scripts/merge_onnx_external_data.py <fp16_model>/$comp
  ./scripts/deploy.sh upload-dir <fp16_model>/$comp/ "$PFN" "models\\$DIR\\$comp"
done
# 2. CLIP tokenizer assets (vendored in-repo — the exact files the host test uses).
./scripts/deploy.sh upload-dir diffusion/clip_tokenizer/ "$PFN" "clip"
# 3. Prompt + model selector, then the flag.
printf 'a red sports car on a mountain road at sunset' > /tmp/prompt.txt
./scripts/deploy.sh upload-file /tmp/prompt.txt "$PFN" ""
printf '%s' "$DIR" > /tmp/diffuse-model.txt
./scripts/deploy.sh upload-file /tmp/diffuse-model.txt "$PFN" ""
printf 'diffuse' > /tmp/diffuse.flag
./scripts/deploy.sh upload-file /tmp/diffuse.flag "$PFN" ""
# 4. Launch from Dev Home; wait; fetch diffuse-out.png (512x512) once .done appears.
```

**Looking for**: `diffuse-out.png` is a coherent 512×512 image matching the prompt (compare
to `sd_turbo_onnx.png` from the validated CPU pipeline). The log prints per-session timing;
UNet-step ms is the number that decides whether diffusion is the flagship GPU workload (the
image-spike hypothesis, step 4, at full model scale). If the image is noise, check the log
for a tokenizer/model-shape mismatch (e.g. a non-fp16 model, or `hidden_dim` ≠ 1024).

## Closeout

For each step: append the median CSV row under `bench/results/`, flip the matching
"On-console validation pending" line in `CHANGELOG.md` to the measured number, and update
the relevant `ROADMAP.md` Phase 3.5 / Phase 5 milestone with the verdict. Commit as a
"Measured" changelog entry.
