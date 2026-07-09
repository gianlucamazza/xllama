# Console Validation Runbook

On-console checks for merged, CI-green work (merge-on-CI-green policy). Most of the
Phase 3.5 batch was **measured 2026-07-08** (§1, §3, §4, §6-CPU, §7 below carry their
results); the sections still marked PENDING are §2 (routing A/B, interactive), §5b
(int4 rebuild variants), and §7b (in-process diffusion experiment, 2026-07-09).

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

## 0. One-time — deploy main

Deploy the current `xllama-appx` artifact (version per `uwp/AppxManifest.xml`).
A **version-bump upgrade preserves LocalState** (uploaded models/settings survive);
a same-version reinstall with different contents is **blocked by WDP** — bump the
version for every console deploy.

```bash
source ~/.config/xllama/xbox-env
# Get the xllama-appx artifact from the latest green build-uwp run on main, then:
./scripts/deploy.sh path/to/xllama_*.msix
PFN=$(./scripts/deploy.sh pfn)
./scripts/deploy.sh get-log        # confirm clean startup (App::App → Window activated)
```

> Re-verify the **Game** designation in Dev Home after any reinstall (it can reset; all
> measured figures assume Game-mode — `docs/uwp-constraints.md §5`).

## 1. PR #2 — KV-cache reuse (Stage 2b bench) — ✅ MEASURED 2026-07-08 (4.87×, `phase35-kv.csv`)

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

## 2. PR #2 — CPU/GPU routing (Stage 3, interactive, ~10 min at the console) — ⏳ PENDING

**Validates**: the `routing=2` (auto) path picks GPU (DML fp16) for long prompts and CPU
for decode, sticky per conversation.

Prereqs (both already satisfied on the current console): `smollm2-360m-dml-fp16` in
`LocalState\models\`, and the routing setting. The setting can be **preloaded via WDP**
so the person at the console only pastes a prompt:

```bash
source ~/.config/xllama/xbox-env
PFN=$(./scripts/deploy.sh pfn)
cat > /tmp/settings.json <<'JSON'
{
  "system_prompt": "You are a helpful assistant.",
  "model": "smollm2-360m-cpu-int4",
  "kv_reuse": true,
  "routing": 2,
  "gpu_model": "smollm2-360m-dml-fp16",
  "sampling": { "temperature": 0.70, "top_p": 0.90, "top_k": 40,
                "repetition_penalty": 1.10, "n_predict": 256 }
}
JSON
./scripts/deploy.sh upload-file /tmp/settings.json "$PFN" ""
```

Then, at the console: launch xllama → paste a **long** prompt (>~500 tokens — e.g. the
content of `bench/prompts/standard-512.txt`) → send; then a short follow-up; then **+ New**
and a short prompt. Fetch the log afterwards (`deploy.sh get-log`).

**Looking for** (in the log): the long-prompt conversation's first turn routed to
`smollm2-360m-dml-fp16` (GPU) with a better TTFT than the CPU baseline at that length;
the follow-up stays on the same EP (sticky); the new chat with a short prompt routes back
to CPU. Flip `routing` back to `0` (Settings) afterwards if desired.

## 3. PR #2 — 0.14.1 decode overhead — ✅ MEASURED 2026-07-08 (flat vs v0.3.6, `phase35-014-*.csv`)

**Validates**: the 0.13.2 → 0.14.1 bump reduced CPU-side per-token overhead.

```bash
./scripts/bench-xbox-ort.sh smollm2-360m-cpu-int4 --runs 3 --out bench/results/phase35-014-cpu.csv
./scripts/bench-xbox-ort.sh smollm2-360m-dml-fp16  --runs 3 --out bench/results/phase35-014-dml.csv --gpu-sample
```

**Looking for**: decode tok/s vs the v0.3.6 baselines (CPU int4 68.0 short / 50.9 long;
DML fp16 46.8 / 36.5). Any uplift is the 0.14.x win; flat is also a valid (recorded) result.

## 4. PR #3 — image spike — ✅ MEASURED 2026-07-08 (DML 11.1× CPU, `phase35-imgspike.csv`)

**Validated the flagship hypothesis**: on a compute-bound fp16 batch (one diffusion UNet
step proxy), the RDNA 2 GPU **beat** the CPU 11.1× — the inverse of text decode. This
greenlit the diffusion pipeline (§7), which shipped in v1.0.0.

_The spike tooling (`uwp/image-spike.cpp`, `image.flag` mode, `gen_imgspike_model.py`)
was removed after validation — purpose served; retrieve it from git history at tag
`v1.0.0` if ever needed again. The result CSV stays at
`bench/results/phase35-imgspike.csv`._

## 5. int4 DML — confirm/falsify the §12 desk-check (5a ✅ done; 5b ⏳ PENDING — variants rebuilt 2026-07-09)

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

## 6. 1.7B scale bench — ✅ CPU-int4 MEASURED 2026-07-08 (20.6 tok/s, `phase35-1b-cpu.csv`); fp16-DML blocked (>2 GB protobuf)

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

## 7. Diffusion pipeline → `diffuse-out.png` — ✅ VALIDATED 2026-07-08 (v0.4.2.0)

**Result**: SD-Turbo fp16 generates a coherent 512×512 image on the console GPU in
**6.9 s** — text_encoder 1.0 s, UNet **3.3 s/step** (1 step), VAE 2.6 s; ~7.5 s session
load excluded. CSV `bench/results/phase5-diffuse.csv`; image
`docs/screenshots/diffuse-sd-turbo-xbox.png` (matches the local CPU-validation image for
the same prompt/seed). The flagship-GPU-workload hypothesis holds at full model scale.

**Model artifacts (the validated recipe)**: convert the fp32 `sd-turbo-onnx` export with
`diffusion/convert_fp16.py` (onnxruntime.transformers converter + EXTENDED load-test; see
`diffusion/README.md` for what does NOT work) → each component self-contained < 2 GB
(unet 1.65 GB, text_encoder 0.65 GB, vae 0.09 GB), no external-data merge needed. Then
validate locally end-to-end with `diffusion/validate_pipeline.py` before uploading.

```bash
source ~/.config/xllama/xbox-env
PFN=$(./scripts/deploy.sh pfn)
DIR=sd-turbo-fp16   # LocalState\models\<DIR>\{text_encoder,unet,vae_decoder}\model.onnx

# 1. Model components (already self-contained from convert_fp16.py).
for comp in text_encoder unet vae_decoder; do
  ./scripts/deploy.sh upload-dir <fp16_model>/$comp/ "$PFN" "models\\$DIR\\$comp"
done
# 2. CLIP tokenizer assets (vendored in-repo — the exact files the host test uses).
./scripts/deploy.sh upload-dir diffusion/clip_tokenizer/ "$PFN" "clip"
# 3. Prompt / steps / seed / model selector, then the flag.
printf 'a red sports car on a mountain road at sunset' > /tmp/prompt.txt
printf '%s' "$DIR" > /tmp/diffuse-model.txt
printf '1'  > /tmp/diffuse-steps.txt
printf '42' > /tmp/diffuse-seed.txt
printf 'diffuse' > /tmp/diffuse.flag
for f in prompt.txt diffuse-model.txt diffuse-steps.txt diffuse-seed.txt diffuse.flag; do
  ./scripts/deploy.sh upload-file /tmp/$f "$PFN" ""
done
# 4. deploy.sh start-app; wait for diffuse-out.png.done; fetch PNG + diffuse-result.csv.
```

**Gotchas found on hardware (both fixed in 0.4.2.0, kept for the record)**:

- Keeping all three sessions resident OOM'd the VAE decode (8007000E at 512×512
  InstanceNorm) — ~2.4 GB weights + VAE activations exceed the 3801 MB budget.
  `run_diffuse` now uses per-stage session lifetime.
- If the image is noise, check the log for a tokenizer/model-shape mismatch (non-fp16
  model, `hidden_dim` ≠ 1024, wrong scheduler class).

### 7b. In-process experiment (`diffuse-inproc.flag`) — PENDING

Falsification run for `docs/uwp-constraints.md` §7 "Open experiment": does plain ORT DML
coexist with the XAML compositor device? Same inputs as §7, but write
`diffuse-inproc.flag` **instead of** `diffuse.flag`, then `deploy.sh start-app` — the app
starts the normal XAML UI and runs the pipeline on a background thread in the same
process.

- Poll `diffuse-progress.txt` (`start` → `text_encoder` → `unet s/N` → `vae` → `done`);
  `diffuse-cancel.flag` aborts between UNet steps (progress `cancelled`).
- **PASS**: log shows `diffuse-inproc.flag detected` + per-stage ms + `wrote
diffuse-out.png` with the XAML window still up; PNG matches the §7 headless output for
  the same prompt/seed/steps.
- **FAIL**: `diffuse: ORT error: ... 887A0036` (device conflict also affects plain ORT —
  the headless flow stays) or `8007000E` during VAE (GPU budget with compositor:
  headroom, not device, is the blocker).
- Either outcome: record it in `docs/uwp-constraints.md` §7 and close the experiment.

## Closeout

For each step: append the median CSV row under `bench/results/`, flip the matching
"On-console validation pending" line in `CHANGELOG.md` to the measured number, and update
the relevant `ROADMAP.md` Phase 3.5 / Phase 5 milestone with the verdict. Commit as a
"Measured" changelog entry.
