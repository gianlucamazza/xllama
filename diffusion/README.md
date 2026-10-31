# diffusion/ — image generation on Xbox (SD-Turbo via ONNX Runtime DirectML)

The text-decode work established that the Series S GPU loses at LLM decode because
that workload is its _worst_ case (M=1, dispatch-bound, non-fused int4 on DML —
see `docs/uwp-constraints.md §12`). Image generation (diffusion) is the _opposite_:
compute-bound fp16 batch on 64×64 latents — the case where DML already wins
(prefill). This directory is the model-side toolchain for running a distilled
few-step diffusion model (SD-Turbo, 1 step) on the console GPU.

## Status

- ✅ **Model validated on the owned layer (2026-07-08).** SD-Turbo exported to
  ONNX and run through **ONNX Runtime (CPU EP)** generates a coherent 512×512
  image in ~13 s / 1 step. This is the exact artifact + runtime that runs on the
  Xbox via the DirectML EP — only the EP (and thus the speed) differs.
- ✅ **Plain ORT DirectML foundation proven** by the original image spike (PR
  #3), then replaced by the shipping diffusion pipeline. The removed spike
  source remains available in Git history.
- ✅ **C++ pipeline built + host-validated (2026-07-08).** `uwp/diffuse.cpp` runs
  three ORT DirectML sessions (text_encoder → 1× UNet → VAE decode) behind the
  `diffuse.flag` headless mode. Its correctness-critical logic — the CLIP
  byte-BPE tokenizer (`include/xllama/diffusion/clip_tokenizer.h`), the
  EulerDiscreteScheduler (`euler_scheduler.h`), fp16 conversion (`half.h`), and
  the PNG writer (`png_writer.h`) — is asserted against golden vectors captured
  from this Python reference (`gen_golden_vectors.py`) in `tests/test_diffusion.cpp`
  (638 assertions, all green). The ORT DirectML orchestration is CI-compile-
  validated; runtime validation is on console per
  `docs/console-validation-runbook.md`.
- ✅ **VALIDATED ON CONSOLE (2026-07-08, v0.4.2.0)**: SD-Turbo fp16 generates a
  coherent 512×512 image on the Xbox Series S GPU (DirectML) in **6.9 s** —
  text_encoder 1.0 s, UNet **3.3 s/step** (1 step), VAE 2.6 s
  (`bench/results/phase5-diffuse.csv`,
  `docs/screenshots/diffuse-sd-turbo-xbox.png`). Artifacts from
  `convert_fp16.py` (fp16 recipe below); procedure + hardware gotchas in
  `docs/console-validation-runbook.md`.

## Reproduce the export

The export is first-party (ADR 0003): `export_sd_turbo.py` loads the diffusers
checkpoint and calls `torch.onnx.export(..., dynamo=True)` once per component
(`text_encoder`, `unet`, `vae_decoder` = `AutoencoderKL.decode`), opset 18. It
keeps the I/O contract the optimum-onnx 0.1.0 export had, and checks it on every
graph before writing it:

| Component      | Inputs                                                                                   | Outputs                                  |
| -------------- | ---------------------------------------------------------------------------------------- | ---------------------------------------- |
| `text_encoder` | `input_ids` int64 `[batch, seq]`                                                         | `last_hidden_state`, `pooler_output` f32 |
| `unet`         | `sample` f32 `[batch, 4, h, w]`, `timestep` f32 scalar `[]`, `encoder_hidden_states` f32 | `out_sample` f32 `[batch, 4, h, w]`      |
| `vae_decoder`  | `latent_sample` f32 `[batch, 4, h, w]`                                                   | `sample` f32 `[batch, 3, 8h, 8w]`        |

`convert_fp16.py` turns every float input to fp16 (including `timestep`);
`uwp/diffuse.cpp` and `validate_pipeline.py` read the declared types and ranks.
Components whose weights reach 2 GB are written with one `model.onnx.data`
file next to `model.onnx` (decided from the initializer sizes, not by
serializing the graph). For SD-Turbo that is the fp32 UNet (~3.46 GB); the text
encoder and VAE decoder fit in one file. `convert_fp16.py` reads either layout
and writes self-contained fp16 graphs, which must each stay under 2 GB. For an
external-data input it runs the same ORT fp16 pass on the weight-free graph and
then converts the weights one tensor at a time, because the default ORT path
holds several copies of the UNet and ran out of memory on a 16 GB host; on the
tiny model both paths give byte-identical graphs.
Both scripts strip every `metadata_props` entry and doc string before saving:
the dynamo exporter tags each node with `pkg.torch.onnx.*` metadata (stack
traces with absolute source paths, FX node text), which nothing reads at
runtime and which would publish the exporting user's directories.
`XLLAMA_EXPORT_EXTERNAL_DATA_BYTES` lowers the threshold (the tests use it to
force the external-data layout on a tiny model). One real fp32 UNet export,
its fp16 conversion and their peak memory are recorded in
`receipts/sd-turbo-unet-export_receipt.json`.

`--verify` runs each graph on ORT CPU (`ORT_ENABLE_EXTENDED`) against eager
torch at the console shapes and at a second, smaller batch-2 shape.
`compare_onnx_exports.py REF CAND [--pipeline]` checks two export directories
against the fixed inputs and tolerances recorded in `export_parity.json`; run it
against a reference export before promoting a toolchain bump. It fails if a
component is missing on either side or nothing was compared. Only outputs with a
recorded tolerance are gated; the others are listed as unchecked in its report
(in fp16 that is `pooler_output`, which the console does not read, and the
`--pipeline` image, whose bound is recorded for fp32 only). Use **Python
3.12** (3.11+): onnxruntime 1.31 has no Python 3.10 wheels.

```bash
python3.12 -m venv venv
./venv/bin/pip install -r diffusion/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
./diffusion/export_sd_turbo.sh                 # -> sd-turbo-onnx/ + sd-turbo-onnx-fp16/
./venv/bin/python diffusion/generate_onnx.py   # validate: writes sd_turbo_onnx.png
```

`scripts/export-taesd-asset.sh`, `export_taesd.py` and `gen_golden_vectors.py`
use the same Python 3.12 env; the asset script looks for it at
`~/.cache/xllama-diffusion/venv312` (override with `XLLAMA_DIFFUSION_VENV`).

Artifacts regenerated with this toolchain are not promoted until the console
live gate (runbook §7) passes with peak-memory evidence.

## Memory / deployability

SD-Turbo (SD1.5-class) fits the console GPU budget in fp16 (~2.4 GB total:
UNet ~1.65 GB, text_encoder ~0.65 GB, vae_decoder ~0.1 GB), and — unlike a 1.7B
LLM in fp16 (a single 3.4 GB weight blob that exceeds the 2 GB ONNX protobuf
limit) — each SD component is individually **< 2 GB**, so all save self-contained
and avoid the AppContainer `weakly_canonical` crash (§8). **These sizes are
confirmed.**

**Getting a runnable fp16 model — the working path (verified 2026-07-08)**:
convert each component with **`onnxruntime.transformers`**'
`OnnxModel.convert_float_to_float16(keep_io_types=False)` (the ORT team's
corrected fork of the fp16 pass), then load with graph optimization capped at
**`ORT_ENABLE_EXTENDED`** — `ORT_ENABLE_ALL`'s layout transforms crash session
init (`graph_utils GetIndexFromName`) on these graphs; EXTENDED loads and runs
them cleanly (same cap set in `uwp/diffuse.cpp`). Validate end-to-end with
`validate_pipeline.py` before deploying.

What does **not** work (all falsified 2026-07-08, kept for the record):
`onnxconverter_common`'s `convert_float_to_float16` leaves mixed-type nodes in
**all three** SD components regardless of options — `/time_proj/Mul` (UNet),
`/text_model/.../self_attn/Add` (text encoder), `.../attentions.0/Cast_2`
(VAE) — with `keep_io_types` either way, with shape inference on or off, and
**even with `/time_proj/*` in `node_block_list`** (the UNet then fails on a
`Gemm` instead). Do not chase per-node block lists; switch converter.

⚠️ **Trap (verified 2026-07-08)**: the ORT-team pre-exports
([`onnxruntime/sd-turbo`](https://hf.co/onnxruntime/sd-turbo),
`tlwu/sd-turbo-onnxruntime`) look ideal on paper (fp16, each component
self-contained < 2 GB) but their UNet/VAE use `com.microsoft.NhwcConv`, which has
**no CPU kernel** (`NOT_IMPLEMENTED` at session init) — they are CUDA-optimized
graphs. They cannot be validated on our owned layer and may not map to DML
either; do not deploy them unvalidated. Other sources for a clean fp16 graph:

1. **Pre-exported fp16 DirectML models** (recommended, no GPU needed) — e.g.
   [`nmkd/stable-diffusion-1.5-onnx-fp16`](https://huggingface.co/nmkd/stable-diffusion-1.5-onnx-fp16),
   [`sharpbai/stable-diffusion-v1-5-onnx-directml-fp16`](https://huggingface.co/sharpbai/stable-diffusion-v1-5-onnx-directml-fp16),
   or the [Amblyopius/Stable-Diffusion-ONNX-FP16](https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16)
   toolkit. **Verified 2026-07-08**: the nmkd SD1.5 fp16 components all load in
   ORT (UNet input `sample: tensor(float16)`, 1.7 GB → < 2 GB) — as does the
   output of the current `convert_fp16.py`, which uses the ORT team's
   `OnnxModel.convert_float_to_float16` and load-tests every component at
   `ORT_ENABLE_EXTENDED`. These ship with **external data** (`unet/weights.pb`),
   so run `scripts/merge_onnx_external_data.py` on each component to make it
   self-contained before deploying (AppContainer `weakly_canonical`, §8).
2. **GPU export**: Microsoft **Olive**'s SD DirectML optimization.

Speed note: SD1.5 needs ~20–25 steps (slow); the console target is a **few-step**
model — SD-Turbo/SDXL-Turbo (1 step) or an LCM variant (2–4 steps) in fp16. The
fp32 SD-Turbo path here is validated for quality; convert it with
`diffusion/convert_fp16.py` for the console — that output is what ships as
`sd-turbo-fp16` and is console-validated (see Status). Sources (1) and (2) are
fallbacks only.

## Run on console

The app downloads `sd-turbo-fp16` from the catalogue on the first image
generation. In the live UI open `[*] Image`, keep one step for SD-Turbo, enter a
prompt and select **Generate**. The in-process pipeline writes
`diffuse-out.png`, `diffuse-result.csv` and progress state under `LocalState`.

For automated WDP benchmarks, the retained `diffuse.flag` path consumes the
same model and implementation without launching the UI. Provisioning and
current acceptance gates are documented in
`docs/console-validation-runbook.md`. SD-Turbo uses Stability's
research/community license; review licensing before redistributing derived
weights or public assets.
