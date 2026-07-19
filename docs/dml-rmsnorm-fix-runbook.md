# DML RMSNorm fix — #91 plan B outcome (2026-07-19)

**Finding.** The #91 broken text logits on the Xbox Series S DML driver were
never an attention problem: the broken kernel is
**`(Skip)SimplifiedLayerNormalization`** (the fused RMSNorm contrib op the
GenAI builder emits in the default domain). Decomposing only those 65 nodes
into primitive ops fixes text logits on DML — with the fused GQA attention
kept, the stock release `genai_config.json` (graph capture on,
`past_present_share_buffer=true`) and the **shipping pinned DLLs**. The fix
is data-only: republish the model asset, no runtime/DLL change.

## Evidence — escalation matrix (on-console logit parity vs llama.cpp golden)

Runtime for rows 1–4: probe #94 build (fork genai DLL, `enable_graph_capture:"0"`,
`past_present_share_buffer:false`, MHA export). Rows 5–6: fused GQA release
asset, stock config. Golden: `tests/golden/logits-smol-short.bin`
(prompt "The capital of France is", top-1 " Paris").

| #   | attention     | rotary     | RMSNorm    | runtime                             | NMSE     | top1     | verdict  |
| --- | ------------- | ---------- | ---------- | ----------------------------------- | -------- | -------- | -------- |
| 1   | primitives    | fused      | fused      | probe                               | 9.79e-01 | " the"   | FAIL     |
| 2   | primitives    | primitives | fused      | probe                               | 9.80e-01 | " the"   | FAIL     |
| 3   | primitives    | primitives | primitives | probe                               | 1.66e-02 | " Paris" | **PASS** |
| 4   | **fused MHA** | fused      | primitives | probe                               | 1.72e-02 | " Paris" | **PASS** |
| 5   | **fused GQA** | fused      | primitives | probe build, stock config           | 1.72e-02 | " Paris" | **PASS** |
| 6   | **fused GQA** | fused      | primitives | **shipping 1.3.0.563, pinned DLLs** | 1.72e-02 | " Paris" | **PASS** |

Rows 4–6 isolate the culprit exactly: everything previously blamed
(GQA #91, MHA #94, driver metacommands, graph capture) was innocent — the
garbage appeared "after attention" only because RMSNorm wraps every block.
Host CPU-EP equivalence for every surgered variant vs its source model:
NMSE ≤ 1.2e-05, top-10 overlap 1.00 (via `scripts/ort-prefill-logits.py`).

## The fix

```bash
# venv with onnx + numpy (e.g. ~/.cache/xllama-venvs/oga-builder)
python scripts/decompose_attention.py -i model.onnx -o model-rmsfix.onnx \
    --skip-attention --also-skipln
```

`--also-skipln` rewrites each `(Skip)SimplifiedLayerNormalization` into
Add (residual, preserving the `output[3]` sum) + Cast fp32 → Mul/ReduceMean/
Add(eps)/Sqrt/Div → Cast fp16 → Mul(gamma), mirroring the contrib op's
`stash_type=1` fp32 accumulation. `--skip-attention` leaves GQA untouched
(also what makes the script applicable to the shipping GQA export). Size
cost: +0.8 MB on the 725 MB asset. Full escalation flags (`--also-rotary`,
attention decomposition, `--fp32-qk`) remain available for future probes.

## Validation chain (all PASS, 2026-07-19)

1. Host equivalence: surgered vs source model, same GenAI CPU runtime.
2. Host golden: `ort-prefill-logits.py` + `compare-logits.py` vs llama.cpp
   golden (NMSE 1.6e-02, " Paris").
3. On-device `validate-logit-parity.sh` — rows 3–6 above; row 6 is the
   shipping configuration end-to-end.

## Follow-ups

- [x] Bench (2026-07-19, standard-512, t8, median of 2):
      **prefill 234 tok/s, decode 43.9 tok/s, peak 1215 MB, GPU mem 793 MB** —
      within the historical dml-fp16 range (169–353 prefill), so the RMSNorm
      decomposition costs little. CPU int4 decode (71 tok/s) stays faster →
      the existing hybrid routing (GPU prefill above `token_threshold`, CPU
      decode) is the right shape. Confirmed on the published asset under its
      catalogue name (2026-07-19, PR #112 build): **236.7 prefill / 44.4
      decode / 1268 MB** — the canonical `phase2-dml` row the benchmark
      summary now selects. Routing re-enabled by PR #112
      (`dml_text_model_ok` behind `token_threshold`).
- [x] Published as **`smollm2-360m-dml-fp16-v2`** on `models-v1` (#109 closed,
      2026-07-19 — new name, never clobber: broken pre-fix copies may survive
      in LocalState and the routing allowlist is keyed on the name; remote
      sizes verified byte-exact vs manifest `approx_bytes`; post-publish
      on-device parity PASS via `provision-models.sh` download).
- [x] Routing re-enable PR (#110 → **PR #112, merged 2026-07-19**):
      `dml_text_model_ok` allowlist in `routing_policy.h` behind
      `token_threshold`, manifest `-v2` entry, `validate-console.sh` §2
      expectations flipped back to auto→gpu. On-console merge gate ALL PASS
      (first legitimate auto→gpu: 959 tok → GPU, decode 38.8 tok/s).
- [ ] Upstream report (#111): draft pending explicit OK; see the repro
      campaign below for what the report can and cannot claim.
- [ ] Re-test fused RMSNorm after future Xbox GameOS/driver updates.

## Repro campaign (#111, 2026-07-19 — PR #113)

Source analysis (ORT v1.24.4): both `SimplifiedLayerNormalization` and
`SkipSimplifiedLayerNormalization` lower to
`DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2` with `UseMean=false`
(`DmlOperatorLayerNormalization.cpp`, `DmlOperatorSkipLayerNormalization.cpp`).

Tooling: `uwp/op-repro.cpp` (`oprepro.flag` headless plain-ORT runner, CPU
session vs DML session on the same payload, optimization level from
`repro-opt.txt`), `scripts/make-op-repro.py` (single-op models, `--scale`),
`scripts/make-chain-repro.py` (decoder-shaped 25-node stack),
`scripts/validate-op-repro.sh` (per-variant driver).

**Every plain-ORT probe MATCHES on the Series S driver** (CPU-vs-DML NMSE
≤ 2e-06 in all cases):

| probe                                                                | opt level                | verdict |
| -------------------------------------------------------------------- | ------------------------ | ------- |
| simplified / skip / layernorm single op, scale 2                     | disable + all            | MATCH   |
| same at scale 30 / 100 (Σx² far beyond fp16 max)                     | disable                  | MATCH   |
| 25-node chain (SimplifiedLN→MatMul→skip residual ×8, real eps/shape) | disable + extended + all | MATCH   |

(The high-scale `skip` "mismatches" under an absolute tolerance are 1-ulp
fp16 divergence on the fp16 residual-sum output — not corruption.)

**Interpretation.** The op — including the fused-graph DML compilation path —
is correct standalone on this device. The corruption manifests only in the
**ORT-GenAI DML execution context** (DML1 shared device/queue, GenAI's
binding/allocation pattern, the decoder graph with past-KV I/O): fused
RMSNorm there produces NMSE ~0.98, and decomposing only those nodes inside
that same context fixes it. Any upstream fix or fork-level workaround must
therefore target the GenAI-DML interaction, not the standalone kernel; the
graph-level decomposition (this runbook's fix) remains the correct
product-side workaround.

Runner gotcha worth keeping: `Ort::TypeInfo` must outlive the
`GetTensorTypeAndShapeInfo()` view (a temporary dangles → garbage shapes →
`bad_alloc`), and per-input buffers must be reserved before wrapping their
`data()` in `Ort::Value` (vector reallocation dangles earlier tensors).

History: `docs/dml-metacommands-runbook.md` (metacommands experiment, FAIL,
2026-07-19) and #94 (MHA probe) document the prior exonerated suspects.
