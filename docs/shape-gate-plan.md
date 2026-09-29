# Shape gate — does a Series S–shaped SLM beat LFM2.5?

> **Status: proposed (2026-09-30).** Not a ROADMAP phase yet; zero console
> sessions spent. This file owns the question, the predeclared gates, and the
> verdict. Measured numbers land in `bench/results/shape-gate.csv` and, if they
> are ever compared, flow through
> [`generate-benchmark-summary.py`](../scripts/generate-benchmark-summary.py)
> into [benchmarks.md](benchmarks.md). Nothing here is a product claim.

## Question

Can the _hyperparameters_ of an `lfm2` model (depth, width, vocabulary) be
chosen for Series S so that it decodes materially faster than the shipped
LFM2.5 shapes at the same parameter budget? If not, a custom SLM is not worth
a training budget, and the answer is a recorded kill.

Out of scope on purpose: new operators or a new architecture (would need
`llama.cpp` patches and a second decode path), MoE (H2 closed FAIL,
[phase15-re-opt.md](phase15-re-opt.md) "Do not reopen"), speculative or
multi-token heads (W2 console A/B, same file), GPU decode
([uwp-constraints.md](uwp-constraints.md) §12, Phase 15 WS-E parked).

## Why random weights are enough

CPU decode and prefill time in llama.cpp depend on tensor shapes, quant types
and the graph — not on weight values. There is no data-dependent fast path on
the CPU backend; `GGML_USE_CPU_REPACK` repacks by type and shape. So a
random-weight GGUF of a candidate shape, quantized with the same
`llama-quantize` mix, costs the console what a trained model of that shape
would. Gate **C0** below tests this assumption instead of trusting it.

Two things random weights do change, and the gates handle both:

- **Stop behaviour.** A random model may emit EOS or a stop sequence early.
  Rows with `n_gen_tok != n_predict` are invalid (C1).
- **Tokenizer efficiency.** A truncated vocabulary tokenizes the same text
  into more tokens. Tok/s from `v*` shapes is converted to text throughput
  before comparison (G2).

## Tooling

| Piece        | Path                                                                          |
| ------------ | ----------------------------------------------------------------------------- |
| Builder      | [`scripts/make-shape-gguf.py`](../scripts/make-shape-gguf.py)                 |
| Sweep spec   | [`bench/configs/shape-sweep.json`](../bench/configs/shape-sweep.json)         |
| Unit tests   | [`tests/test_make_shape_gguf.py`](../tests/test_make_shape_gguf.py)           |
| Console run  | [`scripts/bench-xbox-ort.sh`](../scripts/bench-xbox-ort.sh) (GGUF model dirs) |
| Model upload | `scripts/deploy.sh upload-dir`                                                |

The builder writes an f16 GGUF with random weights (norms = 1), copies the
tokenizer and chat template from a real LFM2 GGUF, then runs `llama-quantize`.
Output files are tagged `random-weights` / `not-a-model` in `general.tags` and
carry `-rand-` in the filename. The desk estimate reproduces the released
LFM2.5 parameter counts exactly (unit-tested), and a Q4_K_M `ref-350m`
reports the same size and parameter count as the released Q4_K_M in
llama-bench (216.41 MiB, 354.48 M).

**Quant = Q4_0, the shipped layout.** Since #270 the catalogue pins LiquidAI's
QAD Q4_0 files for `lfm25-230m/350m/1.2b-instruct`. Their layout is every
matmul Q4_0 and `token_embd` Q6_K, which is exactly what `llama-quantize Q4_0`
produces on a tied-embedding `lfm2`. Q4_0 takes a different CPU repack path
from Q4_K, so the grid must use the reference's quant or C0 is invalid by
construction.

## Grid

Desk estimate from `--dry-run` (`read_MB/tok` = block matmuls at Q4_0 plus
the tied head at Q6_K; an estimate, not a measurement):

| id       |   L |    D |     F |     V | attn | params M | read MB/tok | KV KiB/tok | role                                  |
| -------- | --: | ---: | ----: | ----: | ---: | -------: | ----------: | ---------: | ------------------------------------- |
| ref-230m |  14 | 1024 |  2560 | 65536 |    6 |    229.7 |       146.5 |         12 | calibration vs `lfm25-230m`           |
| ref-350m |  16 | 1024 |  4608 | 65536 |    6 |    354.5 |       216.7 |         12 | calibration vs `lfm25-350m`           |
| ref-1.2b |  16 | 2048 |  8192 | 65536 |    6 |   1170.3 |       692.9 |         12 | calibration vs `lfm25-1.2b-instruct`  |
| d8       |   8 | 3072 | 10496 | 65536 |    3 |   1230.1 |       743.8 |          6 | iso-body, half depth                  |
| d12      |  12 | 2560 |  8192 | 65536 |    4 |   1195.5 |       715.7 |          8 | iso-body, shallower                   |
| d24      |  24 | 1536 |  7168 | 65536 |    9 |   1091.7 |       640.0 |         18 | iso-body, deeper                      |
| d32      |  32 | 1536 |  5120 | 65536 |   12 |   1120.1 |       656.0 |         24 | iso-body, deep-thin (MobileLLM-style) |
| v32k     |  16 | 2048 |  8192 | 32768 |    6 |   1103.2 |       637.9 |         12 | `ref-1.2b` body, half vocabulary      |
| v16k     |  16 | 2048 |  8192 | 16384 |    6 |   1069.7 |       610.3 |         12 | `ref-1.2b` body, quarter vocabulary   |

"Iso-body" means the non-embedding parameters are within 7% of `ref-1.2b`.
Embedding size moves with `D`, so every comparison below also reports
effective read bandwidth (file MB × decode tok/s). That separates "fewer bytes"
from "less fixed per-layer cost".

## Host evidence so far (2026-09-30)

- **Layout match.** The Q4_0 `ref-350m` clone has the same tensor names,
  types and shapes as `LFM2.5-350M-QAD-Q4_0.gguf` (sha256 matches the
  manifest). The three clone files are within 4 KB of the manifest
  `approx_bytes` of their references (metadata only).
- **Tokenizer copy.** On all five `bench/prompts/*.txt` the clone tokenizes
  identically to the real model.
- **Vocabulary cost** ([`shape-gate-tokenizer.csv`](../bench/results/shape-gate-tokenizer.csv)):
  the same 7912 characters take 1549 / 1709 / 1924 tokens at 65536 / 32768 /
  16384 vocab (×1.000 / ×1.103 / ×1.242).
- **G2 desk bound.** To clear G2, decode must rise ≥1.21× (`v32k`) and ≥1.37×
  (`v16k`) over `ref-1.2b`. If decode scaled with bytes read alone, the file
  sizes cap the gain at 1.09× and 1.14×. So G2 is predicted to FAIL. It
  survives only if per-token costs that scale with vocabulary (the output
  logits and sampling over V) are much larger than the bytes suggest. The
  console run keeps `v*` in to test exactly that.

## Procedure

1. **Build** (host, ~1 min per shape plus quantize; use a Release
   `llama-quantize`, a Debug one takes minutes per GB):

   ```bash
   python3 scripts/make-shape-gguf.py --spec bench/configs/shape-sweep.json \
     --template ~/.cache/xllama-gguf/LFM2.5-350M-Q4_K_M.gguf --out-dir build/shape-gguf
   ```

2. **Host smoke** (load + 48 tokens; host tok/s is discarded):

   ```bash
   llama-completion -m build/shape-gguf/shape-<id>/*.gguf -p Hello -n 48 --temp 0 -no-cnv
   ```

3. **Console** (needs explicit OK; one session, current shipping MSIX, t6):
   provision the three real references with `provision-models.sh`, upload
   each shape with
   `deploy.sh upload-dir build/shape-gguf/shape-<id> $PFN "models\\shape-<id>"`,
   then per model:

   ```bash
   ./scripts/bench-xbox-ort.sh <model> --threads 6 --runs 4 --n-predict 128 \
     --out bench/results/shape-gate.csv
   ```

   Order: the three real references and their `ref-*` clones interleaved
   first (C0), then `d*`, then `v*`. Verify each upload by listing the
   directory (WDP writes can fail silently).

4. **Tokenizer ratio** (host, for G2): token count of every
   `bench/prompts/*.txt` under each vocab via `llama-tokenize`, reported as
   tokens per 1000 characters.

## Gates (predeclared)

Medians of runs 2..4. Prefill and decode are recorded separately; only decode
gates.

| Gate   | Test                                                                                             | PASS                               | On FAIL                                                                         |
| ------ | ------------------------------------------------------------------------------------------------ | ---------------------------------- | ------------------------------------------------------------------------------- |
| **C0** | each `ref-*` clone vs its real model, same session: decode and prefill                           | both within ±5%                    | method invalid → stop, record why random weights diverge                        |
| **C1** | every row: `n_gen_tok == n_predict`, quant column `Q4_0`, model column = uploaded dir            | row kept                           | row dropped and re-run; two drops on one shape → shape recorded as unmeasurable |
| **G1** | best of `d8/d12/d24/d32` decode ÷ `ref-1.2b` decode                                              | ≥ 1.30×                            | **kill**: at this budget LFM2.5's shape already sits on the Series S frontier   |
| **G2** | `v*` text throughput ÷ `ref-1.2b` text throughput (tok/s ÷ tokens-per-char ratio vs 65536 vocab) | ≥ 1.10×                            | vocabulary is not a lever; keep 65536                                           |
| **R**  | peak working set of any passing shape                                                            | ≤ `lfm25-1.2b-instruct` peak + 20% | shape disqualified regardless of speed                                          |

Why 1.30× for G1: below that, a pruned-and-distilled student is unlikely to
survive its quality loss against simply shipping `lfm25-1.2b-instruct`.

## Outcomes

- **C0 FAIL** → the random-weight method is recorded as invalid on this
  backend, with the gap.
- **G1 and G2 FAIL** → close: "custom `lfm2` shape is not a Series S lever at
  ~1B". Fold into [model-matrix.md](model-matrix.md) §F so it is not
  re-searched.
- **G1 or G2 PASS** → next step is a separate distillation plan (teacher,
  token budget, rented-GPU cost). That spend is a user decision. This gate
  says nothing about quality; the distilled model still climbs the normal
  ladder (host smoke → console bench → manifest,
  [architecture.md](architecture.md)).

## Known limits

- Truncating the LFM2 vocabulary approximates a smaller trained BPE (ids are
  merge-ordered) but is not identical to one; G2 is an upper bound.
- `default_attn_layers` keeps LFM2.5's 6-in-16 attention ratio; the ratio
  itself is not swept. It mostly moves KV size and long-context prefill, which
  this gate does not measure.
- One thread count (t6, the shipped default). The winner, if any, gets a
  thread re-sweep before any claim.
