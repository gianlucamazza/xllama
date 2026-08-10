# Phase 16 — model scouting

> **Living SSOT for the model-scouting campaign.** Phase 15 owns _how fast the
> resident model runs_ ([phase15-re-opt.md](phase15-re-opt.md)); this file owns
> _which models exist at all_ — the search, the screening funnel, the validation
> ladder, and the per-workstream verdicts. Status of anything that ships still
> lands in [model-matrix.md](model-matrix.md); numbers still live in
> [benchmarks.md](benchmarks.md) / `bench/results/`; the checklist lives in
> [`../ROADMAP.md`](../ROADMAP.md) Phase 16.

**Currency:** 2026-08-10. **Campaign complete.** One model shipped
(`lfm25-230m`, floor tier); WS-B/C/D/G closed on their own kills; WS-E and WS-F
blocked on a product decision and an unwritten probe. 3 of ≤9 console sessions
spent.

## Goal

The last survey is dated **2026-07-27** ([model-matrix.md](model-matrix.md) §F)
and was a desk pass over a shortlist. Since then v1.5.4.0 shipped and Phase 15
closed its levers. The catalogue still reflects families picked in July, and the
`llama.cpp` pin (`6d5a910c5`, tag b10094, 2026-07-22) is the implicit gate on
which 2026 architectures are even loadable.

This campaign re-explores the landscape across four classes — text GGUF, ORT
GenAI / DirectML, diffusion, and three surfaces the product does not have today
(embedding, ASR, vision) — and carries survivors all the way to Series S
measurement. Negative results are deliverables: every dropped candidate is
recorded with its reason so the next campaign does not re-search it.

Expected shape: ~90 raw candidates → ≤24 with a funnel row → ≤12 downloaded →
**≤2 shipped**.

## Method

Inherited verbatim from [phase15-re-opt.md](phase15-re-opt.md) "Method" — it is
not restated here. Two campaign-specific additions:

7. **Surface before speed.** For a capability the product does not have
   (embedding, ASR, vision), the first gate is a written argument that a product
   surface exists — not a benchmark. A workstream may be closed on that argument
   alone, with no download.
8. **A drop is evidence.** Every candidate rejected at desk gets a named reason
   in the funnel table, and the reason is folded into
   [model-matrix.md](model-matrix.md) §F when the campaign closes.

## Screening funnel (T0 — desk, zero downloads)

Cheapest and most-lethal filter first. A candidate enters T1 only with five
explicit passes recorded in its funnel row.

### F0 — dedup

```bash
grep -n -i "<candidate>" docs/model-matrix.md docs/model-selection.md \
     docs/phase7-hypotheses.md docs/phase15-re-opt.md
python3 -c "import json;print([m['name'] for m in json.load(open('uwp/models/manifest.json'))['models']])"
```

A hit in §F, in either "Do not reopen" list, or in the manifest is a **reject**,
unless the candidate carries a **new fact** (new quant, new pin, new
measurement) written into the cell. No re-litigation of a closed call.

### F1 — size arithmetic vs the peak gate

```bash
curl -s "https://huggingface.co/api/models/<repo>/tree/main?recursive=1" \
 | python3 -c "import json,sys;[print(f['size'],f['path']) for f in json.load(sys.stdin) if f['path'].endswith('.gguf')]"
```

`est_console_peak_MB = weights_MB × 1.12`. The 1.12 factor is the **measured**
GGUF load overhead already on record (Coder-3B 1840 → 2116, LFM2-2.6B
1491 → 1623 — [phase7-hypotheses.md](phase7-hypotheses.md)).

- Gate: **≤ 3584 MB** (the 3.5 GB product peak gate).
- The 4864 MB heap ceiling (`bench/results/phase15-ramceil.csv`) is a physics
  bound, **not headroom** — cite it, never quote it as budget.
- ONNX only: merged `model.onnx` < 2 GB ([uwp-constraints.md](uwp-constraints.md)
  §8). This does **not** apply to GGUF (`gemma4-e2b` is 2.45 GB and loads).
- Disk is not binding (90 GB Dev Mode).

### F2 — architecture supported by the current pin

```bash
sed -n '/LLM_ARCH_NAMES/,/^};/p' llama.cpp/src/llama-arch.cpp | grep -o '"[^"]*"'
ls llama.cpp/src/models/ | wc -l   # 139 at b10094
curl -s "https://huggingface.co/<repo>/raw/main/config.json" | python3 -m json.tool | grep model_type
```

PASS = the arch string is in `LLM_ARCH_NAMES` **and** `src/models/<arch>.cpp`
exists; UWP compilation is then automatic, because `uwp/ggml-uwp.vcxproj`
includes `src\models\*.cpp` as a wildcard. **A FAIL here is the only legitimate
trigger for WS-B.**

### F3 — chat template vs the four renderers

`chat_format_for()` (`src/bridge/chat_prompt.cpp`) dispatches on a **substring of
the model-id basename**: `gemma` → Gemma, `llama` → Llama-3, `phi` → Phi-3,
otherwise ChatML; plus `qwen3` (empty-`<think>` prefill) and `thinking` / `a1b`
(reasoning stripped from display). Three checks:

1. Does the upstream `chat_template` reduce to one of the four? If not, it is a
   **new renderer** — cost it in the card, never disguise it with a rename.
2. Does the intended catalogue id accidentally match a predicate? Any id
   containing `llama` gets Llama-3 tokens.
3. **Does the model reason on every turn without saying so in its name?** That
   was the H2 surprise and it costs ~4× perceived latency. If yes, the id must
   contain `thinking` or `a1b`.

The catalogue id is a design act, and its reasoning belongs in the
[model-matrix.md](model-matrix.md) row.

### F4 — licence

| Licence                        | Verdict                                                                                             |
| ------------------------------ | --------------------------------------------------------------------------------------------------- |
| Apache-2.0 / MIT               | may be re-hosted on the `models-v1` release (asset ≤ 2 GB)                                          |
| Gemma Terms of Use             | **never mirror** — `hf_base_url` points at the original HF repo (`gemma3-270m` precedent)           |
| LFM Open License v1.0          | redistributable per §4 **only** if the `LICENSE` ships as a `files[]` entry; §5 caps commercial use |
| `gated: true`                  | **reject** — the in-app downloader is anonymous and cannot accept terms                             |
| non-commercial / research-only | reject unless it survives an explicit §4-style read, written out                                    |
| any, single asset > 2 GB       | HF-hosted, not the GitHub release                                                                   |

Details and precedents: [model-selection.md](model-selection.md).

### F5 — surface fit

WS-E / WS-F / WS-G only: no candidate enters T1 before its workstream S-gate has
PASSed.

**T0 budget:** unbounded brainstorm → **≤24** candidates get a funnel row →
**≤12** pass all five filters.

## Validation ladder

| Tier                 | Entry                                                           | Exit criteria                                                                                                                     | Cap     |
| -------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- | ------- |
| **T0** desk          | named candidate                                                 | 5/5 funnel passes, funnel row written                                                                                             | 24 → 12 |
| **T1** host smoke    | T0 pass                                                         | loads; `general.architecture` as predicted; coherent greedy answer; stops on the right sequence; `can_shift=` observed in the log | 12 → 8  |
| **T2** host quant    | T1 pass                                                         | a Q4_K_M (or a justified IQ) exists; greedy output byte-stable across two runs; ORT/DML only: logit parity                        | 8 → 6   |
| **T3** console bench | T2 pass + a **LocalState manifest override** (not yet a commit) | 3 recorded runs, median + min–max; `peak_ws_mb` ≤ 3584                                                                            | 6 → 4   |
| **T4** capability    | T3 pass                                                         | H9 recorded                                                                                                                       | 4 → 3   |
| **T5** ship          | T4 pass + a product role it actually wins                       | every SSOT write below, coherence clean, 10/10 console gates                                                                      | 3 → ≤2  |

Invocations, in order:

```bash
./build/bin/xllama-cli --chat --greedy -m models/<file>.gguf -t 6 -n 64       # T1
./scripts/quantize.sh in.gguf out.gguf Q4_K_M                                 # T2
./scripts/validate-logit-parity.sh                                            # T2, ORT/DML only
./scripts/provision-models.sh <name>                                          # T3
./scripts/bench-xbox-ort.sh <name> --runs 4 --n-predict 96 \
    --out bench/results/phase16-<ws>.csv                                      # T3 (+ --gpu-sample for DML)
./scripts/eval-xbox-models.sh --models a,b --out bench/results/phase16-h9.jsonl  # T4
python3 scripts/check-coherence.py && ./scripts/validate-console.sh all       # T5
```

Predeclared ladder rules:

- **Host RSS never leaves T1/T2.** The host mmaps; the console does not
  ([uwp-constraints.md](uwp-constraints.md) §1). Console peak comes from
  `peak_ws_mb` in the CSV or it does not exist.
- **Host tok/s are never product numbers** (Method rule 3).
- **One variable per CSV:** `phase16-gguf.csv`, `phase16-dml.csv`,
  `phase16-diffusion.csv`, `phase16-pin-before.csv` / `phase16-pin-after.csv`.
- **H9 on a reasoning model is N/A, not FAIL** — the suite caps generation at
  16–80 tokens (H2 precedent). Declare it at T3.
- `scripts/check-coherence.py` compares `bench/results/phase7-h9.jsonl` against a
  hardcoded expectation table. Writing to `phase16-h9.jsonl` is CI-neutral; if a
  Phase 16 model **ships** with a documented H9 score, its rows move into
  `phase7-h9.jsonl` and the expectation table gains an entry in the same PR.

### Which build path each tier uses

Most of this campaign needs no build at all: a GGUF candidate is **provisioned**
into `LocalState`, so WS-A runs T1–T4 against the shipping CI MSVC package.
Three workstreams do need a rebuild — WS-B (pin bump), and WS-C / WS-G if they
reach engineering — and there the two assembly lines split by purpose:

| Purpose                                                                                  | Path                                                      |
| ---------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| Iterate on build breakage (patch rebase, source list, link errors, does it still launch) | Linux crossbuild — `scripts/crossbuild-uwp.sh` + openappx |
| Produce any number that enters a CSV or a card                                           | CI MSVC `build-uwp` → `xllama-appx`                       |

The crossbuild path compiles, packages and **launches** on Series S as of
2026-08-08 ([crossbuild-console.md](crossbuild-console.md)), which makes it the
right tool for WS-B's R1/R2 failure modes: a patch that no longer applies or a
`ggml/src` source added upstream shows up as a local link error in minutes
instead of a CI round trip. It is **not** the measurement path — ORT/GenAI
backends, first-boot provisioning and long uptime are unproven on it, and a
crossbuild tok/s figure compared against a CI-MSVC baseline would be the
comparability error this campaign already guards against (see the WS-B note).
WS-C is the worst fit of all, since ORT/GenAI is precisely what is unproven
there.

## Workstreams

Cards are predeclared — claim / measure / PASS / FAIL / kill written before any
engineering, per Method rule 1.

| WS       | Card  | Subject                                 | Console budget      | Status                                 |
| -------- | ----- | --------------------------------------- | ------------------- | -------------------------------------- |
| **WS-A** | H16.1 | Text GGUF scouting (chat/coding/reason) | 3 of ≤4 spent       | **closed — 1 shipped** (`lfm25-230m`)  |
| **WS-B** | H16.2 | `llama.cpp` pin bump evaluation         | 0 — never triggered | **closed, not motivated** (2026-08-10) |
| **WS-C** | H16.3 | ORT GenAI / DirectML text re-evaluation | 0 — kill fired      | **closed — surface saturated**         |
| **WS-D** | H16.4 | Diffusion successor to SD-Turbo         | 0 — kill fired      | **closed — nothing exportable**        |
| **WS-E** | H16.5 | Embedding surface (S-gate)              | 0                   | **blocked — S-gate unowned**           |
| **WS-F** | H16.6 | ASR surface (S-gate)                    | 0                   | **blocked — mic probe unwritten**      |
| **WS-G** | H16.7 | Vision / VLM surface (S-gate)           | 0 — S-gate FAIL     | **closed — cost not bought**           |

**Total console bench budget: ≤9 sessions.** Anything beyond is scope creep and
needs a decision-log entry.

### H16.1 — WS-A, text GGUF

- **Claim:** the §F survey is ~6 months stale; at least one un-catalogued GGUF
  beats a current role holder on its own axis at ≤3.5 GB console peak.
- **Measure:** `bench/results/phase16-gguf.csv` (17-column schema from
  [`../bench/README.md`](../bench/README.md)), `--runs 4` → 3 recorded
  `run_index`, median + min–max; H9 in `bench/results/phase16-h9.jsonl`.
- **PASS:** `peak_ws_mb` ≤ 3584 **and** (H9 ≥ incumbent **and** decode ≥ 0.9×
  incumbent) **or** (decode ≥ 1.3× incumbent **and** H9 ≥ incumbent − 1).
- **FAIL:** every survivor is within noise of an incumbent on both axes, or
  breaks 3.5 GB.
- **Kill:** no candidate clears the T0 funnel → close as a desk refresh, restamp
  §F, stop. No downloads.

### H16.2 — WS-B, `llama.cpp` pin bump

- **Claim:** moving off `6d5a910c5` unlocks ≥1 architecture a T0 survivor needs,
  at a cost of ≤1 patch rebase and ≤1 vcxproj edit, with zero regression on the
  shipping baseline.
- **Measure:** (a) arch delta —
  `git -C llama.cpp log --oneline b10094..<new> -- src/models/ src/llama-arch.cpp`
  plus the `LLM_ARCH_NAMES` diff; (b) `scripts/apply-uwp-patches.sh` applies
  clean; (c) `scripts/check-uwp-sources.sh`; (d) a **manual diff of the
  `ggml/src` source list** against `uwp/ggml-uwp.vcxproj` (not covered by CI —
  see R2); (e) an A/B of `lfm25-350m` Q4_K_M at t6 into
  `bench/results/phase16-pin-before.csv` / `phase16-pin-after.csv`.
- **PASS:** the needed arch is present, the patch applies (or rebases in ≤20
  lines), 10/10 console gates, and prefill/decode stay within ±3% of
  `phase13b-threadsbatch-after`.
- **FAIL:** any gate regresses, or the baseline A/B moves >3% unexplained.
- **Kill:** **WS-B does not start speculatively.** It starts only once a
  candidate has passed F1, F3, F4 and F5 and failed **only** F2. No demand →
  closed "not motivated", one decision-log line.
- **Note:** a bump creates a new benchmark comparability boundary, to be recorded
  in [`../bench/README.md`](../bench/README.md) alongside the repack and
  `n_threads_batch` boundaries. Sequence WS-B entirely before or entirely after
  WS-A's console phase — never interleaved.
- **Status: CLOSED, not motivated (2026-08-10).** The kill fired exactly as
  written. T0 produced two `arch:not-in-pin` flags and adversarial verification
  refuted both: `Qwen3.5-2B` converts to `general.architecture = qwen35`, which
  the pin carries, verified from the published GGUF header rather than from a
  `config.json` `model_type` (the two are different namespaces, and that
  confusion was the entire flag). No survivor fails F2 and only F2, so the pin
  bump has no demand. Reopening needs a candidate that clears F1/F3/F4/F5 and
  fails on architecture alone — not a newer upstream release on its own.

### H16.3 — WS-C, ORT GenAI / DirectML text

- **Claim:** with GenAI pinned at 0.14.1 and the model builder frozen at
  Qwen3/Gemma3, no ONNX text model beats `smollm2-360m-dml-fp16-v2` inside the
  GPU budget and the ~360–500M practical DML text ceiling.
- **Measure:** desk first (arch inside the builder's frozen set × ≤500M params ×
  merged `model.onnx` < 2 GB); then build, `scripts/merge_onnx_external_data.py`,
  then `scripts/validate-logit-parity.sh` **before any tok/s claim** (Method rule
  4), then `scripts/bench-xbox-ort.sh <name> --runs 4 --gpu-sample --out bench/results/phase16-dml.csv`.
- **PASS:** parity OK **and** decode ≥1.2× the incumbent in the long-prompt band,
  within the GPU budget.
- **FAIL:** parity fails (the #91 class — silent CPU fallback) or no throughput
  win.
- **Kill:** zero arch inside the frozen set at ≤500M → close as "ORT text surface
  saturated at 0.14.1". Reopening needs a GenAI version bump, which belongs to
  [vendor-lifecycle-plan.md](vendor-lifecycle-plan.md), not this campaign.
- A winner also edits `dml_text_model_ok()` (`include/xllama/routing_policy.h`, a
  literal id) **and** the matching assertion in `scripts/check-coherence.py`, in
  the same PR.

### H16.4 — WS-D, diffusion successor

- **Claim:** a distilled few-step successor beats `sd-turbo-fp16` within the GPU
  budget and the current 3-stage wall-clock.
- **Measure:** desk (a 3-component ONNX export is feasible, each component
  self-contained and < 2 GB); host `diffusion/validate_pipeline.py` against
  `diffusion/golden_vectors.json`; console `scripts/validate-console.sh taesd`
  plus wall-clock.
- **PASS:** faster than the incumbent per image, inside the GPU budget, gate PASS.
- **FAIL:** slower or over budget.
- **Kill:** no candidate exports to a 3-component graph the DirectML EP accepts →
  close. A monolithic DiT diffusion model is **a new backend**, out of scope.

### H16.5 — WS-E, embedding surface (S-gate)

- **Claim (S-gate, not tok/s):** a product surface for embeddings exists. Today
  none does — `uwp/api-server.cpp` routes no `/v1/embeddings`, no UI consumer
  exists, and there is no vector store (`kv_store.h` is a KV-cache snapshot
  store, not a vector DB).
- **Measure:** a written surface spec answering three questions — who calls it,
  where the vectors live, and how it survives the **one-resident-model** rule
  (`include/xllama/session_hub.h`), since an embedding model is a second
  `llama_context`.
- **PASS:** a named consumer plus a memory plan that keeps total peak ≤3.5 GB
  with one resident model.
- **FAIL / kill:** no consumer, or the design needs two resident contexts → close
  before downloading anything, and record that the capability exists in the pin
  while the product surface does not.

### H16.6 — WS-F, ASR surface (S-gate)

- **Claim (S-gate):** speech input is a product surface on a console with **no
  text input** — the reason the whole autopilot apparatus exists.
- **Measure:** three desk preconditions, **in this order**: (1) **microphone** —
  does the Xbox AppContainer grant `MediaCapture` / `AudioGraph`?
  [uwp-constraints.md](uwp-constraints.md) has no entry, so this is the RE item,
  and it is the one that kills or saves the workstream; (2) **backend** —
  whisper.cpp is not vendored, so ASR means a second submodule and a second
  hand-maintained UWP source list, i.e. the "no third backend" line;
  (3) **budget** — a small encoder sits alongside the chat model.
- **PASS:** microphone capture proven on device **and** a backend plan that adds
  no third inference stack **and** a combined peak ≤3.5 GB.
- **FAIL / kill:** mic denied under AppContainer, or ASR requires a new submodule
  → close. Even on FAIL the microphone finding is a deliverable: a new section in
  [uwp-constraints.md](uwp-constraints.md).

### H16.7 — WS-G, vision / VLM surface (S-gate)

- **Claim (S-gate):** an image-in chat turn is a product surface reachable
  without a new backend, because the pin already carries `llama.cpp/tools/mtmd/`.
- **Measure:** (1) **build cost** — `uwp/ggml-uwp.vcxproj` references no mtmd
  file and its `ggml` source list is hand-maintained; (2) **input path** — where
  an image comes from with no file picker (Device Portal upload into LocalState,
  the LAN API, or a generated image); (3) **budget** — mmproj plus LM ≤3.5 GB;
  (4) **template** — VLM templates carry image sentinels none of the four
  renderers emit.
- **PASS:** a named input path, a candidate that fits, and a decided template
  answer.
- **FAIL / kill:** no input path, or template work exceeding one new renderer →
  close.

**Cross-workstream cap:** at most **one** of WS-E / WS-F / WS-G may proceed past
T2 to the console this campaign. Predeclared tiebreak: whichever S-gate passes
with the fewest new C++ surfaces.

## Out of scope (do not expand into these)

FIM / completions as a second prompt surface; a third inference backend
(whisper.cpp, a monolithic DiT); a vector store as a product feature; and any ORT
GenAI version bump, which belongs to
[vendor-lifecycle-plan.md](vendor-lifecycle-plan.md).

## T0 desk sweep — run A (2026-08-10)

Eight search modalities run in parallel (architectures the pin already compiles
but xllama has never run; recency since the pin date; leaderboards; quantiser
organisations; new surfaces; vision; diffusion; papers), then deterministic
screening against the funnel, then semantic dedup against the shipping,
rejected and closed-FAIL lists.

Funnel: **59** raw candidates → 48 after exact/family dedup → 47 after the
deterministic gates → 41 after semantic dedup → **12** shortlisted. Only one
candidate was dropped by a gate (a gated repo, F4): the search itself was run
with the envelope in hand, so the screening had little left to reject. The
substantive output is therefore the negative half, below.

**Provenance caveat — read before citing any of this.** Everything in this
section is a desk assertion from a search agent, not a measurement and not yet
adversarially verified. Run B verifies the twelve survivors on four independent
lenses (architecture, size, licence, template); the negatives below are
verified only to the extent that each carries a URL. Nothing here may move into
[model-matrix.md](model-matrix.md) §F until the campaign closes, and nothing
here is a performance claim — projected decode figures use `DECODE_MODEL_v0`
(below), which is a screening heuristic, not doctrine.

### Negative results worth keeping

Grouped, each with the evidence that closes it. These are why the search stops
where it does — the point is that the next campaign does not re-run them.

| Bucket                                            | Why it is empty                                                                                                                                                                                                                                                                                                                    |
| ------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Coding model under 3B newer than Qwen2.5-Coder-3B | **None exists.** The Qwen coder line jumps from Coder-3B straight to Qwen3-Coder-30B-A3B; Qwen3.5 shipped no Coder variant at any size. The 2026 coding releases (KAT-Coder-V2.5-Dev, Laguna-XS-2.1, North-Mini-Code) are all far past the gate. The best sub-4B coding option is a generalist.                                    |
| 4B-class dense chat (Qwen3.5-4B tier)             | Fails the **decode** bound, not the RAM gate: Q3_K_M is 2293 MB → under the interactive target before any KV cache. Only sub-3-bit files clear it, which is the recorded IQ2_M→EOG failure.                                                                                                                                        |
| 4B-class VLM (Qwen3-VL-4B tier)                   | Same shape: leanest sane 4-bit is 2271 MB before a 454 MB projector. A bandwidth rejection, not a memory one.                                                                                                                                                                                                                      |
| Dense Gemma-4 below E2B                           | Google shipped none — the lineup is E2B / E4B / 12B / 26B-A4B / 31B, and E2B is already catalogued.                                                                                                                                                                                                                                |
| New ORT GenAI / DirectML text                     | The builder is frozen at Qwen3/Gemma3 and **every** 2026 sub-4B model moved off those architectures (Qwen3.5 → `qwen3_5`, LFM2.5 → `lfm2`, SmolLM3 → `smollm3`, Falcon-H1 → `falcon_h1`). Direct evidence for the WS-C kill criterion.                                                                                             |
| Diffusion successor to SD-Turbo                   | SDXL-Turbo is excluded by the **2 GB protobuf limit**, not the GPU budget (UNet external data 5.1 GB). SANA-Sprint, DMD2 and Tiny-SD have no usable ONNX export at all; SD3.5 and Flux are monolithic DiT, i.e. a new backend.                                                                                                     |
| ASR as GGUF under the pin                         | Empty by construction: the ASR GGUFs that exist declare `whisper.cpp` or `transcribe.cpp` as their runtime, neither of which is a submodule here. Any GGUF ASR route means adopting a new C++ submodule.                                                                                                                           |
| ASR on DirectML                                   | The ORT GenAI Whisper path documents CPU and CUDA only. Every ASR candidate is therefore a CPU-EP surface competing for the same 6 threads as the resident LLM.                                                                                                                                                                    |
| Embedding under 200 MB from an LLM backbone       | Arithmetic, not search: the smallest Qwen3-Embedding-class member is 0.6B (~600 MB at Q8). The 47–300M encoder class is the only one that fits — which is why every embedding candidate comes from it. EmbeddingGemma misses the target even at Google's own QAT-Q4_0 (278 MB), because the 262K-token vocabulary is the floor.    |
| New small-MoE worth reopening H2                  | No new fact. The two in-window releases are a 5.72B custom-arch model and a Danbooru prompt generator; neither carries a measurement beating the recorded H2 result.                                                                                                                                                               |
| New low-bit / ternary worth feeding H5            | The trending ternary model is 20.2B on a custom MoE arch whose only GGUF needs a llama.cpp **fork**, not a pin bump. Nine months of QAT literature releases recipes and code, never a downloadable sub-4B quantised checkpoint. H5's survey therefore has no artifact to survey yet — which is itself the survey's answer for now. |

### Shortlist entering run B (unverified)

Twelve candidates, ranked by the screening score. Peak and decode are
**projections**, not measurements.

| Candidate                                  | Class     | Quant  | Est. peak MB | Est. tok/s | Would displace                |
| ------------------------------------------ | --------- | ------ | -----------: | ---------: | ----------------------------- |
| `nomic-ai/nomic-embed-text-v1.5`           | embedding | Q8_0   |          156 |          — | (no incumbent)                |
| `ibm-granite/granite-embedding-english-r2` | embedding | Q8_0   |          171 |          — | (no incumbent)                |
| `ggml-org/SmolLM3-3B-GGUF`                 | text      | Q4_K_M |         2046 |       14.8 | `llama32-3b`                  |
| `ggml-org/embeddinggemma-300M-GGUF`        | embedding | Q8_0   |          356 |          — | (no incumbent)                |
| `Maincode/Maincoder-1B-GGUF`               | text      | Q4_K_M |          718 |       42.1 | `qwen25-coder-1.5b`           |
| `Qwen/Qwen3.5-2B`                          | text      | Q4_K_M |        ~1370 |        ~21 | `qwen3-1.7b`                  |
| `openbmb/MiniCPM5-1B`                      | text      | Q4_K_M |          735 |       41.1 | `qwen3-1.7b` / `smollm2-1.7b` |
| `unsloth/LFM2.5-230M-GGUF`                 | text      | Q4_K_M |          164 |      184.6 | `lfm25-350m` (floor tier)     |
| `PaddlePaddle/PaddleOCR-VL-1.6-GGUF`       | vlm       | Q4_K_M |         1263 |       24.0 | (no incumbent)                |
| `SupraLabs/Supra2-100M-Instruct`           | text      | F16    |          216 |      139.8 | `gemma3-270m` (floor tier)    |
| `intfloat/multilingual-e5-small`           | embedding | Q8_0   |          141 |          — | (no incumbent)                |
| `openai/whisper-base`                      | asr       | int8   |          107 |          — | (no incumbent)                |

Three known defects were in that table; all three were handed to run B rather
than patched by hand, and all three are resolved below.

## T0 adversarial verification — run B (2026-08-10)

Each of the twelve went to four blind skeptics, one per lens (architecture,
size, licence, template), each instructed to refute rather than assess and each
unable to see the others. Size and licence are treated as **single veto** — they
are third-party-checkable facts, and taking a majority vote on a fact is a
category error. Architecture and template carry _cost_, not truth: one failure
degrades, two kill. An unevidenced FAIL is demoted to UNSURE before the tally.

Outcome: **6 PASS, 5 DEGRADE, 1 KILL.** No candidate needed adjudication.

### What the verification actually caught

**The size check fired on 7 of 12 candidates**, correcting the record in both
directions. Five were overstated because the screening rule had selected an
f16/bf16 variant (MiniCPM5-1B 2066 → 656 MB, LFM2.5-230M 440 → 146, Supra2-100M
241 → 103, Whisper-base 278 → 73, Maincoder-1B 1046 → 641). One was
**understated**: PaddleOCR-VL 892 → 1316 MB, because the 841 MiB vision
projector had not been counted against the same budget. Understatement is the
dangerous direction, and only a byte-level check finds it.

**Both `arch:not-in-pin` flags were refuted, and this closes WS-B.**
`Qwen3.5-2B` converts to `general.architecture = qwen35`, which is in the pin —
verified by parsing the published GGUF header, not by reading a config file. No
surviving candidate fails the architecture filter _and nothing else_, so the
predeclared trigger for a pin bump never fires.

**Two candidates claimed ChatML and would have been silently wrong** — the most
valuable catch, because both look correct at desk and fail only in the product:

- `SmolLM3-3B` opens every prompt with a mandatory metadata block carrying a
  `Reasoning Mode: /think` line, and enables extended thinking by default. Our
  ChatML renderer emits none of it, and `model_is_thinking()` is false for that
  id, so chain-of-thought would print verbatim into the chat UI. This is the
  #223 failure mode with the polarity reversed.
- `MiniCPM5-1B` has the inverse of the double-BOS trap: its template emits `<s>`
  while `add_bos_token=false` propagates into the GGUF, so the prompt loses a
  token every training sample carried. It is also post-trained on thinking data
  and lands in an undocumented undefined state when the thinking flag is neither
  set nor cleared.

**`embeddinggemma-300M` is killed on licence, and the record was wrong about
why.** The scout recorded `gated: no`; the origin repo is `gated: manual` — a
human review, not a click-through — and an anonymous HEAD returns 401. The
Gemma-Terms fallback of fetching from Google's own repo is therefore closed to
the app's anonymous downloader, and the ungated community mirror ships no
licence or notice file at all, so the artefact would arrive without the terms
Gemma requires be passed on.

**Two arch failures that are not about the architecture:**

- `multilingual-e5-small` is `bert`, which the pin has. Its published GGUF is
  the problem: the tokenizer is written as SPM over a 250K XLM-R Unigram
  vocabulary, which needs UGM, so segmentation diverges from the reference and
  the published MTEB numbers do not transfer. Self-converting does not rescue it
  at this pin either.
- `whisper-base` resolves the contradiction the sweep left open. The pin _does_
  contain a Whisper encoder, but only as an mmproj audio tower for other
  projectors — there is no `whisper` entry in the arch list and no Whisper text
  decoder. GGUF ASR is genuinely empty by construction. The ORT route also fails
  as published: the `onnx-community` export is a transformers.js artefact with
  no `genai_config.json`, which ORT GenAI cannot load; a loadable build is a
  different artefact.

### Verified state entering T1

| Candidate                                  | Class     | Verdict  | Ship quant | Peak MB | Cost           |
| ------------------------------------------ | --------- | -------- | ---------- | ------: | -------------- |
| `nomic-ai/nomic-embed-text-v1.5`           | embedding | **PASS** | Q8_0       |     156 | config-only    |
| `ibm-granite/granite-embedding-english-r2` | embedding | **PASS** | Q8_0       |     171 | new-renderer   |
| `Maincode/Maincoder-1B-GGUF`               | text      | **PASS** | Q4_K_M     |     718 | config-only    |
| `Qwen/Qwen3.5-2B`                          | text      | **PASS** | Q4_K_M     |    1368 | config-only    |
| `unsloth/LFM2.5-230M-GGUF`                 | text      | **PASS** | Q4_K_M     |     164 | config-only    |
| `SupraLabs/Supra2-100M-Instruct`           | text      | **PASS** | Q4_K_M     |     116 | config-only    |
| `ggml-org/SmolLM3-3B-GGUF`                 | text      | DEGRADE  | Q4_K_M     |    2046 | new-renderer   |
| `openbmb/MiniCPM5-1B`                      | text      | DEGRADE  | Q4_K_M     |     735 | new-renderer   |
| `PaddlePaddle/PaddleOCR-VL-1.6-GGUF`       | vlm       | DEGRADE  | Q8_0 + f16 |    1474 | new-renderer   |
| `intfloat/multilingual-e5-small`           | embedding | DEGRADE  | —          |     141 | arch-port      |
| `openai/whisper-base`                      | asr       | DEGRADE  | —          |      82 | new-dependency |
| `ggml-org/embeddinggemma-300M-GGUF`        | embedding | **KILL** | —          |       — | licence: gated |

DEGRADE means it survives carrying a named engineering cost and ranks below
every clean candidate — not that it is admitted. Admission to T1 is still a
per-workstream decision, and for WS-E/F/G it remains behind the S-gate.

### A flaw in the screening itself

The size lens made a correction that is about our method, not about any
candidate: **`DECODE_MODEL_v0` was applied to encoder models, where it is
meaningless.** An embedding model has no autoregressive decode, so `27000 /
weightsMB` and the 12 tok/s floor do not apply to `clazz: embedding` at all —
the throughput that matters there is prefill-bound. Every decode figure quoted
for the five embedding candidates in run A should be read as absent, not as
generous. The screening code gates by class in future runs.

### `DECODE_MODEL_v0` — a screening heuristic, not doctrine

Back-fitting the console rows in [model-matrix.md](model-matrix.md) §A gives
`decode ≈ 27000 / weightsMB` (±20%, low on Q3_K_S), which is far more stable
across models than the raw effective-bandwidth figure. It is used **only** to
reject candidates before download: it implies weights ≤ ~2250 MB for the
interactive target, a tighter bound than the memory gate, and it is what rejects
the whole 4B class above.

It is labelled and confined on purpose. It is a fit to six points, it has never
been validated forward, and it must not appear in [benchmarks.md](benchmarks.md)
or in any product claim. If WS-A's T3 measurements agree with it, promoting it
to a documented screening rule in [`../bench/README.md`](../bench/README.md) is
a separate proposal.

## Candidate cards (T0 synthesis — run C, 2026-08-10)

Eleven verified survivors went to eleven independent card writers, then to one
editor for id assignment, duplicate-bet removal and criticism of the cards
themselves. Ids are suffixed under the owning workstream card (H16.1a under
WS-A's H16.1, and so on), because a flat H16.8 would collide with a future
workstream.

**A card that failed and was recovered.** The `SmolLM3-3B` writer exhausted its
schema retries and errored out, so the first synthesis was built from ten cards,
not eleven — the candidate dropped out by accident, not by judgement. The cause
was ours: `maxLength` limits on the card fields that nine of twelve agents hit
and recovered from by retrying. The card was rewritten against a relaxed schema
and put through an amendment pass, which judged it on merit and closed it (see
"Do not reopen"). Recording this because a silent eleven-to-ten is
indistinguishable from a decision, and it is not one.

### Shortlist

| Card       | Candidate (catalogue id)                                     | WS   | Ship quant                         | Peak MB (predicted / gate)                            | Engineering cost                                                         | Displaces                                   |
| ---------- | ------------------------------------------------------------ | ---- | ---------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------- |
| **H16.1a** | `Qwen/Qwen3.5-2B` (`qwen35-2b`)                              | WS-A | Q4_K_M (upstream, text-only)       | 1368 / ≤1398                                          | config-only                                                              | `qwen3-1.7b` — chat-upgrade role            |
| **H16.1b** | `Maincode/Maincoder-1B-GGUF` (`maincoder-1b`)                | WS-A | Q4_K_M (upstream)                  | 780 / ≤900                                            | config-only (+ new 8-task coding eval file)                              | `qwen25-coder-1.5b` — coding balanced role  |
| **H16.1c** | `unsloth/LFM2.5-230M-GGUF` (`lfm25-230m`)                    | WS-A | Q4_K_M (upstream)                  | 247 ±15 / ≤300                                        | config-only                                                              | `gemma3-270m` — floor role                  |
| **H16.1d** | `openbmb/MiniCPM5-1B` (`minicpm5-1b`)                        | WS-A | Q4_K_M (self-converted at the pin) | 735 / ≤811                                            | new-renderer (collapses to config-only if the `<s>` half is dropped)     | `lfm25-1.2b-instruct` — balanced chat role  |
| **H16.7a** | `PaddlePaddle/PaddleOCR-VL-1.6-GGUF` (`paddleocr-vl16-0.9b`) | WS-G | Q8_0 LM + BF16 mmproj              | 1474 / ≤1600                                          | new-renderer — 5 new C++ surfaces against a ≤1 budget                    | nothing (new OCR axis; desk close expected) |
| **H16.6a** | `openai/whisper-base` (`whisper-base-ort-cpu-fp32`)          | WS-F | ORT GenAI fp32 export              | 424 / ≤3584                                           | new-renderer (no new submodule; ORT GenAI 0.14.1 already vendored)       | nothing (null ASR surface)                  |
| **H16.5a** | `nomic-ai/nomic-embed-text-v1.5` (`embed-nomic-v15`)         | WS-E | Q8_0                               | 156 / ≤156 (host projection; no console encoder path) | config-only model load; the embedding surface itself is new product code | nothing (new retrieval axis)                |

_Seven rows, unchanged: WS-A stays at its predeclared four sessions and no fifth was added._ **Considered and not scheduled:** `ggml-org/SmolLM3-3B-GGUF` (`smollm3-3b`, WS-A, peer dense-3B vs `llama32-3b`) — card written, judged, and closed at desk for cost; see the card below and model-matrix §F.

### Recommended order of work

Ranked by what closes uncertainty earliest per unit of console time — not by
score.

1. **PaddlePaddle/PaddleOCR-VL-1.6-GGUF** — H16.7a — zero downloads, zero console, and its own ledger (5 surfaces vs a ≤1 budget) already settles it. One decision-log line by 2026-08-17 retires the VLM question and frees the E/F/G allocation before any console time is booked.
2. **nomic-ai/nomic-embed-text-v1.5** — H16.5a — also zero console (no encoder path in bench-xbox-ort.sh), so it runs in parallel with everything below. Assign the surface-spec owner now; unowned by 2026-08-17 it closes early instead of expiring on 08-24.
3. **unsloth/LFM2.5-230M-GGUF** — H16.1c — first console session: config-only, no artefact to build, and its family-anchored prediction is the campaign's test of the peak model itself. Whether x1.12 or the fixed-cost model wins here re-scores every other card's projection, so it must land before them.
4. **Qwen/Qwen3.5-2B** — H16.1a — highest-value displacement, and its desk kill is a byte-count on the download. One session settles a 30 MB margin, and the paired T4 also creates the qwen3-1.7b H9 baseline that does not currently exist.
5. **openai/whisper-base** — H16.6a — the single E/F/G console session, and one Dev-Mode probe answers a whole workstream binary. The mic result becomes a permanent uwp-constraints.md entry either way, so it earns the session even on FAIL. No artefact work before it PASSes.
6. **Maincode/Maincoder-1B-GGUF** — H16.1b — its desk kill (does a 1,500-token file + 256-token reply fit n_ctx 2048?) is free and probably decisive, so run that first; the console session is gated behind building the 8-task coding eval file, which no other card needs.
7. **openbmb/MiniCPM5-1B** — H16.1d — heaviest desk gate (self-convert at the pin, no-think prefill, ≤40-line renderer diff) and now the only WS-A renderer bought: SmolLM3's was refused. Last of the four; no session is booked until all three desk conditions pass.

## Candidate cards

Seven survivors, in shortlist rank order. Each was written before any engineering
(Method rule 1); the editor has applied ids, resolved the duplicate bets, and added
the amendments marked **Editor**. Console allocation: four sessions to WS-A (its
whole budget), **one** session across WS-E/WS-F/WS-G, and it is H16.6a's mic probe.

### H16.1a — WS-A · `qwen35-2b` (`Qwen/Qwen3.5-2B`, unsloth Q4_K_M)

- **Claim:** `qwen35-2b` (text-only file, 1221.5 MiB) displaces `qwen3-1.7b` in the **chat-upgrade** role on the **capability** axis: strictly better H9 at no more console peak than the incumbent's **1398 MB** and no worse than **0.9×** its **21.8 tok/s** decode. Not a speed claim and not a new surface — the 637.3 MiB `mmproj-F16` sidecar is WS-G's subject, out of scope here.
- **Measure:** **T1** `xllama-cli --chat --greedy -m models/Qwen3.5-2B-Q4_K_M.gguf -t 6 -n 64` — record `general.architecture`, the `can_shift=` line, byte-stability over two runs; host tok/s and host RSS are never quoted. **T2** none, the upstream Q4_K_M is the ship file. **T3** `provision-models.sh qwen35-2b` (LocalState override, not a commit) then `bench-xbox-ort.sh qwen35-2b --runs 4 --n-predict 96 --out bench/results/phase16-gguf.csv` → 3 `run_index`, median + min–max, `peak_ws_mb` the only admissible peak. **T4** `eval-xbox-models.sh --models qwen35-2b,qwen3-1.7b --out bench/results/phase16-h9.jsonl` — one paired invocation, because the incumbent has no recorded H9 and the comparison does not exist until this run creates it.
- **PASS:** `peak_ws_mb` ≤ **1398** and decode median ≥ **19.6 tok/s** and H9 ≥ H9(`qwen3-1.7b`) **+1/8** from that jsonl. If T1 logs **`can_shift=0`** (expected — arch `qwen35` measured 0, imrope, §D; the incumbent shifts), the bar rises to **+2/8** and `validate-console.sh longchat` must stay PASS on the #169 fail-fast+trim path: losing context shift is a product regression the capability win has to outrank.
- **FAIL:** H9 delta ≤ 0, or decode median < 19.6, or `peak_ws_mb` > 1398. Above 1398 yet under the 3584 gate is still a **FAIL of this card** — that is a catalogue addition, not a displacement, and the campaign ships ≤2.
- **Kill (desk/T1, by 2026-08-17, zero console):** the fetched file is not exactly **1,280,835,840 B** (then it is the mmproj-bundled VLM artefact, +637.3 MiB → 2082 MB, WS-G's); the load log does not report `general.architecture = qwen35`; a literal `<think>` or a `<|vision_start|>`-class sentinel prints in the 64-token greedy smoke on ≥1 of 3 fixed prompts; greedy output is not byte-stable over two runs. Any hit → one "Do not reopen" line.
- **Cost / id:** config-only — one manifest entry (`hf_base_url` at unsloth, as `qwen3-1.7b` does; Apache-2.0 also permits a `models-v1` mirror at 1.28 GB if Qwen's LICENSE ships as a `files[]` entry attributing Alibaba Cloud and unsloth), plus §A2/§D/§E and benchmarks rows. No renderer, no `dml_text_model_ok` change, no pin bump. Id `qwen35-2b`: the `qwen3` substring fires `model_is_qwen3()`, whose empty-`<think>` prefill is byte-exact against this model's default generation prompt; no `gemma`/`llama`/`phi`; deliberately no `thinking`/`a1b`, which would strip nothing here and kill the KV snapshot (#170b). No clash with `qwen35-0.8b`.
- **Predeclared:** 1221.5 MiB × 1.12 = **1368 MB** — ~30 MB of margin, the thinnest in the shortlist, and the factor may **understate** here (248,320-token vocabulary with tied embeddings; an F32 recurrent state for the 18 gated-delta-net layers beside the KV of the 6 full-attention ones). The memory half of PASS is measured `peak_ws_mb`, never the projection. Screening decode 22.1 is `DECODE_MODEL_v0` and appears in no claim. Dedup vs §F "Qwen3-1.7B/4B general — defer": different model, released 2026-03-02, arch `qwen35` verified at the GGUF header, vendor +15.1 MMLU-Pro over the incumbent — vendor benchmarks motivate the card, only H9 decides it.

### H16.1b — WS-A · `maincoder-1b` (`Maincode/Maincoder-1B-GGUF`, Q4_K_M)

- **Claim:** `maincoder-1b` (641 MiB, Apache-2.0, arch `maincoder`, in pin b10094) displaces `qwen25-coder-1.5b` in the **coding balanced** role on the **decode** axis at no loss of coding capability: **≥1.3×** its **26.1 tok/s** console decode and **≤900 MB** `peak_ws_mb` against its 1179 MB — while provisioning at **`n_ctx` 2048**, half the 4096 every coding incumbent gets, because the trained window is 2048. At Q8_0 the projection is 25.8, a tie; the claim rests entirely on Q4_K_M.
- **Measure:** **T1** `xllama-cli --chat --greedy -m models/Maincoder-1B-Q4_K_M.gguf -t 6 -n 64` (arch, ChatML render, stop on `<|im_end|>`, `can_shift=`). **T3** LocalState override + `bench-xbox-ort.sh maincoder-1b --runs 4 --n-predict 96 --out bench/results/phase16-gguf.csv` → 3 `run_index`, median + min–max, `peak_ws_mb` only. **T4** has no harness — the H9 suite is generalist and scores no coder — so this card also buys `bench/eval/phase16-coding.json`, 8 deterministic pattern-scored tasks, `max_tokens` ≤160, run as `eval-xbox-models.sh --models maincoder-1b,qwen25-coder-1.5b --tasks bench/eval/phase16-coding.json --out bench/results/phase16-h9.jsonl`.
- **PASS:** median decode ≥ **33.9 tok/s** with `peak_ws_mb` ≤ **900** (predicted 780) and ≤ 3584; coding score ≥ `qwen25-coder-1.5b` on the same 8 tasks; T1 holds a ~1,500-token source prompt + 256-token reply inside `n_ctx` 2048 and stops cleanly.
- **FAIL:** median decode < 33.9, or coding score below the incumbent by ≥1 task, or `peak_ws_mb` > 900. H16.1a's parity branch is **not** available here: halving the code window is a product regression only the ≥1.3× decode win pays for. On FAIL the 1.5B keeps the role and Maincoder is not catalogued at all — a second model at the same coding tier is menu noise.
- **Kill (desk/T1, by 2026-08-17, zero console):** at `n_ctx` 2048 a representative coding turn (~1,500-token file + instruction + 256-token reply) does not fit `fit_prompt` — prompt-too-long, or a front-drop that eats the code it was asked about — or T1 greedy output is incoherent or fails to stop. Stopping is load-bearing: the GGUF `eos` is `<|endoftext|>` (151643), not the `<|im_end|>` (151645) the template emits, and decode terminates only because the pinned `llama.cpp` name-matches `<|im_end|>` into `special_eog_ids`. Fires → "Do not reopen: a 2,048-token window cannot hold the coding role".
- **Cost / id:** config-only for the model — ChatML is the default branch of `chat_format_for()`, `maincoder` is in the pin, `n_swa=0` so context shift stays on; one manifest entry (`kind: gguf`, `role: coding`, **`n_ctx: 2048`** — the only deviation from the role default) plus Apache-2.0 text authored from our side, since neither repo checks in a LICENSE. The 8-task coding file is counted here, not treated as free. Id `maincoder-1b` hits no dispatch substring; the near miss is **`a1b`**, which any `…-a1b` id would trip, disabling the KV snapshot for nothing. The vendor filename is clean on the same rules, which matters because host dispatch keys off the basename.
- **Predeclared:** 641 MiB weights + **96 MiB** fp16 KV at the full 2048 ctx + graph/logits ≈ **780 MB**. The campaign's ×1.12 rule gives 718 MB and is **8.6% optimistic** here because it does not cover this KV — recorded as a method result either way. Not a draft model (speculative rejected at pre-gate, 0.81×); prompt-lookup stays default-OFF; no low-bit probe.

### H16.1c — WS-A · `lfm25-230m` (`unsloth/LFM2.5-230M-GGUF`, Q4_K_M)

- **Claim:** `lfm25-230m` Q4_K_M displaces `gemma3-270m` Q4_K_M as the **floor** role at t6 / `n_ctx` 2048 — strictly lower console `peak_ws_mb` **and** strictly higher decode than the floor's measured **368 MB / 76.78 tok/s** (`phase6-gemma.csv`), at capability no worse than one H9 task below it. Only Q4_K_M is on test: BF16 is rejected at desk (493 MB / ~61 tok/s is dominated on both axes by the shipping `lfm25-350m`).
- **Measure:** **T1** `xllama-cli --chat --greedy -m models/LFM2.5-230M-Q4_K_M.gguf -t 6 -n 64` (load, ChatML, `<|im_end|>` stop, `can_shift=`); host tok/s recorded nowhere. **T3** `provision-models.sh lfm25-230m` (LocalState override) then `bench-xbox-ort.sh lfm25-230m --runs 4 --n-predict 96 --out bench/results/phase16-wsa.csv` at n_ctx 2048 / t6 → 3 `run_index`, median + min–max, `peak_ws_mb` only. **T4** `eval-xbox-models.sh --models lfm25-230m,gemma3-270m --out bench/results/phase16-h9.jsonl` — the floor has **no recorded H9** anywhere in `bench/results/`, so it is re-measured head-to-head, not quoted.
- **PASS:** median `peak_ws_mb` ≤ **300** and median decode ≥ **100 tok/s** (≥1.3× the floor) and H9 ≥ H9(`gemma3-270m`) − 1 in the same jsonl.
- **FAIL:** median `peak_ws_mb` > **320** or median decode < **95** — either lands inside the shipping `lfm25-350m` default (320 MB, 94.87 tok/s), which is bigger _and_ smarter, so a second LFM2.5 entry beneath it buys nothing — or H9 ≥ 2 tasks below `gemma3-270m`.
- **Kill:** **desk, by 2026-08-17** — any of the 3 declarative prompts hitting EOG within 8 tokens (the IQ2_M collapse mode, a live risk at 230M) or failing to stop on `<|im_end|>` and running to the `-n` cap → close at desk, no provisioning, no session. **T1, by 2026-08-24** — first console CSV with median decode < 95 or median `peak_ws_mb` > 320 → close after **one** session; WS-A's budget is not spent on a second.
- **Editor (imported from the dropped Supra2-100M card):** an **absolute** floor is added, because "H9 ≥ incumbent − 1" against an incumbent with no recorded H9 is satisfiable at 0/8. **Absolute H9 ≤ 1/8 closes this card regardless of the delta** — a floor model that answers one task in eight has no slot even if the incumbent answers none.
- **Cost / id:** config-only. Manifest entry `lfm25-230m`, `kind: gguf`, ChatML via `chat_format_for`, CPU-only, KV-reuse on, routing disabled; T3 runs off a LocalState override. Ship path reuses the `lfm25-350m` pattern: mirror the 153,406,656 B Q4_K_M **plus** `LFM2.5-230M_LICENSE.txt` (10,574 B) onto `models-v1`, since the unsloth repo ships no LICENSE and LFM Open License v1.0 §4(a) requires the copy to travel. Secondary benefit on PASS: the floor stops being an unmirrorable Gemma-Terms direct download. Id matches the shipping dotless convention asserted in `tests/test_chat_prompt.cpp`; **trap avoided**, any id naming what it displaces (`gemma3-230m-floor`) would fire `model_is_gemma()` and emit `<start_of_turn>` tags.
- **Predeclared:** weights 146.3 MiB + the family's measured same-`n_ctx` runtime overhead (`lfm25-350m` +101 MB; `gemma3-270m` +127; `lfm25-1.2b` +114) → **247 ± 15 MB**. The 164 MB weights×1.12 figure is a host residency number and is **not** the gate. Decode: `DECODE_MODEL_v0` gives 184.5 but over-predicts this class 1.30× (`lfm25-350m`) to 1.46× (`gemma3-270m`), so the family-corrected screen is ≈142 — screening only. Arch `lfm2` is in the pin with all blobs predating it; no WS-B demand.

### H16.1d — WS-A · `minicpm5-1b` (`openbmb/MiniCPM5-1B`, self-converted Q4_K_M)

- **Claim:** MiniCPM5-1B Q4_K_M, rendered **no-think**, displaces `lfm25-1.2b-instruct` in the **balanced chat** role on capability-per-MB: H9 ≥ **6/8** at median decode ≥ **34.1 tok/s** (0.9× the incumbent's 37.9) and `peak_ws_mb` ≤ **811**. It does not target the fast tier, and it is not a reasoning candidate: if it reasons per turn it is disqualified, not re-scoped.
- **Measure:** **T2** the artefact is **self-converted at the pin** (`convert_hf_to_gguf.py` → f16 → `quantize.sh … Q4_K_M`), because the vendor GGUF stamps `tokenizer.ggml.pre = "llama-bpe"` while the dedicated MiniCPM5 pre-tokenizer landed two days later (PR #23384, present in the pin) — record a token-split diff of one fixed prompt, vendor vs self-converted. **T1** `xllama-cli --chat --greedy -m models/minicpm5-1b-Q4_K_M.gguf -t 6 -n 64` on 3 fixed prompts, `can_shift=` observed. **T3** `provision-models.sh minicpm5-1b` then `bench-xbox-ort.sh minicpm5-1b --runs 4 --n-predict 96 --out bench/results/phase16-gguf.csv`, median + min–max over 3 `run_index`. **T4** `eval-xbox-models.sh --models minicpm5-1b,lfm25-1.2b-instruct --out bench/results/phase16-h9.jsonl`.
- **PASS:** median decode ≥ **34.1** and `peak_ws_mb` ≤ **811** and H9 ≥ **6/8**. The alternate H16.1 branch (decode ≥ 49.3 = 1.3×, H9 ≥ 5/8) is available but not expected at a screening estimate of 41.1. Both branches additionally require **zero** `<think>`/`</think>` in visible output across the H9 suite.
- **FAIL:** H9 ≤ 5/8, or median decode < 34.1, or `peak_ws_mb` > 811. A peak between 811 and the 3584 envelope is FAIL, not pass — more memory at equal capability does not displace an 811 MB incumbent. Any chain-of-thought reaching the display makes H9 N/A (the suite caps generation at 16–80 tokens, H2 precedent) and the card FAIL; it is not re-filed as a reasoning candidate, because as a per-turn reasoner it inherits H2's bar and cannot clear it at this size.
- **Kill:** **no T3 session is booked** unless, by **2026-08-17**, host T1/T2 shows all three of — (a) **0 of 3** greedy runs emit `<think>` with the no-think prefill; (b) greedy output byte-stable across two runs on the self-converted Q4_K_M; (c) the renderer diff is ≤ **40 lines**, confined to `chat_prompt.{h,cpp}` plus the #169 `n_keep` call site. Any one missing → FAIL, "Do not reopen" within a day, zero console spent. **Editor:** clause (c) is graded by whoever writes the diff, so the diff and its line count are pasted into the decision-log entry — the number is not self-reported in prose.
- **Cost / id:** **new-renderer, staged.** (1) The no-think prefill is byte-identical to `empty_think_block() + "\n\n"`, but `model_is_qwen3("minicpm5-1b")` is false, so it needs a new id predicate feeding `ChatFormat::gen_suffix` in the ChatML branch. (2) The literal `<s>` the template emits while `add_bos_token=false` propagates into the GGUF needs a new `bos` field on `ChatFormat` plus #169 `n_keep` accounting (`render_system_prefix` must stay byte-identical to `render_prompt`); if T1 shows no output delta with vs without a hand-prepended `<s>`, half (2) is dropped and the cost falls to config-only — record that decision either way. Id `minicpm5-1b` falls to ChatML, the correct family (`<|im_start|>`, EOG 130073); it omits `thinking`/`a1b` on purpose, either would set `strip_thinking_content` and kill the KV snapshot; **never** spell it `…-llama` though `general.architecture` _is_ `llama` — that forces the Llama-3 renderer and echoes `<|eot_id|>` as text.
- **Predeclared:** 656.2 MiB × 1.12 = **735 MB** at `n_ctx` 2048 (KV ≈ 24.6 KB/token f16 → ~49 MB at 2048, ~98 at 4096, so 4096 lands ≈784 and still under 811). The factor is not extensible to the native 131072 ctx — 32k alone adds ~786 MiB. The vendor F16 is rejected: dominated 3.15× on bytes by Q4_K_M in the same repo. Both HF repos are Apache-2.0 and ungated but ship no LICENSE, so the OpenBMB text is copied into the catalogue entry at T5. `DECODE_MODEL_v0`'s 41.1 justified the download only.

### H16.7a — WS-G · `paddleocr-vl16-0.9b` (`PaddlePaddle/PaddleOCR-VL-1.6-GGUF`, Q8_0 LM + BF16 mmproj)

- **Claim (S-gate first, not tok/s):** there is **no VLM incumbent and no image input path**, so the claim is about the surface — an OCR turn on a console with no file picker, no camera and no text input is a product surface reachable without a new backend. Only if H16.7's S-gate PASSes does this candidate claim the newly created OCR axis (1316 MiB resident, **1474 MB** predicted peak), displacing nothing. **Nothing is downloaded before the S-gate is written and passed.**
- **Measure:** the S-gate is a written argument — (1) a named **shipping** (non-Dev-Mode) path that puts a document image into `LocalState`; (2) a new-C++-surface ledger; (3) a memory plan under the one-resident-model rule (`session_hub.h`). Only on PASS: `bench-xbox-ort.sh paddleocr-vl16-0.9b --runs 4 --n-predict 96 --out bench/results/phase16-wsg.csv` (3 `run_index`, median + min–max, `peak_ws_mb`) — **text-only**, it proves the projector is resident, it does not measure OCR. Accuracy has **no harness**: `eval-xbox-models.sh` is a text suite (H9 **N/A**) and host smoke cannot load an mmproj (zero `mtmd` hits across `CMakeLists.txt`, `src/`, `uwp/`, `scripts/`), so a 20-page CER harness is counted as cost.
- **PASS:** S-gate — one named non-Dev-Mode input path **and** a ledger of exactly **1** new C++ surface **and** a memory plan ≤3584 MB with one resident model. Then T3: `peak_ws_mb` ≤ **1600** (median of 3), decode ≥ **15 tok/s**, CER ≤ **5%** on the 20-page desk set.
- **FAIL:** S-gate — the only input path is Device Portal / Dev Mode, or the ledger exceeds 1 surface, or the 840.9 MiB BF16 projector plus a resident chat model needs two contexts. T3 — `peak_ws_mb` > 1600, decode < **12 tok/s** (a 1200-token page > 100 s), or CER > **10%**.
- **Kill:** by **2026-08-17** the S-gate must name one shipping input path and count ≤1 new C++ surface. The honest count today is **5**: mtmd/clip build integration (hand-maintained `ggml-uwp.vcxproj` + `CMakeLists.txt`), a fifth renderer (ERNIE-4.5 `User:`/`Assistant:`, EOG `</s>`, bare system) **plus an image slot `ChatFormat` does not have**, an image input path in `api-server.cpp`/`MainPage`, a two-publisher two-file catalogue entry, and the OCR/CER harness.
- **Editor:** this kill is **written as already firing**, so treat H16.7a as a **desk close, not an open test** — the deliverable is one decision-log line by 2026-08-17 recording that OCR has a model and no surface, at zero downloads and zero console sessions. Its value is that it retires the VLM question for the campaign and frees the E/F/G allocation before any console time is booked. Reopen only on a _new input path_, not on a better model.
- **Cost / id:** new-renderer, five surfaces. Id `paddleocr-vl16-0.9b` misses every dispatch substring; the accidental match to avoid is the backend name — any id or on-disk filename containing `llama.cpp`/`llamacpp` matches `llama` and silently renders Llama-3. Host dispatch keys off the **file basename**, so the provisioned file must retain `paddleocr`, the fifth renderer's substring; until that branch exists the id falls to ChatML, whose `<|im_end|>` is not in this sentencepiece vocab, and decode runs to `n_predict` every turn.
- **Predeclared (screening only):** Q8_0 LM 475.2 MiB + **official BF16 mmproj 840.9 MiB** — published at BF16 only, an irreducible floor no LM quant trades down, 74.6% of resident bytes, on a Zen2 core with no AVX512-BF16 path. 1316.1 MiB → 1474 MB at ×1.12. `DECODE_MODEL_v0` has **no VLM form**: 20.5 charges per-token cost for a tower that runs once per image; the OCR axis is **seconds per page**, not chat tok/s. Apache-2.0, ungated, but one entry spans two publishers and neither GGUF repo ships a LICENSE.

### H16.6a — WS-F · `whisper-base-ort-cpu-fp32` (`openai/whisper-base` via ORT GenAI)

- **Claim:** S-gate first — speech is the input surface a console with **no text entry** actually lacks. `whisper-base` displaces the **null** ASR surface and wins on **integration cost, not speed**: the pinned GenAI 0.14.1 DLL already contains `WhisperModel`/`WhisperProcessor`/`WhisperDecoderState`/`WhisperTokenizer` and `ort_genai_c.h` already declares `OgaLoadAudio`/`OgaCreateMultiModalProcessor`/`OgaProcessorProcessAudios`, so ASR needs no third backend and no new submodule. Falsifiable after the gate: median **RTF ≤ 1.0**, **WER ≤ 15%** on a fixed 5-clip fixture, `peak_ws_mb` ≤ 3584.
- **Measure:** S-gate, **no download** — (1) **microphone**: extend the existing `[caprec]` `ApiInformation::IsTypePresent` probe in `uwp/App.cpp` with `MediaCapture`/`AudioGraph`, plus a real 3 s `AudioGraph` capture (`<DeviceCapability Name="microphone"/>` in `AppxManifest.xml`), WAV read back from `LocalState`; (2) **backend**: the DLL/header symbol evidence, already at desk; (3) **surface spec**: a named transcript consumer plus residency under `session_hub.h`. No ASR harness exists — `bench-xbox-ort.sh` measures text decode — so a new ASR bench harness (none exists — building it is part of this card's cost) (RTF + `peak_ws_mb`, `--runs 4` → 3 `run_index` → `bench/results/phase16-wsf.csv`) and a 5-clip reference fixture are **part of the cost**.
- **PASS:** S-gate — mic type present **and** a 3 s console capture with RMS > 1e-3 **and** a named transcript consumer **and** a residency plan holding combined peak ≤ 3584 under one-resident-model. Post-gate — median RTF ≤ **1.0** at t ≤ 4 on the fixture, WER ≤ **15%**, `peak_ws_mb` ≤ 3584 across 3 recorded runs.
- **FAIL:** mic type absent, capture denied or silent under AppContainer, or a design needing a new submodule → S-gate FAIL, workstream closed. Post-gate: RTF > 1.0 (transcribes slower than speech), WER > 25%, or `peak_ws_mb` > 3584. Host RTF and host RSS are not results; only the console CSV counts.
- **Kill:** **console kill in ONE Dev-Mode session by 2026-08-24** — mic type absent, or a 3 s capture with RMS < 1e-4 → close WS-F that day with a "Do not reopen" line and a new **microphone** section in `uwp-constraints.md`. Zero model downloads before it fires.
- **Editor:** this card takes the **single WS-E/F/G console session**, and it is the only one of the three whose gate the console can settle — WS-G closes at desk and WS-E is host-measured. The card's second (desk) kill, "no GenAI-format artefact loads under 0.14.1", is struck as a gate: the card itself says `genai_config.json` and `audio_processor_config.json` must be hand-written and the published `onnx-community` export is an unloadable transformers.js artefact, so that kill can only fire against our own unfinished work. **The mic probe alone decides WS-F**, and artefact work does not start until it PASSes.
- **Cost / id:** **new-renderer**, and the survey's `new-dependency` label is wrong — nothing new is vendored. New code, all ours: (1) a WinRT `AudioGraph` capture path writing 16 kHz mono PCM into `LocalState`; (2) an `OrtAsrSession` beside the text-only `OrtSession` in `src/bridge/` — multimodal processor plus the forced decoder prompt (50258 / language / 50359 / 50364), `begin_suppress_tokens [220, 50257]`, eos 50257, a token-array prepend rather than a renderer; (3) the two hand-written configs, the model builder being frozen at Qwen3/Gemma3 (that is H16.3's finding about the **builder**, not the runtime); (4) the ASR bench harness. It degrades to new-dependency only if mic capture needs a non-WinRT path. Id `whisper-base-ort-cpu-fp32` matches no `chat_format_for()` predicate; `xllama-whisper`/`llama-whisper` were rejected because `llama` binds Llama-3 and `chat_format_for()` is called on the model **path** at some call sites. Unmatched ids fall to ChatML **silently**, so a `check-coherence.py` assertion must fence this id from `chat_format_for()` and `dml_text_model_ok()`.
- **Predeclared:** the artefact we would load is a GenAI-format **fp32** export (~378 MiB) → **424 MB** at ×1.12, not the 73 MiB `onnx-community` int8 pair; if an int8 GenAI export is produced, re-predeclare at ~82 MB **before** measuring. Concurrent with the 320 MB default chat model that is ~744 MB — memory is not the binding constraint, the microphone is. No `est_decode_tok_s`: `DECODE_MODEL_v0` is a fit to GGUF dense decoders, the product metric is RTF. Licence: `onnx-community/whisper-base` declares none at all; upstream is apache-2.0 by card tag only — ungated, so **do not re-host on `models-v1`**; fetch at runtime or re-export from an apache-2.0-tagged source.

### H16.5a — WS-E · `embed-nomic-v15` (`nomic-ai/nomic-embed-text-v1.5`, Q8_0)

- **Claim:** WS-E has no incumbent, so this is a **surface** claim first: H16.5's S-gate must PASS in writing before anything is downloaded. Conditional on that, `nomic-embed-text-v1.5` Q8_0 is WS-E's single entrant — chosen over the co-shortlisted `granite-embedding-english-r2` and `multilingual-e5-small` on **cost and storage, not score**: config-only load, 156 MB projected peak, and Matryoshka truncation to 256-dim shrinks the persisted index 3× at MTEB 62.28.
- **Measure:** no embedding harness exists (no `/v1/embeddings` route, no vector store, no bench script), so **building one is this card's cost**. S-gate: a written surface spec, no script, no download. **T1′** load-only smoke via `llama-embedding` on the host — the standard T1 (`--chat --greedy -n 64`) is invalid for a bidirectional encoder (no decode, no `can_shift`, and `chat_format_for()` would emit ChatML). **T2′** new a new embedding eval harness (none exists — building it is part of this card's cost) → `bench/results/phase16-embed.jsonl`, 50 fixed query/doc pairs, recall@5 at 768-dim vs Matryoshka-256, cosines byte-stable over two runs. **T3** does not exist: `bench-xbox-ort.sh`'s 17-column decode schema cannot time an encoder.
- **PASS:** S-gate — a named in-product consumer plus a memory plan in which the encoder is **never co-resident** with the chat model (`session_hub.h`). Then T2′ — recall@5 at 256-dim ≥ 0.95× the same model's 768-dim recall@5, cosines byte-stable, load peak ≤ 156 MB, and the model cost stays config-only.
- **FAIL:** no named consumer; the design needs a second resident `llama_context`; 256-dim recall@5 < 0.95× 768-dim, which voids the storage argument the candidate is chosen on; the embedding path reuses `chat_format_for()` and wraps text in `<|im_start|>`; or the consumer's documents exceed the **2048-token** context this GGUF declares (no rope-scaling keys — v1.5's advertised 8192 window is not in this file).
- **Kill:** desk, date-bounded — if the surface spec (named consumer + non-co-resident memory plan) is not written **and accepted** by **2026-08-24**, WS-E closes as "capability in the pin, no product surface": zero downloads, zero console. T1′ kill — if the path needs a second resident `llama_context` or any new C++ backend, close.
- **Editor:** two amendments. (1) A kill that fires by inaction cannot discriminate, so the spec is **assigned an owner when this card is filed**; unowned by 2026-08-17 it closes early rather than expiring on 08-24. (2) WS-E is **desk- and host-only this campaign** — there is no console encoder path and building one is out of budget — so it does **not** consume the single E/F/G console session, which goes to H16.6a's mic probe. A console `peak_ws_mb` for an encoder is a Phase 17 item.
- **Cost / id:** the model load is config-only (`nomic-bert` in the pin since PR #5468, fused `attn_qkv` accepted, `token_type_count=2`, no KV cache). Everything else is new product code the enum does not capture: an embeddings entry point that bypasses `chat_format_for()`, the mandatory `search_document: `/`search_query: ` prefixes (inside the embedding call, never in a renderer), app-side Matryoshka truncate+renormalise (llama.cpp has none), a store, the eval harness, and an Apache-2.0 notice we author (no LICENSE in either repo). Id `embed-nomic-v15` matches no dispatch substring; the `embed-` prefix exists so one literal guard can bar it from chat routing. **Accidental match avoided:** this class invites gemma-style names — the killed `embeddinggemma-300M` would have fired the Gemma renderer on an encoder.
- **Predeclared:** Q8_0 = 146,146,432 B = 139.4 MiB → **156.1 MB** at ×1.12, 4.4% of the gate. Q8_0 over the Q4_K_M standard is a justified departure: on a 137M-param bidirectional encoder Q4 rounding distorts cosine geometry for a 59 MiB saving. `DECODE_MODEL_v0` **does not apply** — an encoder has no autoregressive decode, and the survey's 193.7 tok/s is to be read as absent. `gated: false`, anon HEAD → 302. Trap: the sibling `nomic-embed-vision-v1.5` is CC-BY-NC; the text model is not.

### SmolLM3-3B — WS-A, dense 3B chat, costs a 5th renderer (**no id assigned — closed at desk 2026-08-10**)

- **Claim:** `ggml-org/SmolLM3-3B-GGUF` Q4_K_M (1827 MB weights, predicted peak 2046 MB), rendered through a new 5th renderer that forces `Reasoning Mode: /no_think`, displaces **Llama-3.2-3B Q3_K_S** (14.2 tok/s, 1824 MB, H9 5/8) in the peer dense-3B chat slot on capability-per-byte. Because the renderer is real engineering, the bar is raised past that slot: only **H9 8/8** — which also beats the quality incumbent **LFM2-2.6B** (18.4 tok/s, 1623 MB, H9 7/8) — buys it. Not a coding claim. The scout record's 64k-context pitch is struck: 36 full-attention layers × 4 KV heads × head_dim 128 = **72 KiB/token** f16 KV, so 64k is 4608 MB of KV before a weight byte loads; envelope max ~20–24k, bench at the catalogue default n_ctx 4096. `27000/1826.6 = 14.8` is DECODE_MODEL_v0 screening only and never appears in a product claim.
- **Measure:** T1 `./build/bin/xllama-cli --chat --greedy -m models/SmolLM3-Q4_K_M.gguf -t 6 -n 64` (arch `smollm3`, stop `<|im_end|>`, `can_shift=1`; host tok/s and host RSS stay in T1) **plus a renderer A/B** — 5 fixed greedy prompts, ChatML fallback vs the new branch, diffed for `<think>` and for a literal `## Metadata` line, with a `tests/test_prompt_budget.cpp`-style unit test that the branch fires for id `smollm3-3b`. T3 `./scripts/provision-models.sh smollm3-3b` then `./scripts/bench-xbox-ort.sh smollm3-3b --runs 4 --n-predict 96 --out bench/results/phase16-gguf.csv` (3 recorded `run_index`, median + min–max, `peak_ws_mb` the only admissible peak). T4 `./scripts/eval-xbox-models.sh --models smollm3-3b,llama32-3b --out bench/results/phase16-h9.jsonl` — H9 is admissible **only because /no_think is the shipped default**; under /think the 16–80 token cap makes it N/A (H2).
- **PASS:** `peak_ws_mb` ≤ **2200** (≤ +250 MB vs 1824, inside 3584) **and** median decode ≥ **12.8** (0.9 × 14.2), spread ≤ 10% of median **and** H9 = **8/8** **and** zero `<think>` / `## Metadata` / `Reasoning Mode` text in all 8 H9 transcripts and 3 bench outputs. Ship extras are mandatory, not optional: an Apache-2.0 text **we author** in the catalogue naming HuggingFaceTB and ggml-org (neither repo ships a LICENSE file — the grant is card frontmatter only, s4(a)/(d) still binds), and no 64k claim in docs or store copy.
- **FAIL:** peak > 2200; or decode < 12.8; or **H9 ≤ 7/8** — 7/8 fails _by design_, because it ties LFM2-2.6B while being 4.2 tok/s slower and 423 MB heavier, so no product role is won and T5 cannot be satisfied; or any CoT/metadata leak; or the renderer needs more than one predicate + one branch (a **live** per-turn date breaks the KV-snapshot prefix diff — that is a cost FAIL, not a feature).
- **Kill:** (1) **by 2026-08-17, at T1, no download beyond the smoke file** — with the frozen-date header and the `<think>\n\n</think>\n` prefill (ONE trailing newline; `empty_think_block()` emits two), if ≥1 of the 5 fixed prompts still shows a `<think>` trace or a `## Metadata` line, close: a 3B reasoning every turn is H2 restated (~4× perceived latency), and the fix — putting `thinking` in the id — also disables KV snapshots. (2) **by 2026-08-24** — WS-A has ≤4 console slots; if the four have taken them, SmolLM3 closes as "cost not bought" with no bench. _(The card named the four takers as Qwen3.5-2B / Maincoder-1B / LFM2.5-230M / Supra2-100M; the editor dropped Supra2 as a duplicate bet and slot 4 went to MiniCPM5-1B. The operative trigger is the slot count, not the names — see Disposition.)_
- **Catalogue id:** `smollm3-3b`. No `gemma`/`llama`/`phi`/`qwen3`/`a1b` substring (the double-l is followed by `m`, not `ama`), so it falls through to ChatML today — precisely the silent-wrongness; the `model_is_smollm3()` branch goes **before** that fallback. `thinking` is deliberately excluded: it would strip CoT _and_ disable the KV snapshot for a mode we ship OFF. Exposing /think later needs a second entry `smollm3-3b-thinking` and a new card.
- **Cost:** new-renderer — one predicate, one branch (metadata header merged with the system prompt + one-newline think prefill, frozen date), one unit test, one `model-matrix.md` row, one authored licence file. No pin bump (`LLM_ARCH_SMOLLM3` and `src/models/smollm3.cpp` exist at 6d5a910c5; `n_swa=0`, context shift stays on). Unlike H16.1d's renderer, this one **cannot collapse to config-only**: the metadata block is mandatory and without it CoT leaks (T0 run B).
- **Disposition — dropped, no id, zero console spent.** Kill 2 fires on its operative clause: WS-A's four slots are all allocated (H16.1a–H16.1d). Because slot 4 went to a DEGRADE rather than to the fourth clean PASS the card anticipated, the drop was not left to rest on that wording — the head-to-head was run. Against **H16.1d**: 2046 MB vs 735, a renderer that cannot collapse vs one that can, and a self-set PASS bar of H9 **8/8** — a score no shipping model has recorded — vs the ordinary WS-A bar. It loses on every axis, so displacement is refused on the merits and no budget exception is requested. Two facts are carried out of the close and into §F so they are not re-bought: the **72 KiB/token** KV arithmetic that kills the 64k pitch for any smollm3-class candidate, and the id trap — `smollm3-3b` matches no renderer predicate, so any smollm3 GGUF provisioned without `model_is_smollm3()` leaks chain-of-thought verbatim into the UI (#223, polarity reversed).

### Editor findings about the cards

The editor was asked to criticise the cards, not only to typeset them. Four
findings that change the work:

1. **The cards disagree about the peak model.** H16.1a and H16.1d project
   `weights × 1.12`; H16.1c falsifies that below ~400 MB and uses a measured
   fixed cost; H16.1b shows it 8.6% optimistic once KV is counted. Resolution:
   **H16.1c runs first**, its `peak_ws_mb` settles the factor, and every other
   projection is provisional until it lands. The delta goes into
   [`../bench/README.md`](../bench/README.md).
2. **Three kill criteria could not fire as written.** H16.7a's is a close
   disguised as a test (it is already satisfied). H16.5a's fires by inaction, so
   without a named owner it does not fire — it expires. H16.6a's WER/RTF target
   is unmeasurable this campaign: no fixture, no harness, no loadable artefact —
   struck, leaving the microphone probe to decide WS-F alone.
3. **One claim was too weak to be worth measuring.** "H9 ≥ `gemma3-270m` − 1" at
   the floor tier, where the incumbent has no recorded H9 and likely scores 1–2/8,
   is satisfiable at 0/8. Replaced with an absolute kill at ≤1/8.
4. **A discrepancy the editor put on the record rather than papering over.**
   SmolLM3's kill 2 names four slot-takers that do not match the four actually
   assigned; read by the letter it does not fire, read by its operative clause it
   does. The head-to-head was run anyway, so the drop does not rest on a wording
   accident.

### Additions to `model-matrix.md` §F — **applied 2026-08-10**

The campaign has closed, so these are now **live in**
[model-matrix.md](model-matrix.md) §F. Kept here as the derivation:

**Append to the Phase 16 rows of §F (desk 2026-08-10). One new row — the other Phase 16 rejections are unchanged:**

| Candidate                  | Decision                 | Reason                                                                                                                                                                                                                                                                            |
| -------------------------- | ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| SmolLM3-3B GGUF (ggml-org) | reject — cost not bought | T0 run B **DEGRADE**. WS-A's ≤4 slots all allocated; loses the 4th head-to-head to `minicpm5-1b` (2046 MB vs 735; the only WS-A renderer that cannot collapse to config-only; self-set PASS bar of H9 **8/8**, catalogue best is 7/8). Card's kill 2 fired at desk, zero console. |

**And the first entry in the Phase 16 "Do not reopen" list** (`docs/phase16-model-scouting.md`), replacing the "empty until this campaign produces a FAIL" placeholder:

- **`SmolLM3-3B` — closed at desk 2026-08-10, "cost not bought."** Reopening needs a _free_ WS-A slot and a renderer budget, not merely a better model; the card was judged, not lost. Two facts survive the close and must not be re-bought:
  1. **64k context is arithmetically outside the envelope.** 36 full-attention layers × 4 KV heads × head_dim 128 = **72 KiB/token** f16 KV, so 64k ctx alone is **4608 MB** — past the 3584 MB working-set gate before a single weight byte loads. Envelope max is ~20–24k; the catalogue default n_ctx 4096 stands. No future card re-buys the 64k pitch for this family.
  2. **`smollm3-3b` matches no renderer predicate.** No `gemma`/`llama`/`phi`/`qwen3`/`a1b` substring (the double-l is followed by `m`, not `ama`), so it falls through to the ChatML default, which emits neither the mandatory `## Metadata` block nor `Reasoning Mode: /no_think`; `model_is_thinking()` is also false for the id. Any smollm3-class GGUF provisioned without a new `model_is_smollm3()` branch **leaks chain-of-thought verbatim into the chat UI** — #223 with the polarity reversed. This is a live trap, not a closed one.

## T1 host smoke — results (2026-08-10)

Host: `build/linux-release/bin/xllama-cli`, `-t 6 -n 64 --chat --greedy`. Per
Method rule 3 **no host timing appears here** — T1 proves load, template, stop
and determinism, nothing else.

| Candidate      | `arch`      | `n_swa` | Context shift | Stop            | Greedy byte-stable | T1                     |
| -------------- | ----------- | ------: | ------------- | --------------- | ------------------ | ---------------------- |
| `lfm25-230m`   | `lfm2`      |       0 | OK            | EOG @31 tok     | yes                | **pass**               |
| `qwen35-2b`    | `qwen35`    |       0 | **disabled**  | EOG @26 tok     | yes                | **pass**, bar raised   |
| `maincoder-1b` | `maincoder` |       0 | OK            | EOG @28 tok     | yes                | **pass**               |
| `minicpm5-1b`  | `llama`     |       0 | OK            | **never stops** | yes                | renderer work required |

### H16.1a — kill does not fire, and the naming trap is settled on the artefact

The fetched file is exactly **1,280,835,840 B**, so the mmproj-confusion kill
cannot fire; and the byte count was available from the HF tree API _before_
downloading, which is how it should have been written. `print_info: arch =
qwen35` on the real GGUF closes the `model_type` vs `LLM_ARCH_NAMES` question
that WS-B was contingent on — the pin loads it, and no pin bump is motivated.
No `<think>`, `<|vision_start|>`, `<|image_pad|>` or `<|vision_end|>` on three
fixed prompts.

**Context shift is disabled**, as the card predicted from the imrope path
(`llama-kv-cache.cpp` returns false when `n_pos_per_embd() > 1`). Measured, not
inferred: `llama-completion --context-shift` emits `KV cache shifting is not
supported for this context`. So H16.1a's PASS bar is the raised branch — H9
**+2/8** over `qwen3-1.7b`, and `validate-console.sh longchat` must stay PASS on
the #169 fail-fast+trim path.

### H16.1b — the window holds, with the margin measured

`fit_prompt` requires `tokens + max(n_predict, 250) + 1 ≤ n_ctx`, i.e. **≤ 1791
prompt tokens** at `n_ctx` 2048 with a 256-token reply. Measured with the
model's own tokenizer:

| Source file                    |  Bytes | Rendered prompt | Fits ≤1791 |
| ------------------------------ | -----: | --------------: | ---------- |
| `src/bridge/prompt_budget.cpp` |  3,812 |       1,000 tok | yes        |
| `src/bridge/chat_prompt.cpp`   | 11,232 |       2,923 tok | no         |

The boundary is **~6.9 KB of source**, ~4.0 chars/token measured. A ~1,500-token
file (~6 KB) fits with roughly 290 tokens of margin, so **the kill does not
fire** — but the margin is thin enough to be worth stating, and one thing the
card did not: the incumbent `qwen25-coder-1.5b` ships at **`n_ctx` 4096**, twice
this card's 2048. Comparing a coding model against it at half the window is
unequal on the axis coding cares about most. Either the card moves to 4096 (peak
rises ~50 MB by its own KV arithmetic) or the T4 comparison carries the caveat.

### H16.1d — both halves of the renderer cost are required, measured

The card allowed that the `<s>` half might be dropped, collapsing the cost to
config-only. **Measurement falsifies that.** Isolating the render with
`llama-completion -sp` on a hand-built ChatML prompt:

| Prompt                                                          | Result                                                                         |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| ChatML without `<s>` (what the app renders today)               | model emits `<\|im_end\|>` immediately — an empty turn                         |
| ChatML with `<s>` prepended                                     | correct answer, but opens `<think>` and reasons                                |
| ChatML with `<s>` **and** the `<think>\n\n</think>\n\n` prefill | **0 of 3** prompts leak `<think>`, coherent answers, clean `<\|im_end\|>` stop |

So kill clause **(a) passes**, and it passes _only_ with both halves. The vendor
GGUF stamps `tokenizer.ggml.pre = 'llama-bpe'` and `add_bos_token = false` as the
card predicted, while the template emits `bos_token` — that mismatch is the whole
failure, and the raw completion path (`The capital of France is Paris…`) proves
the weights and quantisation are fine.

Consequence: **no T3 session is booked for `minicpm5-1b`** until clause (c) — the
≤40-line renderer diff — is written and its line count pasted into the decision
log. That is its own card's condition, and it is now the only WS-A candidate
carrying unbought engineering.

### A seventh card defect, same class as the first six

`fit_prompt` lives in the UWP app only (`uwp/api-server.cpp`, `uwp/MainPage.cpp`);
`xllama-cli` never calls it. H16.1b's kill is phrased as "does not fit
`fit_prompt`", which the CLI cannot evaluate any more than it can print
`can_shift`. It was evaluated here by computing the bound from
`src/bridge/prompt_budget.cpp` and measuring the token count with
`llama-tokenize` — which is the honest substitute, and what the card should say.

### A finding about the working tree, not about any model

The `llama.cpp` submodule was carrying `patches/0001-uwp-appcontainer-guards.patch`
applied. That patch **breaks the Linux build**: `llama-mmap.cpp` guards on
`WINAPI_FAMILY_PARTITION(...)`, which a Linux preprocessor cannot parse. The patch
is for UWP builds only and CI never applies it to the Linux job. T1 therefore
required reversing it, building, and re-applying — the submodule was restored
byte-identical afterwards.

Worth recording separately: the tree also carries an **uncommitted, undocumented**
change to `ggml/src/ggml-cpu/ops.h` that is _not_ in `patches/`. It excludes
clang-cl from `__cpp_lib_hardware_interference_size` so uwp-crossbuild can compile
ggml without a private CRT patch. It is harmless on Linux and was preserved, but
it is real crossbuild work living outside `patches/`, where the next person will
not find it.

## T3 console bench — results (2026-08-10, Series S, t6)

Raw evidence: `bench/results/phase16-gguf.csv`, 3 recorded `run_index` each after
the discarded warmup, `n_predict` 96, `n_ctx` 2048, `standard-512` prompt.
`peak_ws_mb` is the only admissible peak.

| Card   | Model          | Prefill (median) | Decode (median) | Spread        | `peak_ws_mb` | Load ms | Verdict             |
| ------ | -------------- | ---------------: | --------------: | ------------- | -----------: | ------: | ------------------- |
| H16.1c | `lfm25-230m`   |            740.2 |      **119.77** | 119.55–120.06 |      **241** |     625 | **T3 PASS**         |
| H16.1a | `qwen35-2b`    |             64.1 |           19.25 | 19.24–19.27   |         1421 |    6949 | **FAIL**            |
| H16.1b | `maincoder-1b` |            152.9 |           33.49 | 33.47–33.51   |          843 |    3334 | **FAIL as written** |

`minicpm5-1b` (H16.1d) was **not benched**: its own kill gates the booking of a
T3 session on a renderer diff that has not been written (see T1 above).

### H16.1c — passes the memory half decisively, T4 outstanding

Predicted 247 ± 15 MB, measured **241** — inside the band, so the campaign's stop
condition did not trigger. Against the two incumbents it could displace:

|                                           |     decode | peak MB |
| ----------------------------------------- | ---------: | ------: |
| `lfm25-230m` (candidate, `unsloth` build) | **119.77** | **241** |
| `gemma3-270m` (floor incumbent)           |       76.8 |     368 |
| `lfm25-350m` (default chat)               |       94.9 |     320 |

It beats both on **both** axes — 1.56× and 1.26× the decode, at 127 MB and 79 MB
less peak. **These are the `unsloth` artefact's numbers**; the build that
actually shipped is LiquidAI's, re-measured at 119.17 (1.55× the floor) with the
same 241 MB peak — see "One artefact-identity correction" below. The shipping
figures are the ones every other document quotes. The capability half is unproven: H9 (T4) has not been run, and the
card's absolute kill (H9 ≤ 1/8 closes it) is still live.

### H16.1a — FAIL on both halves of its own bar

PASS required `peak_ws_mb` ≤ 1398 **and** decode ≥ 19.6. Measured **1421** and
**19.25**: over by 23 MB, under by 0.35 tok/s. Thin margins, but they were
predeclared precisely so they could not be renegotiated afterwards, and it is a
double miss rather than a borderline single one. The card is explicit that a peak
between 1398 and the 3584 envelope is still a FAIL — that is a catalogue
addition, not a displacement of `qwen3-1.7b`. No T4 was run; the session is spent
and the result is recorded.

Load time is worth noting for whoever revisits this: **~6.9 s**, against 0.6 s for
the 230M and 3.3 s for the 1B.

### H16.1b — memory passes, speed misses a self-set bar by 1.3%

PASS required decode ≥ **33.9** with peak ≤ 900. Measured **33.49** and **843**.
The memory half passes with 57 MB to spare; decode lands at **1.283×** the
incumbent `qwen25-coder-1.5b` (26.1) against a bar of 1.3×.

This is a FAIL **as the card is written**, and it is recorded as one. But the card
set a stricter conjunction than H16.1 itself allows: H16.1's branch 1 admits a
candidate at decode ≥ 0.9× incumbent when H9 wins, and 33.49 clears that by a
wide margin. Whether to re-open H16.1b at the softer branch is a judgement about
what the coding role is for, not a measurement — and it is moot until someone
builds `bench/eval/phase16-coding.json`, which does not exist and which its own
T4 requires.

### The peak-model dispute, settled: `× 1.12` is optimistic

The editor recorded that the cards disagreed about how to project peak, and made
H16.1c's measurement the arbiter. Three measurements now exist:

| Model          | Weights MB | Peak MB | Overhead | Overhead % | `×1.12` predicted |    Error |
| -------------- | ---------: | ------: | -------: | ---------: | ----------------: | -------: |
| `lfm25-230m`   |        146 |     241 |       95 |        65% |               164 |  **−77** |
| `maincoder-1b` |        641 |     843 |      202 |        32% |               718 | **−125** |
| `qwen35-2b`    |       1222 |    1421 |      200 |        16% |              1368 |  **−53** |

**`× 1.12` under-predicts in all three cases**, by 4% to 47%. It is not a runtime
constant: the overhead is a large fixed floor (~95 MB at 146 MB of weights,
dominated by the runtime and compute buffers) plus an architecture-dependent KV
term. The existing catalogue shows the same split — at nearly identical weights,
`lfm2-2.6b` carries 132 MB of overhead (9%) while `llama32-3b` carries 353 MB
(24%), because a hybrid recurrent stack holds far less KV than dense attention.

Two consequences, both recorded rather than acted on here:

1. **Every remaining `× 1.12` projection in this campaign is a lower bound.**
   `minicpm5-1b`'s predeclared 735 MB should be read as "at least 735"; its own
   T1-measured overhead class (dense `llama` arch) suggests closer to 840.
2. The factor's provenance (`docs/phase7-hypotheses.md`, from two dense 3B-class
   models) explains why it holds there and fails at 146 MB. A replacement rule
   belongs in [`../bench/README.md`](../bench/README.md) as a screening aid, with
   the same confinement `DECODE_MODEL_v0` has — not in a product claim.

### `DECODE_MODEL_v0` against measurement

| Model          | Predicted `27000/weightsMB` | Measured |             Ratio |
| -------------- | --------------------------: | -------: | ----------------: |
| `lfm25-230m`   |                       184.6 |   119.77 | over by **1.54×** |
| `maincoder-1b` |                        42.1 |    33.49 | over by **1.26×** |
| `qwen35-2b`    |                        22.1 |    19.25 | over by **1.15×** |

(The `lfm25-230m` row is the `unsloth` measurement, which is what the screening
ran against; the shipping build's 119.17 moves the ratio to 1.55× and changes
nothing.) The heuristic over-predicts everywhere, and worst at the small end — the editor
anticipated 1.30–1.46× on floor rows and measured is 1.54×. It did its job as a
pre-download screen (it rejected the whole 4B class, and the direction of its
error means that rejection was conservative in the right way: real decode is
_lower_ than it promised). It must not graduate into any claim.

## T4 capability (H9) — results (2026-08-10)

`scripts/eval-xbox-models.sh --models lfm25-230m,gemma3-270m,lfm25-350m --out bench/results/phase16-h9.jsonl`
— the deterministic 8-task suite through the LAN API, temperature 0, seed 42.

**The harness was validated before the result was read.** `lfm25-350m` was run as
a control and scored **4/8**, reproducing exactly the value recorded in
[model-matrix.md](model-matrix.md) §A1. A capability comparison whose harness has
not been shown to reproduce a known score is not evidence.

| Model                                | H9                       | Median latency |
| ------------------------------------ | ------------------------ | -------------: |
| `lfm25-350m` (control, recorded 4/8) | **4/8** ✓ reproduced     |         507 ms |
| `gemma3-270m` (floor incumbent)      | **3/8** — newly measured |         392 ms |
| `lfm25-230m` (candidate)             | **2/8**                  |         406 ms |

`gemma3-270m`'s H9 had never been recorded (§A1 carries `—`). This run fills that
gap, and it is what makes H16.1c's capability bar computable at all — the editor
had flagged the original "≥ incumbent − 1" as unmeasurable for exactly this
reason.

| Task                   | `lfm25-230m` | `gemma3-270m` | `lfm25-350m` |
| ---------------------- | ------------ | ------------- | ------------ |
| `it_exact_capital`     | fail         | PASS          | PASS         |
| `arithmetic_multistep` | fail         | fail          | fail         |
| `json_extraction`      | fail         | fail          | fail         |
| `grounded_qa`          | PASS         | PASS          | PASS         |
| `constrained_summary`  | **PASS**     | fail          | PASS         |
| `translation_fidelity` | fail         | fail          | fail         |
| `multi_turn_memory`    | fail         | PASS          | PASS         |
| `unknown_from_context` | fail         | fail          | fail         |

The two models do not fail in the same places: the candidate wins
`constrained_summary` and loses `it_exact_capital` and `multi_turn_memory`. Three
tasks (`arithmetic_multistep`, `json_extraction`, `translation_fidelity`,
`unknown_from_context`) defeat every model in this weight class, which is a
property of the tier, not of any candidate.

### H16.1c — PASS, at the exact capability floor

| PASS condition      | Bar                               | Measured           |     |
| ------------------- | --------------------------------- | ------------------ | --- |
| median `peak_ws_mb` | ≤ 300                             | **241**            | ✓   |
| median decode       | ≥ 100 (≥1.3× floor)               | **119.17** (1.55×) | ✓   |
| H9                  | ≥ H9(`gemma3-270m`) − 1 = **2/8** | **2/8**            | ✓   |

All three predeclared conditions hold, and none of the FAIL conditions fires
(peak > 320, decode < 95, or H9 two tasks below the incumbent). **`lfm25-230m` is
the campaign's first and only T5-eligible candidate.**

State the trade plainly rather than glossing it: the candidate is **worse than
the incumbent on capability** — 2/8 against 3/8 — and passes only because the
card predeclared that one task of capability was purchasable for 1.55× the decode
and 127 MB less peak. That was written before the measurement, which is the whole
point of writing it first; but anyone reading "PASS" should know it landed
exactly on the floor, with no margin.

### What T4 did not cost

`qwen35-2b` and `maincoder-1b` failed at T3 and no T4 was run against them —
their sessions are closed. `minicpm5-1b` never reached T3. WS-A therefore spent
**three** of its four console sessions, and the fourth remains unspent because
H16.1d is gated on engineering that has not been written.

## Campaign outcome (2026-08-10)

**Shipped: one.** `lfm25-230m` (H16.1c) is in the catalogue as the **floor**
tier. Every other card is closed, blocked, or deferred with a named cost. The
campaign spent **3 of its ≤9 console sessions**.

| WS   | Card       | Outcome                                                                           |
| ---- | ---------- | --------------------------------------------------------------------------------- |
| WS-A | **H16.1c** | **SHIPPED** — 119.2 tok/s / 241 MB / H9 2/8; displaces `gemma3-270m` as the floor |
| WS-A | H16.1a     | **FAIL, measured** — 1421 MB > 1398 and 19.25 < 19.6 tok/s                        |
| WS-A | H16.1b     | **FAIL as written** — 33.49 < 33.9 tok/s (memory passed at 843 ≤ 900)             |
| WS-A | H16.1d     | **deferred** — needs both renderer halves; no console session spent               |
| WS-A | SmolLM3-3B | closed at desk, "cost not bought"                                                 |
| WS-B | H16.2      | closed, never triggered — **no pin bump**                                         |
| WS-C | H16.3      | **closed** — its kill fired at desk                                               |
| WS-D | H16.4      | **closed** — its kill fired at desk                                               |
| WS-E | H16.5      | blocked — the S-gate has no owner                                                 |
| WS-F | H16.6      | blocked — the microphone probe is unwritten UWP work                              |
| WS-G | H16.7a     | **closed** — S-gate FAIL, 5 new C++ surfaces against a ≤1 budget                  |

### The two workstreams closed by their own predeclared kills

**WS-C (H16.3) — "ORT text surface saturated at 0.14.1."** The kill was: zero
architectures inside the frozen builder set at ≤500M. T0 established exactly
that, with evidence — the GenAI model builder is frozen at Qwen3/Gemma3 and
**every** 2026 sub-4B model moved off those architectures (Qwen3.5 → `qwen3_5`,
LFM2.5 → `lfm2`, SmolLM3 → `smollm3`, Falcon-H1 → `falcon_h1`), while the only
ONNX text repo in the 360–500M fp16 window is the already-catalogued
`gemma3-270m`. Reopening requires a **GenAI version bump**, which belongs to
[vendor-lifecycle-plan.md](vendor-lifecycle-plan.md), not to model scouting.

**WS-D (H16.4) — no candidate exports to a 3-component ONNX graph.** SDXL-Turbo
is excluded by the 2 GB protobuf limit rather than the GPU budget (UNet external
data 5.1 GB); SANA-Sprint, DMD2 and Tiny-SD have no usable ONNX export at all;
SD3.5 and Flux are monolithic DiT, a new backend and out of scope. The incumbent
`sd-turbo-fp16` stands unchallenged.

### The two blocked on a decision, not a measurement

**WS-E (embedding)** produced a verified, config-only candidate —
`nomic-ai/nomic-embed-text-v1.5`, arch `nomic-bert` which the pin carries, 156 MB
at Q8_0, no conversion and no new backend. Its S-gate is the blocker and it is
**not technical**: nobody has named who consumes the vectors, where they live, and
how a second `llama_context` survives the one-resident-model rule. The card's kill
fires by inaction, so without an owner it does not fire — it expires. This is the
single highest-value item the campaign leaves open, and it is a product decision.

**WS-F (ASR)** has the strongest motivation in the repo — the console has no text
input — and the cheapest decisive test: does the Xbox AppContainer grant
`MediaCapture` / `AudioGraph`? That is a small change to the existing `[caprec]`
probe in `uwp/App.cpp`, and the answer becomes a permanent
[uwp-constraints.md](uwp-constraints.md) entry either way. It was not written. T0
already settled the backend half: GGUF ASR is empty by construction — the pin's
Whisper encoder exists only as an mmproj audio tower for other projectors, with no
`whisper` arch and no text decoder — so any ASR route is ORT GenAI with a
purpose-built artefact.

### Corrections applied to the cards

Seven instructions were not executable as written. Each is recorded where it was
found; the substitutions actually used were `deploy.sh upload-dir` for
non-catalogue provisioning, `llama-completion --context-shift` plus
`print_info: n_swa` for the `can_shift` line the CLI never prints,
`build/linux-release/bin/llama-quantize` for `scripts/quantize.sh`,
`llama-completion --jinja` for a no-think prefill that does not exist yet, a
computed `fit_prompt` bound plus `llama-tokenize` for calling `fit_prompt` from
the CLI, a pre-download byte check for H16.1a's post-download one, and
`build/linux-release/` for the CLI path.

The pattern is worth naming because it will recur: **cards written by an agent
that cannot execute name tools by what they sound like they do.** All seven named
a script that really exists; none of them did the job the card assumed. The
cheapest guard is to treat the first execution of a card as a test of the card,
not only of the candidate.

### One artefact-identity correction, caught before shipping

The first bench and H9 ran against `unsloth`'s LFM2.5-230M Q4_K_M. The
**shippable** artefact is LiquidAI's own build, because that repo carries the
`LICENSE` that LFM Open License §4 requires to travel with the weights. The two
files are 352 bytes apart and **produce different greedy output** — different
quantisations of the same model, not copies. Everything was re-measured on the
shipping artefact: decode 119.17 (against 119.77), `peak_ws_mb` **241**
(identical), H9 **2/8** (identical). The superseded rows are kept as a quantiser
A/B in `bench/results/phase16-quantiser-ab.csv`, which is now the evidence that
the distinction is immaterial to peak and capability while being real in the
bytes.

## Do not reopen

_(Phase 15's list stays in [phase15-re-opt.md](phase15-re-opt.md) and is not
amended from here.)_

- **`Qwen3.5-2B` as a chat-upgrade displacement — closed 2026-08-10, measured
  FAIL.** `peak_ws_mb` **1421** against a ≤1398 bar and decode **19.25** against
  ≥19.6 (`bench/results/phase16-gguf.csv`). Both halves missed, so this is not a
  borderline single reading, and nothing about it is noisy — the three runs span
  0.03 tok/s. It also loads in **~6.9 s** against 0.6 s for the 230M. Reopening
  needs a quant that changes the memory arithmetic, not a re-run. The
  architecture question is settled and must not be re-asked:
  `general.architecture = qwen35`, which the pin carries.
- **`Maincoder-1B` as the coding-balanced displacement — closed 2026-08-10 on the
  card as written.** Decode **33.49** against a self-set 33.9 bar (1.283× the
  incumbent against a 1.3× requirement); memory passed at 843 ≤ 900. Recorded
  with its caveat: H16.1's own branch 1 would have admitted it at ≥0.9× decode
  had it won on capability, and capability was never measured because
  `bench/eval/phase16-coding.json` does not exist — the H9 suite is generalist
  and scores no coder. Reopening means building that harness first and deciding
  whether the coding role wants 1.3× throughput or capability parity. The arch
  `maincoder` loads cleanly at the pin; that is not the obstacle.

- **`SmolLM3-3B` — closed at desk 2026-08-10, "cost not bought."** Judged, not
  lost: WS-A's four console slots were allocated and it lost the fourth
  head-to-head to `minicpm5-1b` (2046 MB against 735; the only WS-A renderer that
  cannot collapse to config-only; and a self-set PASS bar of H9 **8/8**, where the
  catalogue best is 7/8 and 7/8 was written as a FAIL by design). Reopening needs
  a _free_ WS-A slot and a renderer budget, not merely a better model. Two facts
  survive the close and must not be re-bought:
  1. **64k context is arithmetically outside the envelope for this family.** 36
     full-attention layers × 4 KV heads × head_dim 128 = **72 KiB/token** of f16
     KV, so 64k context alone is **4608 MB** — past the working-set gate before a
     single weight byte loads. Envelope max is ~20–24k; the catalogue default
     `n_ctx` stands. No future card re-buys the 64k pitch here.
  2. **`smollm3-3b` matches no renderer predicate — and this one is live, not
     closed.** There is no `gemma` / `llama` / `phi` / `qwen3` / `a1b` substring
     (the double-l is followed by `m`, not `ama`), so it falls through to the
     ChatML default, which emits neither the mandatory `## Metadata` block nor
     `Reasoning Mode: /no_think`, and `model_is_thinking()` is false for the id.
     Any smollm3-class GGUF provisioned today **leaks chain-of-thought verbatim
     into the chat UI** — #223 with the polarity reversed.

Inherited bars that a Phase 16 candidate must respect:

- **MoE is not banned, but it inherits H2's bar.** A small-MoE candidate must
  state in its funnel row the new fact distinguishing it from `lfm25-8b-a1b`
  (fewer active params **and** not a per-turn reasoner), and it must **beat**
  H2's measured decode at no more memory — not merely run.
- **BitNet / 1.58-bit (H5) is NOT closed** — it is an open desk survey with no
  engineering yet (`ROADMAP.md`, [phase15-re-opt.md](phase15-re-opt.md) WS-D,
  milestone M8 "go/no-go"). A low-bit candidate is therefore welcome, and WS-A
  scouting is the natural way to feed that survey. It does not get a free pass:
  it still clears the same funnel, and the IQ2_M precedent (immediate EOG on
  long declarative prompts) is the known failure mode to check first.
- Draft-model speculative and prompt-lookup as a product default
  are closed. See [phase7-hypotheses.md](phase7-hypotheses.md) and
  [phase15-re-opt.md](phase15-re-opt.md).

## Decision log

| Date       | WS             | Decision                                                                                                                                                                                                                                                                                                                                                                                            | Evidence                                            |
| ---------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------- |
| 2026-08-10 | —              | Campaign opened; seven workstreams predeclared, console budget capped at ≤9.                                                                                                                                                                                                                                                                                                                        | —                                                   |
| 2026-08-10 | WS-A/C/D/E/F/G | T0 desk sweep run A: 59 candidates screened to a shortlist of 12, plus the negative results above. Nothing admitted to T1 yet — the shortlist is unverified.                                                                                                                                                                                                                                        | this file, "T0 desk sweep — run A"                  |
| 2026-08-10 | WS-A/E/F/G     | T0 verification run B: 6 PASS, 5 DEGRADE, 1 KILL. Size corrected on 7 of 12 (PaddleOCR-VL understated by 424 MB); `SmolLM3-3B` and `MiniCPM5-1B` caught claiming ChatML while both would leak chain-of-thought to the UI.                                                                                                                                                                           | this file, "T0 adversarial verification — run B"    |
| 2026-08-10 | WS-B           | **Closed, not motivated.** Both `arch:not-in-pin` flags refuted at the GGUF header; no survivor fails the architecture filter and nothing else, so the predeclared trigger never fired. No pin bump.                                                                                                                                                                                                | H16.2 kill criterion                                |
| 2026-08-10 | WS-E           | `embeddinggemma-300M` KILLED on licence: origin repo is `gated: manual` (anon HEAD 401), so the Gemma-Terms direct-download fallback is closed to the app, and the ungated mirror ships no licence or notice.                                                                                                                                                                                       | H16.5 / funnel F4                                   |
| 2026-08-10 | WS-A           | T1 host smoke, 4 candidates: all load, all byte-stable. `qwen35-2b` arch confirmed `qwen35` at the artefact and context shift measured **disabled**; `minicpm5-1b` needs BOTH renderer halves (`<s>` BOS and no-think prefill) — measured, not assumed.                                                                                                                                             | this file, "T1 host smoke — results"                |
| 2026-08-10 | WS-A           | T3: `lfm25-230m` **PASS** (241 MB, 119.77 tok/s — beats `gemma3-270m` and `lfm25-350m` on both axes); `qwen35-2b` **FAIL** (1421 > 1398 MB and 19.25 < 19.6); `maincoder-1b` **FAIL as written** (33.49 < 33.9, memory passes).                                                                                                                                                                     | `bench/results/phase16-gguf.csv`                    |
| 2026-08-10 | —              | Peak model settled: **`× 1.12` under-predicts everywhere** (−4% to −47%). Overhead is a ~95 MB floor plus an arch-dependent KV term, not a constant factor. All remaining projections are lower bounds.                                                                                                                                                                                             | this file, "T3 console bench — results"             |
| 2026-08-10 | WS-A           | T4 H9: control `lfm25-350m` reproduced its recorded 4/8 (harness validated); `gemma3-270m` measured **3/8** for the first time; `lfm25-230m` **2/8**. **H16.1c PASSES all three predeclared bars** (241 MB / 119.77 tok/s / 2÷8 against a 2÷8 floor) and is the only T5-eligible candidate.                                                                                                         | `bench/results/phase16-h9.jsonl`                    |
| 2026-08-10 | WS-A           | **T5: `lfm25-230m` shipped** as the floor tier. Artefact switched to LiquidAI's own Q4_K_M so the LICENSE travels with the weights, and everything re-measured on it: 119.17 tok/s, 241 MB, H9 2/8 — verdict unchanged. Manifest, model-matrix §A1/§E/§F, benchmark selector, `phase7-h9.jsonl` + `expect_h9`, CHANGELOG.                                                                           | `bench/results/phase16-gguf.csv`, `phase7-h9.jsonl` |
| 2026-08-10 | WS-C, WS-D     | **Closed on their own predeclared kills** from T0 evidence: no ORT-GenAI-buildable arch at ≤500M, and no diffusion candidate with a usable 3-component ONNX export.                                                                                                                                                                                                                                 | this file, "Campaign outcome"                       |
| 2026-08-10 | WS-E, WS-F     | **Blocked, not closed.** WS-E has a verified config-only candidate and no named consumer for the vectors — a product decision. WS-F needs the AppContainer microphone probe, which is unwritten UWP work.                                                                                                                                                                                           | this file, "Campaign outcome"                       |
| 2026-08-10 | —              | Console suite **10/10 ALL PASS** on MSIX 1.5.4.887 after the campaign's LocalState churn. The first run failed `coderpaste` on a **missing model, not a regression**: the gate needs `qwen25-coder-0.5b`, which `provision-models.sh --all-test` did not seed even though the runbook tells operators to seed with it and then run the suite. `ALL_TEST` now includes it and `lfm25-1.2b-thinking`. | `scripts/provision-models.sh`                       |
| 2026-08-10 | WS-A/E/F/G     | T0 synthesis run C: 7 cards assigned (H16.1a–d, H16.5a, H16.6a, H16.7a); 3 candidates dropped as duplicate bets; `SmolLM3-3B` judged and closed at desk. WS-A stays at its predeclared 4 sessions — no budget exception.                                                                                                                                                                            | this file, "Candidate cards"                        |
| 2026-08-10 | —              | Method defect: a card writer errored against our own schema limits and one candidate left the synthesis silently. Recovered, judged, limits relaxed.                                                                                                                                                                                                                                                | this file, "Candidate cards"                        |

## See also

[model-matrix.md](model-matrix.md) · [model-selection.md](model-selection.md) ·
[phase7-hypotheses.md](phase7-hypotheses.md) ·
[phase15-re-opt.md](phase15-re-opt.md) · [benchmarks.md](benchmarks.md) ·
[`../bench/README.md`](../bench/README.md) · [`../ROADMAP.md`](../ROADMAP.md)
