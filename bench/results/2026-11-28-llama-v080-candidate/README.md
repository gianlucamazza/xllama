# llama.cpp v0.8.0 pin qualification

**Pin decision — HOLD.** The paired Linux CPU comparison fails on the catalogue Qwen3.5-0.8B. The identical checkpoint and prompt produce the same greedy text while their full-vocabulary distributions cross the frozen thresholds. Reciprocal experiments isolate the difference to upstream's more accurate CPU softplus. Full-model reference distributions, paired quality and the original cross-pin threshold still require a review. Other open product gates include package-level integration, catalogue quality and Series S acceptance. [Pin tracker](https://github.com/gianlucamazza/xllama/issues/522) · [Merge gates](../../../docs/superpowers/plans/2026-11-24-moe-engineering-program.md#pin-bump-522--after-m0-before-any-m1-gpu-work).

## Versions and API surface

| Source revision                                                                                                                                                                                                    | Scope             | Finding                                                                                                                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [xLlama baseline](https://github.com/gianlucamazza/xllama/tree/e2abab1f4510ca8cc96fa47f378bc00f7b4bc7e8), [llama.cpp d812350](https://github.com/ggml-org/llama.cpp/tree/d81235049384534c167caea52b85a694f6103d14) | Baseline          | llama.cpp 0.6.0, GGML 0.26.0, six local patches.                                                                                                                                                                                                                                                                                                                                                                |
| [xLlama candidate](https://github.com/gianlucamazza/xllama/tree/708402f63d310ddb4ad6862ecb80cac736fe9c15), [upstream v0.8.0](https://github.com/ggml-org/llama.cpp/tree/8642541b255bfb5dc39cd351a178be09013a55d6)  | Tested candidate  | Earliest release admitting LFM3, EmbeddingGemma 2 and K2 together. GGML 0.28.0. The internal GGML backend interface header is byte-identical to the baseline (git blob `dc7367dec4fe3246eb255332c45c23af34fcb959`, API 3). Rebuild both the application and its static dependencies because `llama_model_params` gained `moe_cache_size`. The upstream default leaves device expert caching off.                |
| [upstream v1.1.0](https://github.com/ggml-org/llama.cpp/tree/171527522610996f5f0acf12334e185156ca8f44)                                                                                                             | Source comparison | Latest published release observed on 2026-11-28. GGML 1.1.1, changed internal backend header (blob `482d6b33cf376f3788c7d5129ff1c625df9e96ca`) and deprecated `llama_batch` API. Adds Domyn Edge and Inkling 2 graphs. Review D3D12's internal dispatch tables and the new upstream batch API before qualifying this larger change. [Release notes](https://github.com/ggml-org/llama.cpp/releases/tag/v1.1.0). |

The v0.8.0 branch rebases the AppContainer partition guards and LFM2's strict named/required grammar, preserves the recurrent zero-fill and tool parser backports, and retires the upstreamed no-RoPE K-shift guard. The bridge now consumes the structured upstream chat prompt and owned parser arena. Text tool proposals retain xLlama's existing string interface. The UWP core project adds `llama-moe-cache.cpp`. The two consumers of the new `common/file-lock.cpp` are the excluded download and HF cache modules. Source inventories match 228 core and 47 chat files. The upstream device cache holds host-resident experts and requires backend execution support. The SSD-backed CPU provider and same-quant resident/streamed parity remain separate gates. [Patch inventory](../../../patches/README.md) · [Native provider](https://github.com/gianlucamazza/xllama/issues/480).

## Paired catalogue screen

All 16 GGUF files in the shipped manifest passed their integrity checks. Of the 14 generative models, 13 returned byte-identical full-vocabulary logits at every prompt position. Qwen3.5-0.8B diverged at later positions. Across the 14 models the test compared 107 complete rows. Both catalogue embedding models returned byte-identical complete vectors, 768 float32 elements for Nomic v2 MoE and 1,024 for BGE-M3.

The [machine-readable result](catalogue.json) records the metrics, file SHA-256, prompt token IDs and distribution digests per checkpoint. The Linux x86-64 host ran kernel 6.18.57 on an AMD EPYC 9V45 with GCC 14.2.0, CMake 3.31.10 and portable AVX2/F16C/FMA kernels. Both CLIs were built with the same Release C/C++ flags `-O1 -DNDEBUG`. Every model file matched the committed manifest's byte count and SHA-256 before use. Runtime timing was excluded from the paired verdict because other workloads shared the host.

For each generative GGUF the two CLIs received the same raw question (prompt text below), context 256, CPU only, two threads, upstream batch/microbatch defaults, greedy decoding and a three-token cap. Each prompt position produced a complete float32 row. Match the row count and vocabulary, require finite values, then run the repository's [comparison](../../../scripts/logit_comparison.py) per row with maximum absolute error ≤ 0.5, NMSE ≤ 0.001, matching top-1 and top-10 overlap ≥ 0.8. Also require identical generated stdout. Token IDs were independently compared with each pin's `llama_tokenize(vocab, text, length, ..., true, false)` on the matching GGUF vocabulary. The embedding CLI checked stability on two identical inputs. A separate call to the same public `Session::embed` path saved the complete native-dimension vector for byte comparison across pins. Retrieval quality remains unmeasured. The three ONNX text models and SD-Turbo require their separate Windows runtimes.

```sh
# Run on each pin using the same validated weights and compiler flags.
scripts/apply-llama-patches.sh
scripts/apply-llama-patches.sh --check
scripts/check-uwp-sources.sh
cmake --preset linux-test -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_FLAGS_RELEASE='-O1 -DNDEBUG' -DCMAKE_C_FLAGS_RELEASE='-O1 -DNDEBUG'
cmake --build build/linux-test --target xllama-cli xllama-tests test-sdk-consumer \
  test-decode-faults test-admission-allocation test-context-admission xllama-cpu-smoke
ctest --test-dir build/linux-test --output-on-failure
python3 scripts/run-cpu-smoke.py --build-dir build/linux-test --model build/pin-smoke/model.gguf
build/linux-test/bin/xllama-cli -m "$MODEL" -p 'What is the capital of Italy?' \
  -t 2 -c 256 --seed 42 --temp 0 -n 3 --dump-logits "$OUT" --dump-logits-stride 1
build/linux-test/bin/xllama-cli --embed -m "$EMBEDDING" -c 256 -t 2 \
  -p 'What is the capital of Italy?' -p 'What is the capital of Italy?'
```

Compare the position sidecars for identical vocabulary size, prompt and consecutive positions. The following reproduces the row thresholds using the two captures `baseline.bin` and `candidate.bin` from the command above, with NumPy and the existing xLlama comparator. A threshold breach exits 1.

```sh
PYTHONPATH=scripts python3 - <<'PYTHON'
import json
import numpy as np
from logit_comparison import compare_logits

def load(prefix):
    with open(prefix + ".bin.positions.json") as file:
        meta = json.load(file)
    shape = (len(meta["positions"]), meta["vocab_size"])
    rows = np.fromfile(prefix + ".bin.positions.bin", dtype="<f4").reshape(shape)
    assert meta["positions"] == list(range(meta["n_prompt_tokens"]))
    assert np.isfinite(rows).all()
    return meta["prompt"], rows

prompt_a, baseline = load("baseline")
prompt_b, candidate = load("candidate")
assert prompt_a == prompt_b and baseline.shape == candidate.shape
measures = [compare_logits(a, b, 10) for a, b in zip(baseline, candidate)]
for position, measure in enumerate(measures):
    print(position, measure)
raise SystemExit(int(not all(m.passes(0.5, 0.001, 0.8) for m in measures)))
PYTHON
```

To compare complete embedding vectors, compile the [probe](embedding-probe.cpp) against each build, run it on the same validated file and compare the resulting binary files with `cmp`.

```sh
B=$(realpath build/linux-test)
c++ -std=c++17 -O1 -Iinclude \
  bench/results/2026-11-28-llama-v080-candidate/embedding-probe.cpp \
  "$B/libxllama.a" -L"$B/bin" -Wl,-rpath,"$B/bin" -lllama-common \
  "$B/llama.cpp/common/libllama-common-base.a" -lllama -lggml -lggml-cpu \
  -lggml-base -pthread -o /tmp/embedding-probe
/tmp/embedding-probe "$EMBEDDING" 'What is the capital of Italy?' "$OUT.f32"
```

Each pin produced byte-identical Qwen3.5 output rows on its own fresh-process repeat. Position 0 is byte-identical across pins. Four of the following six rows breach the thresholds, with worst absolute error 0.604359 and worst NMSE 0.00144747. Every top-1 ID matches. Microbatch 1 still gives worst NMSE 0.00135344. `GGML_CPU_DISABLE_FUSION=1` leaves each pin's default-batch bytes unchanged. A separate layer-output capture agrees on layer 0, differs by NMSE less than 3e-13 through layer 3, then reaches 2.29e-5 at layer 4 and 1.22e-4 at layer 5. The last-layer capture has different shapes, so the all-node capture comparison rejects that pair. The recurrent attention output at layer 4 differs by NMSE 1.87e-9, rising to 7.24e-6 in its output projection and 1.32e-4 in the feed-forward output. A float64 reference checked 39 quantized weight multiplications per pin through layer 4 using each arm's own quantized operand. Worst NMSE was 5.59e-13 (declared limit 1e-6). This bounds arithmetic error in those projections. Full-model reference logits remain open. For each pin, use `--capture-tensors "$CAPTURE" --capture-spec 'l_out-4;matmul_inputs=l_out-4;mode=full;max_mb=128'` with the same prompt, then run `scripts/check-matmul-reference.py "$MODEL" "$BASE_CAPTURE" "$CANDIDATE_CAPTURE" --nmse-limit 1e-6`.

### Qwen3.5 root cause

The first observed difference at layer 1 is the gate's softplus. Its input projection and the checkpoint bias match across pins. Upstream replaced `logf(1.0f + expf(x))` with `log1pf(expf(x))` in the scalar CPU kernel. Both retain the same `x > 20` cutoff. [Upstream change #31086](https://github.com/ggml-org/llama.cpp/commit/74bca3950054eed1795f525cb9cf06bfe796492d) · [Upstream review](https://github.com/ggml-org/llama.cpp/pull/31086).

| Isolated CPU library experiment              | Positions × vocabulary | Observed full-logit result                                                                                             |
| -------------------------------------------- | ---------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| Baseline, candidate softplus expression only | 7 × 248,320            | Byte-identical to the **stock candidate**, SHA-256 `e7f3fe9d9259fe7959cd82f181d86d37a84339d2d56261965a8430f6783d17f9`. |
| Candidate, baseline softplus expression only | 7 × 248,320            | Byte-identical to the **stock baseline**, SHA-256 `8510ac5a295f8340b73eb1b313a2d5d789c337480a4d44a7da40923b63b009c8`.  |

Each experiment rebuilt `ggml/src/ggml-cpu/unary-ops.cpp` in an isolated build using the original pin's compile flags. The rest of each CPU library and application came from its corresponding stock arm. To reproduce, substitute just the return expression in `op_softplus` in each disposable source checkout, rebuild `xllama-cli` and repeat the same `--dump-logits-stride 1` invocation. The reciprocal substitutions account for all 1,738,240 logits in this prompt comparison.

For the 112 observed softplus inputs (7 tokens × 16 heads), add the GGUF's layer-1 `ssm_dt.bias` to the identical projections in float32, then independently evaluate `ln(1 + exp(x))` at 65 decimal digits. The baseline gives 64 correctly rounded float32 results and reaches **13,274 ULP** at worst. The candidate gives 102 correctly rounded results and stays within **1 ULP** throughout. This establishes a local accuracy improvement on the tested inputs. The original full-model gate still records **FAILED**. A checkpoint-level teacher-forced reference and a representative paired quality suite are the next acceptance gates.

Capture `alpha-1` and `a_softplus-1` in each unmodified CLI by appending `--capture-tensors "$CAPTURE" --capture-spec 'alpha-1,a_softplus-1;mode=full;max_mb=4'` to the same prompt run, omitting `--dump-logits-stride`. Run the [decimal reference](softplus-reference.py) from the repository root. With no arguments it replays the 112 captured inputs, 16 bias values and both outputs directly from `catalogue.json` using Python and NumPy. With the following arguments it reads the original GGUF and the fresh captures.

```sh
PYTHONPATH=llama.cpp/gguf-py python3 \
  bench/results/2026-11-28-llama-v080-candidate/softplus-reference.py \
  "$MODEL" "$BASE_CAPTURE" "$CANDIDATE_CAPTURE"
```

### Build and runtime checks

Both UWP package configurations built on GitHub Actions. The Linux build, sanitizers and CodeQL runs also passed on the tested candidate commit. Package installation, inference and suspend/resume on a physical console remain open. [UWP builds](https://github.com/gianlucamazza/xllama/actions/runs/46140767881) · [Linux build](https://github.com/gianlucamazza/xllama/actions/runs/46140767930) · [Sanitizers](https://github.com/gianlucamazza/xllama/actions/runs/46140767955) · [CodeQL](https://github.com/gianlucamazza/xllama/actions/runs/46140767963).

The candidate also passed all four registered CTest targets and the independent synthetic CPU smoke covering prefill logits, incremental decode, session reuse, refusal and recovery. On the catalogue LFM2.5-350M QAD-Q4_0 the Session selection passed (33 cases, 255 assertions in this invocation, including model-conditional cases). On LFM2.5-1.2B-Instruct QAD-Q4_0 the real tool selection passed 5 cases and 79 assertions. Granite's official 3,062,452,832-byte Q3_K_S (SHA-256 `5ec3f4af77fffead37b09984a97625f0f532f6ae2ead65b5019bce6dd8b4d0cd`) passed the no-RoPE overflow case with 18 assertions and repeated evictions on the candidate. [Granite input and prior device measurements](../2026-11-24-granite40-h-tiny/README.md).

## Newly recognized architectures

Weight size is a disk fact. The console product ceiling depends on measured peak working set at product settings. The table concerns the `v0.8.0` qualification branch. GPU execution, SSD streaming, package acceptance, Series S/X timings and a catalogue verdict are **NOT_RUN** for these specific artifacts on this pin.

| Checkpoint                                                                                                              | Total / active parameters, licence                                                         | Quantization and exact weight bytes                               | Host observation and next gate                                                                                                                                                                                                                                                                                                                                                               |
| ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [LFM3-1.7B](https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/tree/cd0c74c575b5c416f3011b5b4c298077f26ca8c2)               | 1,728,801,536 total, active count unreported, LFM Open License v1.0                        | 1,049,615,680 (QAD-Q4_0) and 1,099,627,808 (Q4_K_M)               | **MEASURED Linux** on QAD-Q4_0. 287 tensors, raw prefill of 7 tokens then 12 greedy decode steps at context 256. The CPU log recorded KV 3.00 + 2.50 MiB, recurrent state 0.61 MiB, compute 135.26 MiB and peak RSS 1,860 MiB. Output repeated the question. Check the exact publisher template and thinking controls, tokenizer IDs, multi-turn quality and device budget before admission. |
| [EmbeddingGemma 2](https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/tree/6ea3e3a309d12a4779cfda98b8c19800d2d5da71) | 744,371,512 across the publisher checkpoint, active text-path count unreported, Apache-2.0 | 309,855,456 (Q8_0 text) and 554,821,056 (optional Q8_0 projector) | **MEASURED Linux text-only** on the `gemma-embedding2` graph. Two 9-token inputs returned 768 normalized dimensions with repeat cosine 1.000000. The CPU reserved 26.76 MiB of compute space. Sampled process VmHWM reached 511,296 KiB (499.3 MiB). Retrieval quality, vectors/s, projector and device gates remain open.                                                                   |
| [K2-Horizon-3.7B](https://huggingface.co/IFM/K2-Horizon-3.7B-GGUF/tree/839b82cdd4e8b7c824b509172ae0cd1fab56576e)        | 3.7B named core, 5,058,255,360 publisher parameters, active count unreported, Apache-2.0   | Q4_K_M: 3,156,598,144                                             | **MEASURED Linux** on the `k2-horizon` graph. Eight prompt tokens and 12 greedy decode steps used 36.00 MiB KV and 257.38 MiB compute. RSS reached 4,856 MiB at context 256. The first sampled answer was “Rome” followed by another question. Measure console admission explicitly against the 3,584 MiB product limit before quality or offload promotion.                                 |

The K2 Q4_K_M SHA-256 was `07773aed93890d4f5d08b421430000d5c8d73cefc74fbdc32e67d820f23207b2`. The 250,624-float final-prompt logit row had SHA-256 `0e656816455bc556e23e6893e8bda997426bae58fa8d614bd2515b932522bb96`. On this single shared-host run, K2 loaded in 4,423 ms and processed 16.5 prefill and 14.10 decode tokens/s. LFM3 loaded in 4,682 ms and processed 7.3 prefill and 4.08 decode tokens/s. The samples do not establish latency percentiles or device performance.

The text EmbeddingGemma 2 Q8_0 SHA-256 was `2188ac1deca4b77dffefd603c2776a9d76d9d74ec01841392982ebb840b09135`. The 768-float `Session::embed` result for the same question had SHA-256 `29b46364d1cd16cbf2587de2581771fae74f68ab50001bf946365c30fa8f414e`.

The LFM3 QAD-Q4_0 byte SHA-256 was `43a7d09a0e9cba91b63eb7052488b9564569bddfd65754d2e426465e572a7452`. A plain-prompt preflight on the candidate produced a 128,000-element float32 final logit row (SHA-256 `2a439b7efe607936be01dd7687e69a7b82f075a09561de0fba23e2815e7c32fb`). F0–F5, checkpoint quality and Xbox acceptance stay with [#522](https://github.com/gianlucamazza/xllama/issues/522).
