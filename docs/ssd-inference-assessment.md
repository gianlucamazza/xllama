# SSD-streamed inference — assessment (2026-08)

> Claim under test: "SSDs can be used for inference." Verdict up front: **true,
> but only as weight/KV streaming bounded by file-read bandwidth — never as
> compute inside the drive — and on Series S it buys capacity, not speed.**
> The one number this repo can add is the sandboxed NVMe read rate (§4).

## 1. What has actually been demonstrated

| Work                                                                                           | Mechanism                                                                                     | Hardware                                                                | Measured                                                        | Source                                                                                                                                                                                                           |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Flash-MoE (@anemll, 2026-03)                                                                   | MoE expert streaming from NVMe, 2–4-bit experts, hot cache; active experts cut 10→4           | iPhone 17 Pro                                                           | Qwen3.5-397B-A17B, **0.6 tok/s**, 5.5 GB resident               | [Willison](https://simonwillison.net/2026/Mar/18/llm-in-a-flash/), [wccftech](https://wccftech.com/iphone-17-pro-successfully-runs-400b-llm-locally/)                                                            |
| Flash-MoE, same stack                                                                          | as above                                                                                      | M3 Max 48 GB (~17.5 GB/s SSD)                                           | same model (209 GB on disk), **4.4–5.5 tok/s**                  | [Willison](https://simonwillison.net/2026/Mar/18/llm-in-a-flash/)                                                                                                                                                |
| Apple "LLM in a flash" (2023-12)                                                               | dense-model analogue: ReLU sparsity → load ~2–5% of FFN rows; windowing + row-column bundling | OPT-6.7B / Falcon-7B class                                              | models up to ~2× DRAM; 4–5× vs naive paging (CPU), 20–25× (GPU) | [arXiv 2312.11514](https://arxiv.org/pdf/2312.11514)                                                                                                                                                             |
| llama.cpp mmap (merged, default)                                                               | page weights from disk on demand                                                              | community: DeepSeek-R1 671B on 96 GB RAM + 24 GB VRAM, overflow on NVMe | **~1–2 tok/s**; 2–5 GB/s NVMe read observed during decode       | [HF discussion](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/discussions/13)                                                                                                                                  |
| llama.cpp [PR #25294](https://github.com/ggml-org/llama.cpp/pull/25294) (**open, not merged**) | explicit per-layer expert cache + I/O thread pool (`--moe-stream-cache`, O_DIRECT), mmap off  | ~754B GLM MoE                                                           | **~1.8–2.2 tok/s** decode (2.4× vs mmap+cpu-moe)                | PR thread                                                                                                                                                                                                        |
| FlexGen / ZeRO-Inference (2023)                                                                | layer-by-layer weight fetch amortized over huge batches                                       | OPT-175B, 16 GB T4 + SSD                                                | ~1 tok/s **aggregate at batch 144**; interactive latency dismal | [arXiv 2303.06865](https://arxiv.org/abs/2303.06865)                                                                                                                                                             |
| Phison aiDAPTIV+ (2026-01 PC push)                                                             | middleware swaps weights/KV to proprietary SSDs; "10× TTFT" = KV-cache persistence            | gpt-oss-120B on 32 GB laptop (claim)                                    | **no tok/s published**; no independent benchmarks               | [Blocks&Files](https://www.blocksandfiles.com/ai-ml/2026/01/07/phison-extends-aidaptiv-to-boost-ai-inference-on-pcs/4090377)                                                                                     |
| Computational storage (SmartSSD, KVNAND, InstInfer, SK hynix HBF)                              | compute in/near the drive                                                                     | research prototypes / roadmap                                           | nothing purchasable runs a transformer in-drive today           | [InstInfer](https://arxiv.org/pdf/2409.04992), [KVNAND](https://arxiv.org/pdf/2512.03608), [HBF](https://www.blocksandfiles.com/flash/2026/02/16/sk-hynix-proposes-hbm-and-hbf-hybrid-for-llm-inference/4091326) |

Pattern: every real demo is **offload + streaming**; the drive never computes.
And every real number is ~0.5–5 tok/s — proofs of capacity, not of speed.

## 2. The physics

Decode is a bandwidth-bound M=1 GEMV: on a purely storage-bound path every
token streams every **active** weight byte once, so
`tok/s ≈ effective read GB/s ÷ active GB` — an upper bound over the bytes
actually fetched from storage (a hot cache shrinks the denominator to the
missed bytes, §2 escapes). The hierarchy this repo has already measured
(Series S):

| Tier                         | Read bandwidth                   | Where measured                                      |
| ---------------------------- | -------------------------------- | --------------------------------------------------- |
| CPU DRAM                     | 12.35 GB/s @1t, ~30 GB/s ceiling | `membw` ([benchmarks.md](benchmarks.md))            |
| GPU pool                     | 119.07 GB/s                      | `gpubw` W3 ([phase15-re-opt.md](phase15-re-opt.md)) |
| NVMe (PCIe 4.0 x2, raw spec) | ~2.4 GB/s                        | vendor spec; sandboxed rate in §4                   |

Dense streaming is therefore dead on arrival for chat: a 4 GB Q4 model at even
the full raw 2.4 GB/s is **~0.6 tok/s** before any compute. The only escapes:

- **MoE sparsity + hot expert cache** — only cache-_misses_ touch the SSD, so
  the denominator shrinks from "model size" to "missed expert bytes/token".
- **KV/prompt-cache on disk** — not decode at all: it removes prefill recompute
  (this is the honest content of Phison's "10× TTFT").
- **Batch amortization** (FlexGen) — irrelevant to a single interactive chat.

## 3. Applicability to xllama (Series S, UWP)

Constraints already on file:

- RAM budget **3801 MB** ([uwp-constraints.md](uwp-constraints.md)) — streaming
  would buy _capacity_ (models > RAM), not speed.
- **No transparent paging**: POSIX mmap is unavailable and the AppContainer
  mapping was tried and reverted with no benefit (uwp-constraints §1); GGUF is
  read into heap. Any streaming must be explicit (PR #25294 shape: expert
  cache + I/O threads over Win32 file APIs).
- **H2 MoE FAIL** ([phase7-hypotheses.md](phase7-hypotheses.md)): at
  console-admissible scale the MoE truly reads only active experts and is
  still as slow as the 3B dense — MoE brings no _speed_ headroom here even
  from RAM.

Scenarios:

| Scenario                            | Verdict                    | Why                                                                                                                                                                                                                                                 |
| ----------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Dense weights streamed from NVMe    | **No**                     | §2 arithmetic: sub-1 tok/s at best                                                                                                                                                                                                                  |
| MoE expert streaming (> RAM models) | **Gated on §4 + upstream** | needs measured random-read ≥ ~1.5 GB/s to clear ~1 tok/s for a plausible target (active set ~1 GB, miss rate ~2/3); also needs PR #25294 (or a port) and a console-shaped MoE catalogue entry — and H2's shadow says the win would be capacity only |
| KV/prompt-cache persistence on disk | **Yes, cheap**             | pure TTFT win, no bandwidth cliff; candidate follow-up independent of this assessment                                                                                                                                                               |

## 4. Measured: sandboxed NVMe read bandwidth (`diskbw`)

Probe: `diskbw.flag` → `diskbw-result.csv` (pattern of `membw`/`gpubw`;
`scripts/bench-diskbw.sh`). 4 GiB incompressible file — larger than the
documented 3801 MB app RAM budget, which makes full cache service _unlikely_,
not impossible (the OS file cache is not bound by the app budget) — sequential
8 MiB blocks (bulk-load shape) and random 2 MiB blocks (expert-fetch shape),
1 and 4 I/O threads. The probe prefers `FILE_FLAG_NO_BUFFERING`/`O_DIRECT`,
which bypasses the file cache outright; the `unbuffered` column records
whether that mode actually ran (it did in every row below), so the cache
question only affects the buffered fallback, which additionally drops the
cache per pass on POSIX (`posix_fadvise DONTNEED`, best-effort).

Host reference (Linux laptop, `xllama-cli --diskbw`):

| pattern | threads | unbuffered | first GB/s | best GB/s |
| ------- | ------: | ---------: | ---------: | --------: |
| seq 8M  |       1 |          1 |  1.13–2.21 |      4.47 |
| seq 8M  |       4 |          1 |  2.71–5.96 |      5.96 |
| rnd 2M  |       1 |          1 |  1.51–1.98 |      1.98 |
| rnd 2M  |       4 |          1 |  2.48–3.05 |      3.05 |

(Samsung PM9A1 512 GB, btrfs, O_DIRECT in effect; ranges span two consecutive
runs, 2026-08-18 — btrfs + background I/O make host runs noisy; the host table
is a reference, not a gate input. ⚠️ Gotcha: the first attempt ran in `/tmp`,
which is **tmpfs** on Arch — it measured DRAM (12+ GB/s "disk") and O_DIRECT
did not refuse the open. Always point `XLLAMA_DISKBW_FILE` at a real
filesystem.)

Console (Series S, Dev Mode):

| pattern | threads | unbuffered | first GB/s | best GB/s |
| ------- | ------: | ---------: | ---------: | --------: |
| seq 8M  |       1 |          1 |       2.00 |      2.03 |
| seq 8M  |       4 |          1 |       1.80 |      1.85 |
| rnd 2M  |       1 |          1 |       1.55 |      1.55 |
| rnd 2M  |       4 |          1 |       1.75 |      1.76 |

(Series S Dev Mode, package `1.5.4.900` **crossbuild**, 2026-08-18; raw CSV:
`bench/results/diskbw.csv`. Deliberate deviation from the CI-MSVC-only rule
for console benchmarks: this probe measures NVMe read bandwidth through
Win32 file APIs, not code generation — the compiler cannot move the device's
read rate, so the crossbuild number stands for the §4 gate. Any tok/s or RAM
claim would still require a CI MSVC package.) Three facts fall out:

1. **`FILE_FLAG_NO_BUFFERING` is accepted in the AppContainer** — the sandbox
   does not block unbuffered reads, so an explicit streaming engine could do
   O_DIRECT-style I/O on console.
2. **Sequential ≈ 2.0 GB/s** — ~83% of the PCIe 4.0 x2 raw spec (2.4 GB/s);
   the sandbox tax is real but small. Extra I/O threads do not help (1.8–1.85
   GB/s @4t): the device, not the submitter, is the bottleneck.
3. **Random 2 MiB ≈ 1.55–1.76 GB/s** — the expert-fetch shape loses 22–24%
   vs sequential at 1 thread and 3–5% at 4. The §4 gate (≥ 1.5 GB/s)
   **passes, marginally**.

**Read gate** (why 1.5 GB/s): a hypothetical console-admissible MoE with ~1 GB
active per token and a hot cache absorbing ⅔ of expert reads needs
`1.5 GB/s ÷ (1 GB × ⅓) ≈ 4.5 tok/s` I/O-side headroom to leave room for the
CPU side to still land ≥ 1 tok/s end-to-end. Below 1.5 GB/s random read, the
I/O alone caps the pipeline under usability before compute even starts.

## 5. Verdict and reopen conditions

**Verdict (measured): the claim is real but narrow.** The console I/O gate
passes on the number alone — 1.55 GB/s random read would give a ~4.7 tok/s
I/O-side ceiling at 1 GB active × ⅓ miss — but only marginally, and the other
two conditions (an upstream streaming engine, a console-shaped > RAM MoE worth
running) are still absent, so the recommendation below stands unchanged. Nothing
computes inside an SSD today; streamed-weights inference exists and works, at
~0.5–5 tok/s, only for sparse MoE giants with a hot cache — a capacity play.
On Series S the RAM budget makes the capacity argument tempting, but the H2
measurement, the missing upstream (PR #25294 unmerged), and the NVMe ceiling
make it a poor near-term investment. The cheap real win in this family is
KV/prompt-cache persistence (TTFT), not weight streaming.

Reopen if **any** of:

1. llama.cpp PR #25294 (or successor) merges — the explicit-streaming engine
   arrives upstream instead of needing a port;
2. a catalogue-shaped MoE with active set ≲ 1 GB and > 3.8 GB total exists
   worth running (the I/O half of this condition is already met: §4 measured
   1.55 GB/s random read on console);
3. the product accepts sub-1 tok/s for an offline "big model" mode (changes the
   gate arithmetic entirely).
