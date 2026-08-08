# Launch copy — xllama

Publication-ready copy for announcing the project. Benchmark figures are copied
from the generated summary (`docs/benchmarks.md`); when they change there, they
change here, and no number is invented for an announcement.

## Posting history and what it taught

**Attempt 1 — r/LocalLLM, 2026-07-24, link post: removed by Reddit's site-wide
spam filter.** A new account whose only content is a self-link to its own GitHub
is exactly the shape that filter targets. Two lessons, both baked into the copy
below: **text post, not link post**, and **few outbound links** — a wall of them
from a fresh account is itself a spam signal.

**Sequencing decided for attempt 2**: Show HN first, r/LocalLLaMA a few days
later. Hacker News has no self-promotion ratio rule and "I built this" is its
native format, while r/LocalLLaMA Rule 4 caps self-promotion at ~10% of an
account's content — and this account has none. A Show HN thread also gives the
Reddit post something real to reference.

**Two claims that must not be made**, because they are false or unprovable:

- _"First LLM on an Xbox."_ Andrei David ran `llama2.c` on an **Xbox 360** in
  January 2025 and it was widely covered. Cite it first — it preempts the top
  comment, and it is true that it showed the class of project was worth doing.
- _"The only project doing this."_ I searched and found no runtime for Series
  S\|X, but absence of search results is not proof. The defensible line is the
  distinction between a **proof of concept** and an **installable application**.

---

## A. Show HN (primary launch) — PUBLISHED 2026-07-27

**Live thread: <https://news.ycombinator.com/item?id=49069732>**

**Submission type:** URL post pointing at the repository — not a text post.
**URL:** `https://github.com/gianlucamazza/xllama`
**Title:** `Show HN: Xllama – local LLM chat and Stable Diffusion on an Xbox Series S`
(HN title-cases automatically; the title cannot be edited after submitting.)

No numbers and no editorializing in the title; those get rewritten. The real
content goes in the author's first comment, posted immediately after submitting.

**HN formatting, learned in the act**: comments take no Markdown. `**bold**`
renders literally — the version below is deliberately plain — and a code block
is made by indenting two or more spaces, which is what keeps the benchmark table
aligned.

### First comment

I built this over the last two months. It turns an Xbox Series S|X into a local
inference box: a gamepad chat UI, a model catalogue the app downloads on demand,
Stable Diffusion image generation, and an optional OpenAI-compatible endpoint on
the LAN. Nothing leaves the console.

The barrier first, because it is real: this needs **Xbox Dev Mode**, a one-time
~$19 activation through Partner Center. There is no retail-console path, and I am
not affiliated with Microsoft. If you don't have a console, the same inference
core builds as a Linux CLI — that is what I use for CI and A/B checks.

Prior art I know of: Andrei David ran `llama2.c` on an Xbox 360 in January 2025.
That was a proof of concept — 700 lines of C, one model, one prompt — and it is
what convinced me this was worth doing properly. What I could not find was a
runtime you could install and actually use on the current consoles.

Why the hardware is interesting: 8 Zen 2 cores at 3.6 GHz with AVX2, an RDNA 2
GPU, and 10 GB of unified memory, in a box that has been shipping since 2020. The
constraints are the fun part. The sandbox leaves ~6 usable cores, and ggml's
spin-wait threadpool livelocks at 7–8, so 6 is a hard ceiling. There is no `mmap`
and no `dlopen`. The GPU budget is a fixed 3801 MB.

Measured on the console — decode / prefill / peak RAM, Q4_K_M through llama.cpp
on the CPU unless noted:

    LFM2.5-350M     94.9 tok/s   438.1    320 MB
    Gemma-3-270M    76.8         395.0    368 MB
    SmolLM2-360M    74.8         262.4    708 MB   (int4, ONNX Runtime CPU)
    LFM2.5-1.2B     37.9          76.2    811 MB
    LFM2-2.6B       18.4          32.0   1623 MB
    Llama-3.2-3B    14.2          19.5   1824 MB   (Q3_K_S)

SD-Turbo renders 512×512 in about 6.9 s.

Three things I got wrong, which are the parts that generalize beyond consoles:

**The GPU loses at decode, and it is not close.** I assumed the RDNA 2 GPU would
be the main route. Autoregressive decode is a batch-1, memory-bound workload, and
at this scale CPU int4 kernels beat DirectML: 74.8 tok/s on the CPU against 44.4
for the same model in fp16 on the GPU. DirectML int4 is worse still, because that
path dequantizes to fp16 before the GEMM instead of using a fused low-bit kernel.
The GPU earns its place on batched work — prompt prefill and diffusion.

**I shipped numerically wrong logits for weeks, because I was only measuring
tokens per second.** A DirectML asset produced plausible text at good throughput
while its logits were wrong; the root cause was a broken lowering of RMSNorm in
DirectML, not attention, which is where I looked first. Now no ONNX text asset
reaches a release without passing a logit-parity check against llama.cpp golden
vectors, on rank-based metrics rather than raw diffs. If you take one thing from
this project, take that one: throughput is not correctness, and a plausible
sentence is not evidence.

**Two one-line build bugs were costing 74% of prompt throughput.**
`GGML_USE_CPU_REPACK` was compiled but never defined, so the repacked-weight GEMM
path was dead code: 241 → 394 tok/s to enable it. Then `n_threads_batch` was
never set, and its llama.cpp default is 4 regardless of `n_threads`, so prefill
ran on 4 of the 6 usable cores while decode used all 6: 394 → 438. Neither showed
up in a profile, because nothing was slow — it was just less parallel than I had
assumed.

Most recently I stopped throwing prompt processing away. A long conversation used
to lose its KV cache permanently once it passed the prompt-trimming budget, so
every later turn re-read ~1800 tokens (4532 ms → ~280 ms once fixed), and
switching conversations dropped the cache entirely (551 tokens re-read → 19, by
persisting it to disk). A trap worth knowing if you use llama.cpp's per-sequence
state API: on hybrid attention+recurrent models `llama_memory_seq_rm` refuses a
partial _tail_ erase and returns `false` **without mutating anything** — ignore
that return value, as I did, and you decode on top of a stale tail.

It is research-grade and honest about it: Dev Mode only, the LAN endpoint is
unauthenticated and off by default, on-device fine-tuning is a constrained
last-block partial fine-tune rather than anything general, and most benchmark
rows are single runs (the headline one is n=3). The repo carries the raw CSVs,
the falsified hypotheses, and the postmortems — including the two times the
Device Portal silently lost files on me.

I would most like to hear from people doing low-memory inference or KV-cache
management. What would you run on a fixed 10 GB unified-memory box with no NPU?

---

## B. r/LocalLLaMA (a few days after Show HN)

**Post type:** text post. **Flair:** Resources or Discussion, whichever fits.

**Title options** — none names the project, because titles shaped like product
announcements get filtered and downvoted:

1. `Local LLMs on a 2020 Xbox Series S: 94.9 tok/s decode, and why the GPU loses at generation`
2. `I built a local LLM runtime for the Xbox Series S — measurements, and three assumptions it broke`
3. `Two one-line build bugs were costing me 74% of prompt throughput on an Xbox Series S`

### Body

Reuse the Show HN comment above with three changes:

1. Open with one line that places it for this audience: what it runs (GGUF
   through llama.cpp; ONNX Runtime GenAI for CPU int4 and DirectML) and that the
   catalogue is 11 models from 270M to 3B.
2. Keep the link count at **one** (the repo). Everything else goes in a reply.
3. Append the two disclosures below, verbatim and unhidden.

### Required disclosures (Rules 3 and 4)

> Disclosure: I'm the author of the project.
>
> I'm not a native English speaker and used an LLM to help draft and polish this
> write-up. The work, the measurements and the conclusions are mine.

Rule 4 caps self-promotion at ~10% of an account's content. Before posting, build
some genuine participation in the sub — answer questions where you actually know
the answer. This is the part of the plan that cannot be delegated or rushed.

---

## Reference facts (verified 2026-07-30)

Every countable figure below carries the command that produces it. That is the
point: the previous version of this block said "176 host test cases" for long
enough that it drifted 16 cases behind, and a bare number nobody can re-derive
cannot be spot-checked — only trusted or rewritten. Re-run the command instead of
re-deriving the fact by hand.

- **Generated benchmark table**: `docs/benchmarks.md`. Headline `lfm25-350m`
  438.1 prefill / 94.9 decode (94.8–95.1, n=3) / 320 MB. Only the headline GGUF
  row and the two ORT CPU rows are n=3; every other row is a single run.
- **Console-measured KV work**: trimmed turn 4532 ms → ~280 ms (#169);
  conversation switch 551 → 19 prompt tokens (#170b); turn-2 KV reuse 15–16×.
- **Build fixes**: repack 241.9 → 393.2 tok/s (#155); `n_threads_batch`
  390.7 → 438.1 at P=298 (#168).
- **Scale**: **~19.9k lines of own C++ across 90 files**, of which 3.5k in 23
  test files; **215 host test cases / 2880 assertions**; **10 console validation
  gates** (`scripts/validate-console.sh` — routing, settings, gguf, longchat,
  kvsnap, coderpaste, thinkcut, thinkdone, genroom, taesd); **17 product
  releases** since 2026-05-19, current **1.5.4.0**. Derivations:

  ```bash
  # lines and files — src/ + include/ + uwp/ + tests/, tracked .cpp/.h only
  git ls-files 'src/**' 'include/**' 'uwp/*.cpp' 'uwp/*.h' 'tests/**' |
    grep -E '\.(cpp|h)$' | xargs wc -l | tail -1
  # test cases — from the binary, NOT from a grep: `grep -c TEST_CASE` says 199
  # because seven cases sit behind #if, so a static count would overstate by 7
  cmake --preset linux-test && cmake --build build/linux-test -j"$(nproc)"
  ./build/linux-test/tests/xllama-tests | tail -2
  # gates
  grep -cE '^\w+\) run_gate ' scripts/validate-console.sh
  # product releases — total tags minus the two asset-only ones
  gh release list --limit 60 --json tagName --jq 'length'
  ```

  The test-case figure is additionally guarded in CI: `build-linux.yml` compares
  the number in this file against what doctest reports, right after `ctest`.

- **Hardware limits**: ~6 usable cores (livelock at 7–8), GPU budget 3801 MB, no
  `mmap` / `dlopen`, 2 GB per-file ONNX ceiling (which does not apply to GGUF).
- **Demo video**: cite the **v1.5.2** capture, never the v1.2.0 one. The old
  clip predates the performance campaign, shows one 350M chat and an image, and
  is what `README.md` linked until 2026-07-30. The current capture is stills at
  the rate the Device Portal actually sustains, encoded at the rate achieved so
  **playback is real time** — a demo of a performance claim must not be sped up.
  `docs/screenshots/demo-manifest.json` records the version, frame count and
  achieved fps, and `check-coherence.py` fails if the README link disagrees with
  that manifest on **version or filename**. The filename half exists because the
  version check alone once passed on a link to an asset that had never been
  uploaded — a stale link replaced by a broken one.

The earlier long-form drafts (the measurement-first version and the original
link-post copy) are in git history at `docs/reddit-announcement.md`.
