# MoE upstream readiness — maintainer preparation, 2026-11-16

Status: **preparation verified; independent console reproduction of the fork's CPU stage done on a project Series S (2026-11-16); runtime integration and full-checkpoint parity remain pending**. Trackers: [#479](https://github.com/gianlucamazza/xllama/issues/479), [#480](https://github.com/gianlucamazza/xllama/issues/480), [#481](https://github.com/gianlucamazza/xllama/issues/481). The proposed PR boundaries are in [#482](https://github.com/gianlucamazza/xllama/pull/482).

## Current stage and documentation ownership

Preparation is published for review in [PR #483](https://github.com/gianlucamazza/xllama/pull/483).
The planning and preparation PRs are separate from all four implementation PRs.
Completed preparation below describes work actually checked; it is not a merge,
release or device-acceptance claim.

| Work item                                               | Current state                                                                                                                       | Next gate                                                                                         |
| ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Source review, cache harness and comparison-tool checks | Verified preparation; published for review                                                                                          | Review/merge preparation independently of runtime changes                                         |
| Numerical campaign and CPU input analysis               | Prepared; campaign not executable                                                                                                   | Supply exact token/reference/build bindings and an adequate resident-reference host               |
| PR 1 / #479: cache contribution                         | Pending integration                                                                                                                 | Review a concrete coherent extraction and its upstream tests                                      |
| PR 2 / #480: native CPU integration                     | Pending                                                                                                                             | Complete PR 1, native fixtures and real-checkpoint parity                                         |
| Independent fork console reproduction                   | Done for the CPU stage, 2026-11-16: A2 and B512 match the fork's report ([evidence](evidence/2026-11-16-moe-24b-console/README.md)) | Repeat on the upstream PR 3 package; the run used a test build of the fork's patch route          |
| PR 3 / #481 CPU stage: upstream console acceptance      | Pending                                                                                                                             | Complete native integration, Windows build and package-bound console lifecycle checks             |
| PR 4 / #481 GPU stage: selected-expert correctness      | Pending                                                                                                                             | Console identity/dispatch, primitive/model numerical proof and resource closure before benchmarks |

The [roadmap](../../ROADMAP.md#optional-ssd-moe-contribution--state-at-2026-11-16)
links this state instead of duplicating measurements. The original
[SSD assessment and its dated addendum](../ssd-inference-assessment.md#6-contribution-update-2026-11-16)
separate historical conclusions from the newer contribution. Numerical inputs and
missing bindings remain owned by the prepared campaign; test receipts retain the
source revision actually checked.

When a stage changes, update this table, its tracker (#479/#480/#481), and the
corresponding roadmap checkbox in the same implementation PR. Complete a runtime
stage only after its implementation has merged and its stated validation gates
are evidenced. Retain original failed/pending records; add dated outcomes rather
than rewriting them. Update `docs/model-matrix.md` only if catalogue admission is
actually proposed, using its existing product gates.

## Source-to-review map

All contributor references below refer to `vetax-by/xllama` at `61bf63ad541510a03a36993e60ec204f98bc0b3e`, not a moving branch. Its preparation base is xLlama `4db51522c850e62d0bb24234dfeb48e55f4017d7`; the reviewed upstream base is `87d5a4635abca01a3b484e1488a2a9d043de9be2`. Both pin llama.cpp `d81235049384534c167caea52b85a694f6103d14`.

| Existing source                                                                                        | What it already implements                                                                                 | Maintainer review / remaining proof                                                                                    |
| ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `ua50/patch/runtime-llama.patch`: `src/ua50/expert_cache.h`                                            | Synchronous Reader, fixed owned arena, LRU, move-only leases, pinned-entry exclusion, extent/budget checks | Independent cache harness now exercises eight groups; source extraction remains outside the product                    |
| `ua50/patch/cache-ab-llama.patch`                                                                      | Request/byte accounting, timing and trace additions                                                        | Cache accounting checked for mixed Q4_K/Q6_K payloads; trace replay and timing correctness not covered by this harness |
| `runtime-llama.patch`: `native_provider.{h,cpp}`, `ggml-ua50.h`                                        | Metadata-only ggml buffers and one active expert lease                                                     | Requires native buffer/allocator tests; not equivalent to the portable cache test                                      |
| `runtime-llama.patch`: loader, model and CPU changes                                                   | Provider installation before allocation, routed tensor deferral, worker acquire/release barriers           | Generated native GGUF fixtures and worker/failure tests must enter with PR 2                                           |
| `src/platform/ua50/staged_reader_core.h`, `windows_held_reader.{h,cpp}`                                | Staging and stop/identity/terminal-I/O ownership                                                           | Portable fake-boundary and actual Windows cancellation tests remain separate PR 3 gates                                |
| `include/xllama/cpu_expert_session.h`, application/controller patches                                  | Session ownership, admission, invalidation and teardown                                                    | Needs SessionHub regression and real upstream package lifecycle evidence                                               |
| `cpu-pilot-root.patch`, `first-answer-root.patch`, `finished-answer-root.patch`, `cache-ab-root.patch` | Model pin, exact prompt/template, bounded CPU and cache profiles                                           | Inputs resolved below; no package was installed or activated in this preparation                                       |
| GPU native/root patches and identity successor                                                         | Selected Q4_K/Q6_K bridge, identity and proof stages                                                       | Console device/dispatch, primitive/model numerical correctness and fence closure remain pending                        |

The published `ua50/PROVENANCE.json` explicitly excludes the author's private tests and verification copies. We can reuse published source, but cannot claim to have replayed an unpublished author suite. The independent review harness is our own narrow check, not a replacement for native integration tests.

## Checks completed here

- All seventeen patch sizes and SHA-256 values matched the published `PATCH-LOCK.json`. This verifies local source integrity, not patch applicability to every product revision.
- Eight independent cache contract groups passed with ASan and UBSan instrumentation. The harness compiles against the extracted, unmodified contributor cache, including its cache-accounting patch, and the pinned ggml packed layouts.
- A deliberate removal of the short-read refusal was detected: the harness exited 1 at `short_read`. No contributor source was changed.
- The existing numerical comparator/tensor/control-parity suite passed: 36 tests. Synthetic equal rows passed the strict diagnostic; a changed row failed even though top-token/top-10 still matched.
- The first sanitizer execution failed because LeakSanitizer cannot operate under the runtime's tracing. The successful run used `ASAN_OPTIONS=detect_leaks=0`; no leak-free claim is made.

The [review harness and reproduction recipe](../../bench/moe-streaming/README.md), [prepared campaign](../../bench/moe-streaming/parity-preparation.json) and [machine-readable receipt](evidence/2026-11-16-moe-preparation/receipt.json) keep these checks reproducible. This work does not add native provider code to xLlama or alter its shipping build.

## CPU console inputs resolved from source

| Input                | Source-derived value / qualification                                                                                                                                               |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model                | `LFM2.5-24B-A2B-Q4_K_M.gguf`; exact size/hash are bound in the prepared campaign from `cpu-pilot-root.patch`                                                                       |
| Model folder         | Application LocalFolder `models/lfm25-24b-a2b-q4-k-m/`                                                                                                                             |
| Cache profiles       | `ua50-cpu-lfm24-cache-a2-v1` and `ua50-cpu-lfm24-cache-b512-v1`, each with its source-defined flag, persistent claim, stop, output and trace leaves                                |
| CPU envelope         | One worker, CPU-only/nonfused path, context 512, at most 384 generated samples; prompt must be at most 128 tokens and fit together with generation                                 |
| Prompt and template  | Russian greeting in `first-answer-root.patch`; same exact template/BOS rules retained by the cache profiles; greeting bytes included in the prepared campaign                      |
| Cache procedure      | A normal complete A2 result/trace precedes B512's fresh admission; never delete one-shot claims to force another run                                                               |
| Current public build | Run `43825061482` succeeded at contributor commit `61bf63ad…`; this is the newer seventeen-patch generation successor, not proof that its console prerequisites have been accepted |
| Signing template     | The published template binds a historical identity artifact, not the latest generation artifact. Do not reuse its package hash/version as a current install selection              |

Sources: [pinned preparation recipe](https://github.com/vetax-by/xllama/blob/61bf63ad541510a03a36993e60ec204f98bc0b3e/ua50/README.md), [native/application patches](https://github.com/vetax-by/xllama/tree/61bf63ad541510a03a36993e60ec204f98bc0b3e/ua50/patch), [signing template](https://github.com/vetax-by/xllama/blob/61bf63ad541510a03a36993e60ec204f98bc0b3e/ua50/signing/input-template.json). Stan's later [Discussion report](https://github.com/gianlucamazza/xllama/discussions/435#discussioncomment-19652093) adds bounded CPU observations; retain its revision/measurement scope rather than relabeling historical metadata.

## Remaining prerequisites

1. **Independent console reproduction:** done for the CPU stage on 2026-11-16. Instead of the fork's signed artifact, the seventeen patches (byte-identical to `PATCH-LOCK.json`) were built by this repository's CI at tag `evidence/2026-11-16-moe-24b-console` and installed in place, keeping LocalState. Both cache profiles completed with identical tokens and figures matching the fork's report ([evidence](evidence/2026-11-16-moe-24b-console/README.md)). Acceptance of the upstream PR 3 package remains a separate gate. A second checkpoint, LFM2.5-8B-A1B Q4_K_M, ran the same profiles once the pin carried its shape tuple and cache ceiling ([evidence](evidence/2026-11-16-moe-8b-console/README.md)).
2. **Real model parity:** the pinned GGUF is 14,526,258,464 bytes; this host has about 9.7 GiB total RAM and no swap. It cannot provide an entirely resident reference of these quantized weights here. A smaller checkpoint of the same `lfm2moe` architecture (LFM2.5-8B-A1B, 5,155,564,768 bytes at Q4_K_M) fits this host and can carry the mechanism's parity; the 24B itself still needs a larger host.
3. **Campaign finalization:** exact rendered IDs, resident-derived continuation IDs, native row-export support and prepared source/build receipts remain required. The JSON is deliberately `PREPARED_NOT_EXECUTABLE`; missing identities cannot be treated as wildcards.
4. **Upstream contribution:** review the first concrete extraction when available; avoid implementing a competing copy. NativeProvider and loader changes remain proposed PR 2 work.
5. **GPU:** no claim of console GPU execution, correctness, fit or speed follows from these CPU/cache/tool checks.

These are artifact, environment and implementation prerequisites, not a request for Stan to repeat information already in the repository. Record new test results on their actual source/package revision and keep fork reproduction separate from upstream acceptance.
