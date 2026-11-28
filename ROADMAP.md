# xllama Roadmap

## Current product state

The current Developer Mode release is **1.11.0.0**. Its package uses the
`GianlucaMazza.xllama` identity. It supports Xbox Series S and Series X.
The project measures on Series S, with independent Series X results scoped
in the [positioning guide](docs/positioning.md#series-x).
See the [release notes](docs/releases/v1.11.0.md),
[installation guide](docs/install-release.md), and [changelog](CHANGELOG.md)
for released and unreleased changes.

xllama targets local inference for games and AI assistants. The app provides
a gamepad UI and an opt-in LAN integration surface. The
[model matrix](docs/model-matrix.md) tracks catalogue status, and the
[benchmark summary](docs/benchmarks.md) holds measured performance.

## Current follow-up priorities

1. **Engineering consolidation.** 1.11 shipped. Finish the reproducibility,
   runtime and client validation work in
   [#395](https://github.com/gianlucamazza/xllama/issues/395). The versioned
   client journey ([#634](https://github.com/gianlucamazza/xllama/issues/634))
   and MSIX provenance ([#635](https://github.com/gianlucamazza/xllama/issues/635))
   are closed; every main build of the unified MSIX is attested since
   [#637](https://github.com/gianlucamazza/xllama/pull/637). Open follow-ups:
   chat fields that are ignored instead of refused
   ([#665](https://github.com/gianlucamazza/xllama/issues/665)), Gemma 4
   reasoning split ([#671](https://github.com/gianlucamazza/xllama/issues/671)),
   a per-request reasoning token budget
   ([#680](https://github.com/gianlucamazza/xllama/issues/680)), and the
   unreachable upstream runtime repositories
   ([#646](https://github.com/gianlucamazza/xllama/issues/646)).
2. **Agent clients on the LAN API.** Done in 1.11: the tool path reasons only
   when the model does ([#659](https://github.com/gianlucamazza/xllama/pull/659)),
   a request can switch reasoning with `reasoning_effort`,
   `chat_template_kwargs.enable_thinking` or Ollama `think`
   ([#670](https://github.com/gianlucamazza/xllama/pull/670)), `lfm25-8b-a1b`
   opens at `n_ctx` 8192 ([#660](https://github.com/gianlucamazza/xllama/pull/660)),
   native Ollama `/api/chat`, `/api/generate` and `/api/ps` are served while
   the console app keeps its model
   ([#663](https://github.com/gianlucamazza/xllama/pull/663),
   [#666](https://github.com/gianlucamazza/xllama/pull/666)),
   `/api/catalog` lists the downloadable catalogue
   ([#662](https://github.com/gianlucamazza/xllama/pull/662)), and a replayed
   LFM tool call renders before its result
   ([#669](https://github.com/gianlucamazza/xllama/pull/669), issue
   [#664](https://github.com/gianlucamazza/xllama/issues/664)). Still open:
   multi-step agent runs measured on Series S at the larger context (in
   progress). Routes and residency rules are in the
   [LAN API reference](docs/api-endpoint.md).
3. **Release readiness.** Use the [release readiness plan](docs/release-readiness.md)
   for the Store SKU, licence review and App vs Game decision. Developer Mode
   is the supported installation path. The research release still needs
   physical power and thermal evidence before a publication-grade Series S
   baseline, and matched device evidence before cross-platform claims.
   See the [research package](paper/README.md) and
   [benchmark protocol](bench/README.md).
4. **Catalogue and platform exploration.** Follow the
   [catalogue milestone](https://github.com/gianlucamazza/xllama/milestone/4),
   the [model matrix](docs/model-matrix.md), and the
   [new model types milestone](https://github.com/gianlucamazza/xllama/milestone/7).
   Keep each candidate behind host smoke and console qualification. Keep the
   models in [#522](https://github.com/gianlucamazza/xllama/issues/522)
   blocked on the separately tracked llama.cpp pin work. The owner approved a
   post-M0 evaluation. Selecting a revision and running the full-catalogue
   regression need a separate PR. M1 also requires its own approval and
   either a landed pin bump or an explicit deferral.
5. **Vendor lifecycle.** Follow
   [#85](https://github.com/gianlucamazza/xllama/issues/85) and the
   [vendor lifecycle plan](docs/vendor-lifecycle-plan.md) for the patched
   runtime rebuild, package qualification, and retirement gates. Track upstream
   source availability and verified mirrors in
   [#646](https://github.com/gianlucamazza/xllama/issues/646). Keep
   [DirectML routing](docs/uwp-constraints.md#5-directml-works-on-gpu-headless-but-loses-to-cpu-on-small-models)
   policy changes tied to the measured workload and the product decision.
   Widening the on-device training filter also remains blocked on runtime
   support. See the [training architecture](docs/training-architecture.md).

## Hardware-aware performance engineering

[Program #694](https://github.com/gianlucamazza/xllama/issues/694) coordinates the [performance report and host measurements](docs/research/gpu-acceleration-report.md), [hardware matrix](docs/research/gpu-acceleration-baseline.md), [technology decisions](docs/phase15-re-opt.md), and [benchmark methodology](bench/README.md). The numbered phases below organize evidence; each owner retains its approval and numerical gates. The report holds the program's [benchmark coverage](docs/research/gpu-acceleration-report.md#benchmark-coverage).

| Phase                     | Prerequisite / owner                                                                                                                                 | Deliverable and acceptance                                                                                                                                                                                                                                                                                 |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0. Discovery and baseline | Fixed source, model, package, OS, designation and device receipt. #395, #421, #422, #686.                                                            | Trace the timed intervals end to end. Reproduce archived baselines and classify evidence by host, operator/device, replay and hypothesis. Preserve null, refused and failed runs.                                                                                                                          |
| 1. Runtime efficiency     | Attributable phase trace, CPU reference and fail-closed measurements. #421, #703, #714.                                                              | Separate model open and prefill. Account for timed and untimed GPU work. Evaluate one dispatch/barrier/residency change per paired campaign with numerical, cleanup and reset checks.                                                                                                                      |
| 2. Memory and storage     | Session admission and current per-process budgets. #479 → #480 → #481; proposed research #536.                                                       | Test simultaneous requests, model swap, context growth, cancellation, allocation refusal and a subsequent successful session. Keep Reader/cache bounded; require native resident/streamed logits before target-device acceptance. Use A2/B512 fixed-trace controls, then separate CPU and GPU device runs. |
| 3. Modern inference       | Exact pin/patch review, tokenizer and architecture support, sufficient admitted memory. #522–#526, #589, #590.                                       | Rank changes per workload. Compare source/reference logits, generation, draft acceptance, quality and resource cost. Include open chat and repetitive code in decoding decisions. Reject an optimization on a failed declared cell.                                                                        |
| 4. Adaptive execution     | Measured profiles, deterministic defaults, manual precedence and a bounded search. [#704](https://github.com/gianlucamazza/xllama/issues/704), #422. | Compare threads, ubatch, layers and supported KV types on representative and hold-out prompts. Version results by device/driver/designation, package, model and budget. Exercise expired/unsupported profile and memory-pressure fallback.                                                                 |
| 5. Distributed research   | [#687](https://github.com/gianlucamazza/xllama/issues/687) and #688–#691.                                                                            | Compare independent replicas first. Then assess link costs, partition capacity, numerical equivalence and node loss on an opt-in testbed. Report single-request latency, throughput at concurrency and largest admitted model separately.                                                                  |
| 6. Consolidation          | Candidate tests, exact Linux/Windows build status, independent review and target-device receipts. #395, #421, #422.                                  | Archive matched A/A noise and A/B effect, publish missing counters, restore defaults, and qualify each SKU. Keep fork credit and local patch exit conditions. Maintainer review governs any merge, release or promotion.                                                                                   |

## Xbox GPU acceleration program

Complete attributable profiling in
[#421](https://github.com/gianlucamazza/xllama/issues/421) and the matched
device matrix in [#422](https://github.com/gianlucamazza/xllama/issues/422).
Both still have Series X evidence gates. Review upstream or fork candidates
under [#589](https://github.com/gianlucamazza/xllama/issues/589) and
[#590](https://github.com/gianlucamazza/xllama/issues/590) against the
[GPU baseline](docs/research/gpu-acceleration-baseline.md) and
[GGUF GPU decode gates](docs/gguf-gpu-decode.md) before proposing promotion.

## Optional SSD MoE contribution

The [MoE engineering program](docs/superpowers/plans/2026-11-24-moe-engineering-program.md)
owns sequence and gates. M0 is the approved execution slice. M2 remains the
contributor-led track in [#479](https://github.com/gianlucamazza/xllama/issues/479)
through [#481](https://github.com/gianlucamazza/xllama/issues/481). The
[readiness report](docs/research/moe-upstream-readiness.md) holds the
revision-specific evidence and prerequisites. The portable Reader, cache and lease
candidate for #479 merged in
[#668](https://github.com/gianlucamazza/xllama/pull/668). The next
contribution gate is native CPU integration and numerical parity in
[#480](https://github.com/gianlucamazza/xllama/issues/480). Upstream Xbox CPU
and GPU acceptance remain separate stages in #481. Catalogue admission keeps
its own gate in [#273](https://github.com/gianlucamazza/xllama/issues/273).
The out-of-core plan in [#536](https://github.com/gianlucamazza/xllama/issues/536)
is proposed research.

## Definition of done

Runtime changes require host tests, the relevant Windows CI builds, and
on-console validation for hardware-specific behavior. Store measured evidence
under `bench/results/` and update the owning document listed in the
[documentation index](docs/README.md). Merge through the
[main branch ruleset](docs/adr/0002-main-branch-governance.md).

The [historical roadmap](https://github.com/gianlucamazza/xllama/blob/0432640904fcace3d7649455f29dded87a814153/ROADMAP.md)
preserves earlier phase checklists and decisions.
