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

1. **Engineering consolidation.** Finish the reproducibility, runtime and
   client validation work in [#395](https://github.com/gianlucamazza/xllama/issues/395).
   The [v1.11.0 milestone](https://github.com/gianlucamazza/xllama/milestone/8)
   tracks the next release backlog.
   Check the exact distributed MSIX provenance under
   [#635](https://github.com/gianlucamazza/xllama/issues/635) and record
   versioned LAN SDK and Open WebUI runs under
   [#634](https://github.com/gianlucamazza/xllama/issues/634).
2. **Agent clients on the LAN API.** Multi-step agent tasks fail on Series S at
   the catalogue context of 2048 tokens: the reasoning or the transcript uses
   up the budget, or the model writes its plan as text. Fix the tool path that
   always enables thinking ([#655](https://github.com/gianlucamazza/xllama/issues/655)),
   add per-request reasoning control ([#648](https://github.com/gianlucamazza/xllama/issues/648)),
   and measure a larger context for `lfm25-8b-a1b` on the console before cataloguing
   it ([#649](https://github.com/gianlucamazza/xllama/issues/649), host rung done).
   Native Ollama chat and model residency wait on a product decision: whether a
   LAN client may expire or unload the model the console user has loaded.
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
   blocked on the separately tracked llama.cpp pin work. The issue and MoE
   program describe different scheduling states for the pin bump. Reconcile
   them before scheduling the full-catalogue regression or beginning M1.
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
revision-specific evidence and prerequisites. The next contribution gate is a
concrete cache candidate in #479, followed by native CPU numerical parity in
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
