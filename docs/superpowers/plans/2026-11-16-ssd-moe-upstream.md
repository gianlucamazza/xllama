# SSD MoE upstream integration: proposed PR sequence

**Status:** Proposed; planning only. No runtime integration or console acceptance is delivered by this document.

**Plan of record:** sequencing, gates and acceptance for this work are milestone **M2** of the [evidence-gated MoE engineering program](2026-11-24-moe-engineering-program.md) (#532). This document keeps the proposed PR boundaries; where the two disagree, the program wins.

**Goal:** Bring the existing demand-loaded expert implementation into xLlama through independently reviewable contributions, beginning with the CPU foundation.

**Architecture:** Reuse Stan's Reader, cache leases and native provider. Keep shared weights resident and selected expert matrices in a bounded cache. Separate the portable cache, native loader/execution integration, UWP activation and GPU bridge so each has its own acceptance evidence.

**Stack:** C++17, pinned llama.cpp/ggml, doctest, Python numerical comparators, C++/WinRT UWP and D3D12/HLSL.

**Requirements:** [#479](https://github.com/gianlucamazza/xllama/issues/479), [#480](https://github.com/gianlucamazza/xllama/issues/480), [#481](https://github.com/gianlucamazza/xllama/issues/481), and the [collaboration reply](https://github.com/gianlucamazza/xllama/discussions/435#discussioncomment-19653181). Existing contracts remain owned by [architecture.md](../../architecture.md), [uwp-constraints.md](../../uwp-constraints.md) and [AGENTS.md](../../../AGENTS.md).

## Source baseline and evidence

- Repository inspected: `gianlucamazza/xllama` at `87d5a4635abca01a3b484e1488a2a9d043de9be2`.
- Contributor source inspected: `vetax-by/xllama` at `61bf63ad541510a03a36993e60ec204f98bc0b3e`; its preparation baseline is xLlama `4db51522c850e62d0bb24234dfeb48e55f4017d7`.
- Both use llama.cpp `d81235049384534c167caea52b85a694f6103d14`. Recheck the product pin before implementing each native patch.
- The contributor's [source ledger](https://github.com/vetax-by/xllama/blob/61bf63ad541510a03a36993e60ec204f98bc0b3e/docs/ua50-ssd-moe.md), [preparation recipe](https://github.com/vetax-by/xllama/blob/61bf63ad541510a03a36993e60ec204f98bc0b3e/ua50/README.md) and [ordered patch lock](https://github.com/vetax-by/xllama/blob/61bf63ad541510a03a36993e60ec204f98bc0b3e/ua50/PATCH-LOCK.json) already provide the source inputs. Maintainers inspect these instead of asking the contributor to repeat them.
- [Windows run 43825061482](https://github.com/vetax-by/xllama/actions/runs/43825061482) completed successfully on the inspected contributor commit. This is build evidence, not console execution or GPU correctness.
- Stan's [console report](https://github.com/gianlucamazza/xllama/discussions/435#discussioncomment-19652093) describes bounded CPU completion and cache observations. Full-checkpoint resident/streamed numerical parity and console GPU correctness remain open. Some earlier source-ledger sections describe older revisions; retain revision-specific scope.

The existing [SSD assessment](../../ssd-inference-assessment.md) is historical context. Its conclusion predates this contribution; new evidence needs a dated addendum rather than silently rewriting the original results. Catalogue admission remains governed by [#273](https://github.com/gianlucamazza/xllama/issues/273).

## PR boundaries

| PR  | Proposed title                                         | Tracker                    | Depends on  | Review outcome                                               |
| --- | ------------------------------------------------------ | -------------------------- | ----------- | ------------------------------------------------------------ |
| 1   | `feat: add bounded expert cache foundation`            | #479                       | None        | Portable cache contract and deterministic failure tests      |
| 2   | `feat: integrate optional CPU MoE expert streaming`    | #480                       | PR 1        | Native loader/execution integration and numerical evidence   |
| 3   | `feat: add developer-only Xbox MoE CPU activation`     | #481, CPU stage            | PR 2        | Windows/UWP ownership, quota and console lifecycle evidence  |
| 4   | `feat: add experimental selected-expert GPU execution` | #481, GPU stage; #421/#422 | PRs 2 and 3 | Console dispatch and numerical proof before performance work |

These are proposed contributions, not assignments or delivery promises from Stan. Invite his draft PR for the first coherent extraction; maintainers own upstream compatibility review, CI integration and independent reproduction. GPU work must not block the CPU contribution. Independent tests of the fork may run before PR 1 and must be labelled as fork evidence.

## Global constraints

- Preserve the default resident path, existing SessionHub ownership and shared compute admission.
- Keep the feature explicitly opt-in and experimental. No automatic fallback to a full resident load after streamed admission or decode failure.
- Do not commit edits inside the llama.cpp submodule. Carry native changes as explicit patches against the recorded pin.
- Do not put optional MoE patches in the unconditional `patches/0*-*.patch` path consumed by `scripts/apply-uwp-patches.sh`.
- Reuse existing interfaces and preparation checks; file placement below is the proposed upstream mapping, not an API redesign prerequisite.
- Preserve attribution and MIT notices. Do not import the whole seventeen-patch application history, signing machinery or generated receipts as the first contribution.
- Keep fixture math, full-checkpoint parity, console lifecycle, GPU correctness and performance as separate evidence. No performance table is copied into this plan.

## Review focus

1. All cache entries leased: acquisition must refuse without eviction, hidden allocation or dangling reads (PR 1).
2. Short/throwing read after victim selection: the entry must remain invalid and a later request must read again (PR 1).
3. Workers without assigned chunks: every participating worker must reach the lifetime barriers before lease release (PR 2).
4. Stop or quota pressure during pending I/O: keep backing resources until actual terminal completion (PR 3).
5. Unknown GPU fence completion: retain resources, report incomplete closure and publish no successful result (PR 4).

## PR 1: portable cache foundation

**Proposed files:** `src/bridge/moe/expert_cache.h`, `tests/test_expert_cache.cpp`, `tests/CMakeLists.txt`, and the module contract in `docs/architecture.md`.

**Source:** `expert_cache.h` added by `ua50/patch/runtime-llama.patch`, with the applicable accounting additions from `cache-ab-llama.patch`. Keep the implementation private initially; place it in `xllama::moe` through a mechanical upstream adaptation. Preserve ggml's actual quant block layouts rather than inventing duplicate packed definitions.

**Interfaces retained:** `Reader::read(uint64_t offset, unsigned char* destination, size_t bytes) -> size_t`; `Cache::acquire(uint32_t tensor_id, uint64_t expert) -> Cache::Lease`; move-only `Lease::data()`, `Lease::size()` and `Lease::reset()`; `Cache::required_owned_bytes(size_t slots, size_t max_expert) -> size_t`.

- [ ] Port deterministic tests first: `expert_cache_lru_order`, `expert_cache_pinned_entries`, `expert_cache_all_slots_pinned`, `expert_cache_short_read_not_published`, `expert_cache_throwing_reader`, `expert_cache_budget_and_extent_overflow`, and `expert_cache_mixed_quant_bytes`. Assert payload bytes and actual read calls, not just counters.
- [ ] Verify the tests fail before importing the implementation; reuse contributor fixtures where available. Extract the existing cache and Reader contract with its single-coordinator/external-serialization rule and reader/cache/lease lifetime documented.
- [ ] Prove exactly admitted versus one-byte-short owned budgets, invalid tensor extents, lease move/reset behavior and reacquisition after failed reads. Keep request hits, byte hits and logical read bytes distinct.
- [ ] Register tests in the existing doctest target. Run `cmake --preset linux-test`, the build and `ctest --test-dir build/linux-test --output-on-failure`; run the ownership cases with ASan and UBSan enabled.
- [ ] Commit as `feat: add bounded expert cache foundation`, with source attribution and a concise contract update.

**Exit:** host cache tests and sanitizers pass; no model loader, public configuration, Windows reader or GPU code is activated. NativeProvider, which depends on ggml buffers, enters with PR 2. If Stan's smallest coherent extraction needs a minimal native hook in PR 1, review that explicit boundary adjustment in his draft.

## PR 2: native CPU loader and execution

**Proposed files:** `patches/moe/0001-cpu-expert-provider.patch`, `scripts/apply-moe-patches.sh`, `CMakeLists.txt`, `src/bridge/moe/cpu_expert_loader.{h,cpp}`, `tests/test_moe_native.cpp`, `tests/test_moe_patch_application.py`, `tests/CMakeLists.txt`, `patches/README.md`, and `docs/architecture.md`.

**Source:** existing `NativeProvider`, `OwnedExpertInput`, `OwnedExpertLoad`, `inspect_expert_input` and `load_owned_experts`, plus loader, allocator and CPU `MUL_MAT_ID` changes from the contributor's native patches. Consume PR 1's cache; do not create a second implementation inside the patch.

- [ ] Add patch-application tests for the exact pin, a conflicting source, application twice and a clean ordinary build. The optional helper must refuse an unexpected base or conflict; the ordinary UWP patch route remains unchanged.
- [ ] Add `XLLAMA_ENABLE_MOE_STREAMING`, default `OFF`, to host build configuration and an explicit optional source-preparation route. A requested enabled build without the required native hooks must fail configuration rather than silently disable streaming. Patch scope must remain explicit on both host and Windows builds.
- [ ] Port generated native GGUF fixtures before wiring production loads. Cover resident versus streamed full logit rows, routing, mixed roles/quants, short reads and worker drain. Test one and multiple workers at this layer; that does not expand the console capability envelope.
- [ ] Install the owned provider before tensor allocation; preserve shared resident tensors, non-mmap expert loading and all unsupported-operation refusals. Keep the default factory unchanged. Verify error paths through real native graph execution, including workers that receive no chunks.
- [ ] Reuse `scripts/compare-logits.py` and `tests/test_compare_logits.py` for per-row numerical comparison. Run identical teacher-forced token sequences with the same checkpoint SHA, template, context, quant kernels and threads; export each selected prefill/decode row with provenance. Freeze explicit tolerances before the candidate run; do not inherit comparator defaults without justification or loosen limits after seeing results.
- [ ] Run a full real-checkpoint resident reference on a sufficiently provisioned host, against small and large streamed caches. If resources are unavailable, retain the PR as a draft and record the missing gate; synthetic parity or repeated token IDs cannot substitute for this evidence.
- [ ] Run feature-off resident regressions, feature-on native fixtures and ASan/UBSan. Record separate configurations in the PR. Commit as `feat: integrate optional CPU MoE expert streaming`.

**Exit:** the native patch is reproducible, the CPU fixtures and full-checkpoint gate pass, and the ordinary path has no regression. No catalogue or app activation is added.

## PR 3: developer-only Xbox CPU activation

**Proposed files:** `include/xllama/cpu_expert_session.h`, `src/bridge/cpu_expert_session.cpp`, `src/platform/moe/staged_reader_core.h`, `src/platform/moe/windows_held_reader.h`, `src/bridge/session.cpp`, `uwp/App.cpp`, `uwp/xllama.vcxproj`, `uwp/ggml-uwp.vcxproj`, `tests/test_moe_reader.cpp`, `tests/test_moe_session.cpp`, and `docs/console-validation-runbook.md`.

**Source:** contributor's owned input, staged reader, CPU session/controller and one developer CPU activation mode. Reuse the existing `CpuExpertOptIn` and session lifecycle contracts. Avoid importing the newer LAN generation API or every historical flag profile in this PR.

- [ ] Port fake-boundary tests for pending completion, stop, identity change, short reads, decode failure and quota refusal. Specifically prove that cancellation returning does not release an outstanding I/O buffer and that quota failure cannot invoke the resident factory.
- [ ] Integrate the existing held-file reader with the process-wide writer and compute owners. Charge cache, metadata, staging, shared tensors, model state, workspace and output to fresh app admission; monitor pressure during execution.
- [ ] Wire one opt-in developer activation profile into the existing headless dispatch. Preserve ordinary UI/LAN behavior and one resident-session ownership. Reuse source-defined prompt/template and trace validation instead of inventing a new request protocol.
- [ ] Build/package through the normal Windows MSVC CI path with the explicit native preparation. Run real Windows held-file/I/O lifecycle tests separately from portable fakes.
- [ ] Reproduce CPU completion on the upstream candidate package: record source/package/model identities, effective configuration, native stop reason and normal worker/I/O/file/process closure. Repeat with small and large cache admission. Retain fresh quota and measured-memory scope; distinguish hashing/load time, generation timing and logical read bytes.
- [ ] Add a dated evidence addendum to `docs/ssd-inference-assessment.md` and link evidence from the runbook. Update benchmark summaries only through their existing generator when admissible result data is added. Commit as `feat: add developer-only Xbox MoE CPU activation`.

**Exit:** accepted Windows build plus upstream console lifecycle evidence. Dev Mode failure is an environment blocker, not a model failure. Catalogue promotion, general chat, tools, KV continuation and new LAN endpoints remain separate decisions.

## PR 4: selected-expert GPU correctness

**Proposed files:** `patches/moe/0002-selected-expert-gpu.patch`, focused bridge units under `src/bridge/moe/`, shader units under `shaders/`, their UWP project registrations, `tests/test_moe_gpu_contract.cpp`, `docs/gguf-gpu-decode.md`, and the console validation runbook.

**Source:** contributor's synchronous Q4_K/Q6_K selected-expert bridge and identity/primitive/model proof stages. Preserve one GPU matrix slot and one outstanding submission for the first contribution.

- [ ] Port contract tests for unsupported quant/shape, failed dispatch, stop and unknown fence state. Assert that no successful result or lease release occurs before actual completion.
- [ ] Port the existing explicit bridge without advertising general routed-operator support. Retain CPU graph routing/packing and exact host lease ownership until the fence completes; account for GPU resources within shared UMA admission.
- [ ] Establish actual console adapter identity and dispatch; run primitive checks on identical inputs before model-level proof. Capture enough numerical evidence to reproduce declared comparisons. Do not treat a saved online summary as exported full-row parity.
- [ ] Reuse [ADR 0010](../../adr/0010-logit-parity-reference-spread.md) where its requirements apply. Its existing first-layer `MUL_MAT` coverage is not automatically routed `MUL_MAT_ID` coverage: the MoE proof must exercise the selected expert operators explicitly, with thresholds and coverage declared before measurement.
- [ ] After correctness and closure pass, run matched CPU/hybrid measurements under #421/#422. Preserve model, prompt, cache, worker and warm/cold conditions, and include transfers and read waits in end-to-end results. A correct GPU path can remain slower than CPU.
- [ ] Keep the backend experimental and opt-in. Commit as `feat: add experimental selected-expert GPU execution`; do not bundle automatic routing promotion or catalogue changes.

**Exit:** actual console correctness and lifecycle proof. Performance promotion is a later evidence-based decision; compiling a GPU shader does not satisfy this gate.

## Maintainer workflow and immediate next actions

- [ ] Review Stan's first draft against PR 1; settle only the extraction boundary that the code actually requires. No new questionnaire or external assignment is needed.
- [ ] Derive reproduction inputs from the pinned fork recipe and build receipts; only request genuinely unavailable information after checking those sources.
- [ ] Keep #479 for PR 1, #480 for PR 2, and #481 open until both CPU and GPU stages have explicit outcomes. Do not close #481 when only CPU acceptance lands.
- [ ] Submit each implementation PR independently against current `main` after its prerequisite merges; rebase and rerun native patch checks on every pin change.
- [ ] Required branch checks and squash-only integration follow AGENTS.md. This planning PR may be reviewed now; it does not authorize unverified implementation merges or assert that today's console tests have occurred.
