# xllama excellence audit — 2026-09-30

This is a dated audit, not a new source of truth for catalogue status or performance.
Current contracts remain in the [documentation ownership map](../../README.md).
The deliverables are this report, the [coverage matrix](coverage.md), the
[prioritized backlog](backlog.md), [proposed ADRs](adr-proposals.md), and machine-readable evidence in `evidence/`.

## Verdict

The audited package passes the ordinary Linux, ASan, CI, console and API gates,
but the targeted probes expose correctness and assurance gaps that prevent an
unqualified excellence verdict. There are **eight findings: four P1 and four P2**,
with no confirmed P0 in this bounded scope. Prioritize safe numeric admission,
surface-independent LoRA loading, process-wide compute admission and trustworthy
training oracles. These conclusions rest on the reproductions below, not a score.

## Baseline and method

- Source: `879c9d1d1ff7bcd0d8bed9284d6996e1937e695f`, isolated branch
  `docs/excellence-audit-20260930`; llama.cpp `7fe450e19305b828c199d602c23a8337aaa1f03b`.
- The primary checkout was clean at execution start. The three untracked benchmark
  files seen during planning were no longer untracked; the audit did not adopt the
  planning snapshot as current truth.
- Xbox started on unified `1.6.0.1088` (CI run `36651040136`, source `2274957e4ba320574769f5f7506d35e4c2b31a9c`).
  Audit tests use unified MSVC `1.6.0.1102`, CI run
  [36689739661](https://github.com/gianlucamazza/xllama/actions/runs/36689739661),
  built from the audit source. The package manifest and SHA-256 are recorded in
  [package.json](evidence/package.json). The update was in-place, without uninstall.
- Local static review covered the shared session/budget/sampling/writer boundaries,
  their UWP callers, the LAN parser and handlers, downloader promotion, validation
  harnesses, CI/governance, catalogue/evidence generation, and product contracts.
  This is a risk-directed review, not a claim that every line has been audited.
- No product code, accepted ADR, dependency pin, public interface, release or
  GitHub policy was changed. Standalone reproductions are audit evidence.

Status vocabulary: **PASS** means the stated predicate was observed; **FAIL** means
it was contradicted; **BLOCKED** means a named prerequisite prevented the check;
**NOT TESTED** means no such experiment was performed. A green build is not a
runtime proof. Severity and confidence are separate.

## Confirmed findings

### F01 — P1: unchecked generation numbers can overflow the shared budget

**Confidence: reproduced under UBSan and over the wire on the audit package.**

`uwp/api-server.cpp:479` converts JSON numbers directly to `int`, and
`src/bridge/prompt_budget.cpp:42` evaluates `tokens + reserve + 1` in signed `int`.
An otherwise valid request can provide `max_tokens=2147483647`. A ten-token
counter is already sufficient to overflow; UBSan reports signed integer overflow.
The defect is in the audit baseline, not introduced by this work.
Live requests with `max_tokens=-1`, `1.5` and `2147483647` all returned HTTP 200;
the negative case returned an empty assistant message. Wrong-typed input and tool
requests correctly returned 400. See [wire evidence](evidence/api-boundary.json).

Reproduction source: [prompt-budget-boundary.cpp](evidence/prompt-budget-boundary.cpp).
Compiler/runtime output: [prompt-budget-boundary.log](evidence/prompt-budget-boundary.log).
Compile with `c++ -std=c++17 -fsanitize=undefined -fno-sanitize-recover=undefined
-I include docs/audit/excellence/evidence/prompt-budget-boundary.cpp
src/bridge/prompt_budget.cpp src/bridge/chat_prompt.cpp -o /tmp/xllama-budget-repro`,
then run it. The sanitizer's nonzero exit is the observed failure, not a passing test.

Validate external numbers before narrowing; make the shared budget arithmetic safe
for all public C++ callers as well. Fixing only the HTTP handler leaves the core
contract vulnerable. Acceptance details are in the backlog.

### F02 — P2: HTTP framing is accepted without exact validation

**Confidence: confirmed in source and reproduced over the wire.**

`uwp/api-server.cpp:197` finds a substring rather than parsing a complete header
field, converts with `atoll`, clamps oversized lengths instead of rejecting them,
and marks the request valid after EOF even when fewer body bytes arrived than
declared (`:206–214`). Negative, malformed, duplicate and truncated lengths are not
validated as a framing contract. Partial reads also have no application deadline;
the listener dispatches directly into this blocking reader (`:1564`).

A request declaring a negative Content-Length returned 200. A preferences POST
with a complete JSON value but 100 fewer bytes than its declared length also
returned 200 and accepted the fixture sample. [Wire evidence](evidence/api-boundary.json)
records both responses. CORS also admits arbitrary origins and private-network
preflights; this is recorded exposure of the intentionally unauthenticated LAN
probe, not a claim that every browser permits the request.

This is a local-network service-integrity issue, not evidence of remote code
execution. One bounded wire probe cannot establish resistance to resource
exhaustion; no denial-of-service stress campaign is included.

### F03 — P1: API-created sessions omit catalogue LoRA policy

**Confidence: confirmed policy divergence and controlled device reproduction.**

`uwp/MainPage.cpp:2684–2705` resolves `lora` and `lora_scale` into session parameters.
The API's `CatalogueSessionPolicy` (`uwp/api-server.cpp:339`) contains context,
coding, embedding and backend flags, but no adapter. Both chat (`:452–457`) and
post-pull loading (`:1297–1309`) create sessions without adapter parameters.
`SessionHub::ensure_locked` reuses the session by model ID alone
(`include/xllama/session_hub.h:60–63`).

With identical prompt, system message, greedy sampling and generation cap, the
GUI and a subsequent API request using its session both returned the learned
`XLLAMA-LORA-OK` marker. After restarting on another model, the API created the
same named LoRA model without the adapter and returned a generic refusal. The
controlled requests, settings and device logs are in
[followup-proof.json](evidence/followup-proof.json). This proves load-order
dependence for this fixture, not a quality loss measurement for every adapter.
A shared catalogue-to-session policy and explicit session compatibility are
needed; see Proposal A in the proposed ADRs.

### F04 — P1: training gates can accept historical success

**Confidence: confirmed oracle defect in source.**

The `serve` gate scans all saved chats for the trigger and marker
(`scripts/validate-console-training.sh:226–247`); it does not bind them to the
conversation generated by this run. The `rate` gate accepts any existing `like`
in `training/samples.jsonl` (`:291`). The comments describe stronger protection
than those predicates provide. A historic success can mask a failed current
marker/rating, even if other parts of the script require the current app to run.

The initial merged-model trial also failed its marker under inherited settings.
Repeating with explicit greedy sampling and system/context settings produced a
**new** conversation containing the marker, independently checked rather than
accepted from a historical chat. See [initial trial](evidence/training-serve.log),
[controlled trial](evidence/training-serve-controlled.log) and
`merged_marker_controlled` in [followup-proof.json](evidence/followup-proof.json).
This isolates a non-hermetic gate; it does not establish which inherited setting
was causal, or a model regression. Every gate needs explicit starting settings.

The training script also lacks the expected-package guard present in console and
API suites. Record its live results as supporting observations, not sufficient
proof, until current-run output and package identity are independently checked.

### F05 — P2: console test restoration is not a complete transaction

**Confidence: confirmed harness behavior.**

The shell suites mutate settings, flags and conversations without restoring an
owner snapshot. TAESD restores a downloaded full VAE, not necessarily the decoder
present at entry (`scripts/validate-console.sh:1331–1343`). The Python helper is
stronger but verifies restored bytes only for files that existed: its absent-file
branch issues a best-effort deletion without reading absence back
(`scripts/console_test.py:170–186`), and always starts the app rather than restoring
the original running/stopped state.

The audit uses an independent private backup and final readback. Those safeguards
do not repair the reusable project harness. A passing inference gate alone must
not be interpreted as proof that the user's state survived.

### F06 — P2: known host export dependency alerts remain open

**Confidence: current GitHub alert snapshot.**

[dependabot.json](evidence/dependabot.json) records one high-severity transformers
alert and one low-severity torch alert against `diffusion/requirements.txt`.
These are host export dependencies, not evidence that the affected Python packages
ship inside the MSIX. [ADR 0003](../../adr/0003-diffusion-export-first-party.md)
already proposes the exporter migration and is explicitly **Proposed**.
Do not silently apply it or replace it with a competing ADR. Owner acceptance and
export/numerical/console parity are the existing closure path.

### F07 — P2: one valid concurrent pull returned an unexplained 500

**Confidence: observed live; root cause and frequency unresolved.**

The model-writer trial reached its final pair of concurrent API pulls after the
GUI/API rejection and recovery assertions. One response was HTTP 500 with
`{"error":"download failed"}` instead of success or explicit conflict; the
[trial failed at that assertion](evidence/model-writer.log). The weights were
restored by the harness, and a later controlled repeat of five pairs produced
ten successful responses. The original failure is retained; the successful
repeat neither explains it nor proves it fixed.

The generic catch in `uwp/api-server.cpp:1254–1261` loses the underlying exception
information. Capture bounded stage/error diagnostics before another endurance
campaign. Do not label concurrency as the established cause, or claim corruption
that the evidence did not demonstrate.

### F08 — P1: GUI diffusion bypasses process-wide API admission

**Confidence: source inspection and controlled live overlap.**

`MainPageController::StartDiffusion` (`uwp/MainPage.cpp:2189–2225`) checks only GUI
state and starts `run_diffuse()` without acquiring the hub used by API image/chat
requests. During a four-step GUI image generation, a chat API request returned
HTTP 200 while `diffuse-progress.txt` was `unet 1/4` both before and after the
request. See `gui_image_api_chat_overlap` in
[followup-proof.json](evidence/followup-proof.json).

The documented whole-process single-slot behavior therefore does not cover this
GUI path. This is not evidence of two text Sessions: it is inconsistent admission
of competing compute jobs. Two concurrent image requests could also contend for
shared diffusion files, but the audit deliberately did not run that destructive
variant. GUI training similarly releases the hub after freeing chat memory; its
overlap with API work was reviewed statically, not reproduced live. Resolve job
admission separately from resident model identity; see Proposal C.

## Strengths and limits

The shared core has explicit ownership boundaries, per-backend sampler helpers,
exact-token prompt fitting, staged download promotion and a testable C++ surface.
The catalogue and research generators provide useful reproducibility gates. The
current CI snapshot confirms all four required checks on the selected commit;
the live ruleset is captured independently from its ADR.

The product's stated limits are appropriate: Dev Mode inference for games and
assistants, a showcase UI, and a foreground LAN integration probe. Lack of a retail
certification or game-engine SDK is not misclassified as a broken shipped feature.
The audit does not certify Store readiness, model licensing, Series X behavior,
retrieval relevance, energy efficiency or thermal endurance.

Performance conclusions remain bounded by the actual workload, package and model
artifact. Existing H9 is a small deterministic capability suite, not a general
quality benchmark. Existing schema/provenance checks cannot manufacture missing
power/ambient/thermal measurements or homogeneous cross-device comparisons.
Follow the owning benchmark policy rather than publishing audit timings as new
product headlines.

## Additional observations

- Fresh device training completed eight epochs and evaluated its own marker
  successfully. Runtime LoRA application and current-run preference appends were
  verified separately; neither is substituted for the failed API-first policy.
- Text abort was triggered through the same worker abort flag used by Cancel.
  The pending user message survived and a subsequent API request completed. The
  trial aborted before assistant tokens were persisted; partial-output cancellation
  and physical gamepad navigation are not claimed.
- A download was stopped on an observed intermediate NDJSON progress event. No
  completion marker was published; retry succeeded and the recovered model matched
  the catalogue SHA-256. An earlier attempt based on WDP partial-file sizes missed
  its rendezvous and proved no interruption; that failed attempt remains recorded.
- The raw [default-model timing trial](../../../bench/results/2026-09-30-excellence-default.csv)
  and [H9 trial](../../../bench/results/2026-09-30-excellence-h9.jsonl) are new dated
  evidence, not replacements for published benchmark selectors. The protocol was
  written before running. One timing row ended after very few generated tokens;
  do not interpret mixed lengths as homogeneous sustained decode. All eight H9
  responses were read against their prompts, and the independent reading agreed
  with the four passing and four failing task verdicts. In particular, the default
  model failed capital, arithmetic, extraction and translation tasks in this small
  suite; it is not validated as a general factual assistant by fast generation.

## Final state and restoration

[Restoration proof](evidence/restore-proof.json) records SHA-256 equality for all
22 original non-model files, readback absence for 37 removed fixture files, exact
restoration of the original VAE bytes, and verification of the catalogue-pinned
default weight. Test-created training/model directories were removed; no empty
fixture directories remain. The new trained GGUF is retained privately with its
hash in the proof, not committed to the repository.

The audited in-place package **1.6.0.1102 remains installed**. The app is stopped
and the original absence of `api.flag` is restored. The initial process state was
not independently sampled; its saved log ended with headless benchmark exit, so
no exact process-state equality claim is made. Unaffected model trees were not
exhaustively hashed at entry; the restoration claim is deliberately scoped to
captured files and affected verified weights, not the entire console filesystem.

## Reading the result

Use the [coverage matrix](coverage.md) to distinguish runtime proofs from static
review and gaps. Use the [backlog](backlog.md) for implementation order and
acceptance tests. This audit produces evidence and remediation requirements;
it does not certify the absence of further defects or implement the proposed fixes.
