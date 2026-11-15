# Engineering excellence execution — 2026-11-07

Status: **in progress; stable release blocked**. Tracking: [#395](https://github.com/gianlucamazza/xllama/issues/395).
Milestone ledger: [report](execution-2026-11-07/milestone-report.json), [exact-head integration receipts](execution-2026-11-07/integration-receipt.json), [SHA-256 manifest](execution-2026-11-07/SHA256SUMS). Validate from that evidence directory with `sha256sum -c SHA256SUMS`.

Eight-week planning window: 7 November 2026–1 January 2027. Dates are planning
targets, not automatic approvals or promises of hardware availability.

This is the current program plan and dated discovery receipt. The September
[audit](README.md), its evidence, accepted ADRs and historical benchmark files
remain unchanged. Current technical facts retain their homes in the
[documentation ownership map](../../README.md).

## Discovery boundary and reproducibility

Baseline: `ea40b28deb5c9601863dd5b99709360b1bdee5bf`; llama.cpp remains the
submodule pin recorded in the tree. Discovery inspected all current upstream
branches, open issues and PRs, workflow definitions, release collection and the
ten public forks. Fork branch/commit comparisons and the bjoern shader review
are in [community-forks.md](../../community-forks.md). The machine-readable
[snapshot](execution-2026-11-07/discovery.json) records refs and source URLs.
Reviewing branch scope is not an assertion that every changed line was audited.

Commands to repeat the discovery are ordinary GitHub REST GETs against
`repos/gianlucamazza/xllama/{branches,issues,pulls,releases,actions/workflows}`
with `per_page=100`, plus `forks` and each fork's branches. Follow pagination
when a collection fills a page. Compare immutable SHAs, never a moving branch
name. Use `git rev-list --left-right --count main...<fork-sha>` and inspect the
exclusive log and diff from the merge base; ancestry-only commits may already
be represented by a squash merge. Capture exact-head check runs **and** status
contexts, not an older PR's green badge.

At discovery, the required-check API returned no checks on baseline main or the
two old draft tips. Earlier workflow successes exist on older SHAs; they do not
certify the current commits. The active main ruleset requires an up-to-date
head and four checks: Linux, both UWP package variants and C/C++ CodeQL. Follow
[ADR 0002](../../adr/0002-main-branch-governance.md); do not use an owner bypass.

### Existing tools to retain

| Responsibility                            | Existing owner / tool                                                                                                                     |
| ----------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Linux correctness and sanitizers          | `linux-test`, `linux-asan`, separate UBSan option; `build-linux.yml`                                                                      |
| Windows packaging and vendor verification | `build-uwp.yml`, composite package action, SHA256SUMS; [vendor lifecycle](../../vendor-lifecycle-plan.md)                                 |
| Console / LAN / training gates            | [console runbook](../../console-validation-runbook.md), `validate-console.sh`, `validate-api.sh`, `validate-console-training.sh`          |
| GPU numerical and product acceptance      | [GGUF GPU design](../../gguf-gpu-decode.md), `bench-d3d12-selftest.sh`, `validate-logit-parity.sh`                                        |
| Benchmark capture and admission           | `run-xab.sh`, `bench-xbox-ort.sh`, CSV/sidecar validators; [methodology](../../../bench/README.md)                                        |
| Claims, publication and negative results  | `paper/claims.json`, `build-research-package.py`, [research release](../../../paper/release.md), [hypotheses](../../phase7-hypotheses.md) |
| Generated numbers                         | `benchmark-summary.json` → generated benchmark pages; never edit their numeric tables by hand                                             |

## Complete initial issue and PR triage

Priority is impact on acceptance, not popularity. P0 means an immediate critical
security/correctness defect; none was confirmed in this bounded review. P1
blocks the affected claim/release path; P2 is incremental assurance or optional
performance; P3 is desk research. Open state alone does not mean implementation
has not happened.

| Tracker | Verified state / disposition                                                                                                                                                                         | Priority, owner, acceptance dependency                                                                                                                |
| ------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| #263    | Open; body describes obsolete pins. Accepted ADR 0003 and merged #378 already implement first-party export with patched diffusion pins. Do not migrate twice or close alerts from source inspection. | P1 Security + Release: isolated resolution/export comparison receipts, actual Series S diffusion/TAESD gate and peak-memory evidence, security rescan |
| #84     | Open; shipping GenAI overlay retained. Package availability is a removal candidate, not proof of XAML/DML compatibility.                                                                             | P2 Security: upstream fix inclusion + package build + console smoke                                                                                   |
| #85     | Open; two overlays are independently hash-checked.                                                                                                                                                   | P1 Release dependency: keep both until their individual removal gates pass                                                                            |
| #86     | Open; ORT chunk fix and AppContainer path fix both needed. Issue comments/upstream refs, not July prose, own live upstream status.                                                                   | P2 Security: verify release source inclusion, external-data reload/restart on device                                                                  |
| #313    | Open, explicitly nonblocking multi-column prefill optimization; not implemented by bjoern fork.                                                                                                      | P2 GPU: preregistered own-shader experiment, numerical gate, same-device A/B                                                                          |
| #374    | Open; upstream Series X inference remains unvalidated. Fork has third-party resource-only receipts with labeling/provenance limitations.                                                             | P1 for Series X claim/full program: exact source/package/model binding and device acceptance suite                                                    |
| #273    | Open MoE watch; prior H2 FAIL remains frozen.                                                                                                                                                        | P3 Research: only reopen against its existing artifact/envelope/quality gates                                                                         |
| #274    | Open native low-bit/BitNet watch.                                                                                                                                                                    | P3 Research: artifact compatible with product pin before engineering                                                                                  |
| #275    | Open SSM/RWKV/multimodal watch.                                                                                                                                                                      | P3 Research: bounded surface and pin compatibility before console work                                                                                |
| #381    | Draft; prior tests missed real Windows package naming and wrong-job/invalid-duration evidence. Updated with regressions; current main incorporated.                                                  | P1 QA, owning #397: exact-head CI + live TAESD; preserve sampled-memory vs peak distinction                                                           |
| #369    | Draft; conflicted with main. Conflict resolved without force; stale #370 and version/source wording updated. TBD and live-evidence TODOs retained.                                                   | P1 Release: current receipts, fresh docs review, exact-package approval                                                                               |

Changes discovered or opened during this execution:

| Issue / PR  | Scope and disposition                                                                                                                                       |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| #393 / #399 | Dependabot omitted host training requirements; regression-first coverage added, security monitoring enabled, vulnerable pins not silently upgraded          |
| #394 / #400 | Provenance validator accepted malformed identity/hash/count/nonfinite metadata; shared writer/validator now reject it before replacement                    |
| #398 / #401 | Host trainer unconditionally trusted model-selected Python code; both loaders now refuse custom code; mocked Hub/local policy regressions                   |
| #397 / #381 | TAESD sampler recognized an impossible package-name ordering; fresh rows could be wrong jobs; invalid durations accepted. Regression-first correction       |
| #396        | Concurrent independent CSV validator PR; do not duplicate or overwrite. Review required checks; retain its tests when resolving CI-line conflicts with #400 |
| #395        | Program tracking only; does not replace domain issues                                                                                                       |

### Upstream branches

| Branch                                | Disposition                                                                                                 |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------- |
| `main`                                | Protected baseline; no direct edits                                                                         |
| `cursor/taesd-gate-stale-452e`        | Existing #381 updated by forward merge; keep draft until its gates clear                                    |
| `cursor/docs-v170-release-notes-4bfd` | Existing #369 reconciled; keep draft, no premature tag/date                                                 |
| `cursor/decision-models-spike-8b6e`   | #356 closed NO-GO; compare new Laya fork evidence before considering reopening; no automatic import         |
| `feat/floppylm-training`              | #301 closed without merge; respect existing [FloppyLM disposition](../../floppylm.md), no automatic revival |

Hardening PRs #399, #401, #400 and #396 passed all four required exact-head
checks and were squash-merged, ending at `f1007133ba47a5b9ae9b44f0990aee67bdabf45a`.
The [integration receipt](execution-2026-11-07/integration-receipt.json) retains
check IDs, source SHAs and merge commits. Remaining intervention branches are
identified by their PRs above. Completed integration order: #399, #401, #400, #396. Next: #404 and #381 only
after their device gates, then #369 after release receipts. Refresh each PR
against main before integration. Several independently add CI test names and
AGENTS commands; resolve by retaining **all** checks, not by choosing one side.
For every PR: regression before fix, reviewed diff, applicable local checks,
exact-head required CI, squash merge, rollback by revert. A closed milestone
must link source SHA, PR, tests and remaining blockers.

## Risk register and dependency matrix

| Risk                                                             | Impact / likelihood                            | Control and accountable role                                                                                          | Blocks / release condition                                         |
| ---------------------------------------------------------------- | ---------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ |
| R1: host evidence mistaken for Xbox proof                        | High / high                                    | QA keeps platform and package provenance; no inferred device PASS                                                     | #263, #374, #369 until actual target runs                          |
| R2: custom model code execution                                  | High / confirmed policy gap                    | Security #401 explicitly refuses it; no unreviewed opt-in                                                             | Host training trust boundary; exact-head tests                     |
| R3: malformed evidence admitted                                  | High / reproduced                              | QA #396/#400/#381 fail closed; immutable historical evidence                                                          | New claims until validators and receipts pass                      |
| R4: stale security issue body hides shipped migration            | High / confirmed                               | Security separates source implementation, export parity, live proof and advisory status                               | #263 remains open until remaining receipts exist                   |
| R5: vendor overlay removed prematurely                           | High / medium                                  | Release retains SHA-pinned binaries until both source-inclusion and live gates pass                                   | Runtime pin retirement, not ordinary docs-only work                |
| R6: fork ISA/shader downgrade imported as optimization           | High / confirmed incompatible scope            | GPU review rejects automatic import; compiler/profile/blob provenance                                                 | Any bjoern import; separate SKU decision required                  |
| R7: required CI absent or stale                                  | High / observed missing current receipts       | Release checks exact head; no bypass, no inherited green result                                                       | Every merge and stable release                                     |
| R8: training stack outside patched compatibility floor           | High / confirmed range excludes transformers 5 | Security #405 owns remediation; #399 adds monitoring; resolve PEFT/train/merge/parity in isolation before pins change | Host training security remediation; no closure by monitoring alone |
| R9: hardware mislabeled / resource probes mistaken for inference | High / confirmed third-party caveat            | Research preserves raw labels, records observed OS identity separately; acceptance requires inference                 | #374 and cross-device claims                                       |
| R10: multiple PRs overwrite CI commands                          | Medium / observed merge-line overlap           | Principal resolves incrementally and reruns combined tests                                                            | Merge ordering, current-head CI                                    |
| R11: thermal/noise/selection bias                                | High / medium                                  | Research preregisters matched arms, all repeats/failures and uncertainty                                              | Performance promotion and publication-grade claim                  |

Dependency path: trustworthy harness → exact CI package → source/model/device
binding → numerical correctness → matched performance/quality → claim registry
→ release review. #263 export parity and #374 device runs are separate paths;
passing one cannot discharge the other. Security/correctness work precedes GPU
optimization. Optional architecture watches do not delay the shipping fixes.

## Eight-week execution plan

| Week | Dates        | Owner and work                                                                                   | Reviewable acceptance / stop condition                                                                                           |
| ---- | ------------ | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------- |
| W1   | 7–13 Nov     | Principal/QA/Security: discovery, regression fixes, issue/PR ordering                            | Snapshot, risk register, fork verdicts, regression red/green receipts; no unsupported completion claims                          |
| W2   | 14–20 Nov    | Security/Release: diffusion evidence audit, training dependency resolution, vendor source checks | Isolated supported dependency graph; export fixed-input comparison per ADR 0003; device-dependent work stays blocked             |
| W3   | 21–27 Nov    | QA/Release: verify #381 on current MSVC package, Series X tester preparation                     | Live fresh-result gate; report sampled memory honestly and true peak separately; package/hash/device receipt                     |
| W4   | 28 Nov–4 Dec | GPU: own-shader multi-column int8 experiment behind explicit opt-in                              | One-column decode unchanged; host arithmetic/bounds tests, reproducible source/blob build, exact-head Windows compile            |
| W5   | 5–11 Dec     | GPU/Research: matched Series S trial                                                             | Existing numerical/product gates plus preregistered margins; all raw runs, failed trials and CIs; no hardware means no promotion |
| W6   | 12–18 Dec    | Research/QA: independent Series X acceptance and replication                                     | Distinct device dataset; source/model/package checks; no pooling with Series S or host                                           |
| W7   | 19–25 Dec    | Research/Release: evidence package and RC review                                                 | Claims map to admitted raw files; negative/inconclusive results retained; current gate matrix signed off                         |
| W8   | 26 Dec–1 Jan | Release/Principal: stable decision and final report                                              | All critical gates pass or formally scoped out with rationale; full-program completion still requires mandated hardware runs     |

No new structural ADR is needed for the implemented incremental fixes. [Proposed ADR 0009](../../adr/0009-isolated-host-training-toolchain.md) makes the separate future host-training environment decision reviewable under #405; it is not accepted or implemented. ADR 0003
already governs export; ADR 0002 governs main; the existing GPU design governs
#313. A new ADR is required before a second GPU backend, changed arithmetic
contract or separate Xbox One SKU, with alternatives and explicit acceptance.

## Release candidate and approval

The candidate is a **prepared plan**, not an emitted or certified MSIX. Keep
package version and research version ownership in their existing files. After
the fixes are merged through required checks, select an immutable candidate SHA
and exact CI MSVC package. Populate this checklist with linked receipts:

- [ ] Candidate SHA, clean source tree, submodule and dependency pins; exact package SHA-256, identity/revision and CI run.
- [ ] Four required CI checks green on the up-to-date head; Linux tests and applicable sanitizer runs, with skips and scope stated.
- [ ] Security findings disposition; #263 isolated dependency/export/parity/rescan receipts plus console gate, #398 trust policy, monitored training compatibility risk.
- [ ] Series S console `all`, API `all`, affected training gates; actual package and model hashes, no stale-run fallback.
- [ ] True peak-memory evidence for gates that demand it; sampled PrivateWorkingSet from #381 is insufficient by itself.
- [ ] Series X acceptance under #374 for the full mandate; preserve upstream hardware wording until accepted results exist.
- [ ] GPU change numerical parity, H9 and predeclared A/B verdict if promoted; otherwise keep experimental code disabled or excluded.
- [ ] Research claim registry/provenance and generated summaries pass; publication-grade policy in `paper/release.md` satisfied.
- [ ] #369 TODOs resolved by actual same-package receipts; current docs review, rollback and known limitations.

Any critical unchecked item blocks stable publication. Do not upload a stable
release, tag a result as measured, change Series X wording, or dismiss alerts
to make the checklist green. If a reduced release is proposed, document its
explicit scope; it does not complete this full eight-week mandate.

## M0 receipt and external blockers

Host baseline Linux configure/build succeeded. CTest: 1/1; direct doctest:
411 cases, 33,577 assertions, all passed. Three training job JSON files and CLI
help succeeded. Baseline Python checks: 118 tests, 13 optional dependency checks
skipped; coherence, generated benchmark, research package and metadata checks
passed. New branch-specific tests are recorded in their PR descriptions and
the [receipt](execution-2026-11-07/verification.json). These host runs do not
verify UWP, shaders, exports, model quality or Xbox behaviour.

The subsequent [isolated diffusion host receipt](execution-2026-11-07/diffusion-host-receipt.json) records supported dependency resolution/installation, clean `pip check` and 25 export tests without skips. Tests use synthetic/tiny random models, not full SD-Turbo weights. The dependency audit queried 50 installed packages with zero known findings after upgrading the temporary venv bootstrap pip; the CPU Torch wheel was skipped, and a separate base-release advisory lookup is recorded with its narrower scope. None of this closes #263.

Real first-party VAE-only export, eager/ORT verification and fp16 conversion also passed; [raw receipt](execution-2026-11-07/host-vae-receipt.json) and [canonical source binding](execution-2026-11-07/vae-source-binding.json) retain exact checkpoint/asset hashes. New and published legacy fp16 passed the existing fixed-input fp16-vs-new-fp32 VAE check. This is directional component evidence, not the legacy-fp32/full-pipeline parity required by ADR 0003. UNet needs more available host memory, the legacy fp32 reference is not published in models-v1, and all device gates remain pending. No claim registry or shipping asset is promoted.

The [PEFT candidate receipt](execution-2026-11-07/peft-candidate/candidate-receipt.json) records a supported isolated 58-distribution graph and a tiny random built-in Llama train/save/reload/merge feasibility test. Repository training pins remain unchanged: whole-graph security review, CPU installation policy and full first-party SmolLM2/GGUF/marker parity remain gates under #405. A synthetic API smoke cannot certify that production recipe. The [whole-graph candidate audit](execution-2026-11-07/peft-candidate/security-disposition.json) rejects that first candidate: Datasets and fsspec have unresolved advisories, plus bootstrap-tool findings and a Torch audit skip. Keep the successful API result and failed security verdict together; no pin promotion. The [second candidate](execution-2026-11-07/peft-candidate2/receipt.json) explicitly migrates the proposed dataset constraint, passes the same synthetic trial and reports zero known findings for 58 queried distributions; CPU Torch remains skipped. Full first-party recipe, adapter/GGUF parity and profile qualification remain unverified.

A later [sanitizer receipt](execution-2026-11-07/sanitizer-receipt.json) records a passing Debug ASan+UBSan build and strict ctest with leak detection disabled. Default LeakSanitizer fails under ptrace, independently reproduced on an empty program; it is BLOCKED, not passed. llama.cpp/ggml are outside this instrumentation scope.

This environment has no configured `xbox-env`; no console connection, Windows
UWP build, production first-party host training, complete production diffusion export, GPU benchmark or hardware acceptance was
performed. The VAE-only host trial above is the sole real-component export exception. Required remote CI remains a separate gate. Therefore GPU promotion,
#263 closure, #374 closure, final RC approval and stable release are blocked.
No historical result, generated numeric table or accepted ADR was rewritten.

After the four hardening merges, the [combined host suite](execution-2026-11-07/combined-suite-receipt.json) ran all CI-registered Python modules with the export toolchain installed: **136 tests passed, zero skips**. This extends the earlier baseline receipt; it does not replace it or establish device behaviour.

M1 and later receipts must append dated source/package refs, exact commands,
test outcomes/skips, raw evidence hashes, decisions and blockers to this
program's evidence directory. Do not retrofit results into this M0 snapshot.

## M1 continuation — 2026-11-08

The local continuation now has authenticated Device Portal access and exclusive
use of a Series S; the earlier M0 environment limitation remains historical.
The [continuation ledger](execution-2026-11-08/progress.json) and its
[hash manifest](execution-2026-11-08/SHA256SUMS) retain package-bound outcomes.
The observed OS platform receipt is separate from an operator declaration.

The installed baseline exposed the known TAESD preload/fresh-result defect.
The exact CI package for #381 was installed in-place and its corrected TAESD
gate passed; the complete LAN API suite and extended serve/rate/runtime-adapter
suite also passed on that package. Runtime-adapter loading alone does not prove
marker quality. Device training and the new host PEFT recipe remain in progress.

Draft [#407](https://github.com/gianlucamazza/xllama/pull/407), stacked on #381,
adds a process-lifetime peak working-set field for a future fresh-process device
trial. Its code and CI do not yet establish live peak-memory acceptance.
#404 was refreshed against main to obtain a newer in-place candidate.
Full diffusion export/parity, host training qualification, final same-package
release gates and the preregistered GPU experiment remain open. No stable
release or security closure is approved by these partial results.

### Live continuation — first-party graph compatibility

PR #410 merged with the qualified isolated CPU training profile; #405 is closed.
PR #412 merged after all required checks passed and full-toolchain export tests
and native tokenizer parity passed. The correction includes the legacy CLIP
sidecars consumed by the native runner in first-party export packages.

The [fresh full-graph Series S attempt](execution-2026-11-08/firstparty-diffusion-directml-failure.json)
**failed** in DirectML at the text encoder's `Reshape` (`node_view`, HRESULT
8007023E). Host parity is not a device compatibility result. The
[run log](execution-2026-11-08/firstparty-diffusion-directml-failure.txt) is retained;
no image, TAESD or process-peak acceptance is inferred from this attempt. A
fixed-input session specialization is being tested against the upstream
[dynamic Reshape report](https://github.com/microsoft/onnxruntime/issues/33720).
The graph and tolerance contracts remain unchanged. #263 remains open.

### Live continuation — qualified deployment profile

PR #413 merged after all four required checks passed. The final host tool has
[35 passing export/profile tests without skips](execution-2026-11-08/diffusion-profile-qualification-progress.json).
The dynamic-annotation deployment profile produces an exact RGB match to the
reference under identical CPU session settings. The fully static alternative's
failed exact comparison and Series S resource failures remain in the evidence;
its tolerances were not relaxed.

The [fresh full-VAE Series S run](execution-2026-11-08/diffusion-dynamic-profile-live-1466.json)
passes on the identified CI MSVC package. The [image](execution-2026-11-08/diffusion-dynamic-profile-live-1466.png)
is coherent, and all three deployed graphs
[read back byte-for-byte](execution-2026-11-08/diffusion-dynamic-profile-readback-1466.json)
against the final default profile. This is a headless functional acceptance,
not UI-process TAESD/true-peak or final release acceptance; #263 stays open for
those remaining gates. Frozen fp16 component comparisons pass; that comparison
mode explicitly leaves its image/pooler diagnostics unchecked.

The [isolated dependency rescan](execution-2026-11-08/diffusion-rescan-disposition.json)
reports no known findings in the queried graph. The exact CPU-wheel scanner
skip remains separate from its
[official-wheel and installed-payload identity proof](execution-2026-11-08/diffusion-torch-wheel-identity.json)
and the base-release advisory lookup. This is bounded coverage, not a
whole-binary or unknown-vulnerability certification.

PR #414 now contains the disabled-by-default four-column prefill candidate.
Its original-path Series S selftest passes; optimized-arm, UI-process numeric,
logit/H9 and paired performance gates remain pending. Dedicated deterministic
bench and UI-process diagnostic controls are being checked before the campaign.
No new throughput or release claim follows from compilation or the control run.

The continuation now also includes fresh
[UI-process full-VAE acceptance](execution-2026-11-08/diffusion-full-profile-ui-1472.json)
and the [TAESD gate with true process-peak evidence](execution-2026-11-08/diffusion-taesd-profile-ui-1472-binding.json)
on the same identified package and first-party profile. Post-gate full-VAE
restoration and all three graph hashes are
[verified by readback](execution-2026-11-08/diffusion-profile-ui-readback-1472.json).
The final default profile also passes the
[frozen component comparison](execution-2026-11-08/diffusion-qualified-profile-components.json);
its unchecked image/pooler diagnostics remain explicitly unchecked. These
single acceptance trials do not add a throughput headline. #263 closure awaits
integration of the qualification receipts; final programme/package and physical
publication gates remain separate.

### Integration checkpoint — first-party qualification and write integrity

The qualification receipts now include a stronger
[UI TAESD repeat](execution-2026-11-08/diffusion-verified-taesd-ui-1472.json):
the deployed Tiny decoder is read back before execution, the fresh result
contains the true process lifetime working-set peak, and the full decoder is
restored and read back afterwards. All
[original diffusion weights](execution-2026-11-08/original-diffusion-restoration.json)
are subsequently restored byte-for-byte. Together with the full-VAE UI run,
frozen host comparisons and native tokenizer parity, these receipts complete
the scoped first-party toolchain qualification in #263. Failed alternatives,
unchecked comparison fields and publication limits remain explicit. Final
programme and final-package acceptance remain open.

PR #415 is merged after all four required checks. Its final strict-HTTP-200
helper also passes a [reversible live write test](execution-2026-11-08/wdp-upload-final-live-receipt.json)
on package 1.7.0.1475: successful readback, rejection of an invalid file-parent
write, unchanged sentinel and verified cleanup. This validates the harness
write contract; it is not a new inference measurement.

The [current GitHub alert receipt](execution-2026-11-08/diffusion-alert-state.json)
independently reports the two diffusion advisories as `fixed`, with no dismissal.
The isolated rescan and exact CPU-wheel scope limits above still apply; alert
state alone does not replace dependency or runtime qualification.

The [GPU campaign protocol](execution-2026-11-08/prefill-campaign-protocol.json)
and all six hashed prompt files are frozen before any optimized-arm device
result. Exact package/device binding is required before execution. The protocol
keeps the existing numerical/performance margins and explicitly limits any
uninstrumented result to engineering evidence. The optimized gates are pending.

### GPU preflight — stopped on control parity

The exact package 1.7.0.1482 passes the fresh XAML-process selftest:
[control](execution-2026-11-08/prefill-ui-selftest-1482-A.json) and
[experimental arm](execution-2026-11-08/prefill-ui-selftest-1482-B.json).
The package hash, CI checks and independent Series S observation are bound
[before execution](execution-2026-11-08/prefill-campaign-package-binding-1482.json).
Both LFM CPU/GPU comparisons pass; its two GPU dumps are byte-identical.

The Coder-3B CPU versus original GPU-kernel comparison
[fails](execution-2026-11-08/prefill-logits-coder-CPU-A-comparison.txt) the frozen
maximum-error and NMSE limits. Greedy top-token agreement does not override
those failures. The raw float32 dumps and metadata are retained beside the
comparison, so the verdict is independently reproducible. No cause is proven;
this control-arm failure does not establish a four-column regression.

The [campaign is stopped](execution-2026-11-08/prefill-campaign-disposition.json).
No Coder optimized logits, H9 or matched performance campaign followed the
failure. No retry, replacement prompt, relaxed tolerance or speedup claim is
made. PR #414 remains draft and #313 remains open. The app is stopped and the
operator switch is returned to the original one-column path. The proposed
research H9 raw-path metadata was prepared locally but never deployed.

PR #416 is merged and #263 is closed on the scoped first-party qualification
receipts. The known historical LoRA fixture is also
[byte-restored](execution-2026-11-08/historical-adapter-restoration.json).
Final package gates, full user-state cleanup and physical publication gates
remain outstanding. ASan plus UBSan on the candidate host build pass; they do
not replace the failed device logit gate.

### Final candidate checkpoint — 1.7.0.1495

The [exact unified CI MSVC package](execution-2026-11-08/release-package-binding-1495.json)
includes the independent framing and GPU-validator fixes (#418, #419). All four
required checks passed before deployment. The packaged patched runtime DLLs
[match their owning vendor pins](execution-2026-11-08/release-1495-packaged-vendor-verification.json).
The earlier 1492 functional checkpoint, its initial framing-harness failure
and the corrected replay remain separate receipts. Capture normalizations are
bound in the dated evidence directory.

On the same 1495 package, all ten standard console gates and all eleven API
gates pass, including the ONNX tool-proposal refusal. The standard GPU selftest
has [fresh, finite passing rows](execution-2026-11-08/release-1495-selftest-receipt.json),
with old-result absence verified before launch. The
[shipping H9 screen](execution-2026-11-08/release-1495-h9-receipt.json) retains its
prospective reference floors. This small screen is not general model-quality
validation. Nomic's embedding contract covers the live requests;
[optional BGE coverage](execution-2026-11-08/release-1495-bge-coverage.json) was not
run because that model was not provisioned.

Training serve, preference capture, adapter application, both stale-result
negative controls and the full eight-epoch device pipeline pass. The standard
adapter-application gate's missing-log-marker warning is retained. A separate
[stateless API response](execution-2026-11-08/release-1495-canonical-adapter.json)
with the newly qualified host adapter returns the expected marker after base
and adapter byte readback; it does not reclassify the warning.

Both [full-VAE UI](execution-2026-11-08/release-1495-full-vae.json) and
[Tiny-decoder UI](execution-2026-11-08/release-1495-taesd-receipt.json) trials pass,
with true lifetime process peaks recorded independently of sampled private
memory. All three original graphs read back byte-for-byte after the full-VAE
trial. The [full image](execution-2026-11-08/release-1495-full-vae.png) and
[Tiny image](execution-2026-11-08/release-1495-taesd.png) pass
[visual inspection](execution-2026-11-08/release-1495-image-qa.json).
These are single functional trials, not throughput/cold-start headlines.
SD-Turbo attribution remains in [NOTICE](../../../NOTICE).

The real [GUI/API admission probe](execution-2026-11-08/release-1495-gui-admission.json)
observes HTTP 503 `busy` during an active XAML GUI turn, which then completes.
The [first observer attempt](execution-2026-11-08/release-1495-gui-admission-initial-observer.json)
was inconclusive because it watched a completion-only log line; it made no
concurrent LAN request and is retained. The corrected observer uses prefill
start, with the same prompt and sampling. This is not a physical-controller test.

The [Store 1488 smoke](execution-2026-11-08/store-1488-live-receipt.json) is a
separate reduced-SKU trial: chat/save-title and API rejection pass on a
pre-provisioned model. It does not claim a fresh download, retail validation or
a Store build of revision 1495. Developer Mode is restored by higher-revision
in-place updates. [Host checks](execution-2026-11-08/release-1495-host-checks.json)
include Linux test/Release smoke and ASan plus UBSan; optional dependency skips
remain distinct from the earlier no-skip exporter qualification.

[Original-state restoration](execution-2026-11-08/release-1495-state-restoration.json)
verifies every privately backed-up file by readback, removes owned controls and
exclusive-window test chats, and byte-restores the known historical adapter.
The per-file user inventory stays private. Unrelated models/KV and general
training outputs are preserved; empty owned staging directories may remain.
The app is stopped. There was no uninstall or LocalState wipe.

The original progress snapshot stays historical; `latest_continuation` records
this checkpoint. The local candidate bundle is ready for release review, not
published as a stable tag. PR #369 owns the notes. #313 retains its stopped
control-parity failure, Series X remains untested/excluded, physical publication
gates remain unproven, and the vendor/architecture watches remain gated.
