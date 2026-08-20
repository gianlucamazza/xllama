# docs/

Technical notes and design decisions for xllama.

## Documentation principles

1. **One fact, one home (SSOT).** Every current fact has a single authoritative
   document. Other pages may explain a decision or quote a headline, but must
   **link** instead of maintaining a second table.
2. **Code wins over stale prose.** If a behaviour and a doc disagree, fix the
   doc in the same PR (or immediately after). CI gates numbers where automated
   (`generate-benchmark-summary.py --check`); narrative SSOT is human-owned.
3. **User vs operator vs engineer.** End-user steps live in `using-the-app.md` /
   `install-release.md`. Operator console harnesses live in runbooks. Structure
   and contracts live in `architecture.md` + domain SSOTs.
4. **Generated artefacts are not hand-edited.** Benchmark tables/charts come
   from raw evidence + policy; do not reflow them by hand.
5. **English only** for docs, code comments, and commit messages (project
   convention).

## Documentation ownership

| Concern                                                                                | Authoritative home                                                                                                           |
| -------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| System structure (modules, backends, KV, routing, provisioning, LAN, training surface) | [architecture.md](./architecture.md)                                                                                         |
| Training pillar (RE, capability matrix, lanes A/B/C, hybrid loop, Phase 11 in-app arc) | [training-architecture.md](./training-architecture.md)                                                                       |
| Training ops (job JSON, host PEFT, device train CLI, pull samples)                     | [`../training/README.md`](../training/README.md)                                                                             |
| Raw performance evidence                                                               | [`../bench/results/`](../bench/results/)                                                                                     |
| Comparison policy                                                                      | [`../bench/benchmark-summary.json`](../bench/benchmark-summary.json)                                                         |
| Generated performance summary                                                          | [benchmarks.md](./benchmarks.md) + [benchmarks-charts.html](./benchmarks-charts.html)                                        |
| Bench methodology / CSV schema                                                         | [`../bench/README.md`](../bench/README.md)                                                                                   |
| Model catalogue + backend selection                                                    | [model-selection.md](./model-selection.md) (narrative) + [`../uwp/models/manifest.json`](../uwp/models/manifest.json) (data) |
| Full model inventory (tested / shipping / rejected, roles, H9, coding)                 | [model-matrix.md](./model-matrix.md) (status SSOT; numbers still link to benchmarks.md)                                      |
| UWP/AppContainer constraints (§1–§13)                                                  | [uwp-constraints.md](./uwp-constraints.md)                                                                                   |
| SSD-streamed inference assessment (diskbw probe + verdict)                             | [ssd-inference-assessment.md](./ssd-inference-assessment.md)                                                                 |
| Runtime NuGet pins                                                                     | [recommended-config.md](./recommended-config.md) (narrative) + [`../uwp/packages.config`](../uwp/packages.config) (data)     |
| Patched-DLL lifecycle                                                                  | [vendor-lifecycle-plan.md](./vendor-lifecycle-plan.md)                                                                       |
| LAN HTTP protocol                                                                      | [api-endpoint.md](./api-endpoint.md)                                                                                         |
| App gamepad UI (chat, settings, personalize, images)                                   | [using-the-app.md](./using-the-app.md)                                                                                       |
| Version / current product state                                                        | [`../CHANGELOG.md`](../CHANGELOG.md) + [`../ROADMAP.md`](../ROADMAP.md)                                                      |
| Package identity / install & migration path                                            | [install-release.md](./install-release.md) (user) + `AGENTS.md` Versioning (dev)                                             |
| Xbox Store retail readiness (dual SKU, licence matrix, App vs Game gate)               | [store-readiness.md](./store-readiness.md)                                                                                   |
| Privacy policy (Store / end-user draft)                                                | [privacy.md](./privacy.md)                                                                                                   |
| Demo / screenshot capture (how it is produced, and from which build)                   | [`../scripts/capture-demo-video.sh`](../scripts/capture-demo-video.sh) (tool) + [`./screenshots/`](./screenshots/) (assets)  |
| Public claims and launch copy (what may be cited, and what may not)                    | [launch-copy.md](./launch-copy.md)                                                                                           |
| Phase 16 model-scouting campaign (funnel, ladder, workstream cards, verdicts)          | [phase16-model-scouting.md](./phase16-model-scouting.md)                                                                     |
| Agent / contributor quick map                                                          | [`../AGENTS.md`](../AGENTS.md)                                                                                               |

The benchmark flow is intentionally one-way:

```text
raw CSV/JSONL evidence → comparison selectors → generated Markdown + dashboard
```

Run `python3 scripts/generate-benchmark-summary.py` after changing evidence or
selectors. CI runs it with `--check` and rejects drift. Research notes and dated
reports preserve interpretation and history; they are not current metric stores.

## Document index (paths only)

The ownership table above is the SSOT for _where facts live_. Below is only a
path checklist for discovery — descriptions are not repeated.

**Structure / product:** [architecture.md](./architecture.md) ·
[training-architecture.md](./training-architecture.md) ·
[using-the-app.md](./using-the-app.md) · [api-endpoint.md](./api-endpoint.md) ·
[model-selection.md](./model-selection.md) ·
[recommended-config.md](./recommended-config.md)

**Constraints / vendor:** [uwp-constraints.md](./uwp-constraints.md) ·
[vendor-lifecycle-plan.md](./vendor-lifecycle-plan.md) ·
[fp16-extdata-runbook.md](./fp16-extdata-runbook.md) ·
[dml-rmsnorm-fix-runbook.md](./dml-rmsnorm-fix-runbook.md) ·
[dml-metacommands-runbook.md](./dml-metacommands-runbook.md)

**Ops / install:** [install-release.md](./install-release.md) ·
[device-portal.md](./device-portal.md) · [windows-dev-vm.md](./windows-dev-vm.md) ·
[crossbuild-console.md](./crossbuild-console.md) (Linux → Xbox via uwp-crossbuild + openappx) ·
[console-validation-runbook.md](./console-validation-runbook.md) ·
[store-readiness.md](./store-readiness.md) · [privacy.md](./privacy.md) ·
[../training/README.md](../training/README.md) ·
[../bench/README.md](../bench/README.md)

**Evidence / history:** [benchmarks.md](./benchmarks.md) ·
[model-matrix.md](./model-matrix.md) ·
[phase7-hypotheses.md](./phase7-hypotheses.md) ·
[phase15-re-opt.md](./phase15-re-opt.md) (RE + optimization campaign) ·
[phase16-model-scouting.md](./phase16-model-scouting.md) (model-scouting campaign) ·
[technical-report.md](./technical-report.md) (frozen v1.0) ·
[phase1-runbook.md](./phase1-runbook.md) (compat redirect) ·
[../CHANGELOG.md](../CHANGELOG.md) · [../ROADMAP.md](../ROADMAP.md) ·
[../diffusion/README.md](../diffusion/README.md)

### Acceptable headline vs SSOT

| Kind of content                       | Allowed outside SSOT                                                                                                                              | Forbidden                                                                                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Role / status one-liners              | Yes (e.g. README “default chat is LFM2.5-350M”)                                                                                                   | Second full catalogue or perf table                                                                                                         |
| Exact tok/s, peak RAM, speedups       | Only [benchmarks.md](./benchmarks.md) (generated) + raw `bench/results/`                                                                          | Restating figures in README / UI guide without linking                                                                                      |
| Download / catalogue size             | Sum of `approx_bytes` in `uwp/models/manifest.json`                                                                                               | Invented MB that drift from the manifest                                                                                                    |
| Hardware ceilings (GPU budget, etc.)  | [uwp-constraints.md](./uwp-constraints.md)                                                                                                        | Parallel ceilings elsewhere                                                                                                                 |
| NuGet version pins                    | `uwp/packages.config` + recommended-config / vendor-lifecycle                                                                                     | Parallel version tables                                                                                                                     |
| Repo file tree                        | AGENTS.md (agents)                                                                                                                                | Full second tree in README                                                                                                                  |
| Phase checklist                       | ROADMAP.md                                                                                                                                        | Duplicated phase list in README                                                                                                             |
| Phase 15 RE / optimization campaign   | [phase15-re-opt.md](./phase15-re-opt.md) (W2 default OFF; W3 M6 PASS → #228)                                                                      | Second workstream narrative in README                                                                                                       |
| Phase 16 model scouting               | [phase16-model-scouting.md](./phase16-model-scouting.md) (funnel, ladder, WS-A…WS-G cards)                                                        | Second candidate table elsewhere; shipped status outside model-matrix.md                                                                    |
| Linux→Xbox package without Windows VM | [crossbuild-console.md](./crossbuild-console.md) (crossbuild launch observed 2026-08-08, uwp-crossbuild ≥ 0.5.1; product/measured path = CI MSVC) | Restating crossbuild launch status outside that SSOT, or claiming crossbuild bench/product parity (unproven: ORT/GenAI, first boot, uptime) |

### Coherence check (automated)

```bash
python3 scripts/check-coherence.py   # code ↔ catalogue ↔ pins ↔ docs ↔ benchmarks
python3 scripts/generate-benchmark-summary.py --check
```

Both gates run in CI on every PR (`build-linux.yml`).

`check-coherence.py` fails closed on drift (default model, routing threshold,
NuGet pins, API routes, personalize paths, stale download sizes, H9 / Phase 10
evidence, generated decode headlines, broken doc links). Spot matrix below is
the **2026-07-25** baseline (post perf campaign + identity migration; the
original 2026-07-24 matrix seeded the script):

Spot-checked against code + evidence:

| Fact                                                | SSOT                                                      | Status                                                    |
| --------------------------------------------------- | --------------------------------------------------------- | --------------------------------------------------------- |
| Default chat `lfm25-350m` (unified)                 | `MainPage.cpp` `DefaultChatModelId`                       | OK                                                        |
| `token_threshold` 1550                              | `routing_policy.h`                                        | OK                                                        |
| GPU allowlist `-v2` only                            | `dml_text_model_ok`                                       | OK                                                        |
| NuGet 0.14.1 / 1.24.4 / DML 1.15.4                  | `packages.config`                                         | OK                                                        |
| Decode table (94.9 / 37.9 / 18.4 / 74.8 / 44.4 / …) | `generate-benchmark-summary.py --check`                   | OK (2026-07-26: LFM post-#168 median, ORT CPU shipped-t6) |
| Package identity `GianlucaMazza.xllama` (1.5.5.0)   | `uwp/AppxManifest.xml`; migration in `install-release.md` | OK (in-place from 1.5.x; breaking vs ≤1.4.x)              |
| One resident Session (GUI+API)                      | `include/xllama/session_hub.h`                            | OK (PR #161/#164)                                         |
| H9 6/8 · 7/8 · 4/8 · 5/8                            | `phase7-h9.jsonl`                                         | OK                                                        |
| Lane B peak_ws 1195 MB, wall 446 s                  | `phase10-console-devtrain-result.json`                    | OK                                                        |
| Catalogue download sizes                            | `manifest.json` `approx_bytes`                            | Docs corrected to match (was 218/697/1.46 stale)          |
