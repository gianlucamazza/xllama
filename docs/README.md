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

| Concern | Authoritative home |
| ------- | ------------------ |
| System structure (modules, backends, KV, routing, provisioning, LAN, training surface) | [architecture.md](./architecture.md) |
| Training pillar (RE, capability matrix, lanes A/B/C, hybrid loop, Phase 11 in-app arc) | [training-architecture.md](./training-architecture.md) |
| Training ops (job JSON, host PEFT, device train CLI, pull samples) | [`../training/README.md`](../training/README.md) |
| Raw performance evidence | [`../bench/results/`](../bench/results/) |
| Comparison policy | [`../bench/benchmark-summary.json`](../bench/benchmark-summary.json) |
| Generated performance summary | [benchmarks.md](./benchmarks.md) + [benchmarks-charts.html](./benchmarks-charts.html) |
| Bench methodology / CSV schema | [`../bench/README.md`](../bench/README.md) |
| Model catalogue + backend selection | [model-selection.md](./model-selection.md) (narrative) + [`../uwp/models/manifest.json`](../uwp/models/manifest.json) (data) |
| UWP/AppContainer constraints (§1–§13) | [uwp-constraints.md](./uwp-constraints.md) |
| Runtime NuGet pins | [recommended-config.md](./recommended-config.md) (narrative) + [`../uwp/packages.config`](../uwp/packages.config) (data) |
| Patched-DLL lifecycle | [vendor-lifecycle-plan.md](./vendor-lifecycle-plan.md) |
| LAN HTTP protocol | [api-endpoint.md](./api-endpoint.md) |
| App gamepad UI (chat, settings, personalize, images) | [using-the-app.md](./using-the-app.md) |
| Version / current product state | [`../CHANGELOG.md`](../CHANGELOG.md) + [`../ROADMAP.md`](../ROADMAP.md) |
| Agent / contributor quick map | [`../AGENTS.md`](../AGENTS.md) |

The benchmark flow is intentionally one-way:

```text
raw CSV/JSONL evidence → comparison selectors → generated Markdown + dashboard
```

Run `python3 scripts/generate-benchmark-summary.py` after changing evidence or
selectors. CI runs it with `--check` and rejects drift. Research notes and dated
reports preserve interpretation and history; they are not current metric stores.

## Document index (paths only)

The ownership table above is the SSOT for *where facts live*. Below is only a
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
[console-validation-runbook.md](./console-validation-runbook.md) ·
[../training/README.md](../training/README.md) ·
[../bench/README.md](../bench/README.md)

**Evidence / history:** [benchmarks.md](./benchmarks.md) ·
[phase7-hypotheses.md](./phase7-hypotheses.md) ·
[technical-report.md](./technical-report.md) (frozen v1.0) ·
[phase1-runbook.md](./phase1-runbook.md) (compat redirect) ·
[../CHANGELOG.md](../CHANGELOG.md) · [../ROADMAP.md](../ROADMAP.md) ·
[../diffusion/README.md](../diffusion/README.md)

### Acceptable headline vs SSOT

| Kind of content | Allowed outside SSOT | Forbidden |
| --------------- | -------------------- | --------- |
| Role / status one-liners | Yes (e.g. README “default chat is LFM2.5-350M”) | Second full catalogue or perf table |
| Exact tok/s, MB, speedups | Only in benchmarks.md / uwp-constraints / raw CSV | Restating figures in README / UI guide |
| NuGet version pins | packages.config + recommended-config / vendor-lifecycle | Parallel version tables |
| Repo file tree | AGENTS.md (agents) | Full second tree in README |
| Phase checklist | ROADMAP.md | Duplicated phase list in README |
