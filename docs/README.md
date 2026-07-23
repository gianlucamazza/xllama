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

## Document index

| Document | Description |
| -------- | ----------- |
| [architecture.md](./architecture.md) | Dual pillars, modules, backends, KV-reuse, routing, provisioning, LAN, training surface, membw, diffusion |
| [training-architecture.md](./training-architecture.md) | **Training SSOT**: RE inventory, capability matrix, lanes A/B/C, Lane B engine, hybrid loop, Phase 11 UI arc |
| [../training/README.md](../training/README.md) | Training ops: job JSON, host PEFT, device partial FT, personalize preflight |
| [using-the-app.md](./using-the-app.md) | App guide: chat, settings (incl. Train on my feedback), image generation, LAN API toggle |
| [api-endpoint.md](./api-endpoint.md) | LAN HTTP endpoint: enable, protocol (chat + prefs + training status + images), concurrency, validation |
| [install-release.md](./install-release.md) | Install a tagged GitHub Release build on Xbox (cert + VCLibs + MSIX) |
| [uwp-constraints.md](./uwp-constraints.md) | UWP sandbox limitations §1–§13 (GPU budget, AppContainer, training/adapters RE) |
| [technical-report.md](./technical-report.md) | Historical v1.0 narrative snapshot; [Discussion #76](https://github.com/gianlucamazza/xllama/discussions/76) |
| [console-validation-runbook.md](./console-validation-runbook.md) | Automated on-console gates, benchmark evidence rules, troubleshooting |
| [fp16-extdata-runbook.md](./fp16-extdata-runbook.md) | PatchedOrt external-data outcome, rebuild procedure, drop conditions |
| [dml-metacommands-runbook.md](./dml-metacommands-runbook.md) | #91 experiment: `ep.dml.disable_metacommands` |
| [dml-rmsnorm-fix-runbook.md](./dml-rmsnorm-fix-runbook.md) | #91 root cause/fix: DML RMSNorm graph surgery |
| [windows-dev-vm.md](./windows-dev-vm.md) | Windows VM setup for local UWP/MSIX builds |
| [device-portal.md](./device-portal.md) | Dev Mode and Device Portal deploy |
| [phase1-runbook.md](./phase1-runbook.md) | Legacy compatibility entrypoint → current owners |
| [model-selection.md](./model-selection.md) | Choosing/adding models, manifest override, ORT asset publishing |
| [benchmarks.md](./benchmarks.md) | Generated comparison + interpretation; raw data under `bench/results/` |
| [recommended-config.md](./recommended-config.md) | Modern settings: models, genai_config, settings.json, build variants |
| [vendor-lifecycle-plan.md](./vendor-lifecycle-plan.md) | Patched-runtime pins, upstream deps, removal gates |
| [phase7-hypotheses.md](./phase7-hypotheses.md) | Phase 7 research log; metrics still come from the generated summary |

See also [../CHANGELOG.md](../CHANGELOG.md) for release history and
[../diffusion/README.md](../diffusion/README.md) for the SD-Turbo host toolchain.
