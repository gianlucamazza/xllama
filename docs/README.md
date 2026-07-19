# docs/

Technical notes and design decisions for xllama.

## Documentation ownership

Each current fact has one authoritative home. Other documents may explain a
decision or quote one headline, but must link back instead of maintaining a
second table.

- **System structure** (inference + training pillars, modules, provisioning, membw) → [architecture.md](./architecture.md)
- **Training pillar SSOT** (RE inventory, capability matrix, hybrid loop) → [training-architecture.md](./training-architecture.md)
- **Raw performance evidence** → [`../bench/results/`](../bench/results/)
- **Comparison policy** → [`../bench/benchmark-summary.json`](../bench/benchmark-summary.json)
- **Generated performance summary** → [benchmarks.md](./benchmarks.md) +
  [benchmarks-charts.html](./benchmarks-charts.html)
- **Model catalogue + backend selection** → [model-selection.md](./model-selection.md) (narrative) + [`../uwp/models/manifest.json`](../uwp/models/manifest.json) (data)
- **UWP/AppContainer constraints** (§1–§13) → [uwp-constraints.md](./uwp-constraints.md)
- **Runtime NuGet pins** → [recommended-config.md](./recommended-config.md) (narrative) + [`../uwp/packages.config`](../uwp/packages.config) (data); patched-DLL lifecycle → [vendor-lifecycle-plan.md](./vendor-lifecycle-plan.md)
- **Version / current state** → [../CHANGELOG.md](../CHANGELOG.md) + [../ROADMAP.md](../ROADMAP.md)

The benchmark flow is intentionally one-way:

```text
raw CSV/JSONL evidence → comparison selectors → generated Markdown + dashboard
```

Run `python3 scripts/generate-benchmark-summary.py` after changing evidence or
selectors. CI runs it with `--check` and rejects drift. Research notes and dated
reports preserve interpretation and history; they are not current metric stores.

| Document                                                         | Description                                                                                                                                  |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [architecture.md](./architecture.md)                             | Dual pillars (inference + training), modules, backends, KV-reuse, routing, provisioning, membw, diffusion                                    |
| [training-architecture.md](./training-architecture.md)           | **Training SSOT**: reverse-engineering inventory, capability matrix, lanes A/B/C, Lane B device engine, hybrid preference loop               |
| [../training/README.md](../training/README.md)                   | Training ops: job JSON, host PEFT runner, device partial FT, stages                                                                          |
| [install-release.md](./install-release.md)                       | Install a tagged GitHub Release build on your Xbox (cert + VCLibs + MSIX)                                                                    |
| [using-the-app.md](./using-the-app.md)                           | App guide: chat, settings (model picker, routing, KV reuse), image generation                                                                |
| [uwp-constraints.md](./uwp-constraints.md)                       | UWP sandbox limitations §1–§13 (GPU budget, AppContainer, training/adapters RE)                                                              |
| [technical-report.md](./technical-report.md)                     | Historical v1.0 narrative snapshot; published as [Discussion #76](https://github.com/gianlucamazza/xllama/discussions/76)                    |
| [console-validation-runbook.md](./console-validation-runbook.md) | Current automated on-console gates, benchmark evidence rules and troubleshooting                                                             |
| [fp16-extdata-runbook.md](./fp16-extdata-runbook.md)             | Current PatchedOrt external-data outcome, rebuild procedure and drop conditions                                                              |
| [dml-metacommands-runbook.md](./dml-metacommands-runbook.md)     | #91 experiment: `ep.dml.disable_metacommands` vendored knob, on-console parity procedure and outcomes                                        |
| [dml-rmsnorm-fix-runbook.md](./dml-rmsnorm-fix-runbook.md)       | #91 root cause and fix: broken DML `(Skip)SimplifiedLayerNormalization` kernel, RMSNorm graph surgery, escalation evidence                   |
| [windows-dev-vm.md](./windows-dev-vm.md)                         | Windows VM setup for local UWP/MSIX builds                                                                                                   |
| [device-portal.md](./device-portal.md)                           | How to enable Dev Mode and deploy via Device Portal                                                                                          |
| [phase1-runbook.md](./phase1-runbook.md)                         | Legacy compatibility entrypoint routing readers to current installation, Device Portal and benchmark owners                                  |
| [model-selection.md](./model-selection.md)                       | Choosing/adding models: hard limits, evaluation sequence, tested models, manifest override how-to + ORT-asset publishing (logit-parity gate) |
| [api-endpoint.md](./api-endpoint.md)                             | LAN HTTP endpoint (OpenAI-compatible, opt-in, default OFF): enable, protocol, concurrency, validation                                        |
| [benchmarks.md](./benchmarks.md)                                 | Generated current comparison plus interpretation; raw measurements remain under `bench/results/`                                             |
| [recommended-config.md](./recommended-config.md)                 | Correct modern settings: models, genai_config, settings.json, build variants, obsolete myths                                                 |
| [vendor-lifecycle-plan.md](./vendor-lifecycle-plan.md)           | Active patched-runtime pins, upstream dependencies, refresh procedure and removal gates                                                      |
| [phase7-hypotheses.md](./phase7-hypotheses.md)                   | Phase 7 research log: hypotheses, gates and verdicts; current metrics come from the generated benchmark summary                              |

See also [../CHANGELOG.md](../CHANGELOG.md) for the full pivot history (llama.cpp → ORT GenAI → CPU EP → per-workload routing) and [../diffusion/README.md](../diffusion/README.md) for the SD-Turbo model toolchain.
