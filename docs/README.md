# docs/

Technical notes and design decisions for xllama.

**Single sources of truth (SSOT)** — each fact has one authoritative home; other
docs quote a headline and link back:

- **System structure** (modules, backend dispatch, provisioning, membw) → [architecture.md](./architecture.md)
- **Performance numbers** → [benchmarks.md](./benchmarks.md)
- **Model catalogue + backend selection** → [model-selection.md](./model-selection.md) (narrative) + [`../uwp/models/manifest.json`](../uwp/models/manifest.json) (data)
- **UWP/AppContainer constraints** (§1–§12) → [uwp-constraints.md](./uwp-constraints.md)
- **Version / current state** → [../CHANGELOG.md](../CHANGELOG.md) + [../ROADMAP.md](../ROADMAP.md)

| Document                                                         | Description                                                                                                                                |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| [architecture.md](./architecture.md)                             | Component/data-flow map: modules, runtime backend dispatch, chat templates, KV-reuse, routing, provisioning auto-upgrade, membw, diffusion |
| [install-release.md](./install-release.md)                       | Install a tagged GitHub Release build on your Xbox (cert + VCLibs + MSIX)                                                                  |
| [using-the-app.md](./using-the-app.md)                           | App guide: chat, settings (model picker, routing, KV reuse), image generation                                                              |
| [uwp-constraints.md](./uwp-constraints.md)                       | UWP sandbox limitations, measured GPU budget, AppContainer filesystem quirks, and how xllama works around them                             |
| [technical-report.md](./technical-report.md)                     | The measured story: per-workload CPU/GPU verdict, falsified hypotheses, diffusion on console                                               |
| [console-validation-runbook.md](./console-validation-runbook.md) | Ordered on-console validation checklist with measured results per section                                                                  |
| [windows-dev-vm.md](./windows-dev-vm.md)                         | Windows VM setup for local UWP/MSIX builds                                                                                                 |
| [device-portal.md](./device-portal.md)                           | How to enable Dev Mode and deploy via Device Portal                                                                                        |
| [phase1-runbook.md](./phase1-runbook.md)                         | End-to-end developer build, deploy, and benchmark instructions                                                                             |
| [model-selection.md](./model-selection.md)                       | Choosing/adding models: hard limits, evaluation sequence, tested models, manifest override how-to                                          |
| [benchmarks.md](./benchmarks.md)                                 | Consolidated perf for every tested model (decode/prefill/RAM, 3 backends) + comparative charts (`benchmarks-charts.html`)                  |
| [recommended-config.md](./recommended-config.md)                 | Correct modern settings: models, genai_config, settings.json, build variants, obsolete myths                                               |

See also [../CHANGELOG.md](../CHANGELOG.md) for the full pivot history (llama.cpp → ORT GenAI → CPU EP → per-workload routing) and [../diffusion/README.md](../diffusion/README.md) for the SD-Turbo model toolchain.
