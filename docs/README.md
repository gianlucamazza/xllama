# docs/

Technical notes and design decisions for xllama.

**Single sources of truth (SSOT)** — each fact has one authoritative home; other
docs quote a headline and link back:

- **System structure** (modules, backend dispatch, provisioning, membw) → [architecture.md](./architecture.md)
- **Performance numbers** → [benchmarks.md](./benchmarks.md)
- **Model catalogue + backend selection** → [model-selection.md](./model-selection.md) (narrative) + [`../uwp/models/manifest.json`](../uwp/models/manifest.json) (data)
- **UWP/AppContainer constraints** (§1–§12) → [uwp-constraints.md](./uwp-constraints.md)
- **Version / current state** → [../CHANGELOG.md](../CHANGELOG.md) + [../ROADMAP.md](../ROADMAP.md)

| Document                                                         | Description                                                                                                                                  |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| [architecture.md](./architecture.md)                             | Component/data-flow map: modules, runtime backend dispatch, chat templates, KV-reuse, routing, provisioning auto-upgrade, membw, diffusion   |
| [install-release.md](./install-release.md)                       | Install a tagged GitHub Release build on your Xbox (cert + VCLibs + MSIX)                                                                    |
| [using-the-app.md](./using-the-app.md)                           | App guide: chat, settings (model picker, routing, KV reuse), image generation                                                                |
| [uwp-constraints.md](./uwp-constraints.md)                       | UWP sandbox limitations, measured GPU budget, AppContainer filesystem quirks, and how xllama works around them                               |
| [technical-report.md](./technical-report.md)                     | The measured story (v1.0 snapshot); published as [Discussion #76](https://github.com/gianlucamazza/xllama/discussions/76)                    |
| [console-validation-runbook.md](./console-validation-runbook.md) | Ordered on-console validation checklist with measured results per section                                                                    |
| [fp16-extdata-runbook.md](./fp16-extdata-runbook.md)             | Unblock fp16 models >2 GB on the GPU: zero-code USB spike, then the ORT `weakly_canonical` patch (Fase 1) — entrypoint for the scripts       |
| [windows-dev-vm.md](./windows-dev-vm.md)                         | Windows VM setup for local UWP/MSIX builds                                                                                                   |
| [device-portal.md](./device-portal.md)                           | How to enable Dev Mode and deploy via Device Portal                                                                                          |
| [phase1-runbook.md](./phase1-runbook.md)                         | End-to-end developer build, deploy, and benchmark instructions                                                                               |
| [model-selection.md](./model-selection.md)                       | Choosing/adding models: hard limits, evaluation sequence, tested models, manifest override how-to + ORT-asset publishing (logit-parity gate) |
| [api-endpoint.md](./api-endpoint.md)                             | LAN HTTP endpoint (OpenAI-compatible, opt-in, default OFF): enable, protocol, concurrency, validation                                        |
| [benchmarks.md](./benchmarks.md)                                 | Consolidated perf for every tested model (decode/prefill/RAM, 3 backends) + comparative charts (`benchmarks-charts.html`)                    |
| [recommended-config.md](./recommended-config.md)                 | Correct modern settings: models, genai_config, settings.json, build variants, obsolete myths                                                 |
| [project-analysis-2026-07.md](./project-analysis-2026-07.md)     | Project health snapshot (currency **2026-07-17**: 1.2.0 + #91 gate): status matrix, risks — **not** a perf SSOT                             |
| [vendor-lifecycle-plan.md](./vendor-lifecycle-plan.md)           | Piano di risoluzione residuo: drop pin GenAI/ORT, upstream ReadFile, demo video, catalogue opzionale                                         |
| [phase7-hypotheses.md](./phase7-hypotheses.md)                   | Phase 7 research: peer-class model hypotheses (H1–H9), shortlist, campaign results — **not** a perf SSOT                                     |

See also [../CHANGELOG.md](../CHANGELOG.md) for the full pivot history (llama.cpp → ORT GenAI → CPU EP → per-workload routing) and [../diffusion/README.md](../diffusion/README.md) for the SD-Turbo model toolchain.
