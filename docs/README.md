# docs/

Technical notes and design decisions for xllama.

## Documentation ownership

Each current fact has one authoritative home. Other documents may explain a
decision or quote one headline, but must link back instead of maintaining a
second table.

- **System structure** (modules, backend dispatch, provisioning, membw, host-only LoRA personalization) → [architecture.md](./architecture.md)
- **Raw performance evidence** → [`../bench/results/`](../bench/results/)
- **Comparison policy** → [`../bench/benchmark-summary.json`](../bench/benchmark-summary.json)
- **Generated performance summary** → [benchmarks.md](./benchmarks.md) +
  [benchmarks-charts.html](./benchmarks-charts.html)
- **Model catalogue + backend selection** → [model-selection.md](./model-selection.md) (narrative) + [`../uwp/models/manifest.json`](../uwp/models/manifest.json) (data)
- **UWP/AppContainer constraints** (§1–§12) → [uwp-constraints.md](./uwp-constraints.md)
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
| [architecture.md](./architecture.md)                             | Component/data-flow map: modules, runtime backend dispatch, chat templates, KV-reuse, routing, provisioning, membw, diffusion, host LoRA     |
| [../scripts/lora-spike/README.md](../scripts/lora-spike/README.md) | Host-only PEFT LoRA → GGUF merge → `xllama-cli` A/B (toy marker; no on-device training)                                                      |
| [install-release.md](./install-release.md)                       | Install a tagged GitHub Release build on your Xbox (cert + VCLibs + MSIX)                                                                    |
| [using-the-app.md](./using-the-app.md)                           | App guide: chat, settings (model picker, routing, KV reuse), image generation                                                                |
| [demo-video-runbook.md](./demo-video-runbook.md)                 | Phase 6 demo clip: preflight, storyboard, capture, publish — **console session**                                                             |
| [uwp-constraints.md](./uwp-constraints.md)                       | UWP sandbox limitations, measured GPU budget, AppContainer filesystem quirks, and how xllama works around them                               |
| [technical-report.md](./technical-report.md)                     | Historical v1.0 narrative snapshot; published as [Discussion #76](https://github.com/gianlucamazza/xllama/discussions/76)                  |
| [console-validation-runbook.md](./console-validation-runbook.md) | Ordered on-console validation checklist with measured results per section                                                                    |
| [fp16-extdata-runbook.md](./fp16-extdata-runbook.md)             | Unblock fp16 models >2 GB on the GPU: zero-code USB spike, then the ORT `weakly_canonical` patch (Fase 1) — entrypoint for the scripts       |
| [windows-dev-vm.md](./windows-dev-vm.md)                         | Windows VM setup for local UWP/MSIX builds                                                                                                   |
| [device-portal.md](./device-portal.md)                           | How to enable Dev Mode and deploy via Device Portal                                                                                          |
| [phase1-runbook.md](./phase1-runbook.md)                         | End-to-end developer build, deploy, and benchmark instructions                                                                               |
| [model-selection.md](./model-selection.md)                       | Choosing/adding models: hard limits, evaluation sequence, tested models, manifest override how-to + ORT-asset publishing (logit-parity gate) |
| [api-endpoint.md](./api-endpoint.md)                             | LAN HTTP endpoint (OpenAI-compatible, opt-in, default OFF): enable, protocol, concurrency, validation                                        |
| [benchmarks.md](./benchmarks.md)                                 | Generated current comparison plus interpretation; raw measurements remain under `bench/results/`                                           |
| [recommended-config.md](./recommended-config.md)                 | Correct modern settings: models, genai_config, settings.json, build variants, obsolete myths                                                 |
| [project-analysis-2026-07.md](./project-analysis-2026-07.md)     | Dated project-health snapshot: status matrix and risks; current facts live in the owners listed above                                      |
| [vendor-lifecycle-plan.md](./vendor-lifecycle-plan.md)           | Piano di risoluzione residuo: drop pin GenAI/ORT, upstream ReadFile, demo video, catalogue opzionale                                         |
| [phase7-hypotheses.md](./phase7-hypotheses.md)                   | Phase 7 research log: hypotheses, gates and verdicts; current metrics come from the generated benchmark summary                             |

See also [../CHANGELOG.md](../CHANGELOG.md) for the full pivot history (llama.cpp → ORT GenAI → CPU EP → per-workload routing) and [../diffusion/README.md](../diffusion/README.md) for the SD-Turbo model toolchain.
