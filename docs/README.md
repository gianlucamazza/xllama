# docs/

Technical notes and design decisions for xllama.

| Document                                                         | Description                                                                                                    |
| ---------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| [install-release.md](./install-release.md)                       | Install a tagged GitHub Release build on your Xbox (cert + VCLibs + MSIX)                                      |
| [using-the-app.md](./using-the-app.md)                           | App guide: chat, settings (model picker, routing, KV reuse), image generation                                  |
| [uwp-constraints.md](./uwp-constraints.md)                       | UWP sandbox limitations, measured GPU budget, AppContainer filesystem quirks, and how xllama works around them |
| [technical-report.md](./technical-report.md)                     | The measured story: per-workload CPU/GPU verdict, falsified hypotheses, diffusion on console                   |
| [console-validation-runbook.md](./console-validation-runbook.md) | Ordered on-console validation checklist with measured results per section                                      |
| [windows-dev-vm.md](./windows-dev-vm.md)                         | Windows VM setup for local UWP/MSIX builds                                                                     |
| [device-portal.md](./device-portal.md)                           | How to enable Dev Mode and deploy via Device Portal                                                            |
| [phase1-runbook.md](./phase1-runbook.md)                         | End-to-end developer build, deploy, and benchmark instructions                                                 |
| [model-selection.md](./model-selection.md)                       | Choosing/adding models: hard limits, evaluation sequence, tested models, manifest override how-to              |

See also [../CHANGELOG.md](../CHANGELOG.md) for the full pivot history (llama.cpp → ORT GenAI → CPU EP → per-workload routing) and [../diffusion/README.md](../diffusion/README.md) for the SD-Turbo model toolchain.
