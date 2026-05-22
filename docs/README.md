# docs/

Technical notes and design decisions for xllama.

| Document | Description |
|----------|-------------|
| [uwp-constraints.md](./uwp-constraints.md) | UWP sandbox limitations, GPU pool limit, AppContainer filesystem quirks, and how xllama works around them |
| [windows-dev-vm.md](./windows-dev-vm.md) | Windows VM setup for local UWP/MSIX builds |
| [device-portal.md](./device-portal.md) | How to enable Dev Mode and deploy via Device Portal |
| [phase1-runbook.md](./phase1-runbook.md) | End-to-end build, deploy, and benchmark instructions (ORT GenAI / SmolLM2-360M) |

| [model-selection.md](./model-selection.md) | Checklist operativo per scegliere/valutare un modello ONNX GenAI: limiti disco e GPU, sequenza di verifica, modelli testati |

See also [../CHANGELOG.md](../CHANGELOG.md) for the full pivot history (llama.cpp → ORT GenAI → CPU EP).
