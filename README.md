# xllama

xllama is a free, MIT-licensed open-source project for running local language models on Xbox Series X|S in Developer Mode. The main way to use it today is an app you sideload onto the console: chat, an OpenAI-compatible API on your local network, and image generation. Inference runs on the console. Once a model is on the console, the app only goes online to download a model you don't have yet. The first launch needs an internet connection.

[![build-uwp](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml)
[![build-linux](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml)
[![CodeQL](https://github.com/gianlucamazza/xllama/actions/workflows/codeql.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/codeql.yml)
[![Release](https://img.shields.io/github/v/release/gianlucamazza/xllama?filter=v*&label=release)](https://github.com/gianlucamazza/xllama/releases/latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- XLLAMA_DOI_START -->

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22118437.svg)](https://doi.org/10.5281/zenodo.22118437)
<!-- XLLAMA_DOI_END -->

Supported on Series S and Series X. The project's own figures are measured on a Series S; the Series X figures below were measured by an independent tester (Joshua Henne / Eidon Labs Inc.) and are labelled as such ([details](docs/positioning.md#series-x)).
On a Series S, the default model LFM2.5-350M generates text at **97.3 tok/s** (decode speed, llama.cpp on CPU, 6 threads, median of 3 runs; [benchmarks](docs/benchmarks.md)).

On a Series X, measured by an independent tester (Joshua Henne / Eidon Labs Inc., xllama 1.7.0.1495, [discussion #434](https://github.com/gianlucamazza/xllama/discussions/434)), not by the project:

- LFM2.5-350M, about **95.7 tok/s** over the local-network API end to end, averaged over ten 1024-token generations, after the app was set as a Game. Before that change, batches averaged about 35.4 to 38.8 tok/s.
- LFM2.5-1.2B-Instruct (QAD Q4_0, CPU forced), about **37.1 tok/s** over the local-network API end to end, the average of three runs.

xllama is an independent project, not affiliated with or endorsed by Microsoft.

xllama is MIT-licensed, so a studio could reuse its inference code in its own game. That path isn't built or tested: today xllama runs on the console as a Developer Mode app.

## Before you sideload

- **Developer Mode only.** You activate it with your own Partner Center developer account and accept Microsoft's [Developer Mode terms](https://learn.microsoft.com/en-us/legal/windows/agreements/xbox-one-developer-mode-activation). To play retail games or use retail apps, switch the console back to Retail Mode.
- **Sideload only from this repository's [Releases](https://github.com/gianlucamazza/xllama/releases)** and check the SHA-256. The package is signed with a test certificate.
- **The LAN API is off by default and has no authentication.** Turn it on only on a network you trust.
- **Models come from third parties,** each under its own licence, and are downloaded separately.
- Full notes: [docs/before-you-sideload.md](docs/before-you-sideload.md).

## Quick start

You only need a terminal for the checksum step ([full guide](docs/install-release.md)):

1. Put the console in **Developer Mode** ([Microsoft's steps](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/devkit-activation)).
2. In Dev Home → Remote Access Settings, enable **Device Portal** with a username and password, then open the URL it shows in a browser.
3. From the [latest app release](https://github.com/gianlucamazza/xllama/releases/latest), download `xllama_*_x64.msix`, `xllama-test.cer`, `Microsoft.VCLibs.x64.14.00.appx`, `SHA256SUMS` and `release-validation.json`.
4. Verify them: `sha256sum -c SHA256SUMS` (macOS: `shasum -a 256 -c SHA256SUMS`). Every line must print `OK`.
5. In Device Portal → My games & apps, sideload the MSIX with `Microsoft.VCLibs.x64.14.00.appx` as its dependency.
6. Launch xllama from Dev Home. On first launch it downloads the default chat model LFM2.5-350M (~219 MB). The app bundles no model.

> The browser steps follow Microsoft's documentation but have not yet been walked end to end on a console. If something doesn't match, [open an issue](https://github.com/gianlucamazza/xllama/issues).

**Command line:** `./scripts/deploy.sh` path in [docs/install-release.md](docs/install-release.md#install-from-the-command-line) · **Build from source:** [docs/crossbuild-console.md](docs/crossbuild-console.md) · **Linux CLI:** `./scripts/apply-llama-patches.sh && cmake --preset linux-release && cmake --build build/linux-release -j`

## Demo

https://github.com/user-attachments/assets/9494a6a2-f14e-4229-afa4-ff5faf51ba67

Series S, Developer Mode; not a speed claim. Provenance: [`xllama-demo-v1.5.6.mp4`](docs/screenshots/xllama-demo-v1.5.6.mp4) and [capture notes](docs/screenshots/README.md).

## What you can do

- **Chat** — multi-turn pad UI with the default LFM2.5-350M model, plus other catalogue models you download.
- **LAN API** — OpenAI-compatible and native Ollama chat (`/api/chat`, `/api/generate`, `/api/ps`) on your local network, a downloadable-model list (`/api/catalog`) and per-request reasoning control; **off by default and has no authentication**. See [docs/api-endpoint.md](docs/api-endpoint.md).
- **Image generation** — SD-Turbo on the console. **Powered by Stability AI.**
- **Benchmarks** — headless tok/s and platform probes on the console. See [bench/README.md](bench/README.md).
- **Experimental fine-tuning** — on-device last-block fine-tune, host PEFT on a workstation, and serving a merged GGUF. The in-app _Train on my feedback_ flow is not yet walked end to end on a console.

## Models and performance (Series S)

Headline decode figures from [docs/benchmarks.md](docs/benchmarks.md). Hugging Face collection: [xllama: GGUF models run on Xbox Series S](https://huggingface.co/collections/gianlucamazza/xllama-gguf-models-run-on-xbox-series-s-6aca0b32275cbe99dbf02d89).

| Model              | Decode tok/s | Backend         | Peak RAM (MiB) |
| ------------------ | ------------ | --------------- | -------------- |
| LFM2.5-230M        | 126.9        | llama.cpp CPU   | 237            |
| LFM2.5-350M        | 97.3         | llama.cpp CPU   | 311            |
| LFM2.5-1.2B        | 74.0         | llama.cpp d3d12 | 861            |
| Qwen2.5-Coder-1.5B | 26.1         | llama.cpp CPU   | 1176           |
| LFM2-2.6B          | 18.1         | llama.cpp CPU   | 1629           |

Series S, Developer Mode, median of 3 runs on the v1.11.0 release package. LFM2.5-1.2B is the default GPU model on llama.cpp d3d12. None of these run READMEs state a Game designation.

## LAN API

**Off by default and has no authentication.** Turn it on only on a network you trust. Protocol: [docs/api-endpoint.md](docs/api-endpoint.md).

```bash
curl -s http://<console-ip>:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm25-350m","messages":[{"role":"user","content":"hi"}]}'
```

Each chat model carries its own sampling defaults, so a request sets only what it wants to change. Read a model's values from `/v1/models`:

```bash
curl -s http://<console-ip>:11434/v1/models |
  python3 -c 'import json,sys; m={x["id"]:x for x in json.load(sys.stdin)["data"]}; print(json.dumps(m["lfm25-1.2b-instruct"]["sampling"], indent=2))'
```

```json
{
  "temperature": 0.1,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.05,
  "source": {
    "temperature": "model",
    "top_p": "default",
    "top_k": "model",
    "repetition_penalty": "model"
  }
}
```

`model` values come from the model's card in the catalogue ([measured](bench/results/2026-11-27-lfm25-card-sampling/)); `default` is the global default. This request changes only the temperature, and keeps the model's other three values:

```bash
curl -s http://<console-ip>:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm25-1.2b-instruct","temperature":0.7,"messages":[{"role":"user","content":"hi"}]}'
```

A field the request sets applies to that request only. The order of the layers is in [Discovery](docs/api-endpoint.md#discovery-health-and-v1models).

## How it works

Shared C++17 core, WinRT-free and host-testable, with two front ends (Linux CLI and the UWP app).
Two backends, dispatched per model at runtime: GGUF through llama.cpp, everything else through ONNX Runtime GenAI and DirectML.
One resident session in the process — two models do not fit the console budget.
UWP/AppContainer limits apply (no mmap, no dlopen, no registry, no arbitrary paths); measured ceilings live in the constraints doc.

[Architecture](docs/architecture.md) · [Design decisions](docs/design-decisions.md) · [UWP constraints](docs/uwp-constraints.md) · [Reuse in a game](docs/reuse-in-a-game.md)

## Documentation

| Group    | Topic               | Docs                                                               |
| -------- | ------------------- | ------------------------------------------------------------------ |
| Use      | Sideload a release  | [install-release.md](docs/install-release.md)                      |
| Use      | Before you sideload | [before-you-sideload.md](docs/before-you-sideload.md)              |
| Use      | Using the app       | [using-the-app.md](docs/using-the-app.md)                          |
| Use      | LAN API             | [api-endpoint.md](docs/api-endpoint.md)                            |
| Build    | Architecture        | [architecture.md](docs/architecture.md)                            |
| Build    | Build from source   | [crossbuild-console.md](docs/crossbuild-console.md)                |
| Build    | Platform limits     | [uwp-constraints.md](docs/uwp-constraints.md)                      |
| Research | Benchmarks          | [benchmarks.md](docs/benchmarks.md)                                |
| Research | Positioning         | [positioning.md](docs/positioning.md)                              |
| Research | Reuse in a game     | [reuse-in-a-game.md](docs/reuse-in-a-game.md)                      |
| Research | Research package    | [paper/](paper/) · [technical-report.md](docs/technical-report.md) |

Index: [docs/README.md](docs/README.md)

## Upstream contributions

Running xllama on an Xbox Series S in Developer Mode surfaced problems in the
open-source ONNX Runtime projects. These are the pull requests Gianluca Mazza
opened against them, with their status as shown on GitHub:

| Project           | Pull request                                                      | What it does                                                                                      | Status              |
| ----------------- | ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------- |
| onnxruntime-genai | [#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280) | Falls back to the system D3D12 runtime when the Agility SDK device factory cannot create a device | Merged, 13 Jul 2026 |
| onnxruntime-genai | [#2300](https://github.com/microsoft/onnxruntime-genai/pull/2300) | Lets a model opt out of DirectML graph capture through a provider option                          | Merged, 20 Jul 2026 |
| onnxruntime-genai | [#2316](https://github.com/microsoft/onnxruntime-genai/pull/2316) | Adds tests and documentation for the #2300 opt-out                                                | Open                |
| onnxruntime       | [#29732](https://github.com/microsoft/onnxruntime/pull/29732)     | Reduces the Windows `ReadFileIntoBuffer` chunk size to 16 MB (AppContainer error 1450)            | Open                |

xllama ships builds of these libraries with its own patches until an official
package includes the fixes it needs; see [License](#license) and
[`patches/`](patches/README.md). Closed upstream issues are listed in
[docs/upstream.md](docs/upstream.md).

## Contributing

UWP packaging, compact models, benchmark methodology, and non-Windows
developer documentation are the usual places to help. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## Support

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-db61a2?logo=github)](https://github.com/sponsors/gianlucamazza)

xllama is free and MIT-licensed. If it's useful to you, you can support its
development on [GitHub Sponsors](https://github.com/sponsors/gianlucamazza) —
funds go to development time, Xbox Developer Mode costs, and model/storage budgets
for testing.

## Citation

The versioned research report is archived on Zenodo. The concept DOI
10.5281/zenodo.22118437 always resolves to the latest version; to cite 1.0.6 specifically:

```bibtex
@software{xllama_research_1_0,
  author  = {Mazza, Gianluca},
  title   = {Consumer Game Consoles as Local AI Compute},
  version = {1.0.6},
  doi     = {10.5281/zenodo.22119126},
  url     = {https://github.com/gianlucamazza/xllama}
}
```

The DOI badge and version DOI are synchronized from `release.toml`.

## Acknowledgements, licence and trademarks

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) — Microsoft
- The Xbox Developer Mode community
- Andrei David's `llama2.c` port to Xbox 360

<a id="license"></a>

xllama is MIT-licensed; see [LICENSE](LICENSE). The `llama.cpp` submodule is
under MIT and xllama applies its own UWP patches to it at build time
([`patches/`](patches/README.md)).

The `onnxruntime.dll` and `onnxruntime-genai.dll` that xllama ships are
patched by xllama and are not official builds. They are built from the
open-source ONNX Runtime (1.24.4) and ONNX Runtime GenAI (0.14.1) sources with
the patches in [`patches/`](patches/README.md), and pinned by SHA-256 in
[`vendor/`](vendor/). Both projects are under the MIT License, Copyright (c)
Microsoft Corporation. Their licence texts and third-party notices are in
[LICENSES/](LICENSES/) and [NOTICE](NOTICE).

xllama is an independent open-source project by Gianluca Mazza. It is not affiliated with, endorsed, sponsored or approved by Microsoft. Xbox, Xbox Series S and Xbox Series X are trademarks of the Microsoft group of companies. Llama is a trademark of Meta; xllama is not affiliated with Meta or with the Ollama project. Other product names are trademarks of their respective owners and are used only to describe compatibility.
