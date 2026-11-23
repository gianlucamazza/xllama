# Reuse in a game

xllama is MIT-licensed, so a studio could reuse its inference code in its own
game. That path isn't built or tested: today xllama is a Developer Mode app.

This note is the technical picture of that path. Product intent and honest
limits live in [positioning.md](positioning.md). Public headers are summarised
in [sdk/overview.md](sdk/overview.md).

xllama is an independent project, not affiliated with or endorsed by Microsoft.

## The core

The reusable core is a C++17 static library. On Linux it is the CMake target
`xllama` (`src/bridge/` plus public headers in `include/xllama/`). The UWP app
compiles those same bridge sources into the packaged app
(`uwp/xllama.vcxproj`); `cmake -DXLLAMA_TARGET=uwp` is a deliberate fatal
error and points at `scripts/build-uwp.ps1`.

The headers are WinRT-free and host-testable. Two front ends call the core
today: `xllama-cli` on Linux, and the C++/WinRT app under `uwp/`.

The core is built today for Linux and UWP only.

## Portable core versus UWP / Developer Mode

### Portable (the core)

Public types live in `include/xllama/`. Implementations live in `src/bridge/`.
That is the inference session (`Session` / `SessionHub`), parameter and result
types, routing and prompt-budget helpers, sampling defaults, chat-prompt
formatting, embedding helpers, KV snapshot helpers, HTTP framing and SSE
helpers used by the LAN protocol, tool-proposal classification (the process
does not execute tools), diffusion CLIP / Euler / PNG helpers, and
training-job structs.

Linux CMake builds llama.cpp only (`XLLAMA_USE_LLAMA`). UWP ORT GenAI +
DirectML code stays under `#ifdef XLLAMA_USE_ORT` in
`src/bridge/inference.cpp`. Path and log helpers in
`src/bridge/path_utils.cpp` and `src/bridge/platform.cpp` call
`ApplicationData` only when `XLLAMA_UWP` is set.

### UWP- or Developer-Mode-specific (`uwp/`)

These are not the portable core:

- Pad UI (`MainPage.cpp`, XAML)
- LAN API **server** (`api-server.cpp`, WinRT `StreamSocketListener`). Framing
  helpers in the core are not a listener.
- Model downloader (`model-downloader.cpp`, WinRT `HttpClient`)
- LocalState knobs (flag files such as `api.flag`, `bench.flag`, `train.flag`,
  `diffuse.flag`, `gguf_gpu_layers.txt`, `model.txt`)
- `ApplicationData` paths (`LocalFolder` / LocalState)
- Chat history, headless flag dispatch in `App.cpp`, catalogue
  `uwp/models/manifest.json`, AppxManifest / MSIX packaging

`include/xllama/catalog_trust.h` is compiled only with `XLLAMA_UWP` and is
used by the downloader.

## What has not been done

Nobody has built this core with Microsoft's GDK. Nobody has run it inside a
published game. There is no packaged library for a game engine, and there is
no engine integration. Those statements are untested.

Do not read this note as a product claim that xllama is a game-runtime kit.

## Caveats a studio owns

The MIT licence covers xllama's own code. It does not cover models, DirectML,
or ONNX Runtime. A studio that copies inference code still owns the rest:

- **Model licences.** Catalogue models are third-party. LFM Open License
  commercial use is only under US$10M revenue. Stability AI Community License
  needs registration and a US$1M cap. Gemma Terms and the Meta Llama licence
  also apply to those families. Check `LICENSES/` and
  [model-selection.md](model-selection.md) for the exact text; this note does
  not restate it.
- **DirectML / ONNX Runtime notices** in `LICENSES/`. The UWP app ships
  patched `DirectML.dll`, `onnxruntime.dll`, and `onnxruntime-genai.dll`.
  Their licences and third-party notices travel with those binaries.
- **The game's own memory and GPU budget.** The 1024 MB and 5120 MB figures in
  [uwp-constraints.md](uwp-constraints.md) are UWP App/Game caps, not game
  budgets. There are no in-game measurements.
- **Microsoft's certification and content rules for generative AI.** Untested.
  This project has not walked that process.

Fine-tuning is not this path. Host PEFT runs on a workstation. In-app training
is not walked end to end on a console.
