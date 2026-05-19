# xllama

> Running LLM inference on Xbox Series S|X via UWP Dev Mode.

**Status:** experimental · early development
**License:** MIT
**Maintainer:** [Venere Labs](https://github.com/gianlucamazza)

---

## What is this

`xllama` is a port of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) to Xbox Series S|X consoles running in Dev Mode, packaged as a UWP application. It enables local inference of GGUF-quantized language models on the Xbox's Zen 2 CPU and, in later stages, on its RDNA 2 GPU via Vulkan through Mesa/libgallium.

The goal is twofold:

1. Demonstrate that modern consumer console hardware is a viable, underexplored substrate for local LLM inference.
2. Publish a clean, reproducible baseline that future work (gaming AI, on-device assistants, research on heterogeneous inference) can build on.

This is a research-grade hobby project, not a Microsoft-endorsed product. "Xbox" is a Microsoft trademark; this project is not affiliated with Microsoft.

---

## Why Xbox Series S

The Xbox Series S is interesting as an inference target because:

- **Capable CPU**: 8 Zen 2 cores @ 3.6 GHz, AVX2, comparable to a Ryzen 7 3700X.
- **Modern GPU**: RDNA 2, ~4 TFLOPS FP32, with hardware support for INT8/INT4 operations.
- **Unified memory**: 10 GB GDDR6, of which ~8 GB are addressable by a Dev Mode "Game" application.
- **Accessible Dev Mode**: a one-time ~$19 activation via Partner Center unlocks unsigned UWP deployment. No retail dev kit required.
- **Underexplored**: no public llama.cpp port for the platform exists at the time of writing.

A 7B model quantized to Q4_K_M (~4.5 GB) fits comfortably in memory with room for context. Realistic decode targets: 6–10 tok/s CPU-only, higher with the GPU backend.

---

## Architecture

```
┌──────────────────────────────────────────┐
│  Host: Linux (development)               │
│  ├─ llama.cpp fork + UWP patches         │
│  ├─ GGUF conversion / quantization       │
│  └─ Deploy via Device Portal (HTTP)      │
└──────────────────┬───────────────────────┘
                   │
                   ▼  cross-build (CI)
┌──────────────────────────────────────────┐
│  Build: Windows + MSVC + Windows SDK     │
│  └─ Produces signed .appx package        │
└──────────────────┬───────────────────────┘
                   │
                   ▼  sideload
┌──────────────────────────────────────────┐
│  Target: Xbox Series S|X (Dev Mode)      │
│  ├─ UWP container, ~8 GB RAM available   │
│  ├─ CPU backend (Phase 1)                │
│  └─ Vulkan backend via Mesa (Phase 2)    │
└──────────────────────────────────────────┘
```

The development workflow runs on Linux. The final UWP packaging step requires MSVC and the Windows SDK and is performed either in a Windows VM or via a GitHub Actions Windows runner. Deployment to the console uses the Xbox Device Portal REST API, callable from any OS.

---

## Repository layout

```
xllama/
├── llama.cpp/              # upstream submodule, pinned
├── patches/                # UWP-specific patches against upstream
├── uwp/
│   ├── AppxManifest.xml
│   ├── App.cpp             # UWP entry point
│   ├── App.h
│   ├── llama-bridge.cpp    # llama.cpp ↔ UWP lifecycle glue
│   ├── llama-bridge.h
│   ├── pch.h
│   └── xllama.sln
├── cmake/                  # toolchain files for UWP / Linux targets
├── scripts/
│   ├── build-uwp.ps1       # MSVC build (Windows)
│   ├── deploy.sh           # Device Portal upload (Linux)
│   └── quantize.sh         # GGUF preparation
├── bench/                  # benchmark suite + result CSVs
├── docs/                   # technical notes, design decisions
└── .github/workflows/      # CI: Linux dev build + Windows UWP packaging
```

---

## Build

### Requirements

- Linux host (Ubuntu 22.04+ tested) for development
- Windows 11 + Visual Studio 2022 with the "Universal Windows Platform development" workload, for packaging
- Xbox Series S or X with Dev Mode activated
- Xbox Device Portal username and password

### Develop on Linux

```bash
git clone --recursive https://github.com/gianlucamazza/xllama.git
cd xllama
cmake -B build -DXLLAMA_TARGET=linux
cmake --build build -j
./build/bin/xllama-cli -m models/qwen3-1.7b-Q4_K_M.gguf -p "Hello"
```

This validates the bridge code against the Linux build before attempting the UWP packaging.

### Package for Xbox (Windows)

```powershell
cd uwp
./scripts/build-uwp.ps1
```

The output is `xllama_<version>_x64.appx`.

### Deploy to console

With the console in Dev Mode and Device Portal enabled:

```bash
export XBOX_IP=192.168.1.42
export XBOX_USER=devuser
export XBOX_PASS=...
./scripts/deploy.sh xllama_0.1.0_x64.appx
```

The script uploads the package via the Device Portal REST API and triggers installation.

---

## Models

`xllama` consumes standard GGUF files. Tested targets for Phase 1:

| Model           | Quant   | Size    | Fits in 8 GB | Expected tok/s (CPU) |
|-----------------|---------|---------|--------------|----------------------|
| Qwen3 1.7B      | Q4_K_M  | ~1.1 GB | ✅           | 25–40                |
| Llama 3.2 3B    | Q4_K_M  | ~2.0 GB | ✅           | 15–25                |
| Qwen3 8B        | Q4_K_M  | ~4.7 GB | ✅           | 6–10                 |
| Llama 3 8B      | Q4_K_M  | ~4.7 GB | ✅           | 6–10                 |
| Llama 3 13B     | Q4_K_S  | ~7.0 GB | ⚠ tight     | TBD                  |

Numbers are projections based on Zen 2 reference platforms and will be replaced with measured values as Phase 1 lands.

---

## Limitations

- **Sandboxed filesystem**: UWP apps cannot read arbitrary paths. Models must be transferred to the app's local storage via Device Portal or USB.
- **No `mmap`**: the standard llama.cpp memory-mapping path does not work. We use a Win32 `CreateFileMapping`-based equivalent.
- **No JIT, no `dlopen`**: dynamic backend loading is replaced with compile-time selection.
- **DirectML not exposed**: the Xbox's INT8/INT4 acceleration hardware is not directly addressable from a Dev Mode UWP app. We work around this via Vulkan compute (Phase 2).
- **Dev Mode only**: there is no path to running this on a retail-mode console, and that is by design.

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for the full plan. Headline phases:

1. **Phase 1 — CPU baseline.** Working UWP build, CPU-only inference, reproducible benchmarks.
2. **Phase 2 — Vulkan backend.** GPU acceleration via Mesa/libgallium on Xbox.
3. **Phase 3 — Optimization.** Quantization tuning, KV-cache strategies, memory layout.
4. **Phase 4 — Publication.** Technical report, benchmark dataset, demo.

---

## Contributing

Issues and PRs welcome. The project is small enough that there is no formal governance yet — open an issue describing what you'd like to work on and we'll coordinate.

Areas where contributors are particularly useful:

- UWP packaging and Microsoft Store / Dev Mode quirks
- Mesa/libgallium on Xbox (overlap with the wider Xbox homebrew scene)
- Benchmark methodology and reproducibility
- Documentation, especially install guides for non-Windows developers

---

## Acknowledgements

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors — the upstream that makes this possible.
- The Xbox homebrew community, in particular the work on libgallium, SDL, and Mesa for UWP that opens the door to the Vulkan backend.
- Andrei David's `llama2.c` port to the Xbox 360, which demonstrated that this class of project is worth doing.

---

## License

MIT. See [LICENSE](./LICENSE).

`llama.cpp` is included as a submodule under its own MIT license.
