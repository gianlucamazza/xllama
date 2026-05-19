# AGENTS.md — xllama

Guida rapida per agenti AI e nuovi contributor.

## Convenzioni

- **Lingua:** inglese per codice, commenti, nomi file, commit message.
- **Standard C++:** C++17 (anche su UWP / SDK 22621).
- **Stile:** conciso, no over-engineering. Preferire RAII e `unique_ptr`.
- **Formattazione:** auto-format on save se possibile; altrimenti seguire lo stile esistente.

## Struttura directory (post-refactoring)

```
xllama/
├── include/xllama/        # Header pubblici condivisi
│   ├── inference_params.h # InferenceParams / InferenceResult
│   ├── inference.h        # run_inference, write_bench_csv
│   ├── cli.h              # parse_cli_args
│   ├── llama_raii.h       # unique_ptr wrapper per llama.cpp
│   ├── platform.h         # log_output, detect_threads, peak_working_set_mb
│   ├── path_utils.h       # resolve_model_path, resolve_local_path
│   └── utf8_utils.h       # utf8 <-> wstring (Windows)
├── src/bridge/            # Implementazioni condivise (Linux + UWP)
│   ├── inference.cpp
│   ├── bench.cpp
│   ├── cli.cpp
│   ├── platform.cpp
│   ├── path_utils.cpp
│   └── utf8_utils.cpp
├── src/main.cpp           # Entry point Linux (usa getopt_long)
├── uwp/                   # App C++/WinRT + stub UWP
│   ├── llama-bridge.cpp   # Thin wrapper + main_loop()
│   ├── llama-bridge.h     # Alias backward-compat
│   ├── llama-mmap-uwp.cpp
│   └── ...
├── tests/                 # Unit test (doctest)
├── scripts/               # Build/deploy/bench (Bash / PowerShell)
├── bench/                 # Configurazioni benchmark
├── docs/                  # Note tecniche
└── cmake/                 # Toolchain files
```

## Build

### Linux (sviluppo + test)

```bash
# Release
 cmake --preset linux-release
 cmake --build build/linux-release -j$(nproc)

# Debug con test
 cmake --preset linux-test
 cmake --build build/linux-test -j$(nproc)
 ctest --test-dir build/linux-test --output-on-failure

# Smoke test
 ./build/linux-release/bin/xllama-cli --help
```

### UWP (Windows / CI)

```powershell
.\scripts\build-uwp.ps1
```

Usare `-ForceNewCert` solo se si vuole rigenerare il certificato di test.

## Test

- Framework: **doctest** (header-only, fetch via CMake).
- Target: `xllama-tests`.
- Comando: `ctest --test-dir build/linux-test --output-on-failure`.
- Aggiungere test in `tests/test_*.cpp`.

## Note critiche

- **Mai committare `.env`** (contiene credenziali). Usare `.env.example` come template.
- **Mai committare certificati `.pfx` / `.cer`**. Sono in `.gitignore`.
- **UWP constraints:** no mmap POSIX, no dlopen, no registry, no thread affinity desktop API. Le patch in `uwp/patches/llama.cpp/` applicano guardie `#if WINAPI_FAMILY_PARTITION`.
- **Line splicing C++:** evitare backslash `\` alla fine di righe di commento `//`; il preprocessore C++ li interpreta come continuazione di riga e commenta la riga successiva.
