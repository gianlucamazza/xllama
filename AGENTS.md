# AGENTS.md

Agent guide for xllama. Human overview is [README.md](README.md). Facts have
one home — [docs/README.md](docs/README.md) — so link those documents instead
of copying tables, tok/s figures, or constraint numbers into a second place.

## Project overview

xllama runs local chat, SD-Turbo diffusion, and on-device training on Xbox
Series S|X (Dev Mode) and a Linux host CLI. The shared core is C++17 under
`include/xllama/` (WinRT-free, host-testable) and `src/bridge/`. Two front
ends call it: `xllama-cli` on Linux, and the C++/WinRT UWP app in `uwp/`.

Shipping inference is a **unified** build. At runtime, `*.gguf` goes through
llama.cpp and everything else through ONNX Runtime GenAI + DirectML. The
Linux CMake build is llama.cpp only (`XLLAMA_USE_LLAMA`). UWP ORT code stays
under `#ifdef XLLAMA_USE_ORT`.

`llama.cpp/` is a git submodule. Do not commit edits inside it. AppContainer
fixes belong in `patches/` and are applied by `scripts/apply-uwp-patches.sh`.

## Setup

Linux host toolchain (Ubuntu packages match CI):

```bash
git submodule update --init --recursive
sudo apt-get install -y cmake build-essential libcurl4-openssl-dev
```

CMake 3.22 or newer. Presets live in `CMakePresets.json`.

Formatter pins, same versions CI installs:

```bash
pip3 install --user 'clang-format==22.1.5'
npm install -g 'prettier@3.9.6'
# shellcheck v0.11.0 — apt on Ubuntu 22.04 is older and misses findings
```

UWP packaging needs a Windows host with the UWP workload (SDK 22621). See
[docs/windows-dev-vm.md](docs/windows-dev-vm.md). `cmake -DXLLAMA_TARGET=uwp`
is a deliberate fatal error that points at `scripts/build-uwp.ps1`; CI checks
that message.

Console deploy reads `XBOX_IP`, `XBOX_USER`, and `XBOX_PASS` from
`~/.config/xllama/xbox-env`. That file stays outside the repo.

## Development workflow

Day-to-day work is the Linux test preset. It is Debug, tests on, probes on.

```bash
cmake --preset linux-test
cmake --build build/linux-test -j"$(nproc)"
./build/linux-test/bin/xllama-cli --help
```

Release binary (what README smoke uses):

```bash
cmake --preset linux-release
cmake --build build/linux-release -j"$(nproc)"
./build/linux-release/bin/xllama-cli --help
```

Other presets: `linux-debug` (no tests), `linux-asan` (Debug + ASan + tests).
`XLLAMA_ENABLE_UBSAN` is a separate cache option; the asan preset does not
turn it on. `XLLAMA_NATIVE_OPT=ON` tunes ggml for the build machine;
default is portable AVX2.

Where to put a change:

- Behaviour both the CLI and the UWP app need: a WinRT-free header in
  `include/xllama/` plus the `.cpp` in `src/bridge/`, then a
  `tests/test_*.cpp` registered in `tests/CMakeLists.txt`.
- UWP-only UI, LAN server, or headless flags: `uwp/`. You cannot compile
  that tree with the Linux preset. Say so in the PR.
- A new catalogue model is a `uwp/models/manifest.json` entry after the
  ladder in [docs/architecture.md](docs/architecture.md) (host smoke, then
  console bench, then manifest). `n_ctx` and `role` are session knobs, not a
  second backend.
- Training job JSON goes in `training/jobs/` and must pass
  `xllama-cli --validate-train-job`.

## Testing

Host suite is one doctest binary, `xllama-tests` (doctest v2.4.11, fetched by
CMake). Case counts live in
[docs/architecture.md](docs/architecture.md) (Unit test map). If the count
changes, update every doc that cites it — CI prints the total and does not
pin the number.

```bash
ctest --test-dir build/linux-test --output-on-failure
```

One doctest case (wildcard, from the build directory's test binary):

```bash
./build/linux-test/tests/xllama-tests --test-case='*prompt_budget*'
```

ASan (also the manual `workflow_dispatch` job on `build-linux.yml`):

```bash
cmake --preset linux-asan
cmake --build build/linux-asan -j"$(nproc)"
ctest --test-dir build/linux-asan --output-on-failure
```

Python checks that CI runs outside ctest:

```bash
python3 scripts/check-coherence.py
python3 scripts/generate-benchmark-summary.py --check
python3 scripts/build-research-package.py --check
python3 scripts/check-release-metadata.py
python3 -m unittest tests/test_research_package.py tests/test_xab_contract.py \
  tests/test_release_metadata.py tests/test_release_bundle.py
```

After the CLI exists, CI validates every `training/jobs/*.json`:

```bash
./build/linux-test/bin/xllama-cli --validate-train-job training/jobs/<job>.json
```

New tests: `TEST_CASE` in `tests/test_<header>.cpp`, include the header under
test, link nothing beyond `xllama` (already on the target). Register the file
in `tests/CMakeLists.txt`. Prefer a pure helper on Linux over a WinRT-only
branch when the behaviour is not sandbox-specific.

Console and LAN gates need a deployed package and `xbox-env`. They are not
part of the Linux CI job. Procedure:
[docs/console-validation-runbook.md](docs/console-validation-runbook.md).

```bash
source ~/.config/xllama/xbox-env
# gates: routing|settings|gguf|longchat|kvsnap|coderpaste|thinkcut|thinkdone|genroom|taesd|store|all
./scripts/validate-console.sh <gate|all>
# modes: serve|rate|lora-rt|device-train|all
./scripts/validate-console-training.sh <mode>
./scripts/validate-api.sh <spike|chat|budget|embed|prefs|train|all>
```

## Code style

- English for code, comments, filenames, docs, and commit messages.
- C++17. RAII and `std::unique_ptr`. Concise; no speculative abstraction.
- Public types live in namespace `xllama`.
- Match the SPDX header on the neighbouring file
  (`// Copyright (c) 2024 Gianluca Mazza` / `SPDX-License-Identifier: MIT`).
- `.clang-format`: LLVM, 4-space indent, column 100, pointers and references
  on the left, attached braces, sorted includes. Format with
  **clang-format 22.1.5** only — a newer clang-format will fail CI.

```bash
clang-format -i path/to/file.cpp
# CI check (skips llama.cpp/ and build trees):
find . -path ./llama.cpp -prune -o -path ./build -prune -o -path ./build-uwp-test -prune -o \
  \( -name '*.cpp' -o -name '*.h' -o -name '*.c' \) -type f -print |
  xargs clang-format --dry-run --Werror
```

- Markdown: **prettier 3.9.6**, `.prettierrc` (`proseWrap: preserve`). It
  formats every tracked `*.md` / `*.markdown` except paths in
  `.prettierignore` (`llama.cpp/`, `vendor/`, `paper/generated/`, build
  trees).

```bash
npx prettier@3.9.6 --write path/to/file.md
git ls-files '*.md' '*.markdown' | xargs prettier --check
```

- Shell: `shellcheck scripts/*.sh` at v0.11.0.

Regenerating `docs/benchmarks.md` is two steps: run
`python3 scripts/generate-benchmark-summary.py`, then prettier. Do not
hand-edit that file or `docs/benchmarks-charts.html`.

## Repository map

```
include/xllama/     WinRT-free headers. One concern per header.
  diffusion/        CLIP tokenizer, Euler scheduler, PNG writer (host-testable)
src/bridge/         Shared .cpp for Linux and UWP (inference, session, training, probes)
src/main.cpp        Linux CLI (getopt_long)
uwp/                C++/WinRT app, LAN API, headless flags, AppxManifest, vcxproj
  models/manifest.json   Catalogue data
  packages.config        NuGet pins
training/           Job JSON, host PEFT, datasets
tests/              doctest (test_*.cpp) plus a few unittest modules
scripts/            Deploy, bench, validate, crossbuild, coherence
docs/               SSOT map is docs/README.md
shaders/            HLSL and generated DXIL for the GPU probes
bench/              Raw results and comparison policy
patches/            llama.cpp and vendor patches applied at UWP build time
diffusion/          SD-Turbo → ONNX host toolchain (not inside the MSIX)
paper/              Citable research package
llama.cpp/          Submodule. Do not edit in place.
vendor/             Patched ORT / GenAI DLL hashes (vendor/*/SHA256SUMS)
cmake/              CMake helpers
```

Load-bearing headers agents usually touch:

| Header                               | Role                                                 |
| ------------------------------------ | ---------------------------------------------------- |
| `session.h` / `session_hub.h`        | `Session` API; the one process-wide resident session |
| `inference_params.h` / `inference.h` | Params, result, `run_inference`                      |
| `routing_policy.h`                   | Backend pick and prompt budget                       |
| `prompt_budget.h`                    | `fit_prompt` — the only token-budget trimmer         |
| `sampling.h`                         | Sampler defaults shared by CLI, bench, GUI, API      |
| `chat_prompt.h`                      | `ChatFormat` and stop sequences                      |
| `embedding.h`                        | Embedding params and pooling trim                    |
| `api_policy.h`                       | Rejects tool-execution fields on the LAN API         |
| `training.h` / `device_train.h`      | Job validation and Lane B device train               |
| `personalize.h`                      | In-app personalize helpers                           |
| `json_utils.h`                       | Header-only JSON escape / parse (no `.cpp`)          |
| `catalog_trust.h`                    | UWP-only catalogue signature types                   |

Sampler chains are not headers under `include/`: `src/bridge/sampler_chain.h`
(llama.cpp) and `src/bridge/ort_sampling.h` (ORT). Decode loops are
`src/bridge/decode_loop.h` and `decode_loop_ort.h`.

## Build and deployment

Linux CI is `.github/workflows/build-linux.yml` on every pull request
(feature-branch pushes do not build). It formats, shellchecks, runs the
Python gates above, configures `linux-test`, builds, validates training jobs,
and runs ctest.

UWP CI is `.github/workflows/build-uwp.yml` on `windows-2022`:

| Artifact               | What it is                                                |
| ---------------------- | --------------------------------------------------------- |
| `xllama-appx`          | Shipping package: unified + patched GenAI + patched ORT   |
| `xllama-appx-llamacpp` | Bench-only llama.cpp lane, not the pad chat build         |
| `xllama-appx-store`    | Store SKU, only `workflow_dispatch` with `store_sku=true` |

That CI MSVC package is the shipping and measurement path on Series S. A
Linux `scripts/crossbuild-uwp.sh` package can launch; ORT/GenAI, first boot,
and uptime are not the product claim. See
[docs/crossbuild-console.md](docs/crossbuild-console.md).

Local UWP package:

```powershell
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64
```

`-ForceNewCert` only regenerates the test signing certificate.

Deploy and logs (Device Portal):

```bash
source ~/.config/xllama/xbox-env
./scripts/deploy.sh path/to/xllama_*.msix
./scripts/deploy.sh get-log
./scripts/install-latest-build.sh          # gh: latest xllama-appx for this branch
```

No model ships in the MSIX. First launch downloads the default chat model.
Vendor DLL lifecycle: [docs/vendor-lifecycle-plan.md](docs/vendor-lifecycle-plan.md).
Poll with `scripts/check-vendor-nuget-status.sh`.

`DirectML.dll`, `onnxruntime.dll`, and `onnxruntime-genai.dll` need
`<DeploymentContent>true</DeploymentContent>` or the MSIX omits them. Merge
`.onnx.data` with `scripts/merge_onnx_external_data.py` before packaging; CI
does this. See [docs/fp16-extdata-runbook.md](docs/fp16-extdata-runbook.md)
and [docs/uwp-constraints.md](docs/uwp-constraints.md).

## Versioning

`Major.Minor.Build` in `uwp/AppxManifest.xml` (`Identity` `Version`) is the
semantic version. Bump it by hand on a release, together with `CHANGELOG.md`.
CI stamps the fourth component (revision) to `github.run_number` via
`build-uwp.ps1 -BuildRevision`, so each CI package is a unique in-place
update. Local builds leave `.0`.

Current identity is `GianlucaMazza.xllama`. **1.5.0.0** changed the identity
name from `VenereLabs.xllama`, so there is no in-place update across that
boundary (new app, fresh LocalState). `scripts/deploy.sh` keeps
`APP_ID_LEGACY` for the transition. User-facing steps:
[docs/install-release.md](docs/install-release.md).

## Pull request guidelines

Branch from `main` as `feat/…`, `fix/…`, `docs/…`, `chore/…`, or `ci/…`.
Commit subjects use those conventional prefixes. Keep a PR to one change;
put unrelated infra in its own PR.

Fill `.github/pull_request_template.md`: **What & why** and **How verified**.
Before pushing, run the Linux test build, ctest, and the formatters that
touch your files. If you changed evidence or summary policy, run
`generate-benchmark-summary.py --check`. If you changed code, catalogue,
pins, or docs that `check-coherence.py` watches, run that script.

UWP edits compile on `build-uwp` only. Note in the PR when you could not
build them locally.

If you change a contract, update the owning doc in the same PR
([docs/README.md](docs/README.md) ownership table):

| Change                                        | Update                                                       |
| --------------------------------------------- | ------------------------------------------------------------ |
| Module boundaries, routing, session ownership | `docs/architecture.md`                                       |
| Training lanes or Phase 11                    | `docs/training-architecture.md` and `training/README.md`     |
| User-facing UI steps                          | `docs/using-the-app.md`                                      |
| LAN routes                                    | `docs/api-endpoint.md`                                       |
| Bench CSV schema or numbers                   | `bench/README.md`, then regenerate `docs/benchmarks.md`      |
| Release behaviour                             | `CHANGELOG.md` (and `ROADMAP.md` when a phase closes)        |
| Package identity                              | `docs/install-release.md` and this file's Versioning section |
| Repo layout agents rely on                    | this file                                                    |

## Security

- Never commit `.env`, `.pfx`, or `.cer`. They are gitignored. Test certs
  stay on the build machine.
- Xbox credentials stay in `~/.config/xllama/xbox-env`.
- The LAN API is opt-in and off unless `api.flag` is present. Protocol:
  [docs/api-endpoint.md](docs/api-endpoint.md).
- The API has no tool executor. `api_tool_execution_requested` in
  `api_policy.h` rejects `tools`, `functions`, and `tool_choice`.
- Catalogue signing uses the `XLLAMA_CATALOGUE_PRIVATE_KEY` CI secret. Do
  not copy key material into the tree.

## Debugging

- Linux: `./build/linux-test/bin/xllama-cli --help`. Training jobs fail
  closed with `--validate-train-job`. GPU probes (`--gpubw`, `--gpugemv`)
  report `d3d12_ran=false` on Linux; that is expected.
- Console log: `./scripts/deploy.sh get-log` reads `LocalState\xllama.log`.
  Crash dumps: `./scripts/deploy.sh list-dumps`. Portal details:
  [docs/device-portal.md](docs/device-portal.md).
- Headless flags (`bench.flag`, `train.flag`, `diffuse.flag`, …) replace the
  UI process. The registry is in
  [docs/architecture.md](docs/architecture.md). The LAN server does not listen
  while one of those flags is active, because that process exits first.
  `api.flag` is the server itself.
- MSIX uninstall wipes LocalState. Re-provision models
  (`scripts/provision-models.sh` or `install-latest-build.sh --provision`).
- A Linux tok/s number is not a Series S result. Numbers ship only from
  `bench/results/` through `generate-benchmark-summary.py` into
  `docs/benchmarks.md`.

## Invariants

Break these and both front ends drift:

- **One resident session.** `SessionHub` owns the loaded model for GUI and
  API. Two models do not fit the console budget.
- **One budget enforcer.** `fit_prompt` decides in tokens. A chars-per-token
  estimate may bound work; it must not decide what the user receives.
- **One sampler chain per backend.** CLI, bench, GUI, and API share
  `sampler_chain.h` or `ort_sampling.h`.
- **Single home.** A decision both surfaces make lives in one
  `include/xllama/` header.
- **Measured is not shipped.** Host smoke, then a console bench, then a
  manifest entry. Status lives in `docs/model-matrix.md`.

Platform limits (no `mmap`, no `dlopen`, no registry, no arbitrary paths,
GPU budget, per-file cap) are only in
[docs/uwp-constraints.md](docs/uwp-constraints.md). Link a section number;
do not restate the ceilings.
