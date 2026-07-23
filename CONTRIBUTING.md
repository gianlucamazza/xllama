# Contributing to xllama

Thanks for helping. This is a research project targeting **Xbox Series S (Dev
Mode)**; most constraints come from the UWP/AppContainer sandbox, so read
[`AGENTS.md`](AGENTS.md) (conventions + repo map) and
[`docs/uwp-constraints.md`](docs/uwp-constraints.md) first.

## Workflow

1. Branch off `main` (`feat/…`, `fix/…`, `docs/…`, `chore/…`) and open a PR —
   CI builds run on PRs (feature branches don't build on direct push).
2. Keep PRs focused; put orthogonal infra changes in their own PR.
3. Fill in the PR template's "How verified" section.

## Before you push

```bash
cmake --preset linux-test
cmake --build build/linux-test -j"$(nproc)"
ctest --test-dir build/linux-test --output-on-failure
# Formatting gate (CI enforces it with --Werror):
clang-format -i <changed .cpp/.h files>
shellcheck scripts/*.sh   # if you touched shell
# If you touched bench CSVs or summary policy:
python3 scripts/generate-benchmark-summary.py --check
# Cross-check code / catalogue / pins / docs (no console needed):
python3 scripts/check-coherence.py
```

## Conventions

- **C++17**, RAII, `unique_ptr`; concise, no over-engineering. English for code,
  comments, and commits.
- **Formatting** is enforced (`.clang-format`, pinned `clang-format` in CI).
- **Commits**: conventional prefixes (`feat`, `fix`, `docs`, `ci`, `chore`, …).

## Documentation

Project docs follow a **single source of truth** map — see
[`docs/README.md`](docs/README.md) (principles + ownership table).

| If you change… | Update… |
| -------------- | ------- |
| Module boundaries / data flow | `docs/architecture.md` |
| Training contracts / lanes / Phase 11 | `docs/training-architecture.md` + `training/README.md` |
| User-facing UI steps | `docs/using-the-app.md` |
| LAN routes | `docs/api-endpoint.md` |
| Benchmarks / CSV schema | `bench/README.md` + regen summary if needed |
| Release behaviour | `CHANGELOG.md` (and `ROADMAP.md` if a phase closes) |
| Repo layout for agents | `AGENTS.md` |

Do **not** hand-edit generated tables in `docs/benchmarks.md` (run
`generate-benchmark-summary.py`). Prefer linking to the SSOT over copying
numbers into a second file.

Public C++ headers under `include/xllama/` should state what they own in a short
file comment (English). Prefer pure helpers testable on Linux over UWP-only
logic when the behaviour is not WinRT-specific.

## Platform notes

- **UWP code compiles only on the Windows CI** (`build-uwp`) — you usually can't
  build `uwp/` locally on Linux. Say so in the PR if you couldn't.
- **Inference backends**: the shipping Xbox artifact is unified (ORT GenAI +
  llama.cpp); Linux uses llama.cpp. UWP ORT code remains under
  `#ifdef XLLAMA_USE_ORT`; keep both unified dispatch and Linux builds working.
- **Adding a model**: usually just a `uwp/models/manifest.json` entry — check the
  size/RAM/per-file limits in [`docs/model-selection.md`](docs/model-selection.md).
- **Publishing ORT model assets**: every asset must pass the logit-parity gate
  before `gh release upload` — runbook in
  [`docs/model-selection.md`](docs/model-selection.md#publishing-ort-model-assets-models-v1--logit-parity-gate).

## Reporting issues

Use the issue templates. Include the MSIX version (or git SHA) and, for console
bugs, the `xllama.log` from Device Portal → LocalState.
