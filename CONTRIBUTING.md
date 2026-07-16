# Contributing to xllama

Thanks for helping. This is a research project targeting **Xbox Series S (Dev
Mode)**; most constraints come from the UWP/AppContainer sandbox, so read
[`AGENTS.md`](AGENTS.md) (conventions + repo map) and
[`docs/uwp-constraints.md`](docs/uwp-constraints.md) first.

## Workflow

1. Branch off `main` (`feat/…`, `fix/…`, `chore/…`) and open a PR — CI builds
   run on PRs (feature branches don't build on direct push).
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
```

## Conventions

- **C++17**, RAII, `unique_ptr`; concise, no over-engineering. English for code,
  comments, and commits.
- **Formatting** is enforced (`.clang-format`, pinned `clang-format` in CI).
- **Commits**: conventional prefixes (`feat`, `fix`, `docs`, `ci`, `chore`, …).

## Platform notes

- **UWP code compiles only on the Windows CI** (`build-uwp`) — you usually can't
  build `uwp/` locally on Linux. Say so in the PR if you couldn't.
- **Inference backends**: ORT GenAI (Xbox default) and llama.cpp (Linux + GGUF).
  UWP inference lives under `#ifdef XLLAMA_USE_ORT`; keep the Linux path building.
- **Adding a model**: usually just a `uwp/models/manifest.json` entry — check the
  size/RAM/per-file limits in [`docs/model-selection.md`](docs/model-selection.md).
- **Publishing ORT model assets**: every asset must pass the logit-parity gate
  before `gh release upload` — runbook in
  [`docs/model-selection.md`](docs/model-selection.md#publishing-ort-model-assets-models-v1--logit-parity-gate).

## Reporting issues

Use the issue templates. Include the MSIX version (or git SHA) and, for console
bugs, the `xllama.log` from Device Portal → LocalState.
