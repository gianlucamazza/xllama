# Console validation runbook

Current on-device release gates for Xbox Series S Dev Mode. Historical experiment
procedures and measured results live in `CHANGELOG.md` and `bench/results/`.

## Prerequisites

- Install the latest `xllama-appx` unified artifact.
- Set Device Portal credentials in `~/.config/xllama/xbox-env`.
- Confirm the package is designated **Game** in Dev Home.
- Provision models after any uninstall, because uninstall removes `LocalState`.

```bash
source ~/.config/xllama/xbox-env
./scripts/provision-models.sh --all-test
```

## Official automated suite

```bash
./scripts/validate-console.sh all
```

The suite drives the live UI through autopilot and fails unless all current
hardware gates pass:

- **routing** — auto A/B with the parity-validated `-v2` DML asset: the long
  (>600 tok) turn routes to GPU, short turns to CPU (#91 lifted for that asset);
- **GGUF chat** — the default LFM model loads through llama.cpp and generates;
- **TAESD** — image generation completes through DirectML with the fast VAE;
- **API** — when included by the orchestrator, the LAN health/chat contract
  passes through `scripts/validate-api.sh`.

Run an individual gate while debugging:

```bash
./scripts/validate-console.sh routing
./scripts/validate-console.sh gguf
./scripts/validate-console.sh taesd
./scripts/validate-api.sh all
```

## Benchmark evidence

Use `scripts/bench-xbox-ort.sh` and the fixed prompts described in
[`bench/README.md`](../bench/README.md). A comparison row must come from one
atomic CSV record; never combine the best prefill, decode and RAM values from
different runs.

After adding or changing committed evidence:

```bash
python3 scripts/generate-benchmark-summary.py
python3 scripts/generate-benchmark-summary.py --check
```

Record raw CSV/JSONL output under `bench/results/`, update the appropriate
research verdict or changelog entry, and let the generated summary own the
comparison table.

## Troubleshooting

- `./scripts/deploy.sh diagnose-startup` — process state, log and crash dumps.
- `./scripts/deploy.sh get-log` — current `xllama.log`.
- Remove a stale `bench.flag` before UI/autopilot runs. It is uploaded only when
  `install-latest-build.sh --bench` is used.
- Reinstall with a higher package revision for an in-place update; a same-version
  package with different contents is rejected.
- Detailed Device Portal behavior: [device-portal.md](device-portal.md).

## Release acceptance

A hardware-sensitive change is complete only when the relevant automated gate
passes on the target console and the package version, raw evidence and outcome
are recorded. Do not revive closed historical experiments unless new evidence
changes a documented constraint.
