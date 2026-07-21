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
  (>1550 tok) turn routes to GPU, short turns to CPU (#91 lifted for that asset);
- **settings** — the `set_routing` / `set_sampling` / `set_kv_reuse` /
  `set_taesd` / `set_system_prompt` autopilot ops are dispatched and every
  resulting value is asserted against the persisted `settings.json`. The
  baseline is seeded with the opposite of each target, so an op that silently
  does nothing fails rather than inheriting a value that already matched (needs
  `smollm2-360m-cpu-int4` in LocalState; `set_taesd` / `set_system_prompt` need
  an app build >= 1.4.0.632, the rest >= 1.4.0.606);
- **GGUF chat** — the default LFM model loads through llama.cpp and generates;
- **TAESD** — image generation completes through DirectML with the fast VAE.
  This gate swaps `vae_decoder/model.onnx` from a local cache rather than
  flipping the in-app toggle: the toggle makes the console download the asset
  from the models-v1 release, which would make the gate slower and dependent on
  console-side network. The toggle's own writer is covered by the `settings`
  gate instead;
- **API** — when included by the orchestrator, the LAN health/chat contract
  passes through `scripts/validate-api.sh`.

Run an individual gate while debugging:

```bash
./scripts/validate-console.sh routing
./scripts/validate-console.sh settings
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
