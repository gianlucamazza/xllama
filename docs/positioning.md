# Positioning

What xllama is for, and what it does not claim. Structure lives in
[architecture.md](architecture.md); numbers in [benchmarks.md](benchmarks.md);
Release status in [release-readiness.md](release-readiness.md); LAN protocol in
[api-endpoint.md](api-endpoint.md).

## Thesis

xllama is a **local inference stack** for Xbox Series S and Series X consoles.
Supported on Series S and Series X. The project measures on Series S; the Series X runs come from an independent tester. The intended uses
are **games** (NPC dialogue, agents) and **on-console AI assistants**.

The gamepad chat UI is a **demo / showcase shell** so the stack can be
exercised without a game runtime. It is not the product thesis.

## What exists today

- Dual-backend inference: `*.gguf` through llama.cpp; everything else through
  ONNX Runtime GenAI + DirectML. One resident session
  (`SessionHub` — never 2× model in RAM).
- Opt-in OpenAI-compatible **LAN endpoint** — the current integration probe
  for a PC or tool on the same network. Default OFF. Developer Mode / LAN research
  only. Not a game SDK and not a public inbound service.
- Headless benches (tok/s, memory, disk, GPU probes).
- SD-Turbo diffusion and on-device / host training lanes.

## Series X

Series X is supported from 2026-11-15, by the owner's decision, on this
evidence: an independent tester ran package 1.7.0.1495 on a Series X in
Developer Mode and generated text with `lfm25-350m` and `lfm25-1.2b-instruct`
through the LAN API
([field report](https://github.com/gianlucamazza/xllama/discussions/434),
raw runs in `bench/results/series-x-2026-11-13.csv` and
`series-x-end-to-end-2026-11-13.csv`).

What that covers and what it does not:

- **Covered:** install, model download, CPU text generation and the LAN API,
  for those two models. 143 timed runs and a one-hour run with no failed
  generation.
- **Not covered:** image generation, training, the GGUF GPU path, embeddings,
  memory limits and every model other than those two. None of these has been
  run on a Series X.
- **Not project measurements.** The device is operator-declared (rows are
  labelled `xbox-series-x-declared`) and there is no device receipt. The
  project owns no Series X. Every published figure is a Series S figure.
- **Speed.** With the app designated as a Game, the Series X runs are on par
  with a Series S measured the same way, not faster. With the App
  designation the app runs at full speed for about ten seconds after launch
  and then at about 40 % of the Game speed: the system caps an App's CPU,
  and fewer threads make it slower, not faster. That accounts for the 2.4
  times seen in the first runs. The remedy is the Game designation, which the
  install guide asks for
  ([#457](https://github.com/gianlucamazza/xllama/issues/457)).

## Honest limits

- **Developer Mode only.** Sideloaded onto a console in Developer Mode.
- **Not Microsoft-affiliated.** Independent research. Microsoft-published
  runtimes (UWP, DirectML, ORT) are used as any third party would.
- **Not an Xbox / GDK game SDK.** No AAA drop-in plugin, no ID@Xbox, no
  published engine integration. Do not claim otherwise.
- **LAN is a probe, not a product contract.** Unauthenticated,
  foreground-only, single-slot. See [api-endpoint.md](api-endpoint.md).
- **Numbers are measured.** Published tok/s come from `bench/results/` via
  [benchmarks.md](benchmarks.md). Host timings are not Series S results. Do
  not invent throughput.

## Direction

Prove local inference on the console toward game-runtime and assistant use.
The next integration surface is the LAN endpoint, not a richer chat app.
